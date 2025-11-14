# ============================================================
# MODELO NEURAL-ESTOCÁSTICO — Preço da NVDA (NVIDIA)
# Autor: Luiz Tiago Wilcke (LT)
# ============================================================
# - Baixa dados da NVDA via Yahoo Finance
# - Extrai features estocásticas baseadas em SDE (GBM)
# - Treina LSTM bidirecional + Autoatenção
# - Avalia e plota resultados
# ============================================================

import warnings, math, random
from dataclasses import dataclass
from typing import Tuple, Dict, Union

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8")


# ------------------------------------------------------------
# 1. Configurações gerais
# ------------------------------------------------------------
@dataclass
class Configuracoes:
    ticker: str = "NVDA"
    data_inicio: str = "2010-01-01"

    janela_entrada: int = 60       # número de dias usados como entrada
    horizonte: int = 5             # prever 5 dias úteis à frente
    proporcao_treino: float = 0.8

    epocas: int = 60
    batch: int = 64
    taxa_aprendizado: float = 1e-3
    decaimento_peso: float = 1e-4  # weight decay (regularização L2)

    tamanho_oculto_lstm: int = 64
    camadas_lstm: int = 2
    dropout: float = 0.15

    usar_cuda: bool = True
    seed: int = 42


CFG = Configuracoes()


# ------------------------------------------------------------
# 2. Funções auxiliares de reprodutibilidade e normalização
# ------------------------------------------------------------
def fixar_semente(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def padronizar_colunas(matriz: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Padroniza colunas: (x - média) / desvio.
    Retorna matriz_normalizada, medias, desvios.
    """
    medias = matriz.mean(axis=0)
    desvios = matriz.std(axis=0)
    desvios[desvios == 0] = 1.0
    matriz_norm = (matriz - medias) / desvios
    return matriz_norm, medias, desvios


# ------------------------------------------------------------
# 3. Download dos dados e construção das features SDE
# ------------------------------------------------------------
def baixar_fechamento_ajustado(cfg: Configuracoes) -> pd.Series:
    print(f"[INFO] Baixando dados de {cfg.ticker} do Yahoo Finance...")
    dados = yf.download(cfg.ticker, start=cfg.data_inicio, auto_adjust=False, progress=True)

    if dados is None or dados.empty:
        raise RuntimeError("Nenhum dado retornado pelo Yahoo Finance.")

    # Garante que será uma Series, não DataFrame
    if "Adj Close" in dados.columns:
        serie = dados["Adj Close"].astype(float).copy()
    elif "Close" in dados.columns:
        print("[AVISO] Coluna 'Adj Close' não encontrada; usando 'Close'.")
        serie = dados["Close"].astype(float).copy()
    else:
        raise RuntimeError("Colunas 'Adj Close' e 'Close' não encontradas no DataFrame de preços.")

    # Remove NaN
    serie = serie.dropna()

    if serie.empty:
        raise RuntimeError("Série de preços vazia após remoção de valores ausentes.")

    print(f"[INFO] Dados obtidos: {len(serie)} observações após limpeza.")
    return serie


def _forcar_series(serie_precos: Union[pd.Series, pd.DataFrame, np.ndarray, list, float, int]) -> pd.Series:
    """
    Garante que 'serie_precos' vire uma pd.Series 1D com índice bem definido.
    Isso evita o erro 'If using all scalar values, you must pass an index'.
    """
    # Já é Series
    if isinstance(serie_precos, pd.Series):
        s = serie_precos.copy()
    # Se for DataFrame, pega a primeira coluna numérica ou a coluna 'preco'
    elif isinstance(serie_precos, pd.DataFrame):
        if "preco" in serie_precos.columns:
            s = serie_precos["preco"]
        else:
            s = serie_precos.iloc[:, 0]
    else:
        # np.ndarray, list, escalar etc.
        s = pd.Series(serie_precos)

    # Converte para float e ajusta índice (caso tenha vindo sem)
    s = s.astype(float)
    if s.index is None or not isinstance(s.index, (pd.DatetimeIndex, pd.RangeIndex, pd.Index)):
        s.index = pd.RangeIndex(len(s))

    return s


def construir_features_estocasticas(
    serie_precos: Union[pd.Series, pd.DataFrame, np.ndarray, list, float, int]
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Constrói features inspiradas no SDE de Black–Scholes:
      - log_preco
      - retorno_log
      - drift_instantaneo (μ_t aproximado)
      - sigma_instantaneo (σ_t aproximado)
      - variância instantânea σ_t^2
      - termo de Itô (μ_t - 0.5 σ_t^2)
      - índice de tempo normalizado
    """
    # >>> AQUI está a robustez que evita o seu erro <<<
    s = _forcar_series(serie_precos)
    s = s.dropna()

    if s.empty:
        raise RuntimeError("Série de preços vazia dentro de construir_features_estocasticas.")

    df = pd.DataFrame({"preco": s})
    df["log_preco"] = np.log(df["preco"])

    # Retornos logarítmicos (d log S_t)
    df["retorno_log"] = df["log_preco"].diff()

    # Janela para estimar μ_t e σ_t
    janela_rolagem = 20
    delta_t = 1.0 / 252.0  # 1 dia útil ~ 1/252 ano

    # Estimativas móveis
    df["media_retorno"] = df["retorno_log"].rolling(janela_rolagem).mean()
    df["vol_retorno"] = df["retorno_log"].rolling(janela_rolagem).std()

    # Evitar NaNs iniciais
    df = df.dropna()

    if df.empty:
        raise RuntimeError("DataFrame de features ficou vazio após o dropna (poucos dados para janela).")

    # Drift e volatilidade instantâneos
    df["drift_instantaneo"] = df["media_retorno"] / delta_t
    df["sigma_instantaneo"] = df["vol_retorno"] / math.sqrt(delta_t)

    # Variância instantânea
    df["variancia_instantanea"] = df["sigma_instantaneo"] ** 2

    # Termo de Itô: μ_t - 0.5 σ_t^2 (drift em d log S_t)
    df["ito_drift_log"] = df["drift_instantaneo"] - 0.5 * df["variancia_instantanea"]

    # Índice de tempo normalizado [0, 1]
    df["indice_tempo"] = np.linspace(0.0, 1.0, len(df))

    # Guardar alguns momentos básicos para diagnóstico
    estatisticas = {
        "media_retorno_diario": float(df["retorno_log"].mean()),
        "vol_retorno_diario": float(df["retorno_log"].std())
    }

    return df, estatisticas


def construir_janelas(
    matriz_features: np.ndarray,
    alvo_norm: np.ndarray,
    janela: int,
    horizonte: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Constrói janelas deslizantes:
      X[i]  = [t-janela+1, ..., t] (features)
      y[i]  = alvo_norm[t + horizonte - 1]
      idx[i] = índice temporal do alvo (para mapear datas depois)
    """
    X, y, idx = [], [], []
    n = len(alvo_norm)
    for t in range(janela, n - horizonte + 1):
        X.append(matriz_features[t - janela:t, :])
        y.append(alvo_norm[t + horizonte - 1])
        idx.append(t + horizonte - 1)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(idx, dtype=np.int64)


# ------------------------------------------------------------
# 4. Dataset PyTorch
# ------------------------------------------------------------
class SerieTemporalNvidia(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# ------------------------------------------------------------
# 5. Bloco de autoatenção (single-head)
# ------------------------------------------------------------
class AtencaoSelf(nn.Module):
    """
    Autoatenção simples:
      - Q = W_q h_t
      - K = W_k h_t
      - V = W_v h_t
      - pesos = softmax(Q K^T / sqrt(d))
      - contexto = média temporal de (pesos V)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor):
        # x: (batch, tempo, dim)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        escala = math.sqrt(x.size(-1))
        scores = torch.matmul(Q, K.transpose(1, 2)) / escala  # (B, T, T)
        pesos = torch.softmax(scores, dim=-1)
        contexto = torch.matmul(pesos, V)  # (B, T, dim)

        # pooling global: média sobre o eixo temporal
        contexto_global = contexto.mean(dim=1)  # (B, dim)
        return contexto_global, pesos


# ------------------------------------------------------------
# 6. Modelo Neural-Estocástico (LSTM + Atenção)
# ------------------------------------------------------------
class ModeloNeuralEstocasticoNVDA(nn.Module):
    def __init__(
        self,
        num_caracteristicas: int,
        tamanho_oculto: int,
        camadas_lstm: int,
        dropout: float
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=num_caracteristicas,
            hidden_size=tamanho_oculto,
            num_layers=camadas_lstm,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if camadas_lstm > 1 else 0.0
        )

        dim_lstm = 2 * tamanho_oculto  # bidirecional
        self.atencao = AtencaoSelf(dim=dim_lstm)
        self.dropout = nn.Dropout(dropout)

        self.regressor = nn.Sequential(
            nn.Linear(dim_lstm, dim_lstm),
            nn.ReLU(),
            nn.Linear(dim_lstm, 1)
        )

    def forward(self, x: torch.Tensor):
        # x: (B, T, F)
        saida_lstm, _ = self.lstm(x)        # (B, T, 2H)
        contexto, pesos = self.atencao(saida_lstm)  # contexto: (B, 2H)
        contexto = self.dropout(contexto)
        out = self.regressor(contexto)      # (B, 1)
        return out.squeeze(-1), pesos       # (B,), (B, T, T)


# ------------------------------------------------------------
# 7. Rotina de treino
# ------------------------------------------------------------
def treinar_modelo(
    modelo: nn.Module,
    loader_treino: DataLoader,
    loader_val: DataLoader,
    dispositivo: torch.device,
    cfg: Configuracoes
):
    criterio = nn.MSELoss()
    otimizador = torch.optim.AdamW(
        modelo.parameters(),
        lr=cfg.taxa_aprendizado,
        weight_decay=cfg.decaimento_peso
    )

    historico = {"epoca": [], "loss_treino": [], "loss_val": []}
    melhor_loss = float("inf")
    melhor_estado = None

    for epoca in range(1, cfg.epocas + 1):
        modelo.train()
        perdas_treino = []

        for lotes, alvos in loader_treino:
            lotes = lotes.to(dispositivo)
            alvos = alvos.to(dispositivo)

            otimizador.zero_grad()
            saida, _ = modelo(lotes)
            loss = criterio(saida, alvos)
            loss.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
            otimizador.step()
            perdas_treino.append(loss.item())

        loss_treino_med = float(np.mean(perdas_treino)) if perdas_treino else float("nan")

        modelo.eval()
        perdas_val = []
        with torch.no_grad():
            for lotes, alvos in loader_val:
                lotes = lotes.to(dispositivo)
                alvos = alvos.to(dispositivo)
                saida, _ = modelo(lotes)
                loss = criterio(saida, alvos)
                perdas_val.append(loss.item())
        loss_val_med = float(np.mean(perdas_val)) if perdas_val else float("nan")

        historico["epoca"].append(epoca)
        historico["loss_treino"].append(loss_treino_med)
        historico["loss_val"].append(loss_val_med)

        print(
            f"[EPOCA {epoca:03d}] "
            f"Loss treino = {loss_treino_med:.6f} | "
            f"Loss validação = {loss_val_med:.6f}"
        )

        if loss_val_med < melhor_loss:
            melhor_loss = loss_val_med
            melhor_estado = modelo.state_dict()

    if melhor_estado is not None:
        modelo.load_state_dict(melhor_estado)

    return historico


# ------------------------------------------------------------
# 8. Execução completa do pipeline
# ------------------------------------------------------------
def executar_pipeline(cfg: Configuracoes):
    fixar_semente(cfg.seed)

    dispositivo = torch.device(
        "cuda" if (torch.cuda.is_available() and cfg.usar_cuda) else "cpu"
    )
    print(f"[INFO] Dispositivo em uso: {dispositivo}")

    # 1) Dados
    serie = baixar_fechamento_ajustado(cfg)
    # Aqui garantimos que vai sempre entrar algo "saudável"
    df_feat, estat = construir_features_estocasticas(serie)

    print("[INFO] Estatísticas básicas da série de retornos:")
    print(f"   Média retorno diário: {estat['media_retorno_diario']:.6f}")
    print(f"   Volatilidade diária:  {estat['vol_retorno_diario']:.6f}")

    # 2) Matriz de features e alvo (log_preco)
    colunas_features = [
        "log_preco",
        "retorno_log",
        "drift_instantaneo",
        "sigma_instantaneo",
        "variancia_instantanea",
        "ito_drift_log",
        "indice_tempo"
    ]

    matriz_features = df_feat[colunas_features].values
    alvo_log = df_feat["log_preco"].values

    # Normalizar features e alvo
    matriz_features_norm, medias_feat, desvios_feat = padronizar_colunas(matriz_features)
    alvo_norm, media_alvo, desvio_alvo = padronizar_colunas(alvo_log.reshape(-1, 1))
    alvo_norm = alvo_norm.squeeze(-1)

    # 3) Construir janelas
    X, y, idx_alvo = construir_janelas(
        matriz_features_norm,
        alvo_norm,
        cfg.janela_entrada,
        cfg.horizonte
    )

    n_total = len(X)
    if n_total < 100:
        raise RuntimeError("Poucos dados para treinar; ajuste data_inicio ou parâmetros.")

    n_treino = int(n_total * cfg.proporcao_treino)

    X_treino, X_teste = X[:n_treino], X[n_treino:]
    y_treino, y_teste = y[:n_treino], y[n_treino:]
    idx_treino, idx_teste = idx_alvo[:n_treino], idx_alvo[n_treino:]

    ds_treino = SerieTemporalNvidia(X_treino, y_treino)
    ds_teste = SerieTemporalNvidia(X_teste, y_teste)

    loader_treino = DataLoader(ds_treino, batch_size=cfg.batch, shuffle=True, drop_last=False)
    loader_teste = DataLoader(ds_teste, batch_size=cfg.batch, shuffle=False, drop_last=False)

    # 4) Modelo
    num_caracteristicas = X.shape[-1]
    modelo = ModeloNeuralEstocasticoNVDA(
        num_caracteristicas=num_caracteristicas,
        tamanho_oculto=cfg.tamanho_oculto_lstm,
        camadas_lstm=cfg.camadas_lstm,
        dropout=cfg.dropout
    ).to(dispositivo)

    print(modelo)

    # 5) Treino
    historico = treinar_modelo(modelo, loader_treino, loader_teste, dispositivo, cfg)

    # 6) Avaliação no conjunto de teste
    modelo.eval()
    preds_norm = []
    reais_norm = []

    with torch.no_grad():
        for lotes, alvos in loader_teste:
            lotes = lotes.to(dispositivo)
            saida, _ = modelo(lotes)
            preds_norm.append(saida.cpu().numpy())
            reais_norm.append(alvos.cpu().numpy())

    preds_norm = np.concatenate(preds_norm)
    reais_norm = np.concatenate(reais_norm)

    # Voltar para escala de log-preço e depois preço
    log_pred = preds_norm * desvio_alvo + media_alvo
    log_real = reais_norm * desvio_alvo + media_alvo

    preco_pred = np.exp(log_pred)
    preco_real = np.exp(log_real)

    # Datas correspondentes aos alvos
    datas = df_feat.index.values[idx_teste]

    # Métricas
    rmse_preco = float(np.sqrt(np.mean((preco_pred - preco_real) ** 2)))
    mae_preco = float(np.mean(np.abs(preco_pred - preco_real)))

    print("\n================== RESULTADOS ==================")
    print(f"Ticker:                        {cfg.ticker}")
    print(f"Janela de entrada (dias):     {cfg.janela_entrada}")
    print(f"Horizonte de previsão (dias): {cfg.horizonte}")
    print("------------------------------------------------")
    print(f"RMSE (preço, USD):            {rmse_preco:,.6f}")
    print(f"MAE  (preço, USD):            {mae_preco:,.6f}")
    print("================================================\n")

    # 7) Gráficos
    # 7.1 Curva de perda
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(historico["epoca"], historico["loss_treino"], label="Treino")
    ax1.plot(historico["epoca"], historico["loss_val"], label="Validação", linestyle="--")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Loss MSE (log-preço normalizado)")
    ax1.set_title("Evolução da função de perda")
    ax1.legend()
    ax1.grid(True)

    # 7.2 Preço real vs previsto (teste)
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(datas, preco_real, label="Preço real (teste)")
    ax2.plot(datas, preco_pred, label="Preço previsto (teste)")
    ax2.set_xlabel("Data")
    ax2.set_ylabel("Preço (USD)")
    ax2.set_title(f"{cfg.ticker} — Real vs Previsto (h = {cfg.horizonte} dias)")
    ax2.legend()
    ax2.grid(True)

    # 7.3 Resíduos
    residuos = preco_pred - preco_real
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.hist(residuos, bins=40, edgecolor="k", alpha=0.7)
    ax3.set_title("Distribuição dos resíduos (preço previsto - real)")
    ax3.set_xlabel("Resíduo (USD)")
    ax3.set_ylabel("Frequência")
    ax3.grid(True)

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 9. Ponto de entrada
# ------------------------------------------------------------
if __name__ == "__main__":
    executar_pipeline(CFG)
