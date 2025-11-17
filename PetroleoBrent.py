# ============================================================
# REDE NEURAL -Preço do Brent
# Autor: Luiz Tiago Wilcke (LT)
# ============================================================

import os
import math
import random
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------
# 0) Configurações do experimento
# ------------------------------------------------------------
@dataclass
class Configuracoes:
    usar_csv: bool = False
    caminho_csv: Optional[str] = None  # CSV com colunas: data, preco_brent, [covariaveis...]
    coluna_data: str = "data"
    coluna_preco: str = "preco_brent"

    # Janelas do modelo
    tamanho_janela: int = 60    # L - número de dias passados
    horizonte: int = 5          # H - número de dias futuros

    # Treino
    proporcao_treino: float = 0.8
    batch: int = 128
    epocas: int = 80
    lr: float = 1e-3
    seed: int = 42

    # Arquitetura
    tamanho_oculto_lstm: int = 64
    num_camadas_lstm: int = 2
    dropout_lstm: float = 0.1

    dim_atencao: int = 64
    dim_mlp: int = 128

    # Dispositivo
    usar_cuda: bool = True

    # Outros
    mostrar_graficos: bool = True


# ------------------------------------------------------------
# 1) Fixar semente (reprodutibilidade)
# ------------------------------------------------------------
def fixar_semente(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------------------------------------------
# 2) Carregar dados reais ou gerar dados sintéticos
# ------------------------------------------------------------
def carregar_dados_brent(cfg: Configuracoes) -> pd.DataFrame:
    if cfg.usar_csv and cfg.caminho_csv is not None and os.path.exists(cfg.caminho_csv):
        df = pd.read_csv(cfg.caminho_csv)
        # normalizar nome das colunas
        df.columns = [c.lower() for c in df.columns]
        if cfg.coluna_data in df.columns:
            df[cfg.coluna_data] = pd.to_datetime(df[cfg.coluna_data])
            df = df.sort_values(cfg.coluna_data)
        df = df.reset_index(drop=True)
        return df
    else:
        # ----------------------------------------------------
        # Dados sintéticos (para testar o pipeline completo)
        # ----------------------------------------------------
        n = 1500
        datas = pd.date_range("2010-01-01", periods=n, freq="D")

        # Processo com tendência + sazonalidade + ruído
        t = np.arange(n)
        tendencia = 50 + 0.01 * t                            # leve tendência
        sazonal = 5 * np.sin(2 * np.pi * t / 250)           # ciclo ~1 ano
        choque = np.random.normal(0, 2, size=n)
        preco = tendencia + sazonal + choque
        preco = np.maximum(preco, 5)

        # Covariáveis sintéticas
        wti = preco + np.random.normal(0, 0.8, size=n)
        cambio = 3.0 + 0.0005 * t + np.random.normal(0, 0.05, size=n)
        sp500 = 2000 + 0.5 * t + 50 * np.sin(2 * np.pi * t / 180) + np.random.normal(0, 20, size=n)

        df = pd.DataFrame({
            cfg.coluna_data: datas,
            cfg.coluna_preco: preco,
            "wti": wti,
            "cambio_brlusd": cambio,
            "sp500": sp500
        })
        return df


# ------------------------------------------------------------
# 3) Construir janelas de séries temporais
# ------------------------------------------------------------
def construir_janelas(
    matriz_features: np.ndarray,
    vetor_log_preco: np.ndarray,
    tamanho_janela: int,
    horizonte: int
) -> Tuple[np.ndarray, np.ndarray]:
    X, Y = [], []
    n = len(vetor_log_preco)
    max_t = n - tamanho_janela - horizonte
    for inicio in range(max_t):
        fim = inicio + tamanho_janela
        alvo_ini = fim
        alvo_fim = fim + horizonte
        X.append(matriz_features[inicio:fim])
        Y.append(vetor_log_preco[alvo_ini:alvo_fim])
    return np.array(X), np.array(Y)


# ------------------------------------------------------------
# 4) Dataset PyTorch
# ------------------------------------------------------------
class DatasetBrent(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# ------------------------------------------------------------
# 5) Bloco de Atenção Escalar (self-attention simplificado)
# ------------------------------------------------------------
class BlocoAtencao(nn.Module):
    def __init__(self, dim_input: int, dim_atencao: int):
        super().__init__()
        self.proj_q = nn.Linear(dim_input, dim_atencao)
        self.proj_k = nn.Linear(dim_input, dim_atencao)
        self.proj_v = nn.Linear(dim_input, dim_atencao)
        self.softmax = nn.Softmax(dim=-1)
        self.proj_final = nn.Linear(dim_atencao, dim_input)

    def forward(self, x):
        # x: (batch, tempo, dim)
        Q = self.proj_q(x)   # (B, T, da)
        K = self.proj_k(x)   # (B, T, da)
        V = self.proj_v(x)   # (B, T, da)

        # scores: (B, T, T)
        scores = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(Q.shape[-1])
        A = self.softmax(scores)
        contexto = torch.matmul(A, V)  # (B, T, da)

        # Projeção de volta + residual
        contexto = self.proj_final(contexto)
        saida = x + contexto
        return saida, A


# ------------------------------------------------------------
# 6) Modelo Neural Avançado para Brent
#     - BiLSTM
#     - Self-Attention
#     - MLP
# ------------------------------------------------------------
class ModeloBrentAvancado(nn.Module):
    def __init__(self, dim_entrada: int, horizonte: int, cfg: Configuracoes):
        super().__init__()
        self.horizonte = horizonte

        self.lstm = nn.LSTM(
            input_size=dim_entrada,
            hidden_size=cfg.tamanho_oculto_lstm,
            num_layers=cfg.num_camadas_lstm,
            batch_first=True,
            dropout=cfg.dropout_lstm,
            bidirectional=True
        )

        dim_lstm_saida = 2 * cfg.tamanho_oculto_lstm

        self.bloco_atencao = BlocoAtencao(dim_lstm_saida, cfg.dim_atencao)

        self.norm1 = nn.LayerNorm(dim_lstm_saida)
        self.norm2 = nn.LayerNorm(dim_lstm_saida)

        self.ff = nn.Sequential(
            nn.Linear(dim_lstm_saida, cfg.dim_mlp),
            nn.ReLU(),
            nn.Linear(cfg.dim_mlp, dim_lstm_saida)
        )

        self.head_saida = nn.Sequential(
            nn.Linear(dim_lstm_saida, cfg.dim_mlp),
            nn.ReLU(),
            nn.Linear(cfg.dim_mlp, horizonte)
        )

    def forward(self, x):
        # x: (batch, tempo, dim_entrada)
        lstm_out, _ = self.lstm(x)   # (B, T, 2*hidden)

        # Atenção + residual
        att_out, pesos = self.bloco_atencao(lstm_out)
        x1 = self.norm1(att_out)

        # Feed-forward tipo Transformer + residual
        ff_out = self.ff(x1)
        x2 = self.norm2(x1 + ff_out)

        # Agregação temporal (média global)
        contexto = torch.mean(x2, dim=1)  # (B, 2*hidden)

        saida = self.head_saida(contexto)  # (B, horizonte)
        return saida, pesos


# ------------------------------------------------------------
# 7) Função de treino
# ------------------------------------------------------------
def treinar_modelo(
    modelo: nn.Module,
    loader_treino: DataLoader,
    loader_val: DataLoader,
    dispositivo: torch.device,
    cfg: Configuracoes
):
    criterio = nn.MSELoss()
    otimizador = torch.optim.Adam(modelo.parameters(), lr=cfg.lr)

    historico_treino = []
    historico_val = []

    melhor_val = float("inf")
    melhor_estado = None
    paciencia = 10
    pac = 0

    for epoca in range(1, cfg.epocas + 1):
        modelo.train()
        perdas_treino = []

        for batch_x, batch_y in loader_treino:
            batch_x = batch_x.to(dispositivo)
            batch_y = batch_y.to(dispositivo)

            otimizador.zero_grad()
            saida, _ = modelo(batch_x)  # em log-preço
            perda = criterio(saida, batch_y)
            perda.backward()
            otimizador.step()

            perdas_treino.append(perda.item())

        perda_media_treino = float(np.mean(perdas_treino))

        # Validação
        modelo.eval()
        perdas_val = []
        with torch.no_grad():
            for batch_x, batch_y in loader_val:
                batch_x = batch_x.to(dispositivo)
                batch_y = batch_y.to(dispositivo)
                saida, _ = modelo(batch_x)
                perda = criterio(saida, batch_y)
                perdas_val.append(perda.item())

        perda_media_val = float(np.mean(perdas_val))

        historico_treino.append(perda_media_treino)
        historico_val.append(perda_media_val)

        print(
            f"[Época {epoca:03d}] "
            f"Perda treino (MSE log): {perda_media_treino:.6f}  |  "
            f"Perda val (MSE log): {perda_media_val:.6f}"
        )

        # Early stopping simples
        if perda_media_val < melhor_val - 1e-5:
            melhor_val = perda_media_val
            melhor_estado = modelo.state_dict()
            pac = 0
        else:
            pac += 1
            if pac >= paciencia:
                print("Early stopping acionado.")
                break

    if melhor_estado is not None:
        modelo.load_state_dict(melhor_estado)

    return historico_treino, historico_val


# ------------------------------------------------------------
# 8) Avaliação e gráficos
# ------------------------------------------------------------
def avaliar_modelo(
    modelo: nn.Module,
    loader: DataLoader,
    scaler_preco: StandardScaler,
    dispositivo: torch.device,
    horizonte: int
):
    modelo.eval()
    previsoes = []
    verdadeiros = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(dispositivo)
            batch_y = batch_y.to(dispositivo)
            saida, _ = modelo(batch_x)  # (B, H)

            # Desnormalizar log-preço -> voltar a preço
            saida_np = saida.cpu().numpy()
            y_np = batch_y.cpu().numpy()

            # Como o scaler foi aplicado no log-preço, precisamos inverter
            # mas aqui usaremos diretamente exp() do log sem reusar scaler,
            # assumindo que alvo já é log-preço bruto (sem normalização).
            # Se você escalar o log-preço, adapte conforme sua pipeline.
            # Neste exemplo, NÃO escalamos o alvo para simplificar:
            previsoes.append(np.exp(saida_np))
            verdadeiros.append(np.exp(y_np))

    previsoes = np.concatenate(previsoes, axis=0)   # (N, H)
    verdadeiros = np.concatenate(verdadeiros, axis=0)

    # Métricas (para o primeiro horizonte como indicativo)
    y_true_1 = verdadeiros[:, 0]
    y_pred_1 = previsoes[:, 0]

    rmse = np.sqrt(np.mean((y_true_1 - y_pred_1)**2))
    mape = np.mean(np.abs((y_true_1 - y_pred_1) / y_true_1)) * 100

    print(f"\nMétricas (horizonte 1 dia):")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAPE: {mape:.2f}%")

    return previsoes, verdadeiros


# ------------------------------------------------------------
# 9) Pipeline principal
# ------------------------------------------------------------
def main():
    cfg = Configuracoes()
    fixar_semente(cfg.seed)

    dispositivo = torch.device("cuda" if (cfg.usar_cuda and torch.cuda.is_available()) else "cpu")
    print("Usando dispositivo:", dispositivo)

    # 9.1 Carregar dados
    df = carregar_dados_brent(cfg)
    print("Colunas disponíveis no DataFrame:", df.columns.tolist())

    # 9.2 Montar matriz de features e log-preço
    #    -> sempre incluir log(preço) como primeira feature
    preco = df[cfg.coluna_preco].values.astype(float)
    log_preco = np.log(preco)

    # Covariáveis: todas as colunas numéricas exceto data
    col_excluir = [cfg.coluna_data] if cfg.coluna_data in df.columns else []
    col_excluir.append(cfg.coluna_preco)

    col_covs = [c for c in df.columns if c not in col_excluir]
    matriz_covs = df[col_covs].values.astype(float) if len(col_covs) > 0 else np.empty((len(df), 0))

    # Matriz de entrada: concatena log_preco e covariáveis
    matriz_features = np.concatenate(
        [log_preco.reshape(-1, 1), matriz_covs],
        axis=1
    )

    # Normalizar features (não o alvo)
    scaler_features = StandardScaler()
    matriz_features_norm = scaler_features.fit_transform(matriz_features)

    # Aqui, por simplicidade, vamos usar como alvo o log_preco sem normalizar:
    X, Y = construir_janelas(
        matriz_features_norm,
        log_preco,
        cfg.tamanho_janela,
        cfg.horizonte
    )

    print("Formato de X:", X.shape, "Formato de Y:", Y.shape)

    n_amostras = X.shape[0]
    n_treino = int(n_amostras * cfg.proporcao_treino)

    X_treino, X_val = X[:n_treino], X[n_treino:]
    Y_treino, Y_val = Y[:n_treino], Y[n_treino:]

    ds_treino = DatasetBrent(X_treino, Y_treino)
    ds_val = DatasetBrent(X_val, Y_val)

    loader_treino = DataLoader(ds_treino, batch_size=cfg.batch, shuffle=True)
    loader_val = DataLoader(ds_val, batch_size=cfg.batch, shuffle=False)

    # 9.3 Instanciar modelo
    dim_entrada = X.shape[2]
    modelo = ModeloBrentAvancado(dim_entrada, cfg.horizonte, cfg).to(dispositivo)
    print(modelo)

    # 9.4 Treinar
    hist_treino, hist_val = treinar_modelo(
        modelo,
        loader_treino,
        loader_val,
        dispositivo,
        cfg
    )

    # 9.5 Avaliar no conjunto de validação
    previsoes, verdadeiros = avaliar_modelo(
        modelo,
        loader_val,
        scaler_preco=None,
        dispositivo=dispositivo,
        horizonte=cfg.horizonte
    )

    # 9.6 Gráficos (só para o primeiro horizonte)
    if cfg.mostrar_graficos and previsoes.shape[0] > 0:
        y_true_1 = verdadeiros[:, 0]
        y_pred_1 = previsoes[:, 0]

        plt.figure(figsize=(10, 5))
        plt.plot(y_true_1, label="Real (h=1)", linewidth=1.5)
        plt.plot(y_pred_1, label="Previsto (h=1)", linewidth=1.0)
        plt.title("Preço do Brent — Real vs Previsto (Horizonte 1 dia)")
        plt.xlabel("Índice da janela de validação")
        plt.ylabel("Preço")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # Curva de perdas
        plt.figure(figsize=(8, 4))
        plt.plot(hist_treino, label="Treino (MSE log)")
        plt.plot(hist_val, label="Validação (MSE log)")
        plt.xlabel("Época")
        plt.ylabel("Perda")
        plt.title("Histórico de Treino")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
