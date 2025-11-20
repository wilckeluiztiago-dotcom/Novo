# ============================================================
# MODELO NEURAL-ESTOCÁSTICO — PREVISÃO USD/BRL
# Autor: Luiz Tiago Wilcke (LT)
# ============================================================
# Ideia:
#   - Baixa histórico USD/BRL do Yahoo Finance (yfinance)
#   - Cria features estocásticas a partir de um SDE (GBM)
#   - Treina rede neural LSTM + atenção para prever h passos
#   - Gera métricas e gráficos
#
# Observação:
#   Tudo em um único arquivo para facilitar portfólio/GitHub.
# ============================================================

import os, math, random, argparse, warnings
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# Dependências opcionais
# ------------------------------------------------------------
try:
    import yfinance as yf
except Exception:
    yf = None

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except Exception as e:
    raise RuntimeError(
        "PyTorch não encontrado. Instale com:\n"
        "pip install torch --index-url https://download.pytorch.org/whl/cpu"
    ) from e


# ============================================================
# 1. Configurações
# ============================================================
@dataclass
class Configuracoes:
    ticker: str = "BRL=X"          # USD/BRL no Yahoo Finance
    data_inicio: str = "2005-01-01"
    data_fim: str = None

    janela_entrada: int = 60       # lookback
    horizonte: int = 5             # h passos à frente
    tamanho_treino: float = 0.8

    epocas: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    hidden: int = 64
    camadas: int = 2
    dropout: float = 0.2

    seed: int = 42
    usar_gpu: bool = False

    # Features estocásticas
    janela_vol: int = 21           # ~1 mês
    n_caminhos_mc: int = 50        # Monte Carlo paths
    passos_mc: int = 21            # horizonte simulado
    dt_mc: float = 1/252


def setar_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# 2. Dados
# ============================================================
def baixar_dados(cfg: Configuracoes) -> pd.DataFrame:
    if yf is None:
        print("yfinance não disponível. Usando dados sintéticos.")
        return gerar_dados_sinteticos()

    df = yf.download(cfg.ticker, start=cfg.data_inicio, end=cfg.data_fim, progress=False)
    if df is None or len(df) < 200:
        print("Falha no download ou poucos dados. Usando série sintética.")
        return gerar_dados_sinteticos()

    df = df.rename(columns=str.lower)
    df = df[["close"]].dropna()
    df.index = pd.to_datetime(df.index)
    return df


def gerar_dados_sinteticos(n: int = 2500, S0: float = 5.0, mu: float = 0.02, sigma: float = 0.15):
    dt = 1/252
    eps = np.random.randn(n) * math.sqrt(dt)
    S = [S0]
    for e in eps:
        S.append(S[-1] + mu*S[-1]*dt + sigma*S[-1]*e)
    datas = pd.date_range("2015-01-01", periods=n+1, freq="B")
    return pd.DataFrame({"close": S}, index=datas)


# ============================================================
# 3. Features estocásticas baseadas em SDE (GBM)
# ============================================================
def extrair_features_estocasticas(df: pd.DataFrame, cfg: Configuracoes) -> pd.DataFrame:
    preco = df["close"].values
    logret = np.diff(np.log(preco), prepend=np.log(preco[0]))

    # drift e volatilidade rolling
    mu_roll = pd.Series(logret).rolling(cfg.janela_vol).mean().fillna(0.0).values
    sig_roll = pd.Series(logret).rolling(cfg.janela_vol).std().fillna(0.0).values

    # volatilidade realizada (EWMA)
    lam = 0.94
    var_ewma = np.zeros_like(logret)
    for t in range(1, len(logret)):
        var_ewma[t] = lam*var_ewma[t-1] + (1-lam)*(logret[t]**2)
    vol_ewma = np.sqrt(var_ewma)

    # Feature Monte Carlo: média e quantis do preço futuro via GBM
    mc_media = np.zeros_like(preco)
    mc_p10 = np.zeros_like(preco)
    mc_p90 = np.zeros_like(preco)

    for t in range(len(preco)):
        mu_t = mu_roll[t]
        sig_t = max(sig_roll[t], 1e-8)
        S_t = preco[t]

        caminhos_finais = []
        for _ in range(cfg.n_caminhos_mc):
            S = S_t
            for _ in range(cfg.passos_mc):
                z = np.random.randn()
                S = S * math.exp((mu_t - 0.5*sig_t**2)*cfg.dt_mc + sig_t*math.sqrt(cfg.dt_mc)*z)
            caminhos_finais.append(S)

        caminhos_finais = np.array(caminhos_finais)
        mc_media[t] = caminhos_finais.mean()
        mc_p10[t] = np.quantile(caminhos_finais, 0.10)
        mc_p90[t] = np.quantile(caminhos_finais, 0.90)

    feat = pd.DataFrame({
        "close": preco,
        "logret": logret,
        "mu_roll": mu_roll,
        "sig_roll": sig_roll,
        "vol_ewma": vol_ewma,
        "mc_media": mc_media,
        "mc_p10": mc_p10,
        "mc_p90": mc_p90
    }, index=df.index)

    feat = feat.replace([np.inf, -np.inf], np.nan).fillna(method="bfill").fillna(0.0)
    return feat


# ============================================================
# 4. Dataset supervisionado
# ============================================================
def normalizar_treino_teste(X: np.ndarray, split: int):
    mu = X[:split].mean(axis=0, keepdims=True)
    sig = X[:split].std(axis=0, keepdims=True) + 1e-8
    return (X - mu)/sig, mu, sig


class SerieTemporalDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


def montar_janelas(feat: pd.DataFrame, cfg: Configuracoes) -> Tuple[np.ndarray, np.ndarray]:
    cols_X = ["close","logret","mu_roll","sig_roll","vol_ewma","mc_media","mc_p10","mc_p90"]
    X_raw = feat[cols_X].values
    y_raw = feat["close"].values

    X, y = [], []
    for i in range(cfg.janela_entrada, len(feat) - cfg.horizonte):
        X.append(X_raw[i-cfg.janela_entrada:i])
        # previsão multi-step (vetor de tamanho horizonte)
        y.append(y_raw[i:i+cfg.horizonte])

    return np.array(X), np.array(y)


# ============================================================
# 5. Modelo LSTM + Atenção
# ============================================================
class AtencaoTemporal(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.W = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, 1, bias=False)

    def forward(self, h_seq):
        # h_seq: (B,T,H)
        score = self.v(torch.tanh(self.W(h_seq)))  # (B,T,1)
        pesos = torch.softmax(score, dim=1)        # (B,T,1)
        contexto = (pesos * h_seq).sum(dim=1)      # (B,H)
        return contexto, pesos


class ModeloNeuralEstocastico(nn.Module):
    def __init__(self, n_feat: int, hidden: int, camadas: int, dropout: float, horizonte: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_feat,
            hidden_size=hidden,
            num_layers=camadas,
            batch_first=True,
            dropout=dropout if camadas > 1 else 0.0,
            bidirectional=True
        )
        self.att = AtencaoTemporal(hidden*2)
        self.out = nn.Sequential(
            nn.Linear(hidden*2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, horizonte)
        )

    def forward(self, x):
        h_seq, _ = self.lstm(x)
        contexto, pesos = self.att(h_seq)
        yhat = self.out(contexto)
        return yhat, pesos


# ============================================================
# 6. Treino e avaliação
# ============================================================
def treinar(modelo, dl_treino, dl_valid, cfg: Configuracoes, device):
    opt = torch.optim.Adam(modelo.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    historico = {"loss_treino": [], "loss_valid": []}

    for ep in range(1, cfg.epocas+1):
        modelo.train()
        loss_tr = 0.0
        for Xb, yb in dl_treino:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            yhat, _ = modelo(Xb)
            loss = loss_fn(yhat, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            opt.step()
            loss_tr += loss.item()

        modelo.eval()
        loss_va = 0.0
        with torch.no_grad():
            for Xb, yb in dl_valid:
                Xb, yb = Xb.to(device), yb.to(device)
                yhat, _ = modelo(Xb)
                loss_va += loss_fn(yhat, yb).item()

        loss_tr /= max(len(dl_treino), 1)
        loss_va /= max(len(dl_valid), 1)
        historico["loss_treino"].append(loss_tr)
        historico["loss_valid"].append(loss_va)

        print(f"Época {ep:02d}/{cfg.epocas} | Loss treino: {loss_tr:.6f} | Loss valid: {loss_va:.6f}")

    return historico


def prever(modelo, dl, device):
    modelo.eval()
    preds, alvos = [], []
    with torch.no_grad():
        for Xb, yb in dl:
            Xb = Xb.to(device)
            yhat, _ = modelo(Xb)
            preds.append(yhat.cpu().numpy())
            alvos.append(yb.numpy())
    return np.vstack(preds), np.vstack(alvos)


def metricas(yhat, ytrue):
    # yhat/ytrue: (N, horizonte)
    mse = np.mean((yhat - ytrue)**2)
    mae = np.mean(np.abs(yhat - ytrue))
    mape = np.mean(np.abs((ytrue - yhat) / (ytrue + 1e-8))) * 100
    return {"MSE": mse, "MAE": mae, "MAPE(%)": mape}


# ============================================================
# 7. Plot
# ============================================================
def plotar_resultados(historico, yhat, ytrue, datas_teste, cfg: Configuracoes, pasta_saida="saidas"):
    os.makedirs(pasta_saida, exist_ok=True)

    # curva de loss
    plt.figure()
    plt.plot(historico["loss_treino"], label="treino")
    plt.plot(historico["loss_valid"], label="valid")
    plt.title("Loss por época")
    plt.xlabel("época")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(pasta_saida, "loss.png"), dpi=150)

    # previsão do 1º passo (t+1)
    plt.figure()
    plt.plot(datas_teste, ytrue[:,0], label="real")
    plt.plot(datas_teste, yhat[:,0], label="previsto t+1")
    plt.title("Previsão USD/BRL (1 passo à frente)")
    plt.xlabel("data")
    plt.ylabel("preço")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(pasta_saida, "previsao_t1.png"), dpi=150)

    # fan chart multi-step
    plt.figure()
    plt.plot(datas_teste, ytrue[:,0], label="real")
    for k in range(cfg.horizonte):
        plt.plot(datas_teste, yhat[:,k], label=f"prev t+{k+1}", alpha=0.7)
    plt.title("Previsão multi-step USD/BRL")
    plt.xlabel("data")
    plt.ylabel("preço")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(pasta_saida, "previsao_multistep.png"), dpi=150)

    print(f"Gráficos salvos em: {pasta_saida}/")


# ============================================================
# 8. Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Modelo Neural-Estocástico USD/BRL")
    parser.add_argument("--ticker", type=str, default="BRL=X")
    parser.add_argument("--horizonte", type=int, default=5)
    parser.add_argument("--janela", type=int, default=60)
    parser.add_argument("--epocas", type=int, default=30)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    cfg = Configuracoes(
        ticker=args.ticker,
        horizonte=args.horizonte,
        janela_entrada=args.janela,
        epocas=args.epocas,
        usar_gpu=args.gpu
    )

    setar_seed(cfg.seed)

    device = torch.device("cuda" if (cfg.usar_gpu and torch.cuda.is_available()) else "cpu")
    print("Device:", device)

    df = baixar_dados(cfg)
    feat = extrair_features_estocasticas(df, cfg)
    X, y = montar_janelas(feat, cfg)

    split = int(len(X)*cfg.tamanho_treino)
    Xn, muX, sigX = normalizar_treino_teste(X.reshape(len(X), -1), split)
    Xn = Xn.reshape(X.shape)

    yn, muy, sigy = normalizar_treino_teste(y, split)

    X_treino, X_teste = Xn[:split], Xn[split:]
    y_treino, y_teste = yn[:split], yn[split:]

    ds_treino = SerieTemporalDataset(X_treino, y_treino)
    ds_teste  = SerieTemporalDataset(X_teste,  y_teste)

    dl_treino = DataLoader(ds_treino, batch_size=cfg.batch_size, shuffle=True)
    dl_teste  = DataLoader(ds_teste, batch_size=cfg.batch_size, shuffle=False)

    modelo = ModeloNeuralEstocastico(
        n_feat=X.shape[2],
        hidden=cfg.hidden,
        camadas=cfg.camadas,
        dropout=cfg.dropout,
        horizonte=cfg.horizonte
    ).to(device)

    historico = treinar(modelo, dl_treino, dl_teste, cfg, device)

    yhat_n, ytrue_n = prever(modelo, dl_teste, device)
    # desnormaliza
    yhat = yhat_n*sigy + muy
    ytrue = ytrue_n*sigy + muy

    met = metricas(yhat, ytrue)
    print("\nMétricas finais:")
    for k,v in met.items():
        print(f"{k}: {v:.6f}")

    # datas do teste alinhadas ao primeiro alvo
    datas = feat.index[cfg.janela_entrada: len(feat)-cfg.horizonte]
    datas_teste = datas[split:]

    plotar_resultados(historico, yhat, ytrue, datas_teste, cfg)

    # salva modelo
    os.makedirs("saidas", exist_ok=True)
    torch.save({
        "estado_modelo": modelo.state_dict(),
        "muX": muX, "sigX": sigX,
        "muy": muy, "sigy": sigy,
        "cfg": cfg.__dict__
    }, os.path.join("saidas","modelo_treinado.pt"))
    print("Modelo salvo em saidas/modelo_treinado.pt")


if __name__ == "__main__":
    main()