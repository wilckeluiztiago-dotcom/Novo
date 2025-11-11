# ============================================================
# RNEC — Revisão Neural-Estatística de Combustíveis (Gasolina)
# Autor: Luiz Tiago Wilcke (LT)
# ============================================================

import os, math, random, warnings
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ---------------- PyTorch (rede) ----------------
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# --------------- Statsmodels (Kalman/UCM) -------
import statsmodels.api as sm

# ------------------------------------------------
# 0) Configurações
# ------------------------------------------------
@dataclass
class Config:
    usar_csv: bool = False
    caminho_csv: Optional[str] = None  # colunas: data, preco_anp, brent_usd, cambio_brlusd, icms, pis_cofins, cide, etanol_anidro, margem
    freq: str = "W"                    # "W" semanal (ANP), "M" mensal
    epocas_nn: int = 300
    batch: int = 128
    lr: float = 1e-3
    iters_alternancia: int = 3         # E↔M alternâncias
    dropout: float = 0.15              # p/ incerteza (MC Dropout)
    semente: int = 42
    usar_cuda: bool = True
    # pesos/fatores para base linear
    proporcao_etanol: float = 0.27
    fator_refino_gasolinaA: float = 0.74
    fator_cambio_pass: float = 0.95
    # regularização (se quiser usar mais tarde)
    penalizar_spread: float = 1.0

cfg = Config()

# ------------------------------------------------
# 1) Utilidades
# ------------------------------------------------
def fixar_semente(seed=42, usar_cuda=True):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if usar_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def preparar_dispositivo(usar_cuda=True):
    return torch.device("cuda" if (usar_cuda and torch.cuda.is_available()) else "cpu")

fixar_semente(cfg.semente, cfg.usar_cuda)
dev = preparar_dispositivo(cfg.usar_cuda)

# ------------------------------------------------
# 2) Dados (carrega CSV ou simula)
# ------------------------------------------------
def simular_dados(n=156, freq="W"):
    """Simula série semanal de ~3 anos com drivers plausíveis."""
    idx = pd.date_range("2019-01-06", periods=n, freq=freq)
    t = np.arange(n)

    brent = 65 + 20*np.sin(2*np.pi*t/52) + np.random.normal(0, 4, n)
    brent = np.maximum(25, brent)

    cambio = 3.8 + 0.3*np.sin(2*np.pi*t/40) + np.random.normal(0, 0.15, n)
    cambio = np.maximum(3.2, cambio)

    icms = np.full(n, 1.20) + np.random.normal(0, 0.02, n)
    pis_cofins = np.full(n, 0.79) + np.random.normal(0, 0.01, n)
    cide = np.full(n, 0.10) + np.random.normal(0, 0.005, n)

    etanol = 2.5 + 0.6*np.sin(2*np.pi*t/26+0.7) + np.random.normal(0, 0.12, n)
    etanol = np.maximum(1.8, etanol)

    margem = 0.8 + 0.1*np.sin(2*np.pi*t/52+1.2) + np.random.normal(0, 0.05, n)
    margem = np.maximum(0.5, margem)

    gasolinaA_int = cfg.fator_refino_gasolinaA * brent * cfg.fator_cambio_pass / 159.0 * 5.0
    base_linear = (1-cfg.proporcao_etanol)*gasolinaA_int + cfg.proporcao_etanol*etanol + icms + pis_cofins + cide + margem

    spread = np.zeros(n)
    for i in range(1, n):
        spread[i] = 0.92*spread[i-1] + np.random.normal(0, 0.03)

    nlin = 0.08*np.maximum(0, np.diff(np.r_[cambio[0], cambio])) * (1 + 0.5*(brent>75))
    nlin = np.cumsum(nlin)

    preco_obs = base_linear + spread + nlin + np.random.normal(0, 0.07, n)

    df = pd.DataFrame({
        "data": idx,
        "preco_anp": preco_obs,
        "brent_usd": brent,
        "cambio_brlusd": cambio,
        "icms": icms,
        "pis_cofins": pis_cofins,
        "cide": cide,
        "etanol_anidro": etanol,
        "margem": margem
    }).set_index("data")

    # garante frequência explícita para calar warnings
    if df.index.freq is None:
        try:
            df = df.asfreq(freq)
        except Exception:
            pass
    return df

def carregar_ou_simular(cfg: Config):
    if cfg.usar_csv and cfg.caminho_csv and os.path.exists(cfg.caminho_csv):
        df = pd.read_csv(cfg.caminho_csv, parse_dates=["data"]).set_index("data").sort_index()
        df = df.asfreq(cfg.freq).interpolate()
        return df
    else:
        return simular_dados(freq=cfg.freq)

df = carregar_ou_simular(cfg)

# ------------------------------------------------
# 3) Base linear e UCM auxiliares
# ------------------------------------------------
def construir_base_linear(df: pd.DataFrame, cfg: Config) -> pd.Series:
    gasolinaA = cfg.fator_refino_gasolinaA * df["brent_usd"] * cfg.fator_cambio_pass / 159.0 * 5.0
    base = (1-cfg.proporcao_etanol)*gasolinaA + cfg.proporcao_etanol*df["etanol_anidro"] \
           + df["icms"] + df["pis_cofins"] + df["cide"] + df["margem"]
    return base.rename("base_linear")

df["base_linear"] = construir_base_linear(df, cfg)

def _forcar_freq(serie: pd.Series, freq: Optional[str]) -> pd.Series:
    """Garante um freq no índice para evitar ValueWarning do statsmodels."""
    if hasattr(serie.index, "freq") and serie.index.freq is not None:
        return serie
    try:
        if freq is not None:
            return serie.asfreq(freq)
        # tentativa via inferência
        inf = pd.infer_freq(serie.index)
        if inf:
            serie.index.freq = inf  # type: ignore[attr-defined]
    except Exception:
        pass
    return serie

def ajustar_ucm(resid: pd.Series):
    """Ajusta UCM (nível local + AR(1)) no resíduo e retorna (modelo, resultados)."""
    resid = _forcar_freq(resid, cfg.freq)
    mod = sm.tsa.UnobservedComponents(resid, level="local level", autoregressive=1)
    res = mod.fit(disp=False)
    return mod, res

def obter_level_suavizado(res_ucm, index: pd.Index) -> pd.Series:
    """
    Converte o level.smoothed do statsmodels para Series com índice.
    Em alguns ambientes volta ndarray; em outros, Series. Normalizamos aqui.
    """
    try:
        # caminho "canônico" quando o statsmodels devolve um objeto com .smoothed
        arr = res_ucm.level.smoothed
    except Exception:
        # fallback: usar estado suavizado (primeiro componente costuma ser o nível)
        st = getattr(res_ucm, "smoothed_state", None)
        if st is not None:
            arr = st[0, :]
        else:
            raise RuntimeError("Não foi possível extrair o 'level' suavizado do UCM.")
    # Se já for Series com índice, só reindexamos; se for ndarray, criamos Series
    if isinstance(arr, pd.Series):
        ser = arr
    else:
        ser = pd.Series(np.asarray(arr).ravel(), index=index)
    return ser

# Ajuste inicial (opcional, antes da alternância)
resid_linear = (df["preco_anp"] - df["base_linear"]).dropna()
_, res_ucm_inicial = ajustar_ucm(resid_linear)
spread_ini = obter_level_suavizado(res_ucm_inicial, resid_linear.index)
df["spread_est"] = spread_ini.reindex(df.index).interpolate()
df["resid_ucm"] = resid_linear.reindex(df.index) - df["spread_est"]

# ------------------------------------------------
# 4) Features p/ Rede Neural
# ------------------------------------------------
def montar_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    z = pd.DataFrame(index=df.index)
    z["brent_usd"] = df["brent_usd"]
    z["cambio_brlusd"] = df["cambio_brlusd"]
    z["etanol_anidro"] = df["etanol_anidro"]
    z["icms"] = df["icms"]
    z["pis_cofins"] = df["pis_cofins"]
    z["cide"] = df["cide"]
    z["margem"] = df["margem"]
    for col in ["brent_usd","cambio_brlusd","etanol_anidro"]:
        z[f"d_{col}"] = z[col].diff().fillna(0)
        z[f"{col}_lag1"] = z[col].shift(1).bfill()
    t = np.arange(len(z))
    z["sin52"] = np.sin(2*np.pi*t/52)
    z["cos52"] = np.cos(2*np.pi*t/52)
    mu = z.mean(); sd = z.std().replace(0,1)
    zn = (z - mu)/sd
    return zn, mu, sd

Z, muZ, sdZ = montar_features(df)
target_nn = df["resid_ucm"].fillna(0.0)

class Conjunto(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1,1)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]

class MLP(nn.Module):
    def __init__(self, d_in, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
    def forward(self, x): return self.net(x)

def treinar_nn(X, y, epocas=300, batch=128, lr=1e-3, dropout=0.2, device="cpu"):
    ds = Conjunto(X, y)
    dl = DataLoader(ds, batch_size=batch, shuffle=True)
    modelo = MLP(X.shape[1], dropout=dropout).to(device)
    ot = torch.optim.AdamW(modelo.parameters(), lr=lr)
    crit = nn.SmoothL1Loss(beta=0.1)
    modelo.train()
    for ep in range(epocas):
        loss_ac = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = modelo(xb)
            loss = crit(pred, yb)
            ot.zero_grad(); loss.backward(); ot.step()
            loss_ac += loss.item()*len(xb)
        if (ep+1) % 50 == 0:
            print(f"Época {ep+1}/{epocas} — Loss {loss_ac/len(ds):.6f}")
    return modelo

# ------------------------------------------------
# 5) Alternância (E: UCM ; M: NN)
# ------------------------------------------------
X0 = Z.values.astype(np.float32)
y0 = target_nn.values.astype(np.float32)

for it in range(cfg.iters_alternancia):
    print(f"\n=== Alternância {it+1}/{cfg.iters_alternancia} ===")
    # M: treina NN no resid_ucm
    nn_model = treinar_nn(X0, y0, epocas=cfg.epocas_nn, batch=cfg.batch, lr=cfg.lr,
                          dropout=cfg.dropout, device=dev)
    # Predição não-linear como Series alinhada ao índice
    nn_model.eval()
    with torch.no_grad():
        nn_pred_arr = nn_model(torch.tensor(X0, dtype=torch.float32, device=dev)).cpu().numpy().ravel()
    nn_pred = pd.Series(nn_pred_arr, index=df.index, name=f"nlin_pred_{it}")
    df[nn_pred.name] = nn_pred

    # E: reestimar UCM sobre (preco_anp - base_linear - nn_pred)
    novo_resid_series = (df["preco_anp"] - df["base_linear"] - nn_pred).dropna()
    _, res_ucm = ajustar_ucm(novo_resid_series)
    spread_series = obter_level_suavizado(res_ucm, novo_resid_series.index)
    # traz para o índice completo do df
    df["spread_est"] = spread_series.reindex(df.index).interpolate()
    df["resid_ucm"] = (df["preco_anp"] - df["base_linear"] - nn_pred) - df["spread_est"]
    df["resid_ucm"] = df["resid_ucm"].fillna(0.0)

    # atualiza alvo para próxima iteração
    y0 = df["resid_ucm"].values.astype(np.float32)

# ------------------------------------------------
# 6) MC Dropout para incerteza da parte não-linear
# ------------------------------------------------
def mc_dropout_pred(modelo, X, n=80, device="cpu"):
    modelo.train()  # ativa dropout
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    preds = []
    with torch.no_grad():
        for _ in range(n):
            preds.append(modelo(X_t).cpu().numpy().ravel())
    preds = np.vstack(preds)
    return preds.mean(axis=0), preds.std(axis=0)

nlin_mean, nlin_std = mc_dropout_pred(nn_model, X0, n=80, device=dev)
df["nlin_final"] = pd.Series(nlin_mean, index=df.index)
df["nlin_std"] = pd.Series(nlin_std, index=df.index)

# ------------------------------------------------
# 7) Preço técnico e métricas
# ------------------------------------------------
df["preco_tecnico"] = df["base_linear"] + df["spread_est"] + df["nlin_final"]
df["revisao"] = df["preco_tecnico"] - df["preco_anp"]

def rmse(a,b): return float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))
def mae(a,b):  return float(np.mean(np.abs(np.asarray(a)-np.asarray(b))))

print("\n--- Métricas ---")
print("RMSE(preco_tecnico vs preco_anp):", rmse(df["preco_tecnico"], df["preco_anp"]))
print("MAE (preco_tecnico vs preco_anp):", mae(df["preco_tecnico"], df["preco_anp"]))

# ------------------------------------------------
# 8) Gráficos
# ------------------------------------------------
plt.figure(figsize=(10,4.2))
plt.plot(df.index, df["preco_anp"], label="Preço ANP (observado)")
plt.plot(df.index, df["preco_tecnico"], label="Preço técnico (RNEC)", linewidth=2)
plt.title("Gasolina — Observado vs Preço Técnico (RNEC)")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(df.index, df["base_linear"], label="Base linear")
plt.plot(df.index, df["spread_est"], label="Spread doméstico (latente)")
plt.plot(df.index, df["nlin_final"], label="Não-linear (NN)")
plt.title("Decomposição — Base, Spread, Não-linear")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.show()

plt.figure(figsize=(10,3.8))
plt.plot(df.index, df["revisao"], label="Revisão sugerida (R$/L)")
plt.axhline(0, color="k", linewidth=1)
plt.title("Revisão Sugerida do Preço (RNEC)")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.show()

print("\n--- Últimas observações ---")
print(df[["preco_anp","preco_tecnico","revisao","base_linear","spread_est","nlin_final","nlin_std"]].tail(10).round(4))
