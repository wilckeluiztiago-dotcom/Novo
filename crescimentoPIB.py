# ============================================================
# Previsão de Crescimento Econômico — Neural SDE Probabilística
# Autor: Luiz Tiago Wilcke (LT)
# ============================================================

import os
import math
import time
import random
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------- (Opcional) PyTorch -----------------
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ==========================
# 0) Configurações gerais
# ==========================
@dataclass
class Config:
    usar_csv: bool = False
    caminho_csv: Optional[str] = None  # CSV com colunas: data, pib (nível ou log), inflacao, juros, investimento, comercio, cambio (o que tiver)
    frequencia: str = "Q"              # "M" ou "Q"
    normalizar: bool = True
    proporcao_treino: float = 0.7
    proporcao_valid: float = 0.15
    epocas: int = 500
    batch_size: int = 128
    lr: float = 3e-4
    peso_l2: float = 1e-4
    dropout: float = 0.15
    sementes: int = 42
    horizonte_passos: int = 12         # horizonte de previsão (passos da frequência)
    amostras_mc: int = 2000            # cenários para bandas
    paciencia_es: int = 30             # early stopping
    delta_t: float = 1.0               # passo de tempo em unidades da frequência (1 período)

cfg = Config()

np.random.seed(cfg.sementes)
random.seed(cfg.sementes)
torch.manual_seed(cfg.sementes)

# =============================================
# 1) Carregamento ou simulação de dados
# =============================================
def gerar_dados_sinteticos(n: int = 200, freq="Q") -> pd.DataFrame:
    """
    Gera uma série macro sintética realista:
    - y_t: log(PIB) (nível) com tendência + ruído SDE
    - Δy_t: crescimento (target)
    - Covariáveis: inflação, juros, investimento, comércio, câmbio
    """
    if freq == "Q":
        datas = pd.period_range("2005Q1", periods=n, freq="Q").to_timestamp()
    else:
        datas = pd.period_range("2005-01", periods=n, freq="M").to_timestamp()

    # Processo para Δy_t com regime suave: ARX + choques heteroscedásticos
    inflacao = 0.04 + 0.01*np.sin(np.linspace(0, 10, n)) + 0.005*np.random.randn(n)
    juros    = 0.08 + 0.01*np.cos(np.linspace(0, 8, n))  + 0.01*np.random.randn(n)
    inv      = 0.20 + 0.02*np.random.randn(n) + 0.03*np.sin(np.linspace(0, 6, n))
    comercio = 0.30 + 0.02*np.random.randn(n)
    cambio   = 4.0  + 0.4*np.random.randn(n)

    x = np.column_stack([inflacao, juros, inv, comercio, cambio])
    beta = np.array([ -0.6, -0.2, 0.9, 0.3, -0.05 ])  # coef. "verdadeiros" para drift local

    # Volatilidade estocástica lognormal
    log_vol = -2.0 + 0.85*np.random.randn(n)
    sigma_t = np.exp(log_vol)                          # ~ 0.1 a 0.4 aprox.

    # Drift não-linear (com saturação)
    drift_t = 0.005 + (x @ beta) / (1.0 + np.abs(x @ beta))

    # Δy_t ~ N(drift_t, sigma_t^2)
    dy = drift_t + sigma_t * np.random.randn(n)

    # nível y_t (log PIB)
    y = np.cumsum(dy)
    y = y - y[0] + 5.0   # reescala base

    df = pd.DataFrame({
        "data": datas,
        "log_pib": y,
        "crescimento": dy,
        "inflacao": inflacao,
        "juros": juros,
        "investimento": inv,
        "comercio": comercio,
        "cambio": cambio
    })
    return df

def carregar_ou_simular(cfg: Config) -> pd.DataFrame:
    if cfg.usar_csv and cfg.caminho_csv and os.path.exists(cfg.caminho_csv):
        df = pd.read_csv(cfg.caminho_csv)
        # Esperado: data (yyyy-mm-dd ou similar), pib (nível ou log),
        # covariáveis quaisquer. Se 'pib' for nível, tiramos log.
        if "data" not in df.columns:
            raise ValueError("CSV deve conter coluna 'data'.")
        if "pib" in df.columns and "log_pib" not in df.columns:
            df["log_pib"] = np.log(df["pib"].astype(float))
        elif "log_pib" not in df.columns:
            raise ValueError("CSV precisa ter 'pib' (nível) ou 'log_pib'.")
        df["data"] = pd.to_datetime(df["data"])
        df = df.sort_values("data").reset_index(drop=True)
        df["crescimento"] = df["log_pib"].diff().fillna(0.0)
        return df
    else:
        return gerar_dados_sinteticos(n=220, freq=cfg.frequencia)

df = carregar_ou_simular(cfg)

# Seleção de features disponíveis
colunas_possiveis = ["inflacao","juros","investimento","comercio","cambio"]
colunas_x = [c for c in colunas_possiveis if c in df.columns]
if not colunas_x:
    raise ValueError("Sem covariáveis para condicionar o SDE.")

# ==============================
# 2) Pré-processamento temporal
# ==============================
df = df.dropna().reset_index(drop=True)
y = df["crescimento"].values.astype(np.float32)
X = df[colunas_x].values.astype(np.float32)
datas = pd.to_datetime(df["data"])

# Normalização opcional (guardando parâmetros)
def normalizar_fit(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = x.mean(0)
    s = x.std(0) + 1e-8
    return (x - m)/s, m, s

if cfg.normalizar:
    X_norm, media_x, desvio_x = normalizar_fit(X)
else:
    X_norm, media_x, desvio_x = X, np.zeros(X.shape[1]), np.ones(X.shape[1])

# Splits (respeitando o tempo)
n = len(y)
n_tr = int(n*cfg.proporcao_treino)
n_vl = int(n*cfg.proporcao_valid)
idx_tr = slice(0, n_tr)
idx_vl = slice(n_tr, n_tr+n_vl)
idx_ts = slice(n_tr+n_vl, n)

X_tr, y_tr = X_norm[idx_tr], y[idx_tr]
X_vl, y_vl = X_norm[idx_vl], y[idx_vl]
X_ts, y_ts = X_norm[idx_ts], y[idx_ts]
datas_tr, datas_vl, datas_ts = datas[idx_tr], datas[idx_vl], datas[idx_ts]

# ============================
# 3) Dataset e DataLoader
# ============================
class SerieDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

dl_tr = DataLoader(SerieDataset(X_tr, y_tr), batch_size=cfg.batch_size, shuffle=True)
dl_vl = DataLoader(SerieDataset(X_vl, y_vl), batch_size=cfg.batch_size, shuffle=False)

# ==============================================
# 4) MLP Probabilística (μ(x), σ(x)>0) com Dropout
# ==============================================
class MLP_SDE(nn.Module):
    def __init__(self, d_in: int, d_hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.rede = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.cabeca_mu = nn.Linear(d_hidden, 1)      # drift
        self.cabeca_rho = nn.Linear(d_hidden, 1)     # rho -> sigma = softplus(rho)
        self.softplus = nn.Softplus()

    def forward(self, x):
        h = self.rede(x)
        mu = self.cabeca_mu(h).squeeze(-1)
        rho = self.cabeca_rho(h).squeeze(-1)
        sigma = self.softplus(rho) + 1e-6
        return mu, sigma

# ===========================
# 5) Treino (NLL Gaussiana)
# ===========================
def perda_nll_gaussiana(y_true, mu, sigma, delta_t: float = 1.0):
    # y ~ N(mu*Δt, (sigma^2)*Δt)
    var = (sigma**2) * delta_t
    nll = 0.5 * (torch.log(2*torch.pi*var) + (y_true - mu*delta_t)**2 / var)
    return nll.mean()

dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelo = MLP_SDE(d_in=X_tr.shape[1], d_hidden=192, dropout=cfg.dropout).to(dispositivo)
otimiz = torch.optim.AdamW(modelo.parameters(), lr=cfg.lr, weight_decay=cfg.peso_l2)

melhor_vl = float("inf")
contador_sem_melhora = 0
estado_melhor = None

for epoca in range(1, cfg.epocas+1):
    modelo.train()
    perdas = []
    for xb, yb in dl_tr:
        xb = xb.to(dispositivo)
        yb = yb.to(dispositivo)
        mu, sigma = modelo(xb)
        perda = perda_nll_gaussiana(yb, mu, sigma, delta_t=cfg.delta_t)
        otimiz.zero_grad()
        perda.backward()
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
        otimiz.step()
        perdas.append(perda.item())

    # validação
    modelo.eval()
    with torch.no_grad():
        xb = torch.tensor(X_vl, dtype=torch.float32, device=dispositivo)
        yb = torch.tensor(y_vl, dtype=torch.float32, device=dispositivo)
        mu, sigma = modelo(xb)
        perda_vl = perda_nll_gaussiana(yb, mu, sigma, delta_t=cfg.delta_t).item()

    if perda_vl < melhor_vl - 1e-5:
        melhor_vl = perda_vl
        contador_sem_melhora = 0
        estado_melhor = modelo.state_dict()
    else:
        contador_sem_melhora += 1

    if epoca % 25 == 0 or epoca == 1:
        print(f"Época {epoca:04d} | perda_tr={np.mean(perdas):.6f} | perda_vl={perda_vl:.6f}")

    if contador_sem_melhora >= cfg.paciencia_es:
        print(f"Early stopping na época {epoca}. Melhor validação: {melhor_vl:.6f}")
        break

# Carrega melhor estado
if estado_melhor is not None:
    modelo.load_state_dict(estado_melhor)

# ==============================================
# 6) Avaliação em teste + calibração e fan chart
# ==============================================
modelo.eval()

def previsao_pontual_intervalo(X_in: np.ndarray, delta_t: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        xb = torch.tensor(X_in, dtype=torch.float32, device=dispositivo)
        mu, sigma = modelo(xb)
        mu = mu.cpu().numpy()
        sigma = sigma.cpu().numpy()
    media = mu*delta_t
    desvio = sigma*np.sqrt(delta_t)
    return media, desvio

media_ts, desvio_ts = previsao_pontual_intervalo(X_ts, delta_t=cfg.delta_t)

# RMSE / MAE
rmse = float(np.sqrt(np.mean((y_ts - media_ts)**2)))
mae  = float(np.mean(np.abs(y_ts - media_ts)))
print(f"Teste: RMSE={rmse:.6f} | MAE={mae:.6f}")

# Cobertura empírica de um IC 95%
ic_low = media_ts - 1.96*desvio_ts
ic_high= media_ts + 1.96*desvio_ts
cobertura95 = float(np.mean((y_ts >= ic_low) & (y_ts <= ic_high)))
print(f"Cobertura 95% (empírica) = {100*cobertura95:.2f}%")

# ===========================================
# 7) Previsão multi-passos (Monte Carlo)
# ===========================================
def amostrar_cenarios(X_cond: np.ndarray, amostras: int = 2000, delta_t: float = 1.0, usar_mc_dropout: bool = True):
    """
    Recebe uma sequência de covariáveis X_cond (T x d). Para cada t,
    sorteia Δy_t ~ N(μ(x_t)Δt, σ(x_t)^2 Δt), com dropout ativo (MC) se desejado.
    Retorna matriz (amostras x T) de incrementos simulados.
    """
    modelo.train() if usar_mc_dropout else modelo.eval()
    xb = torch.tensor(X_cond, dtype=torch.float32, device=dispositivo)
    T = xb.shape[0]
    with torch.no_grad():
        # Para eficiência, computamos em blocos
        blocos = []
        rep = max(1, amostras // 100)
        resto = amostras - 100*rep
        tamanhos = [rep]*100 + ([1]*resto if resto > 0 else [])
        for k in range(len(tamanhos)):
            mu, sigma = modelo(xb)  # com dropout se .train()
            mu = (mu*cfg.delta_t).cpu().numpy()
            sig = (sigma*np.sqrt(cfg.delta_t)).cpu().numpy()
            incs = mu[None,:] + sig[None,:]*np.random.randn(tamanhos[k], T)
            blocos.append(incs)
        cenarios = np.vstack(blocos)
    modelo.eval()
    return cenarios  # (amostras x T)

# Horizonte fora da amostra: usamos últimas observações de X_ts como "janela" + repetição de cenários.
# Para um caso real, você passaria X futuro (cenários macro). Aqui faremos:
# - cenário "baseline": repetir últimas linhas de X_ts ou aplicar choque suave.
def construir_cenario_macro_futuro(X_base: np.ndarray, passos: int) -> np.ndarray:
    # mantém último estado e relaxa lentamente (exemplo simples)
    ultimo = X_base[-1:].copy()
    futuros = np.repeat(ultimo, passos, axis=0)
    return futuros

X_fut = construir_cenario_macro_futuro(X_ts, cfg.horizonte_passos)
cenarios = amostrar_cenarios(X_fut, amostras=cfg.amostras_mc, delta_t=cfg.delta_t, usar_mc_dropout=True)

# Constrói níveis de log_PIB a partir do último nível observado
y_nivel_ultimo = float(df["log_pib"].iloc[n_tr+n_vl-1])  # último antes do teste? Prefira último da amostra
y_nivel_ultimo = float(df["log_pib"].iloc[-1])           # aqui: último geral (pós-treino)

# Acumula os incrementos amostrados para cada cenário
acumulado = np.cumsum(cenarios, axis=1)
niveis_futuros = y_nivel_ultimo + acumulado  # (amostras x H)

# Quantis para fan chart
quantis = [0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95]
faixas = {q: np.quantile(niveis_futuros, q, axis=0) for q in quantis}

# Exporta CSV de previsões (nível e crescimento)
datas_fut = pd.date_range(start=datas.iloc[-1] + pd.offsets.QuarterEnd(0 if cfg.frequencia=="Q" else 0),
                          periods=cfg.horizonte_passos, freq=cfg.frequencia)
df_prev = pd.DataFrame({
    "data": datas_fut,
    "mediana_log_pib": faixas[0.5],
    "p05_log_pib": faixas[0.05],
    "p10_log_pib": faixas[0.1],
    "p20_log_pib": faixas[0.2],
    "p80_log_pib": faixas[0.8],
    "p90_log_pib": faixas[0.9],
    "p95_log_pib": faixas[0.95],
})
# converte para crescimento anualizado aproximado (se Q: *4; se M: *12)
fator_anual = 4 if cfg.frequencia=="Q" else 12
df_prev["mediana_crescimento_anual"] = np.r_[np.nan, np.diff(df_prev["mediana_log_pib"])] * fator_anual

df_prev.to_csv("previsoes_pib.csv", index=False)
print("Arquivo salvo:", os.path.abspath("previsoes_pib.csv"))

# ===========================
# 8) Gráficos e diagnósticos
# ===========================
plt.figure(figsize=(11,5))
plt.plot(datas_ts, y_ts, label="Observado (Δ log PIB)", linewidth=1.5)
plt.plot(datas_ts, media_ts, label="Média predita", linewidth=1.2)
plt.fill_between(datas_ts, ic_low, ic_high, alpha=0.25, label="IC 95% (condicional)")
plt.title("Teste — Crescimento (Δ log PIB)")
plt.legend()
plt.tight_layout()
plt.show()

# Fan chart de níveis futuros
plt.figure(figsize=(11,5))
# histórico
plt.plot(datas, df["log_pib"].values, label="Histórico (log PIB)", linewidth=1.5)

# horizonte futuro
cores_alpha = [(0.9, "p10/p90"), (0.7, "p20/p80"), (0.5, "p05/p95")]
for alpha, _ in cores_alpha:
    plt.fill_between(datas_fut, faixas[0.1], faixas[0.9], alpha=alpha*0.25, label=None)
    plt.fill_between(datas_fut, faixas[0.2], faixas[0.8], alpha=alpha*0.25, label=None)
    plt.fill_between(datas_fut, faixas[0.05], faixas[0.95], alpha=alpha*0.18, label=None)
plt.plot(datas_fut, faixas[0.5], label="Mediana (futuro)", linewidth=2.0)

plt.title("Fan chart — Nível de log PIB (SDE neural)")
plt.legend()
plt.tight_layout()
plt.show()

# Calibração simples: quantile PIT
with torch.no_grad():
    xb = torch.tensor(X_ts, dtype=torch.float32, device=dispositivo)
    mu, sigma = modelo(xb)
    mu = (mu.cpu().numpy()*cfg.delta_t)
    sd = (sigma.cpu().numpy()*np.sqrt(cfg.delta_t))
# z-scores
z = (y_ts - mu)/sd
# PIT gaussian (aprox)
from math import erf, sqrt
Phi = lambda zz: 0.5*(1.0 + erf(zz/np.sqrt(2.0)))
pit = np.vectorize(Phi)(z)

plt.figure(figsize=(8,4.5))
plt.hist(pit, bins=20, density=True, alpha=0.8)
plt.axhline(1.0, color="k", linestyle="--", linewidth=1.0, label="Uniforme(0,1)")
plt.title("Diagnóstico de Calibração — Histograma do PIT")
plt.legend()
plt.tight_layout()
plt.show()

# ===========================
# 9) Impressão de resumo
# ===========================
print("\n=== Resumo ===")
print(f"Observações totais: {len(df)} | Treino: {len(X_tr)} | Valid.: {len(X_vl)} | Teste: {len(X_ts)}")
print(f"RMSE teste: {rmse:.6f} | MAE teste: {mae:.6f} | Cobertura 95%: {100*cobertura95:.2f}%")
print("Faixas de log PIB (futuro):")
for q in [0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95]:
    print(f"  Q{int(100*q):02d}: {faixas[q][-1]:.4f} (último passo)")
