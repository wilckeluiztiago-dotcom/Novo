# -*- coding: utf-8 -*-
# ============================================================
# Crescimento do Desemprego — SDE Neural + PyTorch (float64)
# Autor: Luiz Tiago Wilcke (LT)
# ============================================================

import os, math, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim

# ------------------------ Precisão/Dispositivo ------------------------
torch.set_default_dtype(torch.float64)
DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"
SEMENTE = 42
torch.manual_seed(SEMENTE)
np.random.seed(SEMENTE)

# -------------------------- Utilidades --------------------------
def padronizar(col):
    v = np.asarray(col, dtype=np.float64)
    mu, sd = np.nanmean(v), np.nanstd(v) + 1e-12
    return (v - mu) / sd, mu, sd

def quantil_torch(t: torch.Tensor, q: float, dim: int):
    if hasattr(torch, "quantile"):
        return torch.quantile(t, q, dim=dim)
    k = max(0, min(t.shape[dim]-1, int(round(q*(t.shape[dim]-1)))))
    valores, _ = torch.sort(t, dim=dim)
    return valores.select(dim, k)

# ---------------------- Dados -------------------------
CAMINHO = "desemprego.csv"   # opcional (data,desemprego,pi,juros,ipca)
usar_dados_reais = os.path.exists(CAMINHO)

if usar_dados_reais:
    df = pd.read_csv(CAMINHO)
    df["data"] = pd.to_datetime(df["data"]) if "data" in df.columns else pd.date_range("2015-01-01", periods=len(df), freq="M")
    if df["desemprego"].max() > 1.5:
        df["desemprego"] = (df["desemprego"] / 100.0).clip(0.0, 1.0)
    for c in ["pi","juros","ipca"]:
        if c not in df.columns:
            df[c] = df["desemprego"].rolling(3, min_periods=1).mean().bfill()
    df = df[["data","desemprego","pi","juros","ipca"]].dropna().reset_index(drop=True)
else:
    # Série sintética
    n = 240
    datas = pd.date_range("2006-01-01", periods=n, freq="M")
    t = np.arange(n)
    pi    = 0.6*np.sin(2*np.pi*t/48.0) + 0.2*np.sin(2*np.pi*t/12.0) + 0.1*np.random.randn(n)
    juros = 0.3 + 0.15*np.sin(2*np.pi*t/36.0 + 1.1) + 0.05*np.random.randn(n)
    ipca  = 0.2 + 0.12*np.sin(2*np.pi*t/18.0 - 0.7) + 0.04*np.random.randn(n)
    z = np.zeros(n, dtype=np.float64)
    kappa, mu0, dt = 1.2, 0.0, 1.0
    for i in range(1, n):
        alvo = mu0 + 0.6*(-pi[i]) + 0.4*(juros[i]) + 0.3*(ipca[i]-0.2)
        ruido = 0.35*np.sqrt(dt)*np.random.randn()
        z[i] = z[i-1] + kappa*(alvo - z[i-1])*dt + ruido
    u = 1/(1+np.exp(-z))
    y = np.clip(u + 0.02*np.random.randn(n), 0, 1)
    df = pd.DataFrame({"data": datas, "desemprego": y, "pi": pi, "juros": juros, "ipca": ipca})

covs = ["pi","juros","ipca"]
for c in covs:
    df[c], _, _ = padronizar(df[c])

alvo  = df["desemprego"].values.astype(np.float64)
datas = df["data"].values

prop_treino = 0.75
n_total = len(df)
n_treino = int(n_total*prop_treino)
idx_treino = np.arange(n_treino)
idx_teste  = np.arange(n_treino, n_total)

X = torch.tensor(df[covs].values, device=DISPOSITIVO)         # (T, d)
Y = torch.tensor(alvo, device=DISPOSITIVO)                    # (T,)

# --------------------- Módulos NN -------------------------
class RedeDrift(nn.Module):
    def __init__(self, d_cov: int, largura: int = 64):
        super().__init__()
        self.d_cov = d_cov
        self.net = nn.Sequential(
            nn.Linear(1 + d_cov + 1, largura),
            nn.SiLU(),
            nn.Linear(largura, largura),
            nn.SiLU(),
            nn.Linear(largura, 1)
        )
    def forward(self, z, x, t):
        z = z.view(-1, 1)
        x = x.view(-1, self.d_cov)
        t = t.view(-1, 1)
        return self.net(torch.cat([z, x, t], dim=-1))

class RedeDiff(nn.Module):
    def __init__(self, d_cov: int, largura: int = 64):
        super().__init__()
        self.d_cov = d_cov
        self.net = nn.Sequential(
            nn.Linear(1 + d_cov + 1, largura),
            nn.SiLU(),
            nn.Linear(largura, largura),
            nn.SiLU(),
            nn.Linear(largura, 1)
        )
        self.softplus = nn.Softplus(beta=1.0, threshold=20.0)
    def forward(self, z, x, t):
        z = z.view(-1, 1)
        x = x.view(-1, self.d_cov)
        t = t.view(-1, 1)
        return self.softplus(self.net(torch.cat([z, x, t], dim=-1))) + 1e-6

class SDEDesemprego(nn.Module):
    """
    dz_t = [ kappa*(mu(x,t) - z_t) + f_res(z,x,t) ] dt + g(z,x,t) dW_t
    u_t = sigmoid(z_t); y_t ~ N(u_t, sigma_obs^2)
    """
    def __init__(self, d_cov: int):
        super().__init__()
        self.d_cov = d_cov
        self.kappa = nn.Parameter(torch.tensor(0.8))
        self.a  = nn.Parameter(torch.randn(d_cov)*0.1)
        self.b0 = nn.Parameter(torch.tensor(0.0))
        self.b1 = nn.Parameter(torch.tensor(0.0))
        self.b2 = nn.Parameter(torch.tensor(0.0))
        self.S  = nn.Parameter(torch.tensor(12.0))
        self.rede_drift = RedeDrift(d_cov)
        self.rede_diff  = RedeDiff(d_cov)
        self.log_sigma_obs = nn.Parameter(torch.tensor(-3.0))

    def mu_base(self, x, t):
        # x: (B,d), t: (B,1)  -> retorna (B,1)
        t = t.view(-1,1)
        x = x.view(-1, self.d_cov)
        saz1 = torch.sin(2*math.pi*t/self.S)
        saz2 = torch.cos(2*math.pi*t/self.S)
        mu = (x @ self.a) + self.b0 + self.b1*saz1.squeeze(-1) + self.b2*saz2.squeeze(-1)
        return mu.view(-1,1)

    def passo_em(self, z, x, t, dt, xi):
        # z: (B,1), x:(B,d), t:(B,1)
        z = z.view(-1,1)
        x = x.view(-1, self.d_cov)
        t = t.view(-1,1)
        mu = self.mu_base(x, t)                  # (B,1)
        f_res = self.rede_drift(z, x, t)         # (B,1)
        g = self.rede_diff(z, x, t)              # (B,1)
        drift = self.kappa*(mu - z) + f_res      # (B,1)
        z_next = z + drift*dt + g*math.sqrt(dt)*xi.view(-1,1)
        return z_next

    def simular(self, z0, X_seq, t_seq, dt: float, n_passos: int, n_mc: int = 8):
        T = X_seq.shape[0]
        assert T == n_passos, "X_seq e n_passos incompatíveis"
        X_rep = X_seq.unsqueeze(0).repeat(n_mc, 1, 1)     # (MC,T,d)
        t_rep = t_seq.view(1, -1, 1).repeat(n_mc, 1, 1)   # (MC,T,1)
        z = z0.view(1,1).repeat(n_mc,1)                   # (MC,1)
        us = []
        for k in range(T):
            xk = X_rep[:,k,:]                             # (MC,d)
            tk = t_rep[:,k,:]                             # (MC,1)
            xi = torch.randn_like(z)                      # (MC,1)
            z = self.passo_em(z, xk, tk, dt, xi)          # (MC,1)
            u = torch.sigmoid(z)                          # (MC,1)
            us.append(u.squeeze(-1))                      # (MC,)
        U = torch.stack(us, dim=1)                        # (MC,T)
        return U

    def forward(self, X_seq, y_seq, i0: int, dt: float, n_janela: int, n_mc: int):
        T = n_janela
        t_idx = torch.arange(i0, i0+T, device=X_seq.device, dtype=torch.float64)
        t_seq = t_idx.view(-1,1)                          # (T,1)
        if i0 > 0:
            y0 = y_seq[i0-1].clamp(1e-4, 1-1e-4)
            z0 = torch.log(y0/(1.0 - y0))
        else:
            z0 = torch.tensor(0.0, device=X_seq.device)
        U = self.simular(z0, X_seq[i0:i0+T, :], t_seq, dt, T, n_mc=n_mc)  # (MC,T)
        y_obs = y_seq[i0:i0+T].unsqueeze(0).repeat(n_mc, 1)               # (MC,T)
        sigma_obs = torch.exp(self.log_sigma_obs) + 1e-8
        nll = 0.5*(((y_obs - U)**2)/(sigma_obs**2) + 2.0*self.log_sigma_obs)
        return nll.mean()

# ------------------------ Treino -------------------------------
d_cov = X.shape[1]
modelo = SDEDesemprego(d_cov).to(DISPOSITIVO)

otimizador = optim.AdamW(modelo.parameters(), lr=5e-3, weight_decay=1e-4)
agendador  = optim.lr_scheduler.ReduceLROnPlateau(otimizador, factor=0.5, patience=20)

def step_scheduler(ag, metric, opt):
    lr_ant = opt.param_groups[0]['lr']
    ag.step(metric)
    lr_nova = opt.param_groups[0]['lr']
    if lr_nova < lr_ant:
        print(f"[scheduler] LR reduzida: {lr_ant:.2e} -> {lr_nova:.2e}")

n_epocas       = 400
tam_janela     = 24
n_mc_treino    = 16
dt_discreto    = 1.0
clip_grad      = 1.0

historico = {"epoca":[], "loss":[], "loss_val":[]}

def loss_validacao():
    modelo.eval()
    with torch.no_grad():
        i0 = max(1, n_treino - tam_janela)
        return modelo(X, Y, i0=i0, dt=dt_discreto, n_janela=tam_janela, n_mc=32).item()

melhor_val = float("inf")
melhor_estado = None

for ep in range(1, n_epocas+1):
    modelo.train()
    perdas = []
    inicio_max = max(1, n_treino - tam_janela)
    for i0 in range(1, inicio_max, max(1, tam_janela//3)):
        otimizador.zero_grad(set_to_none=True)
        loss = modelo(X, Y, i0=i0, dt=dt_discreto, n_janela=tam_janela, n_mc=n_mc_treino)
        loss.backward()
        nn.utils.clip_grad_norm_(modelo.parameters(), clip_grad)
        otimizador.step()
        perdas.append(loss.item())
    loss_m = float(np.mean(perdas)) if perdas else float("nan")
    loss_v = loss_validacao()
    step_scheduler(agendador, loss_v, otimizador)
    historico["epoca"].append(ep); historico["loss"].append(loss_m); historico["loss_val"].append(loss_v)
    if loss_v < melhor_val:
        melhor_val = loss_v
        melhor_estado = {k: v.detach().cpu().clone() for k,v in modelo.state_dict().items()}
    if ep % 20 == 0 or ep == 1:
        print(f"[época {ep:04d}] loss_treino={loss_m:.6f} | loss_val={loss_v:.6f} | lr={otimizador.param_groups[0]['lr']:.2e}")

if melhor_estado is not None:
    modelo.load_state_dict(melhor_estado)

# ------------------------ Avaliação & Previsão -------------------
modelo.eval()
with torch.no_grad():
    i0 = 1
    T_all = len(df)-i0
    t_seq = torch.arange(i0, len(df), device=DISPOSITIVO, dtype=torch.float64).view(-1,1)
    y0 = Y[i0-1].clamp(1e-4, 1-1e-4)
    z0 = torch.log(y0/(1.0 - y0))
    U_mc = modelo.simular(z0, X[i0:, :], t_seq, dt_discreto, T_all, n_mc=256)  # (MC,T)
    u_med = U_mc.mean(0).cpu().numpy()
    u_p05 = quantil_torch(U_mc, 0.05, dim=0).cpu().numpy()
    u_p95 = quantil_torch(U_mc, 0.95, dim=0).cpu().numpy()
    y_true = Y[i0:].cpu().numpy()
    idx_test_local = idx_teste[idx_teste >= i0] - i0
    mse_teste = float(np.mean((u_med[idx_test_local] - y_true[idx_test_local])**2))
    mae_teste = float(np.mean(np.abs(u_med[idx_test_local] - y_true[idx_test_local])))

plt.figure(figsize=(11,5.5))
plt.plot(df["data"][i0:], y_true, label="observado", linewidth=1.5)
plt.plot(df["data"][i0:], u_med, label="SDE neural (média MC)", linewidth=1.8)
plt.fill_between(df["data"][i0:], u_p05, u_p95, alpha=0.2, label="IC 90% (MC)")
plt.axvline(df["data"].iloc[n_treino], ls="--", lw=1.2, color="k", label="corte treino/teste")
plt.title("Taxa de Desemprego — Ajuste SDE Neural (EM + PyTorch)")
plt.ylabel("desemprego (proporção)")
plt.legend(); plt.tight_layout(); plt.show()

print("\n======== PARÂMETROS APRENDIDOS ========")
print(f"kappa        = {modelo.kappa.item(): .5f}")
print(f"a (covari.)  = {modelo.a.detach().cpu().numpy()}")
print(f"b0,b1,b2     = {modelo.b0.item(): .5f}, {modelo.b1.item(): .5f}, {modelo.b2.item(): .5f}")
print(f"S sazonal    = {modelo.S.item(): .5f}")
print(f"sigma_obs    = {math.exp(modelo.log_sigma_obs.item()): .5f}")
print("\n======== MÉTRICAS (TESTE) ========")
print(f"MSE_teste  = {mse_teste: .5f}")
print(f"MAE_teste  = {mae_teste: .5f}")

# ---------------------- Previsão (12 meses) ----------------------
h = 12
with torch.no_grad():
    ultimos = df.iloc[-tam_janela:][covs].mean().values
    X_fut = torch.tensor(np.tile(ultimos, (h,1)), device=DISPOSITIVO, dtype=torch.float64)
    t0 = torch.tensor([len(df)-1], device=DISPOSITIVO, dtype=torch.float64)
    t_seq_fut = (t0 + torch.arange(1, h+1, device=DISPOSITIVO, dtype=torch.float64)).view(-1,1)
    y0 = torch.tensor(df["desemprego"].values[-1], device=DISPOSITIVO, dtype=torch.float64).clamp(1e-4, 1-1e-4)
    z0 = torch.log(y0/(1.0 - y0))
    U_mc_fut = modelo.simular(z0, X_fut, t_seq_fut, dt_discreto, h, n_mc=1024)
    fut_med = U_mc_fut.mean(0).cpu().numpy()
    fut_p10 = quantil_torch(U_mc_fut, 0.10, dim=0).cpu().numpy()
    fut_p90 = quantil_torch(U_mc_fut, 0.90, dim=0).cpu().numpy()

datas_fut = pd.date_range(df["data"].iloc[-1] + pd.offsets.MonthEnd(1), periods=h, freq="M")
plt.figure(figsize=(11,5.2))
plt.plot(df["data"], df["desemprego"], label="observado", lw=1.5)
plt.plot(datas_fut, fut_med, label="previsão média", lw=1.8)
plt.fill_between(datas_fut, fut_p10, fut_p90, alpha=0.25, label="80% (MC)")
plt.title("Projeção da Taxa de Desemprego — SDE Neural")
plt.ylabel("desemprego (proporção)")
plt.legend(); plt.tight_layout(); plt.show()
