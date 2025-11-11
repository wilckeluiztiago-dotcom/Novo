# ============================================================
# Ações — NSEJD-RV (Neural SDE com saltos e rugosidade)
#  Luiz Tiago Wilcke (LT)
# ============================================================

import os, math, random, time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# --------------------------
# 0) Configurações
# --------------------------
@dataclass
class Config:
    usar_csv: bool = False
    caminho_csv: Optional[str] = None  # CSV: data, preco, [volume, ibov, juros, cambio]
    freq: str = "D"
    proporcao_treino: float = 0.8
    dt: float = 1.0
    epocas: int = 100
    batch: int = 256
    lr: float = 1e-3
    seed: int = 42
    usar_cuda: bool = True
    tam_latente: int = 32
    tam_features: int = 4
    janela_ctx: int = 20
    lambda_mart: float = 0.1
    lambda_suav: float = 0.01
    horizontes_mc: int = 30
    n_amostras_mc: int = 1000
    taxa_risco_rf: float = 0.0  # pode ligar à SELIC diária

cfg = Config()
torch.manual_seed(cfg.seed); np.random.seed(cfg.seed); random.seed(cfg.seed)
device = torch.device("cuda" if (cfg.usar_cuda and torch.cuda.is_available()) else "cpu")
Tensor = torch.Tensor

# --------------------------
# 1) Dados (sintético se CSV ausente)
# --------------------------
def carregar_dados(cfg: Config) -> pd.DataFrame:
    if cfg.usar_csv and cfg.caminho_csv and os.path.exists(cfg.caminho_csv):
        df = pd.read_csv(cfg.caminho_csv)
        if "preco" not in df.columns:
            raise ValueError("CSV precisa conter coluna 'preco'.")
    else:
        n = 2000
        t = np.arange(n)
        preco = 100*np.exp(0.0005*t + 0.02*np.sin(2*np.pi*t/90)) * np.exp(0.2*np.random.randn(n).cumsum()/np.sqrt(n))
        volume = 1e6 + 2e5*np.sin(2*np.pi*t/30) + 1e5*np.random.randn(n)
        ibov = 100000 + 1000*np.sin(2*np.pi*t/240) + 500*np.random.randn(n)
        juros = 0.13 + 0.01*np.sin(2*np.pi*t/365) + 0.002*np.random.randn(n)
        cambio = 5.0 + 0.1*np.sin(2*np.pi*t/70) + 0.05*np.random.randn(n)
        df = pd.DataFrame({"preco": preco, "volume": volume, "ibov": ibov, "juros": juros, "cambio": cambio})
    df = df.dropna().reset_index(drop=True)
    return df

df = carregar_dados(cfg)
df["x"] = np.log(df["preco"].values)
df["ret"] = df["x"].diff()
df = df.dropna().reset_index(drop=True)

# features observáveis φ_t
col_feats = [c for c in ["volume","ibov","juros","cambio"] if c in df.columns][:cfg.tam_features]
X_feats = df[col_feats].values if col_feats else np.zeros((len(df),0))
scaler_feats = StandardScaler()
X_feats = scaler_feats.fit_transform(X_feats) if X_feats.shape[1] > 0 else X_feats

y = df["ret"].values.astype(np.float32)
x_log = df["x"].values.astype(np.float32)

# --------------------------
# 2) Dataset
# --------------------------
class SerieFinanceiraDS(Dataset):
    def __init__(self, ret, feats, x_log, janela_ctx: int):
        self.ret = torch.as_tensor(ret, dtype=torch.float32)                     # Evita UserWarning
        self.feats = torch.as_tensor(feats, dtype=torch.float32) if feats.shape[1] > 0 else torch.zeros((len(ret),0), dtype=torch.float32)
        self.x_log = torch.as_tensor(x_log, dtype=torch.float32)
        self.janela = int(janela_ctx)

    def __len__(self): return len(self.ret)

    def __getitem__(self, i):
        # r_t ~ média de retornos^2 da janela (proxy da rugosidade/vol)
        i0 = max(0, i - self.janela)
        rets2 = self.ret[i0:i]**2
        r_aprox = rets2.mean() if len(rets2) > 0 else torch.tensor(0.0, dtype=torch.float32)

        # IMPORTANTÍSSIMO: usar as_tensor para não recriar Tensor de Tensor
        r_aprox = torch.as_tensor(r_aprox, dtype=torch.float32)

        feat = self.feats[i] if self.feats.shape[1] > 0 else torch.zeros(0, dtype=torch.float32)
        x_prev = self.x_log[i] - self.ret[i]
        return {
            "ret": self.ret[i],
            "feat": feat,
            "r_aprox": r_aprox,
            "x_prev": x_prev
        }

n = len(y)
n_tr = int(cfg.proporcao_treino*n)
ds_tr = SerieFinanceiraDS(y[:n_tr], X_feats[:n_tr], x_log[:n_tr], cfg.janela_ctx)
ds_te = SerieFinanceiraDS(y[n_tr:], X_feats[n_tr:], x_log[n_tr:], cfg.janela_ctx)
dl_tr = DataLoader(ds_tr, batch_size=cfg.batch, shuffle=True)
dl_te = DataLoader(ds_te, batch_size=cfg.batch, shuffle=False)

# --------------------------
# 3) Blocos neurais
# --------------------------
def softplus(x: Tensor) -> Tensor:
    return torch.log1p(torch.exp(x))

class MLP(nn.Module):
    def __init__(self, din, dout, hidden=96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dout)
        )
    def forward(self, x): return self.net(x)

class ModeloNSEJD_RV(nn.Module):
    """
    - GRU (batch_first=True)
    - Parâmetros μ, κ, ξ, η, ψ, α, β, λ, mJ, sJ, (viéses) a partir de [z_t, φ_t, r_t]
    - Likelihood mistura (0 salto vs 1 salto)
    """
    def __init__(self, dim_feat: int, dim_lat: int = 32):
        super().__init__()
        self.dim_lat = dim_lat
        self.dim_feat = dim_feat
        self.gru = nn.GRU(input_size=dim_feat + 2, hidden_size=dim_lat, num_layers=1, batch_first=True)
        self.proj = MLP(dim_lat + dim_feat + 1, 12, hidden=96)
        self.gamma_logit = nn.Parameter(torch.tensor(0.2))  # γ ∈ (0,1)

    def forward_step(self, z: Tensor, feat: Tensor, r_t: Tensor):
        # z: [B, L], feat: [B, F], r_t: [B]
        if feat.ndim == 1: feat = feat.unsqueeze(0)
        if r_t.ndim == 1: r_t = r_t.unsqueeze(-1)
        h = torch.cat([z, feat, r_t], dim=-1)

        p = self.proj(h)
        mu, kappa, xi, eta, psi, alpha, beta, lam, mJ, sJ, vbias, dbias = torch.chunk(p, 12, dim=-1)

        kappa = softplus(kappa) + 1e-4
        xi    = softplus(xi)    + 1e-6
        eta   = softplus(eta)   + 1e-6
        psi   = softplus(psi)
        alpha = softplus(alpha)
        beta  = softplus(beta)  + 1e-4
        lam   = softplus(lam)   + 1e-8
        sJ    = softplus(sJ)    + 1e-6
        gamma = torch.sigmoid(self.gamma_logit)  # escalar 0..1

        mu = mu + 0.0*dbias  # placeholder para futuros vieses

        return dict(mu=mu, kappa=kappa, xi=xi, eta=eta, psi=psi, alpha=alpha, beta=beta,
                    lam=lam, mJ=mJ, sJ=sJ, gamma=gamma)

    def rollout_and_nll(self, lote, dt, rf, lambda_mart, lambda_suav, treinar=True):
        ret = lote["ret"].to(device)                # [B]
        feat = lote["feat"].to(device)              # [B, F] ou [B, 0]
        r_ap = lote["r_aprox"].to(device)           # [B]
        x_prev = lote["x_prev"].to(device)          # [B]

        B = ret.shape[0]
        # Entrada do GRU precisa ser [B, T, F_in] (batch_first=True).
        # Sequência T=1: concatenamos [feat, r_ap, ret]
        if feat.ndim == 1:  # quando F=0
            feat = feat.unsqueeze(-1) * 0.0  # vira [B,1] de zeros
        entrada = torch.cat([feat, r_ap.unsqueeze(-1), ret.unsqueeze(-1)], dim=-1)  # [B, F+2]
        entrada = entrada.unsqueeze(1)  # [B, 1, F+2]

        z0 = torch.zeros(1, B, self.dim_lat, device=device)  # hx: [num_layers, B, H]
        z_seq, zT = self.gru(entrada, z0)                    # z_seq: [B,1,H]
        z = z_seq[:, 0, :]                                   # [B,H]

        pars = self.forward_step(z, feat, r_ap)
        mu   = pars["mu"].squeeze(-1)
        kappa= pars["kappa"].squeeze(-1)
        xi   = pars["xi"].squeeze(-1)
        eta  = pars["eta"].squeeze(-1)
        psi  = pars["psi"].squeeze(-1)
        alpha= pars["alpha"].squeeze(-1)
        beta = pars["beta"].squeeze(-1)
        lam  = pars["lam"].squeeze(-1)
        mJ   = pars["mJ"].squeeze(-1)
        sJ   = pars["sJ"].squeeze(-1)
        gamma= pars["gamma"].mean()

        # v_t proxy inicial = r_ap
        v_t = torch.clamp(r_ap, min=1e-6)

        # Próximo v (usado só para suavidade)
        eps_v = torch.randn_like(v_t)
        v_next = v_t + kappa*(xi - v_t)*dt + eta*(v_t**gamma)*math.sqrt(dt)*eps_v + psi*r_ap*dt
        v_next = torch.clamp(v_next, min=1e-8)

        # Mistura 0 salto vs 1 salto
        mean0 = mu*dt
        var0  = torch.clamp(v_t*dt, min=1e-10)
        mean1 = mu*dt + mJ
        var1  = torch.clamp(v_t*dt + sJ**2, min=1e-10)

        def log_norm_pdf(x, m, v):
            return -0.5*(math.log(2*math.pi) + torch.log(v) + (x-m)**2 / v)

        logp0 = (-lam*dt) + log_norm_pdf(ret, mean0, var0)
        logp1 = (torch.log(lam*dt + 1e-12)) + log_norm_pdf(ret, mean1, var1)

        mstack = torch.stack([logp0, logp1], dim=-1)     # [B,2]
        maxl = torch.max(mstack, dim=-1, keepdim=True).values
        loglik = maxl.squeeze(-1) + torch.log(torch.exp(mstack - maxl).sum(-1) + 1e-12)

        nll = -loglik.mean()
        pen_mart = ((mu - 0.5*v_t - rf)**2).mean()
        pen_suav = ((v_next - v_t)**2).mean()

        loss = nll + lambda_mart*pen_mart + lambda_suav*pen_suav
        saida = dict(nll=nll.detach(), v_t=v_t.detach(), lam=lam.detach())
        return loss, saida

# --------------------------
# 4) Treinamento
# --------------------------
modelo = ModeloNSEJD_RV(dim_feat=(X_feats.shape[1] if X_feats.shape[1] > 0 else 0),
                        dim_lat=cfg.tam_latente).to(device)
opt = torch.optim.AdamW(modelo.parameters(), lr=cfg.lr)

hist = []
for ep in range(cfg.epocas):
    modelo.train()
    ep_loss = 0.0; n_batches = 0
    for lote in dl_tr:
        opt.zero_grad(set_to_none=True)
        loss, _ = modelo.rollout_and_nll(lote, cfg.dt, cfg.taxa_risco_rf, cfg.lambda_mart, cfg.lambda_suav, treinar=True)
        loss.backward()
        nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
        opt.step()
        ep_loss += float(loss.item()); n_batches += 1
    ep_loss /= max(1, n_batches)

    modelo.eval()
    with torch.no_grad():
        val_nll = 0.0; nb=0
        for lote in dl_te:
            _, saida = modelo.rollout_and_nll(lote, cfg.dt, cfg.taxa_risco_rf, cfg.lambda_mart, cfg.lambda_suav, treinar=False)
            val_nll += float(saida["nll"].item()); nb += 1
        val_nll /= max(1, nb)
    hist.append((ep_loss, val_nll))
    if (ep+1) % 10 == 0:
        print(f"Época {ep+1:03d} | loss={ep_loss:.4f} | val_nll={val_nll:.4f}")

# --------------------------
# 5) Previsão Monte Carlo
# --------------------------
def previsao_mc(modelo: ModeloNSEJD_RV, df: pd.DataFrame, X_feats: np.ndarray,
                ult_idx: int, passos: int, n_amostras: int, dt: float):
    modelo.eval()
    x0 = float(df["x"].iloc[ult_idx])

    janela = cfg.janela_ctx
    rets2 = (df["ret"].iloc[max(1, ult_idx-janela):ult_idx].values.astype(np.float32)**2)
    r_t = float(rets2.mean() if rets2.size > 0 else 1e-6)
    v_t = max(r_t, 1e-6)

    # feature do tempo atual
    if X_feats.shape[1] > 0:
        feat0 = torch.as_tensor(X_feats[ult_idx], dtype=torch.float32, device=device).unsqueeze(0)  # [1,F]
    else:
        feat0 = torch.zeros((1,0), dtype=torch.float32, device=device)                              # [1,0]

    # estado inicial do GRU: hx [num_layers, B, H] com B=1
    z = torch.zeros(1, 1, modelo.dim_lat, device=device)

    xs = np.zeros((n_amostras, passos+1), dtype=np.float64); xs[:,0] = x0
    for s in range(n_amostras):
        x = x0; r = r_t; v = v_t
        for tstep in range(1, passos+1):
            # entrada do GRU: [B, T=1, F_in] = concat(feat, r, ret_placeholder)
            r_tensor = torch.tensor([[r]], dtype=torch.float32, device=device)          # [1,1]
            ret_zero = torch.zeros((1,1), dtype=torch.float32, device=device)           # [1,1]
            entrada = torch.cat([feat0, r_tensor, ret_zero], dim=-1)                    # [1, F+2]
            entrada = entrada.unsqueeze(1)                                              # [1, 1, F+2]  << FIX CRÍTICO

            z_seq, z = modelo.gru(entrada, z)      # z_seq: [1,1,H], z(hx): [1,1,H]
            z_t = z_seq[:, 0, :]                    # [1,H]

            pars = modelo.forward_step(z_t, feat0, torch.tensor([r], dtype=torch.float32, device=device))
            mu   = float(pars["mu"].squeeze().item())
            kappa= float(pars["kappa"].squeeze().item())
            xi   = float(pars["xi"].squeeze().item())
            eta  = float(pars["eta"].squeeze().item())
            psi  = float(pars["psi"].squeeze().item())
            alpha= float(pars["alpha"].squeeze().item())
            beta = float(pars["beta"].squeeze().item())
            lam  = float(pars["lam"].squeeze().item())
            mJ   = float(pars["mJ"].squeeze().item())
            sJ   = float(pars["sJ"].squeeze().item())
            gamma= float(torch.sigmoid(modelo.gamma_logit).item())

            # salto (0/1)
            p1 = 1.0 - math.exp(-max(1e-12, lam)*dt)
            salto = (np.random.rand() < p1)
            J = np.random.normal(mJ, max(1e-8, sJ)) if salto else 0.0

            # difusão
            eps_s = np.random.randn()
            dx = (mu - 0.5*max(1e-10, v))*dt + math.sqrt(max(1e-10, v*dt))*eps_s + J
            x = x + dx

            # atualiza r e v
            r = max(1e-8, (1 - beta*dt)*r + alpha*(dx*dx)/max(1e-8, dt))
            eps_v = np.random.randn()
            v = v + kappa*(xi - v)*dt + eta*(max(1e-8, v)**gamma)*math.sqrt(dt)*eps_v + psi*r*dt
            v = max(1e-8, v)

            xs[s, tstep] = x
    return xs

xs = previsao_mc(modelo, df, X_feats, len(df)-1, cfg.horizontes_mc, cfg.n_amostras_mc, cfg.dt)
S = np.exp(xs)
preco0 = float(df["preco"].iloc[-1])
mediana = np.median(S[:, -1]); p5 = np.percentile(S[:, -1], 5); p95 = np.percentile(S[:, -1], 95)

print(f"Preço atual: {preco0:.2f}")
print(f"Projeção {cfg.horizontes_mc}d — mediana: {mediana:.2f}, 90% I.C.: [{p5:.2f}, {p95:.2f}]")

plt.figure(figsize=(9,4))
for k in np.linspace(5,95,19):
    q = np.percentile(S, k, axis=0)
    plt.plot(q, alpha=0.3)
plt.plot(np.median(S, axis=0), lw=2, label='Mediana')
plt.scatter([0],[preco0], label='Atual')
plt.title('Previsão Monte Carlo — NSEJD-RV (corrigido)')
plt.xlabel('Passos (dias)'); plt.ylabel('Preço simulado')
plt.legend(); plt.tight_layout(); plt.show()
