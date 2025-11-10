# ============================================================
# Neural Vasicek — Curva de Juros via SDE + Rede Neural (PyTorch)
# Autor: Luiz Tiago Wilcke (LT)
# ============================================================
# Recursos:
#  - Dois modos:
#      (a) Vasicek paramétrico com parâmetros aprendidos (κ, θ, σ)
#      (b) Neural-SDE geral aprendendo μ(r,t,ctx) e σ(r,t,ctx)
#  - Preço de bonds:
#      (a) Fórmula fechada (Vasicek) com parâmetros neurais
#      (b) Monte Carlo da integral ∫ r_s ds (Euler–Maruyama)
#  - Perdas: MSE da curva, regularização física, suavidade temporal
#  - Treinador: AdamW + CosineAnnealing + Early stopping
#  - Gráficos: RMSE por maturidade e Curva média Observado vs Modelado
# ============================================================

import os, math, time, random, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------------------
# 0) Configurações gerais
# ---------------------------
@dataclass
class Configuracoes:
    semente: int = 42
    usar_cuda: bool = True

    # Dados (sintético ou CSV real)
    usar_csv: bool = False
    caminho_csv: Optional[str] = None  # CSV: data, y_0.25, y_0.5, y_1, y_2, y_3, y_5, y_10 (exemplo)
    n_dias: int = 600
    data_inicial: str = "2018-01-01"
    maturidades_anos: Tuple[float,...] = (0.25, 0.5, 1, 2, 3, 5, 10)

    # Verdadeiro (para gerar dados) — Vasicek
    kappa_true: float = 0.6
    theta_true: float = 0.06
    sigma_true: float = 0.015
    r0_true: float = 0.08
    ruido_y_bps: float = 5.0  # ruído observado (bp)

    # Treino
    epocas: int = 200
    batch: int = 64
    lr: float = 2e-3
    peso_reg_fisica: float = 1e-3
    peso_suavidade: float = 1e-3
    paciencia: int = 20

    # Modo de modelagem
    vasicek_parametrico: bool = True
    neural_sde: bool = False  # defina True para aprender μ e σ diretamente

    # Simulação
    qtd_caminhos_mc: int = 128
    dt_dias: int = 1  # passo em dias p/ EM e integral
    max_contexto_dias: int = 256  # janela p/ contexto

    # Arquitetura NN
    dim_oculta: int = 128
    camadas: int = 3
    dropout: float = 0.10

    # Gráficos / logs
    mostrar_graficos: bool = True
    salvar_figuras: bool = False
    pasta_saida: str = "saida_neural_vasicek"

cfg = Configuracoes()

# ---------------------------
# 1) Seeds e device
# ---------------------------
def fixar_semente(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def pegar_device(usar_cuda=True):
    if usar_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

fixar_semente(cfg.semente)
device = pegar_device(cfg.usar_cuda)
os.makedirs(cfg.pasta_saida, exist_ok=True)

# ---------------------------
# 2) Fórmulas Vasicek (analíticas)
# ---------------------------
def B_vasicek(T: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    return (1.0 - torch.exp(-kappa * T)) / (kappa + 1e-12)

def A_vasicek(T: torch.Tensor, kappa: torch.Tensor, theta: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    B = B_vasicek(T, kappa)
    term1 = (theta - (sigma**2)/(2.0*(kappa**2 + 1e-12))) * (B - T)
    term2 = (sigma**2 * (B**2)) / (4.0 * (kappa + 1e-12))
    return term1 - term2

def preco_bond_vasicek(r0: torch.Tensor,
                       T: torch.Tensor,
                       kappa: torch.Tensor,
                       theta: torch.Tensor,
                       sigma: torch.Tensor) -> torch.Tensor:
    B = B_vasicek(T, kappa)
    A = A_vasicek(T, kappa, theta, sigma)
    return torch.exp(A - B * r0)

def yield_vasicek(r0: torch.Tensor,
                  T: torch.Tensor,
                  kappa: torch.Tensor,
                  theta: torch.Tensor,
                  sigma: torch.Tensor) -> torch.Tensor:
    P = preco_bond_vasicek(r0, T, kappa, theta, sigma)
    return - torch.log(P + 1e-12) / (T + 1e-12)

# ---------------------------
# 3) Geração de dados sintéticos
# ---------------------------
def simular_vasicek_series(n_dias: int,
                           r0: float,
                           kappa: float,
                           theta: float,
                           sigma: float,
                           dt_dias: int = 1) -> np.ndarray:
    n_passos = n_dias // dt_dias
    r = np.zeros(n_passos, dtype=np.float64)
    r[0] = r0
    dt = dt_dias / 252.0
    for t in range(1, n_passos):
        dr = kappa*(theta - r[t-1])*dt + sigma*np.sqrt(dt)*np.random.randn()
        r[t] = max(r[t-1] + dr, -0.02)  # trava inferior leve
    return r

def gerar_base_sintetica(cfg: Configuracoes) -> pd.DataFrame:
    datas = pd.bdate_range(cfg.data_inicial, periods=cfg.n_dias, freq="B")
    r_short = simular_vasicek_series(cfg.n_dias, cfg.r0_true, cfg.kappa_true, cfg.theta_true, cfg.sigma_true, cfg.dt_dias)

    T_list = list(cfg.maturidades_anos)
    y_cols = {f"y_{m:g}": [] for m in T_list}

    # parâmetros verdadeiros constantes (referência)
    kappa_t = torch.tensor(cfg.kappa_true, dtype=torch.float64)
    theta_t = torch.tensor(cfg.theta_true, dtype=torch.float64)
    sigma_t = torch.tensor(cfg.sigma_true, dtype=torch.float64)

    for val in r_short:
        r0 = torch.tensor(val, dtype=torch.float64)
        for m in T_list:
            Tm = torch.tensor(m, dtype=torch.float64)
            y = float(yield_vasicek(r0, Tm, kappa_t, theta_t, sigma_t))
            y_noisy = y + (cfg.ruido_y_bps/1e4)*np.random.randn()
            y_cols[f"y_{m:g}"].append(y_noisy)

    df = pd.DataFrame({"data": datas[:len(r_short)], "r_short": r_short})
    df = pd.concat([df, pd.DataFrame(y_cols)], axis=1)
    return df

# ---------------------------
# 4) Dataset PyTorch (CORRIGIDO usando iloc)
# ---------------------------
class CurvaJurosDataset(Dataset):
    def __init__(self, df: pd.DataFrame, maturidades_anos: Tuple[float,...], janela_contexto: int):
        self.df = df.reset_index(drop=True).copy()
        self.maturidades = maturidades_anos
        self.janela = int(janela_contexto)

        # z-score
        self.media_r = float(self.df["r_short"].mean())
        self.std_r   = float(self.df["r_short"].std() + 1e-9)

        for m in self.maturidades:
            col = f"y_{m:g}"
            mu  = float(self.df[col].mean())
            sd  = float(self.df[col].std() + 1e-9)
            self.df[col+"_norm"] = (self.df[col] - mu)/sd

    def __len__(self):
        return len(self.df) - self.janela - 1

    def __getitem__(self, idx):
        inicio = idx
        fim    = idx + self.janela  # fim exclusivo
        r_ctx      = self.df.iloc[inicio:fim]["r_short"].to_numpy(dtype=np.float32)  # (janela,)
        r_ctx_norm = (r_ctx - self.media_r)/self.std_r

        t_alvo = fim
        y_alvo = []
        y_alvo_norm = []
        for m in self.maturidades:
            col = f"y_{m:g}"
            y_alvo.append(self.df.iloc[t_alvo][col])
            y_alvo_norm.append(self.df.iloc[t_alvo][col+"_norm"])
        y_alvo      = np.asarray(y_alvo, dtype=np.float32)
        y_alvo_norm = np.asarray(y_alvo_norm, dtype=np.float32)

        tempo_norm = np.linspace(0.0, 1.0, self.janela, endpoint=False, dtype=np.float32)
        assert r_ctx_norm.shape == tempo_norm.shape, f"Shapes diferentes: r={r_ctx_norm.shape} vs t={tempo_norm.shape}"
        X = np.stack((r_ctx_norm, tempo_norm), axis=1).astype(np.float32)  # (janela,2)

        return {
            "X": torch.from_numpy(X),                               # (T,2)
            "r_ultimo": torch.tensor(r_ctx[-1], dtype=torch.float32),
            "r_ultimo_norm": torch.tensor(r_ctx_norm[-1], dtype=torch.float32),
            "y_alvo": torch.from_numpy(y_alvo),                     # (Tm,)
            "y_alvo_norm": torch.from_numpy(y_alvo_norm),           # (Tm,)
        }

# ---------------------------
# 5) Redes neurais
# ---------------------------
class MLP(nn.Module):
    def __init__(self, d_in, d_h, d_out, camadas=3, dropout=0.1, ativ=nn.GELU):
        super().__init__()
        cam = []
        din = d_in
        for _ in range(camadas-1):
            cam += [nn.Linear(din, d_h), ativ(), nn.Dropout(dropout)]
            din = d_h
        cam += [nn.Linear(din, d_out)]
        self.net = nn.Sequential(*cam)
    def forward(self, x):
        return self.net(x)

class ParametrizacaoVasicek(nn.Module):
    """Prediz κ(t), θ(t), σ(t) a partir do contexto (vetor resumo)."""
    def __init__(self, dim_in, dim_oculta):
        super().__init__()
        self.mlp = MLP(dim_in, dim_oculta, 3, camadas=3, dropout=cfg.dropout)
        self.softplus = nn.Softplus()
        self.sigmoid  = nn.Sigmoid()
    def forward(self, resumo_ctx):
        bruto = self.mlp(resumo_ctx)  # (B,3)
        kappa = self.softplus(bruto[:,0]) + 1e-5
        theta = 0.01 + 0.15*self.sigmoid(bruto[:,1])     # [~1%, ~16%]
        sigma = 1e-5 + 0.10*self.softplus(bruto[:,2])    # positivo
        return kappa, theta, sigma

class NeuralDriftDifusao(nn.Module):
    """μ(r,t,ctx), σ(r,t,ctx) — Neural-SDE geral (σ>0 com Softplus)."""
    def __init__(self, dim_ctx, dim_oculta):
        super().__init__()
        self.mlp_mu = MLP(dim_ctx+2, dim_oculta, 1, camadas=3, dropout=cfg.dropout)
        self.mlp_si = MLP(dim_ctx+2, dim_oculta, 1, camadas=3, dropout=cfg.dropout)
        self.softplus = nn.Softplus()
    def forward(self, r, t_norm, resumo_ctx):
        x = torch.cat([r.unsqueeze(-1), t_norm.unsqueeze(-1), resumo_ctx], dim=1)
        mu = self.mlp_mu(x).squeeze(-1)
        si = self.softplus(self.mlp_si(x).squeeze(-1)) + 1e-5
        return mu, si

# ---------------------------
# 6) Agregador de contexto (simples; substituível por LSTM/Transformer)
# ---------------------------
class AgregadorContexto(nn.Module):
    """Converte a janela (T,2) em vetor resumo (mean + std + último)."""
    def __init__(self, d_in=2, d_h=64):
        super().__init__()
        self.proj = nn.Linear(d_in, d_h)
        self.ln   = nn.LayerNorm(d_h)
    def forward(self, X):
        # X: (B,T,2)
        H = self.ln(torch.tanh(self.proj(X)))  # (B,T,h)
        h_mean = H.mean(dim=1)
        h_std  = H.std(dim=1)
        h_last = H[:,-1,:]
        resumo = torch.cat([h_mean, h_std, h_last], dim=1)  # (B, 3h)
        return resumo

# ---------------------------
# 7) Modelo principal
# ---------------------------
class ModeloNeuralVasicek(nn.Module):
    def __init__(self, cfg: Configuracoes, n_matur: int):
        super().__init__()
        self.cfg = cfg
        self.agregado = AgregadorContexto(d_in=2, d_h=64)
        self.dim_resumo = 64*3

        if cfg.vasicek_parametrico:
            self.param_vas = ParametrizacaoVasicek(self.dim_resumo, cfg.dim_oculta)
        if cfg.neural_sde:
            self.neural_sde = NeuralDriftDifusao(self.dim_resumo, cfg.dim_oculta)

        self.n_matur = n_matur

    def simular_integral_r(self, r0, resumo_ctx, maturidades, qtd_caminhos, dt_anos):
        """
        Simula ∫_0^T r_s ds por Monte Carlo (Euler–Maruyama) para cada T.
        r_t segue:
          (a) Vasicek com parâmetros neurais (κ, θ, σ)   OU
          (b) Neural-SDE μ(r,t,ctx), σ(r,t,ctx)
        Retorna: E[∫ r_s ds] aproximado, por maturidade -> (B,Tm)
        """
        B = r0.shape[0]
        Tm = maturidades.shape[0]
        max_T = float(maturidades.max())
        n_passos = max(2, int(math.ceil(max_T/dt_anos)))

        r = r0.unsqueeze(1).repeat(1, qtd_caminhos)          # (B,MC)
        area_acum = torch.zeros(B, qtd_caminhos, device=r0.device)
        t_norm_grid = torch.linspace(0, 1, n_passos, device=r0.device).unsqueeze(0).repeat(B,1)

        if self.cfg.vasicek_parametrico:
            kappa, theta, sigma = self.param_vas(resumo_ctx)  # (B,)
            kappa = kappa.unsqueeze(1).repeat(1, qtd_caminhos)
            theta = theta.unsqueeze(1).repeat(1, qtd_caminhos)
            sigma = sigma.unsqueeze(1).repeat(1, qtd_caminhos)

        for i in range(n_passos):
            if self.cfg.vasicek_parametrico and (not self.cfg.neural_sde):
                mu = kappa*(theta - r)
                si = sigma
            else:
                t_norm = t_norm_grid[:, i]  # (B,)
                # usa r médio como estado resumido para estabilidade
                mu_b, si_b = self.neural_sde(r.mean(dim=1), t_norm, resumo_ctx)
                mu = mu_b.unsqueeze(1).repeat(1, qtd_caminhos)
                si = si_b.unsqueeze(1).repeat(1, qtd_caminhos)

            dW = torch.randn_like(r) * math.sqrt(dt_anos)
            dr = mu*dt_anos + si*dW
            r = r + dr
            area_acum = area_acum + r * dt_anos

        # Aproxima E[∫_0^{T_j} r_s ds] como fração de T_j/max_T do total
        frac = maturidades / (max_T + 1e-12)  # (Tm,)
        integrais = []
        for j in range(Tm):
            A_T = area_acum * frac[j]
            exp_A = A_T.mean(dim=1)  # média sobre MC -> (B,)
            integrais.append(exp_A)
        return torch.stack(integrais, dim=1)  # (B,Tm)

    def forward(self, X, r_ultimo, maturidades_anos):
        B = X.size(0)
        resumo_ctx = self.agregado(X)                      # (B,dim_resumo)
        matur = maturidades_anos.to(X.device).float()      # (Tm,)
        r0 = r_ultimo.float()

        dt_anos = (self.cfg.dt_dias/252.0)

        # Modo 1: Vasicek paramétrico + fórmula fechada (rápido/estável)
        if self.cfg.vasicek_parametrico and (not self.cfg.neural_sde):
            kappa, theta, sigma = self.param_vas(resumo_ctx)  # (B,)
            y_pred = []
            for j in range(matur.size(0)):
                Tj = matur[j].repeat(B,1)
                yj = yield_vasicek(r0.unsqueeze(1), Tj, kappa.unsqueeze(1), theta.unsqueeze(1), sigma.unsqueeze(1)).squeeze(1)
                y_pred.append(yj)
            y_pred = torch.stack(y_pred, dim=1)  # (B,Tm)
            info = {"kappa":kappa.detach(), "theta":theta.detach(), "sigma":sigma.detach()}
            return y_pred, info

        # Modo 2: Neural-SDE (ou híbrido): curva via Monte Carlo da integral
        qtd_mc = self.cfg.qtd_caminhos_mc
        integrais = self.simular_integral_r(r0, resumo_ctx, matur, qtd_mc, dt_anos)  # (B,Tm)
        y_pred = integrais / (matur.unsqueeze(0) + 1e-12)
        info = {}
        if self.cfg.vasicek_parametrico:
            kappa, theta, sigma = self.param_vas(resumo_ctx)
            info = {"kappa":kappa.detach(), "theta":theta.detach(), "sigma":sigma.detach()}
        return y_pred, info

# ---------------------------
# 8) Perdas e métricas
# ---------------------------
def perda_curva(y_pred, y_true):
    return F.mse_loss(y_pred, y_true)

def perda_regularizacao_fisica(info: Dict[str, torch.Tensor], peso: float):
    if len(info)==0: 
        return torch.tensor(0.0, device=device)
    kappa = info["kappa"]; theta = info["theta"]; sigma = info["sigma"]
    reg = 0.0
    reg += F.relu(0.03 - kappa).mean()              # empurra κ >= 0.03
    reg += (kappa**2).mean()*1e-3                   # limita κ
    reg += F.relu(theta - 0.18).mean() + F.relu(0.005 - theta).mean()  # θ ~ [0.5%,18%]
    reg += (sigma**2).mean()*1e-2                   # limita σ
    return peso * reg

def perda_suavidade_temporal(param_series: torch.Tensor, peso: float):
    if param_series is None or param_series.numel() < 3:
        return torch.tensor(0.0, device=device)
    dif = param_series[1:] - param_series[:-1]
    return peso * (dif**2).mean()

def rmse_por_maturidade(y_pred, y_true):
    with torch.no_grad():
        err = y_pred - y_true
        rmse = torch.sqrt((err**2).mean(dim=0))
    return rmse.detach().cpu().numpy()

# ---------------------------
# 9) Treinador
# ---------------------------
def treinar_modelo(cfg: Configuracoes, modelo: nn.Module, dl_treino, dl_valid, matur_tensor):
    opt = torch.optim.AdamW(modelo.parameters(), lr=cfg.lr, betas=(0.9,0.999), weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(10, cfg.epocas))
    melhor_valid = float("inf")
    pac = 0
    historico = {"treino":[], "valid":[]}

    for ep in range(1, cfg.epocas+1):
        modelo.train()
        perdas = []
        for batch in dl_treino:
            X = batch["X"].to(device).float()                    # (B,T,2)
            r_ult = batch["r_ultimo"].to(device).float()         # (B,)
            y_alvo = batch["y_alvo"].to(device).float()          # (B,Tm)

            y_pred, info = modelo(X, r_ult, matur_tensor)

            L_fit = perda_curva(y_pred, y_alvo)
            L_reg = perda_regularizacao_fisica(info, cfg.peso_reg_fisica)

            L_suav = 0.0
            if "kappa" in info:
                idx_ord = torch.argsort(r_ult)
                L_suav += perda_suavidade_temporal(info["kappa"][idx_ord], cfg.peso_suavidade)
                L_suav += perda_suavidade_temporal(info["theta"][idx_ord], cfg.peso_suavidade)
                L_suav += perda_suavidade_temporal(info["sigma"][idx_ord], cfg.peso_suavidade)

            L = L_fit + L_reg + L_suav
            opt.zero_grad()
            L.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            opt.step()
            perdas.append(float(L.detach().cpu()))

        sched.step()
        perda_tr = float(np.mean(perdas))

        # validação
        modelo.eval()
        perdas_val = []
        with torch.no_grad():
            for batch in dl_valid:
                X = batch["X"].to(device).float()
                r_ult = batch["r_ultimo"].to(device).float()
                y_alvo = batch["y_alvo"].to(device).float()
                y_pred, _ = modelo(X, r_ult, matur_tensor)
                perdas_val.append(float(perda_curva(y_pred, y_alvo).cpu()))
        perda_val = float(np.mean(perdas_val))
        historico["treino"].append(perda_tr)
        historico["valid"].append(perda_val)

        print(f"[Época {ep:03d}] treino={perda_tr:.6f} | valid={perda_val:.6f} | lr={sched.get_last_lr()[0]:.2e}")

        if perda_val + 1e-6 < melhor_valid:
            melhor_valid = perda_val
            pac = 0
            torch.save(modelo.state_dict(), os.path.join(cfg.pasta_saida, "melhor_modelo.pt"))
        else:
            pac += 1
            if pac >= cfg.paciencia:
                print("Early stopping!")
                break
    return historico

# ---------------------------
# 10) Avaliação e gráficos
# ---------------------------
def avaliar_modelo(cfg: Configuracoes, modelo: nn.Module, dl, matur_tensor, titulo="Validação"):
    modelo.eval()
    y_preds, y_trues = [], []
    with torch.no_grad():
        for batch in dl:
            X = batch["X"].to(device).float()
            r_ult = batch["r_ultimo"].to(device).float()
            y_alvo = batch["y_alvo"].to(device).float()
            y_pred, _ = modelo(X, r_ult, matur_tensor)
            y_preds.append(y_pred.cpu()); y_trues.append(y_alvo.cpu())
    y_pred = torch.cat(y_preds, dim=0)
    y_true = torch.cat(y_trues, dim=0)

    rmse = rmse_por_maturidade(y_pred, y_true)
    print(f"RMSE por maturidade ({titulo}):")
    for m, e in zip(cfg.maturidades_anos, rmse):
        print(f"  T={m:>5}: RMSE={e*1e4:7.3f} bps")

    if cfg.mostrar_graficos:
        plt.figure(figsize=(8,4))
        plt.plot(rmse*1e4, marker="o")
        plt.xticks(range(len(cfg.maturidades_anos)), [str(m) for m in cfg.maturidades_anos])
        plt.ylabel("RMSE (bps)"); plt.xlabel("Maturidade (anos)")
        plt.title(f"Neural Vasicek — RMSE por Maturidade ({titulo})")
        plt.grid(True, alpha=0.3)
        if cfg.salvar_figuras:
            plt.savefig(os.path.join(cfg.pasta_saida, f"rmse_{titulo}.png"), dpi=160, bbox_inches="tight")
        plt.show()

# ---------------------------
# 11) Carregar dados (CSV ou sintético)
# ---------------------------
def carregar_dados(cfg: Configuracoes) -> pd.DataFrame:
    if cfg.usar_csv and cfg.caminho_csv and os.path.exists(cfg.caminho_csv):
        df = pd.read_csv(cfg.caminho_csv, parse_dates=["data"])
        df = df.sort_values("data").reset_index(drop=True)
        return df
    else:
        print(">> Gerando base sintética (Vasicek verdadeiro)...")
        return gerar_base_sintetica(cfg)

# ---------------------------
# 12) Split e DataLoaders
# ---------------------------
def preparar_loaders(df: pd.DataFrame, cfg: Configuracoes):
    ds = CurvaJurosDataset(df, cfg.maturidades_anos, cfg.max_contexto_dias)
    n = len(ds)
    n_tr = int(0.7*n); n_val = int(0.15*n); n_te = n - n_tr - n_val
    ds_tr, ds_val, ds_te = torch.utils.data.random_split(
        ds, [n_tr, n_val, n_te],
        generator=torch.Generator().manual_seed(cfg.semente)
    )
    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch, shuffle=True, drop_last=True)
    dl_val = DataLoader(ds_val, batch_size=cfg.batch, shuffle=False, drop_last=False)
    dl_te  = DataLoader(ds_te,  batch_size=cfg.batch, shuffle=False, drop_last=False)
    return dl_tr, dl_val, dl_te

# ---------------------------
# 13) Main
# ---------------------------
def main(cfg: Configuracoes):
    df = carregar_dados(cfg)
    print(df.head())

    dl_tr, dl_val, dl_te = preparar_loaders(df, cfg)
    matur_tensor = torch.tensor(cfg.maturidades_anos, dtype=torch.float32, device=device)

    modelo = ModeloNeuralVasicek(cfg, n_matur=len(cfg.maturidades_anos)).to(device)

    print("\n=== Treinando modelo ===")
    _ = treinar_modelo(cfg, modelo, dl_tr, dl_val, matur_tensor)

    print("\n=== Avaliando (Validação) ===")
    avaliar_modelo(cfg, modelo, dl_val, matur_tensor, titulo="Validação")

    print("\n=== Avaliando (Teste) ===")
    avaliar_modelo(cfg, modelo, dl_te, matur_tensor, titulo="Teste")

    # Curva média prevista vs observada (teste)
    if cfg.mostrar_graficos:
        modelo.eval()
        y_pred_all, y_true_all = [], []
        with torch.no_grad():
            for b in dl_te:
                X = b["X"].to(device).float()
                r_ult = b["r_ultimo"].to(device).float()
                y_alvo = b["y_alvo"].to(device).float()
                yp, _ = modelo(X, r_ult, matur_tensor)
                y_pred_all.append(yp.cpu()); y_true_all.append(y_alvo.cpu())
        y_pred_all = torch.cat(y_pred_all, dim=0).mean(dim=0).numpy()
        y_true_all = torch.cat(y_true_all, dim=0).mean(dim=0).numpy()

        plt.figure(figsize=(7,4))
        plt.plot(cfg.maturidades_anos, y_true_all*100, "-o", label="Observado (média)")
        plt.plot(cfg.maturidades_anos, y_pred_all*100, "-s", label="Modelado (média)")
        plt.xlabel("Maturidade (anos)"); plt.ylabel("Yield (%)")
        plt.title("Curva Média — Observado vs Modelado")
        plt.grid(True, alpha=0.3); plt.legend()
        if cfg.salvar_figuras:
            plt.savefig(os.path.join(cfg.pasta_saida, "curva_media.png"), dpi=160, bbox_inches="tight")
        plt.show()

if __name__ == "__main__":
    main(cfg)
