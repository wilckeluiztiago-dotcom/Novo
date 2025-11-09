# -*- coding: utf-8 -*-
# ============================================================
# Previsão da Taxa de Inflação — SDE (OU) + Rede Neural Avançada
# Autor: Luiz Tiago Wilcke (LT)
# Descrição (visão geral):
#   • Fator latente estocástico z_t ~ OU discretizado (tendência inflacionária).
#   • Rede temporal híbrida (TCN causal + LSTM + Atenção) para mapear X_t -> inflação.
# ============================================================

import os
import math
import json
import random
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------
# 0) Sementes e dispositivo
# ------------------------------------------------------------
SEMENTE = 42
random.seed(SEMENTE); np.random.seed(SEMENTE); torch.manual_seed(SEMENTE)
DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# 1) Dados
# ============================================================

@dataclass
class ConfigDados:
    usar_csv: bool = False
    caminho_csv: Optional[str] = None
    # Esperado se usar CSV:
    # colunas mínimas: data, inflacao (ex.: IPCA % m/m)
    # colunas exógenas sugeridas: cambio, juros, desemprego, salario, commodities, expectativa, output_gap
    nome_col_data: str = "data"
    nome_col_inflacao: str = "inflacao"
    col_exogenas: Tuple[str, ...] = (
        "cambio", "juros", "desemprego", "salario", "commodities",
        "expectativa", "output_gap"
    )
    freq: str = "MS"    # mensal
    tam_janela: int = 24
    horizonte: int = 1  # previsão 1 passo à frente

cfg_dados = ConfigDados(
    usar_csv=False,
    caminho_csv=None
)

def gerar_dados_sinteticos(n_meses: int = 180, freq: str = "MS") -> pd.DataFrame:
    """Gera série mensal de inflação e exógenas com um fator latente z_t (OU)."""
    datas = pd.date_range("2010-01-01", periods=n_meses, freq=freq)

    # SDE (OU) contínua discretizada: z_{t+1} = z_t + kappa*(mu - z_t)*dt + sigma*sqrt(dt)*eps
    dt = 1.0/12.0
    kappa_true, mu_true, sigma_true = 0.35, 0.004, 0.10
    z = np.zeros(n_meses, dtype=float)
    z[0] = 0.006
    for t in range(n_meses-1):
        eps = np.random.randn()
        z[t+1] = z[t] + kappa_true*(mu_true - z[t])*dt + sigma_true*np.sqrt(dt)*eps

    # Exógenas correlacionadas ao z_t + ruído
    rng = np.random.default_rng(SEMENTE)
    cambio       = 5.0 + 0.8*z + 0.3*rng.normal(size=n_meses)
    juros        = 10.0 + 2.0*z + 0.5*rng.normal(size=n_meses)
    desemprego   = 8.0 - 3.0*z + 0.7*rng.normal(size=n_meses)
    salario      = 3000.0*(1 + 0.002*np.arange(n_meses))*(1 + 0.2*z) + 50*rng.normal(size=n_meses)
    commodities  = 100.0 + 10.0*z + 2.0*rng.normal(size=n_meses)
    expectativa  = 0.004 + 0.5*z + 0.001*rng.normal(size=n_meses)
    output_gap   = 0.2*z + 0.05*rng.normal(size=n_meses)

    # Inflação observada: mistura linear + não linear de (z, exógenas) + ruído
    inflacao = (
        0.6*z
        + 0.02*(cambio - np.mean(cambio))
        + 0.01*(juros - np.mean(juros))
        - 0.015*(desemprego - np.mean(desemprego))
        + 0.000003*(salario - np.mean(salario))
        + 0.008*(commodities - np.mean(commodities))
        + 0.9*(expectativa - np.mean(expectativa))
        + 0.5*(output_gap - np.mean(output_gap))
        + 0.001*np.sin(np.linspace(0, 12*np.pi, n_meses))
        + 0.05*rng.normal(size=n_meses)
    )

    df = pd.DataFrame({
        "data": datas,
        "inflacao": inflacao,
        "cambio": cambio,
        "juros": juros,
        "desemprego": desemprego,
        "salario": salario,
        "commodities": commodities,
        "expectativa": expectativa,
        "output_gap": output_gap,
        "fator_latente_true": z
    })
    return df

def carregar_dados(cfg: ConfigDados) -> pd.DataFrame:
    if cfg.usar_csv and cfg.caminho_csv and os.path.exists(cfg.caminho_csv):
        df = pd.read_csv(cfg.caminho_csv)
        df[cfg.nome_col_data] = pd.to_datetime(df[cfg.nome_col_data])
        df = df.sort_values(cfg.nome_col_data).reset_index(drop=True)
        return df
    # fallback: sintético
    return gerar_dados_sinteticos(n_meses=220, freq=cfg.freq)

df_bruto = carregar_dados(cfg_dados)

# ------------------------------------------------------------
# 1.1) Featurização: janelas, diferenças, lags
# ------------------------------------------------------------
def construir_matriz(df: pd.DataFrame, cfg: ConfigDados) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DatetimeIndex]:
    df = df.copy()
    df = df.sort_values(cfg.nome_col_data).reset_index(drop=True)
    df.set_index(cfg.nome_col_data, inplace=True)

    # Lags da inflação e exógenas
    def add_lags(col, n=12):
        for k in range(1, n+1):
            df[f"{col}_lag{k}"] = df[col].shift(k)

    add_lags(cfg.nome_col_inflacao, n=12)
    for col in cfg.col_exogenas:
        if col in df.columns:
            add_lags(col, n=12)

    # Diferenças / variações % simples nas exógenas selecionadas
    for col in ("cambio", "juros", "commodities", "salario"):
        if col in df.columns:
            df[f"{col}_diff"] = df[col].diff()
            df[f"{col}_ret"]  = df[col].pct_change()

    # Sazonalidade explícita (mês do ano, seno/cosseno)
    df["mes"] = df.index.month
    df["sen"] = np.sin(2*np.pi*df["mes"]/12.0)
    df["cos"] = np.cos(2*np.pi*df["mes"]/12.0)

    df = df.dropna()
    y = df[cfg.nome_col_inflacao].shift(-cfg.horizonte)  # alvo à frente
    df_feat = df.drop(columns=[cfg.nome_col_inflacao], errors="ignore")
    X = df_feat.values
    y = y.loc[df_feat.index].values
    tempo = df_feat.index

    # Limpa últimas linhas (por shift -h)
    mask = ~np.isnan(y)
    X, y, tempo = X[mask], y[mask], tempo[mask]

    nomes = list(df_feat.columns)
    return X, y, nomes, tempo

X, y, nomes_colunas, datas = construir_matriz(df_bruto, cfg_dados)

# ------------------------------------------------------------
# 1.2) Escalonamento e criação de janelas para série temporal
# ------------------------------------------------------------
tam_janela = cfg_dados.tam_janela
h = cfg_dados.horizonte

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_esc = scaler_X.fit_transform(X)
y_esc = scaler_y.fit_transform(y.reshape(-1,1)).ravel()

def criar_tensores_sequencias(Xa: np.ndarray, ya: np.ndarray, janela: int) -> Tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    for t in range(janela, len(Xa)):
        xs.append(Xa[t-janela:t, :])
        ys.append(ya[t])
    Xseq = torch.tensor(np.array(xs), dtype=torch.float32)
    Yseq = torch.tensor(np.array(ys), dtype=torch.float32).unsqueeze(-1)
    return Xseq, Yseq

Xseq, Yseq = criar_tensores_sequencias(X_esc, y_esc, tam_janela)

# Split temporal (80/20)
n_total = len(Xseq)
n_treino = int(0.8*n_total)
Xtr, Ytr = Xseq[:n_treino], Yseq[:n_treino]
Xva, Yva = Xseq[n_treino:], Yseq[n_treino:]
datas_seq = datas[tam_janela:]  # datas alinhadas às sequências
datas_tr, datas_va = datas_seq[:n_treino], datas_seq[n_treino:]

# ============================================================
# 2) Blocos de Rede: TCN causal + LSTM + Atenção
# ============================================================

class BlocoTCN(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, dilatacoes: List[int], kernel: int = 3, dropout: float = 0.1):
        super().__init__()
        camadas = []
        d_in = dim_in
        for d in dilatacoes:
            pad = (kernel-1)*d
            conv = nn.Conv1d(d_in, dim_out, kernel_size=kernel, padding=pad, dilation=d)
            camadas += [conv, nn.GELU(), nn.Dropout(dropout)]
            d_in = dim_out
        self.rede = nn.Sequential(*camadas)

    def forward(self, x):
        # x: (B, T, F) -> (B, F, T) para conv1d
        x = x.transpose(1,2)
        y = self.rede(x)
        # Causal crop: remove lookahead gerado pelo padding
        excesso = y.size(-1) - x.size(-1)
        if excesso > 0:
            y = y[..., :-excesso]
        return y.transpose(1,2)  # volta (B, T, F_out)

class AtencaoEscalar(nn.Module):
    """Atenção temporal simples (score escalar por tempo)."""
    def __init__(self, dim: int):
        super().__init__()
        self.w = nn.Linear(dim, 1)

    def forward(self, h):  # h: (B, T, D)
        scores = self.w(h).squeeze(-1)          # (B, T)
        pesos = torch.softmax(scores, dim=-1)   # (B, T)
        contexto = torch.einsum("btd,bt->bd", h, pesos)  # (B, D)
        return contexto, pesos

# ============================================================
# 3) Fator Latente Estocástico (OU) + Perda Física
# ============================================================

class FatorLatenteOU(nn.Module):
    """
    Modelo latente z_t (um por sequência) estimado de forma diferenciável.
    A dinâmica é penalizada via perda física (resíduo da SDE OU discretizada).
    """
    def __init__(self, tam_janela: int, dt: float = 1.0/12.0):
        super().__init__()
        self.tam_janela = tam_janela
        self.dt = dt
        # Parâmetros treináveis da SDE
        # kappa>0, sigma>0 garantidos via softplus
        self._kappa = nn.Parameter(torch.tensor(0.3))
        self._mu    = nn.Parameter(torch.tensor(0.0))
        self._sigma = nn.Parameter(torch.tensor(0.1))

        # Estado latente inicial por amostra será inferido por uma pequena rede do passado X
        self.encoder_z0 = nn.Sequential(
            nn.Linear(64, 64), nn.GELU(),
            nn.Linear(64, 1)
        )

    def parametros(self):
        kappa = F.softplus(self._kappa) + 1e-5
        mu    = self._mu
        sigma = F.softplus(self._sigma) + 1e-5
        return kappa, mu, sigma

    def forward(self, h_resumo: torch.Tensor) -> torch.Tensor:
        """
        h_resumo: (B, 64) embedding do passado (derivado da backbone)
        Retorna z_seq: (B, T) valores latentes ao longo da janela.
        """
        B = h_resumo.size(0)
        kappa, mu, sigma = self.parametros()
        z0 = self.encoder_z0(h_resumo).squeeze(-1)  # (B,)

        # Simulação diferenciável do OU (E-M sem ruído explícito; ruído é absorvido na penalização)
        # Para estabilidade numérica, geramos z determinístico no forward e
        # cobramos o ruído "implícito" na loss física via incrementos.
        T = self.tam_janela
        dt = self.dt
        z = [z0]
        for t in range(T-1):
            z_next = z[-1] + kappa*(mu - z[-1])*dt
            z.append(z_next)
        z_seq = torch.stack(z, dim=1)  # (B, T)
        return z_seq

    def perda_fisica(self, z_seq: torch.Tensor) -> torch.Tensor:
        """
        Penaliza o resíduo da SDE OU: z_{t+1} - z_t - kappa*(mu - z_t)*dt ~ N(0, (sigma^2 dt))
        Minimização ~ MSE normalizado por var.
        """
        kappa, mu, sigma = self.parametros()
        dt = self.dt
        inc = z_seq[:, 1:] - z_seq[:, :-1] - kappa*(mu - z_seq[:, :-1])*dt  # (B, T-1)
        var = (sigma**2)*dt + 1e-8
        loss = torch.mean((inc**2)/var)
        return loss

# ============================================================
# 4) Modelo Final: Backbone temporal + cabeças (previsão e OU)
# ============================================================

class ModeloInflacao(nn.Module):
    def __init__(self, n_feat: int, tam_janela: int, d_model: int = 128, dropout: float = 0.15):
        super().__init__()
        self.tam_janela = tam_janela

        # Projeção inicial
        self.proj = nn.Linear(n_feat, 64)

        # TCN causal (dilatações 1,2,4,8)
        self.tcn = BlocoTCN(dim_in=64, dim_out=64, dilatacoes=[1,2,4,8], kernel=3, dropout=dropout)

        # LSTM (captura dependência de longo prazo)
        self.lstm = nn.LSTM(input_size=64, hidden_size=d_model, num_layers=2, batch_first=True, dropout=dropout)

        # Atenção temporal
        self.att = AtencaoEscalar(d_model)

        # Cabeça de previsão da inflação (usa também o z_t reconstruído)
        self.fc_pred = nn.Sequential(
            nn.Linear(d_model + tam_janela, 128), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 1)
        )

        # Fator latente OU (usa embedding do passado, aqui tiramos o estado final da LSTM)
        self.mod_ou = FatorLatenteOU(tam_janela=tam_janela, dt=1.0/12.0)

        # Ponte para encoder_z0 do OU (reduz d_model->64)
        self.cond_z0 = nn.Linear(d_model, 64)

    def forward(self, x):  # x: (B, T, F)
        h0 = self.proj(x)                 # (B, T, 64)
        h_tcn = self.tcn(h0)              # (B, T, 64)
        h_lstm, _ = self.lstm(h_tcn)      # (B, T, d_model)

        # Atenção temporal
        contexto, pesos = self.att(h_lstm)  # (B, d_model), (B, T)

        # Fator latente OU condicionado ao estado final (resumo)
        h_resumo = self.cond_z0(contexto)   # (B, 64)
        z_seq = self.mod_ou(h_resumo)       # (B, T)

        # Concatena contexto com z_seq (achatado) para prever y_{t+1}
        z_flat = z_seq  # (B, T)
        saida = self.fc_pred(torch.cat([contexto, z_flat], dim=-1))  # (B,1)
        return saida, z_seq, pesos

    def perda_composta(self, y_pred, y_true, z_seq, alpha_fisica=0.3, l2=1e-4):
        # Erro de previsão
        loss_pred = F.mse_loss(y_pred, y_true)

        # Penalização física do OU
        loss_fis = self.mod_ou.perda_fisica(z_seq)

        # L2 em pesos (exclui biases e LayerNorm)
        l2_reg = torch.tensor(0., device=y_pred.device)
        for n, p in self.named_parameters():
            if p.requires_grad and p.dim() > 1:
                l2_reg = l2_reg + (p**2).mean()

        loss = loss_pred + alpha_fisica*loss_fis + l2*l2_reg
        comp = {
            "loss_total": loss.detach().item(),
            "loss_pred": loss_pred.detach().item(),
            "loss_fis": loss_fis.detach().item(),
            "l2": l2_reg.detach().item()
        }
        return loss, comp

# ============================================================
# 5) Treino
# ============================================================

@dataclass
class ConfigTreino:
    epocas: int = 200
    batch: int = 64
    lr: float = 2e-3
    alpha_fisica: float = 0.4
    l2: float = 1e-4
    paciencia: int = 20
    clip_grad: float = 1.0

cfg_treino = ConfigTreino()

def iter_batches(X, Y, bs):
    n = len(X)
    idx = np.arange(n)
    np.random.shuffle(idx)
    for i in range(0, n, bs):
        sel = idx[i:i+bs]
        yield X[sel], Y[sel]

modelo = ModeloInflacao(n_feat=Xtr.size(-1), tam_janela=tam_janela).to(DISPOSITIVO)
opt = torch.optim.AdamW(modelo.parameters(), lr=cfg_treino.lr)
sched = torch.optim.lr_scheduler.OneCycleLR(
    opt, max_lr=cfg_treino.lr, epochs=cfg_treino.epocas, steps_per_epoch=max(1, len(Xtr)//cfg_treino.batch)
)

historico = {"treino": [], "val": []}
melhor_val = float("inf")
melhor_estado = None
sem_melhora = 0

for ep in range(1, cfg_treino.epocas+1):
    modelo.train()
    perdas = []
    for xb, yb in iter_batches(Xtr, Ytr, cfg_treino.batch):
        xb = xb.to(DISPOSITIVO); yb = yb.to(DISPOSITIVO)
        opt.zero_grad(set_to_none=True)
        yhat, zseq, _ = modelo(xb)
        loss, comp = modelo.perda_composta(yhat, yb, zseq, alpha_fisica=cfg_treino.alpha_fisica, l2=cfg_treino.l2)
        loss.backward()
        nn.utils.clip_grad_norm_(modelo.parameters(), cfg_treino.clip_grad)
        opt.step()
        sched.step()
        perdas.append(comp)
    med_treino = {k: float(np.mean([d[k] for d in perdas])) for k in perdas[0].keys()}

    # Validação
    modelo.eval()
    with torch.no_grad():
        yhat_va, z_va, _ = modelo(Xva.to(DISPOSITIVO))
        loss_va, comp_va = modelo.perda_composta(yhat_va, Yva.to(DISPOSITIVO), z_va,
                                                 alpha_fisica=cfg_treino.alpha_fisica, l2=cfg_treino.l2)

    historico["treino"].append(med_treino)
    historico["val"].append(comp_va)

    if comp_va["loss_total"] < melhor_val - 1e-5:
        melhor_val = comp_va["loss_total"]
        melhor_estado = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in modelo.state_dict().items()}
        sem_melhora = 0
    else:
        sem_melhora += 1

    print(f"Época {ep:03d} | train: {med_treino['loss_total']:.4f} "
          f"(pred {med_treino['loss_pred']:.4f}, fis {med_treino['loss_fis']:.4f}) "
          f"| val: {comp_va['loss_total']:.4f}")

    if sem_melhora >= cfg_treino.paciencia:
        print("Early stopping por paciência.")
        break

# Restaura melhor estado
if melhor_estado is not None:
    modelo.load_state_dict(melhor_estado)

# ============================================================
# 6) Avaliação e gráficos
# ============================================================

modelo.eval()
with torch.no_grad():
    yhat_tr, z_tr, att_tr = modelo(Xtr.to(DISPOSITIVO))
    yhat_va, z_va, att_va = modelo(Xva.to(DISPOSITIVO))

def desscala(v):
    v = v.detach().cpu().numpy().reshape(-1,1)
    return scaler_y.inverse_transform(v).ravel()

y_tr_pred = desscala(yhat_tr)
y_va_pred = desscala(yhat_va)
y_tr_true = scaler_y.inverse_transform(Ytr.cpu().numpy()).ravel()
y_va_true = scaler_y.inverse_transform(Yva.cpu().numpy()).ravel()

# Métricas simples
def rmse(a,b): return float(np.sqrt(np.mean((a-b)**2)))
def mae(a,b):  return float(np.mean(np.abs(a-b)))

print("\n--- Métricas ---")
print(f"Treino: RMSE={rmse(y_tr_true,y_tr_pred):.5f}  MAE={mae(y_tr_true,y_tr_pred):.5f}")
print(f"Val   : RMSE={rmse(y_va_true,y_va_pred):.5f}  MAE={mae(y_va_true,y_va_pred):.5f}")

# Curvas de perda
loss_tr = [d["loss_total"] for d in historico["treino"]]
loss_va = [d["loss_total"] for d in historico["val"]]

plt.figure(figsize=(9,4))
plt.plot(loss_tr, label="treino")
plt.plot(loss_va, label="validação")
plt.title("Perda total por época")
plt.xlabel("época"); plt.ylabel("loss")
plt.legend(); plt.tight_layout(); plt.show()

# Série prevista (validação)
plt.figure(figsize=(10,4))
plt.plot(datas_va, y_va_true, label="inflação observada")
plt.plot(datas_va, y_va_pred, label="previsão (modelo)", linestyle="--")
plt.title("Validação — Inflação observada vs previsão")
plt.xlabel("tempo"); plt.ylabel("inflação (escala original)")
plt.legend(); plt.tight_layout(); plt.show()

# Fator latente (exemplo: média por tempo no conjunto)
z_va_np = z_va.detach().cpu().numpy()
z_tr_np = z_tr.detach().cpu().numpy()
plt.figure(figsize=(10,3.5))
plt.plot(datas_tr[:z_tr_np.shape[1]], z_tr_np.mean(axis=0), label="z_t (média — treino)")
plt.plot(datas_va[:z_va_np.shape[1]], z_va_np.mean(axis=0), label="z_t (média — validação)")
if "fator_latente_true" in df_bruto.columns:
    # para dados sintéticos, mostramos o verdadeiro
    z_true = df_bruto.set_index("data")["fator_latente_true"].iloc[tam_janela: tam_janela+len(datas_seq)]
    plt.plot(datas_seq[:len(z_true)], z_true.values, label="z_t verdadeiro (sintético)", linestyle=":")
plt.title("Fator latente estocástico (OU) — visão média")
plt.xlabel("tempo"); plt.ylabel("z_t")
plt.legend(); plt.tight_layout(); plt.show()

# Importância temporal média (atenção)
att_va_np = att_va.detach().cpu().numpy()
plt.figure(figsize=(10,3.5))
plt.plot(np.arange(1, tam_janela+1), att_va_np.mean(axis=0))
plt.title("Importância temporal média (atenção)")
plt.xlabel("defasagem dentro da janela (t-τ)"); plt.ylabel("peso médio")
plt.tight_layout(); plt.show()

# ============================================================
# 7) Rotina de previsão para novos dados
# ============================================================

def prever_proxima_inflacao(janela_X: np.ndarray, modelo: ModeloInflacao,
                            scaler_X: StandardScaler, scaler_y: StandardScaler) -> float:
    """
    janela_X: matriz (T, F) das últimas T observações já featurizadas e escalonadas.
    Retorna previsão desscalada.
    """
    modelo.eval()
    Xn = scaler_X.transform(janela_X)
    Xseq_ = torch.tensor(Xn[None, ...], dtype=torch.float32).to(DISPOSITIVO)
    with torch.no_grad():
        yhat, _, _ = modelo(Xseq_)
    y_pred = scaler_y.inverse_transform(yhat.cpu().numpy()).ravel()[0]
    return float(y_pred)

print("\nExemplo de uso de inferência (dummy com a última janela de validação):")
ult_janela = X_esc[len(X_esc)-tam_janela:len(X_esc), :]
print("Previsão próxima inflação:", prever_proxima_inflacao(ult_janela, modelo, scaler_X, scaler_y))
