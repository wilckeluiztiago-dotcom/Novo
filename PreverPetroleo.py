# ============================================================
# Preço do Petróleo (WTI) — LSTM + Atenção (PyTorch, compatível)
# Autor: Luiz Tiago Wilcke (LT)
# Correções:
#   - Removido 'verbose' de ReduceLROnPlateau (versões antigas do PyTorch)
#   - Print manual quando LR reduz
#   - Early stopping robusto (inicializa melhor_peso)
# ============================================================

import os
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# ==========================
# 0) Configurações gerais
# ==========================
@dataclass
class Configuracoes:
    semente: int = 42
    usar_cuda: bool = True
    tamanho_janela: int = 64
    horizonte: int = 5
    proporcao_treino: float = 0.75
    proporcao_valid: float = 0.10
    epocas: int = 200
    batch: int = 64
    lr_inicial: float = 1e-3
    dropout: float = 0.2
    num_camadas_lstm: int = 2
    dim_oculta: int = 96
    paciencia_es: int = 20
    fator_lr: float = 0.5
    paciencia_lr: int = 5
    plotar: bool = True

cfg = Configuracoes()

def fixar_semente(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

fixar_semente(cfg.semente)

dispositivo = torch.device("cuda" if (torch.cuda.is_available() and cfg.usar_cuda) else "cpu")
print(f"[INFO] Treinando em: {dispositivo}")

# ============================================================
# 1) Geração de dados sintéticos (realistas)
# ============================================================
def gerar_dados_sinteticos(n_dias: int = 2200) -> pd.DataFrame:
    datas = pd.date_range("2012-01-01", periods=n_dias, freq="D")
    t = np.arange(n_dias)

    tendencia = 0.02 * t / 365.0
    saz_anual = 3.0 * np.sin(2*np.pi*t/365.25)
    saz_sem = 0.6 * np.sin(2*np.pi*t/7.0)

    eps = np.random.normal(0, 1.2, size=n_dias)
    choque = np.zeros(n_dias)
    phi = 0.85
    for i in range(1, n_dias):
        choque[i] = phi*choque[i-1] + eps[i]
    for idx in [500, 820, 1200, 1650, 1900]:
        choque[idx:idx+20] += np.linspace(4.0, 0.0, 20)

    brent = 5.0 + 0.9*(50 + tendencia*8 + saz_anual + 0.6*saz_sem + choque) \
            + np.random.normal(0, 0.7, n_dias)

    usdbrl = 2.2 + 0.4*np.sin(2*np.pi*t/500.0) + 0.002*(t/365.0) \
             + np.random.normal(0, 0.05, n_dias)

    estoques = 400 + 12*np.sin(2*np.pi*t/365.25 + 1.5) \
               - 0.4*choque + np.random.normal(0, 1.8, n_dias)

    opep = 30 + 0.3*np.sin(2*np.pi*t/700.0) + 0.05*(t/365.0) \
           + np.random.normal(0, 0.08, n_dias)

    wti_base = 50 + tendencia*8 + saz_anual + 0.8*saz_sem + 0.9*choque
    wti = (wti_base
           - 0.015*(estoques - np.mean(estoques))
           + 0.008*(brent - np.mean(brent))
           - 0.9*(usdbrl - np.mean(usdbrl))
           + 0.4*(opep - np.mean(opep))
           + 0.015*(brent - 0.5*estoques) * (0.02*opep)
           + np.random.normal(0, 0.7, n_dias))

    df = pd.DataFrame({
        "data": datas,
        "wti": wti.astype(np.float32),
        "brent": brent.astype(np.float32),
        "usdbrl": usdbrl.astype(np.float32),
        "estoques_eua": estoques.astype(np.float32),
        "producao_opep": opep.astype(np.float32)
    })
    return df

dados = gerar_dados_sinteticos()
dados.set_index("data", inplace=True)
print(dados.head())

# ============================================================
# 2) Preparação: janelas e escalonamento
# ============================================================
variaveis_explicativas = ["wti","brent","usdbrl","estoques_eua","producao_opep"]
alvo = "wti"

val_inicio = int(len(dados)*cfg.proporcao_treino)
test_inicio = int(len(dados)*(cfg.proporcao_treino + cfg.proporcao_valid))

dados_np = dados[variaveis_explicativas].values.astype(np.float32)

scaler_x = StandardScaler()
scaler_y = StandardScaler()

x_treino = scaler_x.fit_transform(dados_np[:test_inicio])
y_treino = scaler_y.fit_transform(dados[[alvo]].values[:test_inicio])

x_total = scaler_x.transform(dados_np)
y_total = scaler_y.transform(dados[[alvo]].values)

def criar_janelas(x: np.ndarray, y: np.ndarray, janela: int, horizonte: int,
                  inicio: int, fim: int) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    limite = fim - janela - horizonte + 1
    for i in range(inicio, max(inicio, limite)):
        xs.append(x[i:i+janela, :])
        ys.append(y[i+janela:i+janela+horizonte, 0])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

X_tr, y_tr = criar_janelas(x_total, y_total, cfg.tamanho_janela, cfg.horizonte,
                           0, val_inicio)
X_va, y_va = criar_janelas(x_total, y_total, cfg.tamanho_janela, cfg.horizonte,
                           val_inicio, test_inicio)
X_te, y_te = criar_janelas(x_total, y_total, cfg.tamanho_janela, cfg.horizonte,
                           test_inicio, len(x_total))

print(f"[INFO] X_tr: ({len(X_tr)}, {cfg.tamanho_janela}, {len(variaveis_explicativas)}), "
      f"X_va: ({len(X_va)}, {cfg.tamanho_janela}, {len(variaveis_explicativas)}), "
      f"X_te: ({len(X_te)}, {cfg.tamanho_janela}, {len(variaveis_explicativas)})")

# ============================================================
# 3) Dataset / DataLoader
# ============================================================
class ConjuntoTemporal(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dl_tr = DataLoader(ConjuntoTemporal(X_tr, y_tr), batch_size=cfg.batch, shuffle=True, drop_last=True)
dl_va = DataLoader(ConjuntoTemporal(X_va, y_va), batch_size=cfg.batch, shuffle=False, drop_last=False)
dl_te = DataLoader(ConjuntoTemporal(X_te, y_te), batch_size=cfg.batch, shuffle=False, drop_last=False)

# ============================================================
# 4) Modelo: LSTM com Atenção
# ============================================================
class AtencaoSimples(nn.Module):
    def __init__(self, dim_oculta: int):
        super().__init__()
        self.W = nn.Linear(dim_oculta, dim_oculta, bias=False)
        self.v = nn.Linear(dim_oculta, 1, bias=False)

    def forward(self, H):  # H: (b, L, d)
        score = torch.tanh(self.W(H))               # (b, L, d)
        pesos = torch.softmax(self.v(score), dim=1) # (b, L, 1)
        contexto = torch.sum(pesos * H, dim=1)      # (b, d)
        return contexto, pesos.squeeze(-1)          # (b, d), (b, L)

class ModeloLSTMAttn(nn.Module):
    def __init__(self, dim_entrada: int, dim_oculta: int,
                 num_camadas: int, dropout: float, horizonte: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=dim_entrada,
            hidden_size=dim_oculta,
            num_layers=num_camadas,
            dropout=dropout if num_camadas > 1 else 0.0,
            batch_first=True,
            bidirectional=False
        )
        self.atencao = AtencaoSimples(dim_oculta)
        self.dropout = nn.Dropout(dropout)
        self.projecao = nn.Sequential(
            nn.Linear(dim_oculta, dim_oculta),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_oculta, horizonte)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):  # x: (b, L, d_in)
        H, _ = self.lstm(x)             # (b, L, d_h)
        contexto, pesos = self.atencao(H)
        out = self.projecao(self.dropout(contexto))  # (b, horizonte)
        return out, pesos

dim_entrada = len(variaveis_explicativas)
modelo = ModeloLSTMAttn(
    dim_entrada=dim_entrada,
    dim_oculta=cfg.dim_oculta,
    num_camadas=cfg.num_camadas_lstm,
    dropout=cfg.dropout,
    horizonte=cfg.horizonte
).to(dispositivo)

criterio = nn.MSELoss()
otimizador = torch.optim.AdamW(modelo.parameters(), lr=cfg.lr_inicial)

# ---- ReduceLROnPlateau sem 'verbose' (compatível com versões antigas)
reduz_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
    otimizador, mode="min", factor=cfg.fator_lr, patience=cfg.paciencia_lr
)

def obter_lr(opt):
    for pg in opt.param_groups:
        return pg.get("lr", None)

# ============================================================
# 5) Loop de treino com Early Stopping
# ============================================================
melhor_val = float("inf")
contador_es = 0
historico = {"treino": [], "valid": []}
melhor_peso = {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}  # inicial

for epoca in range(1, cfg.epocas+1):
    # ---- Treino ----
    modelo.train()
    perdas_tr = []
    for xb, yb in dl_tr:
        xb = xb.to(dispositivo)
        yb = yb.to(dispositivo)
        pred, _ = modelo(xb)
        perda = criterio(pred, yb)
        otimizador.zero_grad()
        perda.backward()
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=2.0)
        otimizador.step()
        perdas_tr.append(perda.item())

    # ---- Validação ----
    modelo.eval()
    perdas_va = []
    with torch.no_grad():
        for xb, yb in dl_va:
            xb = xb.to(dispositivo)
            yb = yb.to(dispositivo)
            pred, _ = modelo(xb)
            perda = criterio(pred, yb)
            perdas_va.append(perda.item())

    perda_tr_media = float(np.mean(perdas_tr))
    perda_va_media = float(np.mean(perdas_va)) if len(perdas_va) > 0 else perda_tr_media
    historico["treino"].append(perda_tr_media)
    historico["valid"].append(perda_va_media)

    lr_antes = obter_lr(otimizador)
    reduz_lr.step(perda_va_media)
    lr_depois = obter_lr(otimizador)
    msg_lr = f" | LR: {lr_depois:.2e}"
    if lr_depois is not None and lr_antes is not None and lr_depois < lr_antes:
        msg_lr += " (reduzido)"

    print(f"Época {epoca:03d} | Loss_tr: {perda_tr_media:.6f} | Loss_va: {perda_va_media:.6f}{msg_lr}")

    # Early Stopping
    if perda_va_media < melhor_val - 1e-5:
        melhor_val = perda_va_media
        contador_es = 0
        melhor_peso = {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}
    else:
        contador_es += 1
        if contador_es >= cfg.paciencia_es:
            print("[INFO] Early stopping acionado.")
            break

# Carrega melhor estado
modelo.load_state_dict(melhor_peso)

# ============================================================
# 6) Avaliação — métricas e previsões
# ============================================================
def desescalar_y(y_pad):
    y = scaler_y.inverse_transform(y_pad.reshape(-1, 1)).reshape(y_pad.shape)
    return y

def calcular_metricas(y_true_pad, y_pred_pad):
    y_true = desescalar_y(y_true_pad)
    y_pred = desescalar_y(y_pred_pad)
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mae  = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (np.maximum(np.abs(y_true), 1e-8)))) * 100
    return rmse, mae, mape

modelo.eval()
predicoes_pad, verdadeiros_pad = [], []
with torch.no_grad():
    for xb, yb in dl_te:
        xb = xb.to(dispositivo)
        yb = yb.to(dispositivo)
        pred, _ = modelo(xb)
        predicoes_pad.append(pred.detach().cpu().numpy())
        verdadeiros_pad.append(yb.detach().cpu().numpy())

predicoes_pad = np.vstack(predicoes_pad) if len(predicoes_pad) > 0 else np.empty((0, cfg.horizonte))
verdadeiros_pad = np.vstack(verdadeiros_pad) if len(verdadeiros_pad) > 0 else np.empty((0, cfg.horizonte))

if len(predicoes_pad) > 0:
    rmse, mae, mape = calcular_metricas(verdadeiros_pad, predicoes_pad)
    print(f"\n[MÉTRICAS TESTE] RMSE: {rmse:.4f}  |  MAE: {mae:.4f}  |  MAPE: {mape:.2f}%")

    ultimas_pred = desescalar_y(predicoes_pad[-1])
    print("\nÚltima janela — previsão multi-passo (USD/barril):")
    for h, valor in enumerate(ultimas_pred, start=1):
        print(f"T+{h}: {valor:,.2f}")

# Gráfico 1 passo à frente (opcional)
if cfg.plotar and len(predicoes_pad) > 0:
    y1_true = desescalar_y(verdadeiros_pad[:, 0])
    y1_pred = desescalar_y(predicoes_pad[:, 0])
    plt.figure(figsize=(10,4))
    plt.plot(y1_true, label="Verdadeiro (T+1)")
    plt.plot(y1_pred, label="Previsto (T+1)", alpha=0.8)
    plt.title("Previsão 1 passo à frente — WTI (teste)")
    plt.xlabel("Janela no conjunto de teste")
    plt.ylabel("USD/barril")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ============================================================
# 7) Função de previsão operacional (rolling)
# ============================================================
def prever_multiplos_passos(dados_df: pd.DataFrame, modelo_treinado: nn.Module,
                            janela: int, horizonte: int) -> np.ndarray:
    X_raw = dados_df[variaveis_explicativas].values.astype(np.float32)
    X_scaled = scaler_x.transform(X_raw)
    ult_janela = torch.tensor(X_scaled[-janela:], dtype=torch.float32).unsqueeze(0).to(dispositivo)
    modelo_treinado.eval()
    with torch.no_grad():
        pred_pad, _ = modelo_treinado(ult_janela)
    pred = desescalar_y(pred_pad.cpu().numpy().reshape(-1))
    return pred

previsao_operacional = prever_multiplos_passos(dados, modelo, cfg.tamanho_janela, cfg.horizonte)
print("\n[Previsão operacional a partir do último dia conhecido]")
for i, v in enumerate(previsao_operacional, start=1):
    print(f"Dia +{i}: {v:,.2f} USD/barril")
