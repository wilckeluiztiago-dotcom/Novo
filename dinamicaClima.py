# ============================================================
# Sistema Climático Neural: PDE + CNN (PyTorch)
# Autor: Luiz Tiago Wilcke (LT)
# Descrição resumida:
#   - Domínio 2D retangular com PDE acoplada de Temperatura (T) e CO2 (C):
#       ∂T/∂t = κ_T ∇²T - u·∇T + α*C + F_T + ξ_T
#       ∂C/∂t = κ_C ∇²C - u·∇C - β*T + F_C + ξ_C
#     (difusão κ, advecção u(x,y), acoplamentos α e β, fontes F_• e ruído ξ_•)
#   - Discretização por diferenças finitas (explícito, passo pequeno Δt).
#   - Gera dataset de pares (estado_t -> estado_t+1) e treina CNN para emular a PDE.
#   - Compara rollout PDE vs. CNN, gráficos e métricas (14 dígitos).
# ============================================================

import os
import math
import time
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------- (opcional) PyTorch ---------
tem_pytorch = True
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception as e:
    tem_pytorch = False

# -------------------- Configurações --------------------
np.random.seed(42)

@dataclass
class ParametrosPDE:
    nx: int = 64
    ny: int = 64
    dx: float = 1.0
    dy: float = 1.0
    dt: float = 0.05
    passos: int = 160             # passos totais p/ simulação
    salvar_cada: int = 4          # salvar estado a cada N passos (para dataset)
    kappa_T: float = 0.20         # difusão T
    kappa_C: float = 0.10         # difusão C
    alpha: float = 0.015          # acoplamento C -> T (efeito estufa)
    beta: float = 0.008           # acoplamento T -> C (sumidouros/vegetação)
    ruido_T: float = 0.0005       # ruído térmico
    ruido_C: float = 0.0005       # ruído CO2
    forca_T_base: float = 0.0005  # forçante média T (ex.: balanço radiativo)
    forca_C_base: float = 0.0002  # forçante média C (ex.: emissões médias)

@dataclass
class ParametrosCNN:
    epocas: int = 6
    batch: int = 64
    lr: float = 1e-3
    proporcao_treino: float = 0.8

# -------------------- Campos auxiliares --------------------
def gerar_campo_vento(nx, ny):
    # campo de vento não-divergente simples: rotação fraca
    y, x = np.mgrid[0:ny, 0:nx]
    cx, cy = (nx-1)/2, (ny-1)/2
    dx = (x - cx) / max(cx, 1)
    dy = (y - cy) / max(cy, 1)
    # velocidade tangencial suave
    u = -0.4 * dy
    v =  0.4 * dx
    return u.astype(np.float32), v.astype(np.float32)

def laplaciano(z, dx, dy):
    # fronteira: Neumann (derivada normal ~ 0) via extrapolação simples
    z_ext = z.copy()
    z_ext[0, :]  = z[1, :]     # topo ~ refletido
    z_ext[-1, :] = z[-2, :]
    z_ext[:, 0]  = z[:, 1]
    z_ext[:, -1] = z[:, -2]
    # 5-pontos
    return ((np.roll(z_ext, +1, axis=0) - 2*z_ext + np.roll(z_ext, -1, axis=0)) / (dy*dy) +
            (np.roll(z_ext, +1, axis=1) - 2*z_ext + np.roll(z_ext, -1, axis=1)) / (dx*dx))

def gradiente_adveccao(z, u, v, dx, dy):
    # upwind simples
    dzdx_central = (np.roll(z, -1, axis=1) - np.roll(z, +1, axis=1)) / (2*dx)
    dzdy_central = (np.roll(z, -1, axis=0) - np.roll(z, +1, axis=0)) / (2*dy)
    # poderíamos usar esquema upwind mais estável; central ok com dt pequeno
    return u * dzdx_central + v * dzdy_central

def forca_T(nx, ny, base=0.0005):
    # gradiente latitudinal simples (mais aquecimento próximo ao "equador")
    y = np.linspace(-1, 1, ny)[:, None]
    mapa = base * (1.1 - 0.6*np.abs(y))
    return np.repeat(mapa, nx, axis=1).astype(np.float32)

def forca_C(nx, ny, base=0.0002):
    # hotspots industriais (2 gaussianas)
    y, x = np.mgrid[0:ny, 0:nx]
    g1 = np.exp(-(((x-0.30*nx)**2 + (y-0.35*ny)**2) / (0.06*nx*ny)))
    g2 = np.exp(-(((x-0.70*nx)**2 + (y-0.65*ny)**2) / (0.08*nx*ny)))
    mapa = base * (1.0 + 1.8*g1 + 1.5*g2)
    return mapa.astype(np.float32)

# -------------------- Simulador PDE --------------------
def simular_pde(par: ParametrosPDE):
    nx, ny = par.nx, par.ny
    T = 280.0 + 3.0*np.random.randn(ny, nx).astype(np.float32)  # K (base)
    C = 400.0 + 5.0*np.random.randn(ny, nx).astype(np.float32)  # ppm (base)

    u, v = gerar_campo_vento(nx, ny)
    FT = forca_T(nx, ny, par.forca_T_base)
    FC = forca_C(nx, ny, par.forca_C_base)

    estados = []   # lista de (T, C) ao longo do tempo
    estados.append((T.copy(), C.copy()))

    for t in range(par.passos):
        lap_T = laplaciano(T, par.dx, par.dy)
        lap_C = laplaciano(C, par.dx, par.dy)
        adv_T = gradiente_adveccao(T, u, v, par.dx, par.dy)
        adv_C = gradiente_adveccao(C, u, v, par.dx, par.dy)

        dT = par.kappa_T*lap_T - adv_T + par.alpha*C + FT + par.ruido_T*np.random.randn(*T.shape)
        dC = par.kappa_C*lap_C - adv_C - par.beta*T + FC + par.ruido_C*np.random.randn(*C.shape)

        T = (T + par.dt*dT).astype(np.float32)
        C = (C + par.dt*dC).astype(np.float32)

        if (t+1) % par.salvar_cada == 0:
            estados.append((T.copy(), C.copy()))

    return np.array([np.stack([e[0], e[1]], axis=0) for e in estados], dtype=np.float32)
    # shape: [tempo, 2, ny, nx], canal 0=T, canal 1=C

# -------------------- Dataset para a CNN --------------------
def construir_pares(estados):
    # pares (X_t -> X_{t+1})
    X = estados[:-1]
    Y = estados[1:]
    return X, Y  # [T-1, 2, ny, nx], [T-1, 2, ny, nx]

# -------------------- CNN para passo da PDE --------------------
class EmuladorClimaCNN(nn.Module):
    def __init__(self, canais_in=2, canais_mid=32, canais_out=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(canais_in, canais_mid, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(canais_mid, canais_mid, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(canais_mid, canais_out, kernel_size=3, padding=1)
        )
    def forward(self, x):
        return self.net(x)

def treinar_cnn(X, Y, par_cnn: ParametrosCNN):
    # normalização simples por canal (z-score)
    X_t = torch.from_numpy(X)
    Y_t = torch.from_numpy(Y)

    media = X_t.mean(dim=(0,2,3), keepdim=True)
    desvio = X_t.std(dim=(0,2,3), keepdim=True) + 1e-6
    Xn = (X_t - media) / desvio
    Yn = (Y_t - media) / desvio  # alinha saída com a mesma escala

    n = Xn.shape[0]
    n_treino = int(par_cnn.proporcao_treino * n)

    Xtr, Ytr = Xn[:n_treino], Yn[:n_treino]
    Xva, Yva = Xn[n_treino:], Yn[n_treino:]

    modelo = EmuladorClimaCNN()
    otimizador = optim.Adam(modelo.parameters(), lr=par_cnn.lr)
    criterio = nn.MSELoss()

    hist = {"perda_treino": [], "perda_valid": []}

    for ep in range(par_cnn.epocas):
        modelo.train()
        # mini-batches ao longo do tempo (tratando o "tempo" como batch)
        idx = torch.randperm(Xtr.shape[0])
        perdas = []
        for ini in range(0, len(idx), par_cnn.batch):
            lote_idx = idx[ini:ini+par_cnn.batch]
            xb = Xtr[lote_idx]
            yb = Ytr[lote_idx]

            pred = modelo(xb)
            loss = criterio(pred, yb)

            otimizador.zero_grad()
            loss.backward()
            otimizador.step()
            perdas.append(loss.item())

        perda_t = float(np.mean(perdas))

        # validação
        modelo.eval()
        with torch.no_grad():
            pred_v = modelo(Xva)
            perda_v = float(criterio(pred_v, Yva).item())

        hist["perda_treino"].append(perda_t)
        hist["perda_valid"].append(perda_v)
        print(f"[Ep {ep+1:02d}] perda_treino={perda_t:.6e}  perda_valid={perda_v:.6e}")

    pacote = {
        "modelo": modelo,
        "media": media,
        "desvio": desvio,
        "hist": hist,
    }
    return pacote

# -------------------- Rollout CNN vs. PDE --------------------
def rollout_cnn(pacote, estado_inicial, passos):
    modelo = pacote["modelo"]
    media = pacote["media"]
    desvio = pacote["desvio"]
    modelo.eval()

    estados = [estado_inicial.copy()]
    atual = estado_inicial.copy()

    with torch.no_grad():
        for _ in range(passos):
            x = torch.from_numpy(atual[None, ...])        # [1,2,ny,nx]
            xn = (x - media) / desvio
            ypred = modelo(xn).cpu() * desvio + media
            prox = ypred.numpy()[0].astype(np.float32)
            estados.append(prox.copy())
            atual = prox
    return np.array(estados, dtype=np.float32)  # [passos+1,2,ny,nx]

# -------------------- Métricas --------------------
def rmse(a, b):
    return float(np.sqrt(np.mean((a - b)**2)))

def mae(a, b):
    return float(np.mean(np.abs(a - b)))

def imprimir_metricas(nome, d):
    print(f"\n===== {nome} =====")
    for k, v in d.items():
        print(f"{k}: {v:.14f}")

# -------------------- Gráficos --------------------
def plot_campo(ax, campo, titulo):
    im = ax.imshow(campo, origin="lower")
    ax.set_title(titulo)
    ax.set_xticks([]); ax.set_yticks([])
    return im

def grafico_series_temporais(est, titulo):
    T_med = est[:,0].mean(axis=(1,2))
    C_med = est[:,1].mean(axis=(1,2))
    plt.figure(figsize=(9,4.2))
    plt.plot(T_med, label="T média")
    plt.plot(C_med, label="C média")
    plt.title(titulo)
    plt.xlabel("índice temporal (salvos)")
    plt.ylabel("média espacial")
    plt.legend()
    plt.tight_layout()

def grafico_perdas(hist):
    plt.figure(figsize=(7.2,4.2))
    plt.plot(hist["perda_treino"], label="treino")
    plt.plot(hist["perda_valid"], label="validação")
    plt.title("Perda (MSE) por época")
    plt.xlabel("época")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()

def comparar_mapas(est_pde, est_cnn, t_idx, sufixo=""):
    # t_idx: índice no array salvo (0..Tsalvo)
    fig, axs = plt.subplots(2, 3, figsize=(12,6))
    im0 = plot_campo(axs[0,0], est_pde[t_idx,0], f"T PDE t={t_idx}")
    im1 = plot_campo(axs[0,1], est_cnn[t_idx,0], f"T CNN t={t_idx}")
    im2 = plot_campo(axs[0,2], est_cnn[t_idx,0]-est_pde[t_idx,0], f"T (CNN-PDE) t={t_idx}")
    im3 = plot_campo(axs[1,0], est_pde[t_idx,1], f"C PDE t={t_idx}")
    im4 = plot_campo(axs[1,1], est_cnn[t_idx,1], f"C CNN t={t_idx}")
    im5 = plot_campo(axs[1,2], est_cnn[t_idx,1]-est_pde[t_idx,1], f"C (CNN-PDE) t={t_idx}")
    fig.suptitle(f"Comparação mapas PDE vs CNN {sufixo}")
    plt.tight_layout()

# -------------------- Execução principal --------------------
def main():
    par = ParametrosPDE()
    par_cnn = ParametrosCNN()

    print(">> Simulando PDE...")
    estados = simular_pde(par)       # [Tsalvo, 2, ny, nx]
    Tsalvo = estados.shape[0]

    # Gráficos de série temporal médios
    grafico_series_temporais(estados, "Séries temporais (médias espaciais) — PDE")

    # Snapshots iniciais e finais
    fig, axs = plt.subplots(2, 2, figsize=(10,8))
    plot_campo(axs[0,0], estados[0,0], "T inicial (PDE)")
    plot_campo(axs[0,1], estados[-1,0], "T final (PDE)")
    plot_campo(axs[1,0], estados[0,1], "C inicial (PDE)")
    plot_campo(axs[1,1], estados[-1,1], "C final (PDE)")
    fig.suptitle("Mapas: estado inicial e final — PDE")
    plt.tight_layout()

    # Estatísticas numéricas (14 dígitos)
    T_med_ini = float(estados[0,0].mean());   T_med_fin = float(estados[-1,0].mean())
    C_med_ini = float(estados[0,1].mean());   C_med_fin = float(estados[-1,1].mean())
    met_pde = {
        "media_T_inicial": T_med_ini,
        "media_T_final":   T_med_fin,
        "media_C_inicial": C_med_ini,
        "media_C_final":   C_med_fin,
        "rmse_T_inicial_final": rmse(estados[0,0], estados[-1,0]),
        "rmse_C_inicial_final": rmse(estados[0,1], estados[-1,1]),
    }
    imprimir_metricas("Estatísticas PDE", met_pde)

    # Dataset (X_t -> X_{t+1})
    X, Y = construir_pares(estados)

    if tem_pytorch:
        print("\n>> Treinando CNN para emular o passo da PDE...")
        pacote = treinar_cnn(X, Y, par_cnn)
        grafico_perdas(pacote["hist"])

        # Rollout da CNN: começar do mesmo estado inicial e prever todos os passos salvos
        print(">> Rollout CNN vs PDE...")
        est_cnn = rollout_cnn(pacote, estados[0], Tsalvo-1)  # tamanho igual ao de 'estados'

        # Séries médias (CNN)
        grafico_series_temporais(est_cnn, "Séries temporais (médias espaciais) — CNN")

        # Comparar mapas em alguns tempos
        for tidx in [0, Tsalvo//2, Tsalvo-1]:
            comparar_mapas(estados, est_cnn, tidx, sufixo=f"(t={tidx})")

        # Métricas PDE vs CNN
        met_cmp = {
            "rmse_T_rollout": rmse(estados[:,0], est_cnn[:,0]),
            "mae_T_rollout":  mae(estados[:,0], est_cnn[:,0]),
            "rmse_C_rollout": rmse(estados[:,1], est_cnn[:,1]),
            "mae_C_rollout":  mae(estados[:,1], est_cnn[:,1]),
        }
        imprimir_metricas("PDE vs CNN (rollout completo)", met_cmp)

        # Tabela resumida (pandas) para copiar/colar se quiser
        df_metricas = pd.DataFrame([met_pde | met_cmp])
        # forçar 14 dígitos nas colunas float na impressão
        with pd.option_context('display.float_format', lambda x: f"{x:.14f}"):
            print("\nResumo métrico:\n", df_metricas)
    else:
        print("\n[AVISO] PyTorch não disponível. Execução segue apenas com a PDE (gráficos e métricas PDE gerados).")

    plt.show()

if __name__ == "__main__":
    main()
