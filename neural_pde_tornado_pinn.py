# -*- coding: utf-8 -*-
# ============================================================
# Neural PDEs (PINN) para previsão de tornados — Sul do Brasil
# Autor: Luiz Tiago Wilcke (LT)
# Descrição:
#   - Simula uma PDE de vorticidade 2D (advecção + difusão + fonte CAPE/cisalhamento).
#   - (Opcional) Treina uma CNN tipo PINN para aprender dζ/dt penalizando o resíduo físico.
#   - Gera gráficos (CAPE, cisalhamento, vorticidade t0/tf, energia) e CSVs.
#   - Todas as variáveis e comentários estão em português.
# Dependências mínimas: numpy, matplotlib, pandas
# O PINN usa PyTorch (opcional). Caso não tenha, a parte neural é ignorada.
# ============================================================

import os
import math
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --------------------------------------
# (Opcional) PyTorch para a parte neural
# --------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


# ===============================
# 1) Parâmetros físicos e numéricos
# ===============================
@dataclass
class Parametros:
    # Grade espacial (nx * ny)
    nx: int = 48
    ny: int = 48
    # Resolução (km) — ~10 km por célula (apenas demonstrativo)
    dx_km: float = 10.0
    dy_km: float = 10.0
    # Tempo (minutos)
    dt_min: float = 2.0
    passos: int = 36                # 36*2 = 72 minutos
    # Física efetiva
    viscosidade: float = 1200.0     # m^2/s (difusão efetiva)
    fonte_amp: float = 1.0          # ganho do termo de fonte CAPE/cisalhamento
    seed: int = 42                  # reprodutibilidade

    # ---- Parte neural (se PyTorch disponível) ----
    usar_pinn: bool = True          # se False, ignora treino da rede
    epocas: int = 8                 # poucas épocas para rodar rápido
    lr: float = 2e-3                # taxa de aprendizado
    passos_treino: int = 6          # usar apenas os 6 primeiros steps como dados para o PINN
    rollout_passos: int = 12        # prever 12 passos à frente (24 min)


# Utilidades de conversão
def _conversoes(p: Parametros):
    dx = p.dx_km * 1000.0
    dy = p.dy_km * 1000.0
    dt = p.dt_min * 60.0
    return dx, dy, dt


# =======================================
# 2) Campos ambientais (CAPE, cisalhamento, vento)
# =======================================
def gerar_campos_ambiente(p: Parametros):
    """
    Gera campos sintéticos coerentes com ambiente de tempo severo
    no Sul do Brasil: CAPE, cisalhamento vertical e ventos (u,v).
    """
    X, Y = np.meshgrid(np.linspace(-1, 1, p.nx), np.linspace(-1, 1, p.ny), indexing='xy')

    # CAPE com máximo no noroeste/centro — padrão sintético de convecção
    cape = 2200*np.exp(-((0.1 - X)**2 + (0.2 - Y)**2)/0.30) \
         + 700*np.exp(-((X + 0.6)**2 + (Y - 0.4)**2)/0.35)
    cape += 150*np.random.randn(p.ny, p.nx)

    # Cisalhamento 0–6 km (m/s) — gradiente zonal/meridional simplificado
    cis = 18*(1 + 0.4*X - 0.3*Y) + 1.5*np.random.randn(p.ny, p.nx)

    # Vento de baixos níveis (u, v): jato NW→SE com componente meridional
    u = 9 + 7*X - 1.5*Y
    v = 3 + 3.5*Y + 1.2*X

    return cape, cis, u, v


# =======================================
# 3) Operadores diferenciais (diferenças finitas periódicas)
# =======================================
def gx(f: np.ndarray, dx: float) -> np.ndarray:
    return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2*dx)

def gy(f: np.ndarray, dy: float) -> np.ndarray:
    return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2*dy)

def laplaciano(f: np.ndarray, dx: float, dy: float) -> np.ndarray:
    # Discretização periódica simples (5 pontos)
    return (np.roll(f, -1, 1) + np.roll(f, 1, 1) + np.roll(f, -1, 0) + np.roll(f, 1, 0) - 4*f) / ((dx*dy)/(dx/dy + dy/dx))


# =======================================
# 4) Fontes e inicialização de vorticidade
# =======================================
def termo_fonte(cape: np.ndarray, cis: np.ndarray, ganho: float) -> np.ndarray:
    """
    Proxy simples para baroclinicidade/helicidade: combinação normalizada
    de CAPE e cisalhamento.
    """
    cape_n = (cape - cape.mean())/(cape.std() + 1e-6)
    cis_n  = (cis  - cis.mean()) /(cis.std()  + 1e-6)
    return ganho*(0.002*cape_n + 0.004*cis_n + 0.001*cape_n*cis_n)

def vortex_gauss(nx: int, ny: int, cx: int, cy: int, amp: float, raio: float) -> np.ndarray:
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='xy')
    r2 = (X - cx)**2 + (Y - cy)**2
    return amp*np.exp(-r2/(2*raio**2))

def inicializar_vorticidade(p: Parametros) -> np.ndarray:
    z = np.zeros((p.ny, p.nx))
    z += vortex_gauss(p.nx, p.ny, int(0.35*p.nx), int(0.55*p.ny), 3.5e-3, 4)
    z += vortex_gauss(p.nx, p.ny, int(0.55*p.nx), int(0.45*p.ny), -2.8e-3, 5)
    z += 4e-4*np.random.randn(p.ny, p.nx)
    return z


# =======================================
# 5) Simulação física (PDE): dζ/dt + u·∇ζ = ν∇²ζ + S
# =======================================
def simular_vorticidade(zeta_ini: np.ndarray, u: np.ndarray, v: np.ndarray, S: np.ndarray, p: Parametros) -> Tuple[np.ndarray, np.ndarray]:
    dx, dy, dt = _conversoes(p)
    z = zeta_ini.copy()
    zs = [z.copy()]
    for _ in range(p.passos):
        adv = u*gx(z, dx) + v*gy(z, dy)
        dif = p.viscosidade * laplaciano(z, dx, dy)
        dz = -adv + dif + S
        z = z + dt*dz
        zs.append(z.copy())
    zs = np.array(zs)              # [T, ny, nx]
    tempos_min = np.arange(zs.shape[0]) * p.dt_min
    return zs, tempos_min


# =======================================
# 6) (Opcional) PINN raso via CNN para dζ/dt
# =======================================
class CNN_dZ(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, 1)
        )
    def forward(self, x):
        return self.net(x)


def treinar_pinn(traj: np.ndarray, cape: np.ndarray, cis: np.ndarray, u: np.ndarray, v: np.ndarray, S: np.ndarray, p: Parametros, saida_dir: str):
    """
    Treino curto do PINN (8 épocas por padrão) para fins demonstrativos.
    Retorna: histórico (lista), rollout previsto (np.ndarray ou None).
    """
    if not HAS_TORCH or not p.usar_pinn:
        return [], None

    dx, dy, dt = _conversoes(p)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dados de treino (apenas primeiros 'passos_treino')
    T = min(p.passos_treino, traj.shape[0]-1)
    z_t   = traj[:T]         # [T, ny, nx]
    z_tp1 = traj[1:T+1]
    dz_true = (z_tp1 - z_t)/dt

    # Normalizações
    z_mean, z_std = z_t.mean(), z_t.std() + 1e-6
    dz_mean, dz_std = dz_true.mean(), dz_true.std() + 1e-6
    z_tn  = (z_t - z_mean)/z_std
    dzn   = (dz_true - dz_mean)/dz_std
    cape_n = (cape - cape.mean())/(cape.std()+1e-6)
    cis_n  = (cis  - cis.mean()) /(cis.std()+1e-6)
    cape_b = np.broadcast_to(cape_n, z_tn.shape)
    cis_b  = np.broadcast_to(cis_n,  z_tn.shape)

    # Tensores
    X = np.stack([z_tn, cape_b, cis_b], axis=1)   # [T, 3, ny, nx]
    Y = dzn[:, None, :, :]                        # [T, 1, ny, nx]
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    Y_t = torch.tensor(Y, dtype=torch.float32, device=device)

    u_t = torch.tensor(u, dtype=torch.float32, device=device)[None, None, :, :]
    v_t = torch.tensor(v, dtype=torch.float32, device=device)[None, None, :, :]
    S_t = torch.tensor(S, dtype=torch.float32, device=device)[None, None, :, :]

    # Operadores físicos em torch (periódicos)
    def gx_t(f): return (torch.roll(f, -1, 3) - torch.roll(f, 1, 3))/(2*dx)
    def gy_t(f): return (torch.roll(f, -1, 2) - torch.roll(f, 1, 2))/(2*dy)
    def lap_t(f): return (torch.roll(f, -1, 3) + torch.roll(f, 1, 3) + torch.roll(f, -1, 2) + torch.roll(f, 1, 2) - 4*f) / ((dx*dy)/(dx/dy + dy/dx))

    model = CNN_dZ().to(device)
    opt = optim.Adam(model.parameters(), lr=p.lr)

    history = []
    for ep in range(p.epocas):
        opt.zero_grad()
        pred_dzdt_n = model(X_t)
        loss_data = ((pred_dzdt_n - Y_t)**2).mean()

        zrec = X_t[:, 0:1]*z_std + z_mean
        dz_pred_real = pred_dzdt_n*dz_std + dz_mean

        adv = u_t*gx_t(zrec) + v_t*gy_t(zrec)
        dif = p.viscosidade * lap_t(zrec)
        residuo = dz_pred_real - (-adv + dif + S_t)
        loss_phys = (residuo**2).mean()

        loss = loss_data + 0.5*loss_phys
        loss.backward()
        opt.step()
        history.append((ep, float(loss.item()), float(loss_data.item()), float(loss_phys.item())))

    # Rollout curto
    with torch.no_grad():
        z_cur = torch.tensor(traj[0:1], dtype=torch.float32, device=device)  # [1, ny, nx]
        cb = torch.tensor(cape_b[0:1], dtype=torch.float32, device=device)   # [1, ny, nx]
        sb = torch.tensor(cis_b[0:1],  dtype=torch.float32, device=device)   # [1, ny, nx]
        out = [z_cur.cpu().numpy()[0]]
        for _ in range(p.rollout_passos):
            zn = (z_cur - z_mean)/z_std
            Xin = torch.cat([zn[:,None,:,:], cb[:,None,:,:], sb[:,None,:,:]], dim=1)  # [1,3,ny,nx]
            dzp_n = model(Xin)
            dzp   = dzp_n*dz_std + dz_mean
            z_cur = z_cur + dt*dzp[:,0,:,:]
            out.append(z_cur.cpu().numpy()[0])
        z_roll = np.array(out)  # [roll+1, ny, nx]

    # Salvar histórico
    hist_df = pd.DataFrame(history, columns=["epoca","loss_total","loss_dados","loss_fisica"])
    hist_df.to_csv(os.path.join(saida_dir, "hist_pinn.csv"), index=False)

    # Gráfico de loss
    plt.figure()
    plt.plot(hist_df["epoca"], hist_df["loss_total"])
    plt.xlabel("Época"); plt.ylabel("Loss total"); plt.title("Treino PINN (rápido)")
    plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(saida_dir, "loss_total.png"), dpi=140); plt.close()

    return history, z_roll


# =======================================
# 7) Visualização e exportação
# =======================================
def plot_field(field: np.ndarray, titulo: str, fname: str):
    plt.figure()
    plt.imshow(field, origin="lower", aspect="auto")
    plt.colorbar()
    plt.title(titulo)
    plt.xlabel("lon (índice)"); plt.ylabel("lat (índice)")
    plt.tight_layout()
    plt.savefig(fname, dpi=140)
    plt.close()

def salvar_series(tempos_min: np.ndarray, traj: np.ndarray, saida_dir: str):
    energia = (traj**2).sum(axis=(1,2))
    df = pd.DataFrame({"tempo_min": tempos_min, "energia": energia})
    df.to_csv(os.path.join(saida_dir, "energia.csv"), index=False)

    # gráfico energia
    plt.figure()
    plt.plot(tempos_min, energia)
    plt.xlabel("Tempo (min)"); plt.ylabel("Energia ζ")
    plt.title("Energia integrada — simulação física")
    plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(saida_dir, "energia.png"), dpi=140)
    plt.close()


# =======================================
# 8) Execução principal
# =======================================
def main():
    p = Parametros()
    np.random.seed(p.seed)
    dx, dy, dt = _conversoes(p)

    saida_dir = "resultados_neural_pde_tornado"
    os.makedirs(saida_dir, exist_ok=True)

    # Campos ambientais
    cape, cis, u_bg, v_bg = gerar_campos_ambiente(p)
    plot_field(cape, "CAPE (J/kg) — campo sintético", os.path.join(saida_dir, "cape.png"))
    plot_field(cis,  "Cisalhamento 0–6 km (m/s) — campo sintético", os.path.join(saida_dir, "cisalhamento.png"))

    # Inicialização e simulação física
    z0 = inicializar_vorticidade(p)
    traj, tempos_min = simular_vorticidade(z0, u_bg, v_bg, termo_fonte(cape, cis, p.fonte_amp), p)
    plot_field(traj[0],  "Vorticidade ζ — t0 (simulação)",      os.path.join(saida_dir, "zeta_t0.png"))
    plot_field(traj[-1], "Vorticidade ζ — t_final (simulação)", os.path.join(saida_dir, "zeta_tf.png"))
    salvar_series(tempos_min, traj, saida_dir)

    # Salvar trajetória bruta para pós-processamento
    np.save(os.path.join(saida_dir, "traj_vorticidade.npy"), traj)

    # ----- PINN (opcional) -----
    if HAS_TORCH and p.usar_pinn:
        hist, z_roll = treinar_pinn(traj, cape, cis, u_bg, v_bg, termo_fonte(cape, cis, p.fonte_amp), p, saida_dir)
        if z_roll is not None:
            # Comparação simples em um corte meridional
            j = p.ny//2
            tcmp = min(p.rollout_passos, traj.shape[0]-1)
            plt.figure()
            plt.plot(traj[0, j, :], label="sim t0")
            plt.plot(traj[tcmp, j, :], label=f"sim t{tcmp}")
            plt.plot(z_roll[tcmp, j, :], label=f"PINN t{tcmp}")
            plt.xlabel("lon (índice)"); plt.ylabel("ζ")
            plt.title("Corte meridional — simulação vs PINN")
            plt.grid(True); plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(saida_dir, "corte_comparacao.png"), dpi=140)
            plt.close()
    else:
        # Registrar que a parte neural não foi executada
        with open(os.path.join(saida_dir, "INFO.txt"), "w", encoding="utf-8") as f:
            f.write("PyTorch não disponível OU usar_pinn=False. Parte neural (PINN) não executada.\n")

    print(f"Concluído. Resultados salvos em: {os.path.abspath(saida_dir)}")

if __name__ == "__main__":
    main()
