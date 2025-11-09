# -*- coding: utf-8 -*-
# ============================================================
# Célula Fotovoltaica de Silício — Modelo Quântico Híbrido
# Autor: Luiz Tiago Wilcke (LT)
# Descrição:
#   1) Poisson 1D (diferenças finitas) -> φ(x) da junção p–n.
#   2) TDSE 1D (Crank–Nicolson) -> pacote fotoexcitado e fluxo quântico.
#   3) I–V fenomenológico -> usa fotocorrente vinda do fluxo quântico.
#   4) PINN (opcional) -> solução estacionária Schrödinger–Poisson.
# Obs:
#   - Gráficos com matplotlib (1 plot por figura).
#   - PINN é automático: se 'torch' faltar, essa etapa é pulada.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from math import pi

# ---------- Torch (opcional p/ PINN) ----------
tem_torch = True
try:
    import torch
    import torch.nn as nn
    import torch.autograd as autograd
except Exception:
    tem_torch = False

# ---------- Constantes físicas ----------
hbar = 1.054_571_817e-34  # J*s
q    = 1.602_176_634e-19  # C
m_e  = 9.109_383_7015e-31 # kg
m_ef = 0.26 * m_e         # massa efetiva (Si, condução ~0.26 m_e)
eps0 = 8.854_187_8128e-12 # F/m
epsr_si = 11.7
eps_si  = eps0 * epsr_si
kB   = 1.380649e-23
T    = 300.0
Vt   = kB*T/q

# ---------- Malha 1D ----------
L = 150e-9    # comprimento do domínio (150 nm)
N = 320       # pontos de grade
dx = L/(N-1)
x  = np.linspace(0, L, N)

# ---------- Dopagem (junção p–n simplificada) ----------
Na = 2e24   # aceitadores (m^-3) ~ 2e18 cm^-3
Nd = 5e23   # doadores    (m^-3) ~ 5e17 cm^-3
indice_p = int(0.45*N)
dopagem = np.zeros(N)
dopagem[:indice_p] = -Na
dopagem[indice_p:] = +Nd

# ---------- Poisson 1D: φ'' = -ρ/ε , ρ ~ q*dopagem ----------
V_bi = 0.8  # potencial embutido (aproximação razoável p/ Si dopado)
rho = q*dopagem/1.0
A = np.diag(-2.0*np.ones(N)) + np.diag(np.ones(N-1), 1) + np.diag(np.ones(N-1), -1)
b = -rho/eps_si * (dx**2)

# contorno Dirichlet: φ(0)=0, φ(L)=V_bi
b[0] = 0.0
b[-1] = V_bi
A[0,:] = 0.0; A[0,0] = 1.0
A[-1,:]= 0.0; A[-1,-1] = 1.0

phi = np.linalg.solve(A, b)
E_campo = -np.gradient(phi, dx)
V = q*phi  # potencial em Joules

# ---------- TDSE (Crank–Nicolson): iħ ψ_t = [-(ħ²/2m*)∂² + V(x)] ψ ----------
coef = -(hbar**2)/(2.0*m_ef*(dx**2))
lap  = np.diag(-2.0*np.ones(N)) + np.diag(np.ones(N-1), 1) + np.diag(np.ones(N-1), -1)
H    = coef*lap + np.diag(V)

dt     = 5e-19  # s
passos = 220

I = np.eye(N, dtype=complex)
A_cn = I + 1j*dt*H/(2.0*hbar)
B_cn = I - 1j*dt*H/(2.0*hbar)

# amortecimento (absorvente) nas bordas
amortecimento = np.ones(N)
largura_abs = max(2, int(0.08*N))
rampa = np.linspace(0.0, 1.0, largura_abs)
amortecimento[:largura_abs]     *= np.exp(-3.0*rampa**2)
amortecimento[-largura_abs:]    *= np.exp(-3.0*rampa[::-1]**2)
Amat = np.diag(amortecimento)

# pacote gaussiano -> elétron fotoexcitado no lado p
x0 = 0.22*L
largura = 7e-9
k0 = 1.6e9
psi = (1.0/((2*pi*largura**2)**0.25)) * np.exp(-(x-x0)**2/(4*largura**2) + 1j*k0*(x-x0))
psi = psi/np.sqrt(np.trapz(np.abs(psi)**2, x))

def fluxo_quantico(psi_c):
    dpsi_dx = np.gradient(psi_c, dx)
    # J = (ħ/m*) Im(ψ* dψ/dx)
    return (hbar/m_ef)*np.imag(np.conjugate(psi_c)*dpsi_dx)

fluxo_dir = []
for n in range(passos):
    rhs = B_cn @ psi
    rhs = Amat @ rhs
    psi = np.linalg.solve(A_cn, rhs)
    if (n+1) % 30 == 0:
        # renormalização suave p/ estabilidade numérica
        norma = np.sqrt(np.trapz(np.abs(psi)**2, x))
        if norma > 0:
            psi = psi/norma
    J = fluxo_quantico(psi)
    fluxo_dir.append(np.mean(J[-largura_abs:]))

densidade_final = np.abs(psi)**2
fluxo_dir = np.array(fluxo_dir)

# corrente “proxy” a partir do fluxo quântico (densidade de corrente -> A/m; multiplicar por área efetiva)
corrente_quanto = q * max(0.0, np.mean(fluxo_dir[-20:]))

# ---------- Curva I–V (equação do diodo alimentada por I_ph do modelo quântico) ----------
Aef = 5e-13     # área efetiva (m^2) para converter densidade em corrente total (demo)
I_ph = corrente_quanto * Aef
I0, n_id, Rs, Rsh = 1e-12, 1.4, 1.0, 1e4

def corrente_diodo(Vap):
    I = np.zeros_like(Vap)
    for _ in range(35):
        expo = np.exp((q*(Vap + I*Rs))/(n_id*kB*T))
        I = I_ph - I0*(expo - 1.0) - (Vap + I*Rs)/Rsh
    return I

tensao_var   = np.linspace(0.0, 0.9, 120)
corrente_IV  = corrente_diodo(tensao_var)
potencia     = tensao_var * corrente_IV
idx_mpp      = int(np.argmax(potencia))
V_mpp, I_mpp, P_mpp = tensao_var[idx_mpp], corrente_IV[idx_mpp], potencia[idx_mpp]

# ---------- PINN (opcional): Schrödinger–Poisson estacionário ----------
tem_resultado_pinn = False
if tem_torch:
    x_t = torch.tensor(x, dtype=torch.float32).view(-1,1); x_t.requires_grad = True
    class PINN_SP(nn.Module):
        def __init__(self):
            super().__init__()
            self.rede = nn.Sequential(
                nn.Linear(1, 64), nn.Tanh(),
                nn.Linear(64, 64), nn.Tanh(),
                nn.Linear(64, 2)   # ψ (real) e φ
            )
            self.E = nn.Parameter(torch.tensor([0.05], dtype=torch.float32))
        def forward(self, x):
            out = self.rede(x)
            return out[:,0:1], out[:,1:2]
    modelo = PINN_SP()
    opt = torch.optim.Adam(modelo.parameters(), lr=2e-3)
    eps_t = torch.tensor(eps_si, dtype=torch.float32)
    def perda_pinn(x_t):
        psi_t, phi_t = modelo(x_t)
        dpsi  = autograd.grad(psi_t, x_t, torch.ones_like(psi_t), create_graph=True)[0]
        d2psi = autograd.grad(dpsi,   x_t, torch.ones_like(dpsi), create_graph=True)[0]
        dphi  = autograd.grad(phi_t, x_t, torch.ones_like(phi_t), create_graph=True)[0]
        d2phi = autograd.grad(dphi,  x_t, torch.ones_like(dphi), create_graph=True)[0]
        Epar = torch.abs(modelo.E)
        # Resíduos:
        #   Schr: -(ħ²/2m*) ψ'' + q φ ψ = E ψ
        #   Poiss: φ'' + q|ψ|²/ε = 0
        res_s = -(hbar**2/(2*m_ef))*d2psi + q*phi_t*psi_t - Epar*psi_t
        rho_q = q*psi_t**2
        res_p = d2phi + rho_q/eps_t
        # BC suaves: φ(0)=0, φ(L)=V_bi; ψ(0)=ψ(L)=0
        bc = (phi_t[0]-0.0)**2 + (phi_t[-1]-V_bi)**2 + (psi_t[0]**2 + psi_t[-1]**2)
        return (res_s**2).mean() + (res_p**2).mean() + 1e-2*bc
    for _ in range(160):   # treino leve só p/ demo
        opt.zero_grad()
        Lp = perda_pinn(x_t); Lp.backward(); opt.step()
    with torch.no_grad():
        psi_pinn, phi_pinn = modelo(x_t)
        psi_pinn = psi_pinn.squeeze().numpy()
        phi_pinn = phi_pinn.squeeze().numpy()
        tem_resultado_pinn = True

# ---------- Gráficos (um por figura; sem estilos/cores fixos) ----------
plt.figure(); plt.plot(x*1e9, phi); plt.xlabel("x (nm)"); plt.ylabel("φ (V)"); plt.title("Potencial eletrostático φ (Poisson)"); plt.show()
plt.figure(); plt.plot(x*1e9, E_campo); plt.xlabel("x (nm)"); plt.ylabel("E (V/m)"); plt.title("Campo elétrico interno"); plt.show()
plt.figure(); plt.plot(x*1e9, np.abs(psi)**2); plt.xlabel("x (nm)"); plt.ylabel("|ψ|² (a.u.)"); plt.title("Densidade de probabilidade após fotoexcitação"); plt.show()
plt.figure(); t_fs = np.arange(passos)*dt*1e15; plt.plot(t_fs, fluxo_dir); plt.xlabel("Tempo (fs)"); plt.ylabel("Fluxo quântico (a.u.)"); plt.title("Fluxo quântico no contato direito"); plt.show()
plt.figure(); plt.plot(tensao_var, corrente_IV); plt.scatter([V_mpp],[I_mpp]); plt.xlabel("Tensão (V)"); plt.ylabel("Corrente (A)"); plt.title("Curva I–V fotogerada (fenomenológica)"); plt.show()
plt.figure(); plt.plot(tensao_var, potencia); plt.scatter([V_mpp],[P_mpp]); plt.xlabel("Tensão (V)"); plt.ylabel("Potência (W)"); plt.title("Potência e ponto de máxima potência (MPP)"); plt.show()
if tem_resultado_pinn:
    plt.figure(); plt.plot(x*1e9, psi_pinn); plt.xlabel("x (nm)"); plt.ylabel("ψ (a.u.)"); plt.title("PINN — ψ estacionária (demo)"); plt.show()
    plt.figure(); plt.plot(x*1e9, phi_pinn); plt.xlabel("x (nm)"); plt.ylabel("φ (V)"); plt.title("PINN — φ estacionário (demo)"); plt.show()

# ---------- Resumo ----------
print("\n==== RESUMO ====")
print(f"V_bi ≈ {V_bi:.2f} V | Corrente quântica (proxy) = {corrente_quanto:.3e} A/m")
print(f"I_ph ≈ {I_ph:.3e} A | MPP: V*={V_mpp:.3f} V, I*={I_mpp:.3e} A, P*={P_mpp:.3e} W")
print("PINN:", "OK" if tem_resultado_pinn else "não executada (torch ausente)")
