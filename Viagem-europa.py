# Simulação avançada: lançamento para Europa (lua de Júpiter)
# Modelo 2D com gravidade do Sol e de Júpiter + integração RK4
# - Transferência tipo Hohmann (impulso inicial) e faseamento de Júpiter
# - Amostragem diária em tabela
# Saídas: CSV "simulacao_foguete_europa.csv", gráficos e resumo no terminal
# Autor : Luiz Tiago Wilcke

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ======================== Constantes físicas ========================
AU      = 1.495978707e11          # m
DIA     = 86400.0                 # s
KM      = 1000.0                  # m

mu_sol      = 1.32712440018e20    # m^3/s^2
mu_jupiter  = 1.26686534e17       # m^3/s^2

a_terra   = 1.0*AU
a_jupiter = 5.2044*AU

# Europa (apenas para acompanhar o alvo em torno de Júpiter)
a_europa = 671.1e6                # m (raio orbital ~671 mil km)
n_europa = math.sqrt(mu_jupiter / a_europa**3)  # rad/s

# Movimento de Júpiter (aprox. circular 2D)
n_jup = math.sqrt(mu_sol / a_jupiter**3)  # rad/s

# ======================== Transferência Hohmann Terra->Júpiter ========================
mu = mu_sol
a1, a2 = a_terra, a_jupiter

v_circ_terra = math.sqrt(mu/a1)
v_circ_jup   = math.sqrt(mu/a2)
a_transfer   = 0.5*(a1 + a2)

v_peri_trans = math.sqrt(mu*(2.0/a1 - 1.0/a_transfer))
v_apo_trans  = math.sqrt(mu*(2.0/a2 - 1.0/a_transfer))

delta_v1 = v_peri_trans - v_circ_terra
delta_v2 = v_circ_jup   - v_apo_trans              # (não aplicado — apenas referência)
t_voo    = math.pi * math.sqrt(a_transfer**3 / mu) # meio-período da elipse de transferência

# ======================== Parâmetros de simulação ========================
t0         = 0.0
t_final    = 3.2 * 365.25 * DIA     # ~3.2 anos
dt         = 6 * 3600.0             # 6 horas
amostra    = int(DIA/dt)            # registrar ~1x por dia
delta_v_extra = 0.0                 # ajuste prograde opcional na queima inicial (m/s)

# Faseamento: avance Júpiter por n_jup * t_voo para "chegar junto"
angulo_jup0    = n_jup * t_voo
angulo_europa0 = 0.0

# ======================== Estados iniciais (2D) ========================
# Sonda parte da órbita da Terra (x=a_terra, y=0), tangencial +y e aplica Δv da transferência
r_sc = np.array([a_terra, 0.0], dtype=np.float64)
v_sc = np.array([0.0, v_circ_terra + delta_v1 + delta_v_extra], dtype=np.float64)

def pos_jupiter(t):
    ang = angulo_jup0 + n_jup*t
    return np.array([a_jupiter*math.cos(ang), a_jupiter*math.sin(ang)], dtype=np.float64)

def pos_europa(t):
    rj  = pos_jupiter(t)
    ang = angulo_europa0 + n_europa*t
    re_rel = a_europa*np.array([math.cos(ang), math.sin(ang)], dtype=np.float64)
    return rj + re_rel

# ======================== Dinâmica (equações diferenciais) ========================
def acc_total(r_sc, t):
    """ r'' = a_sol + a_jup, com Júpiter em órbita circular. """
    r_jup = pos_jupiter(t)
    d_sol = r_sc
    d_jup = r_sc - r_jup
    a_sol = -mu_sol     * d_sol / (np.linalg.norm(d_sol)**3)
    a_jup = -mu_jupiter * d_jup / (np.linalg.norm(d_jup)**3)
    return a_sol + a_jup

def rk4_step(r, v, t, h):
    """ Um passo de Runge–Kutta 4 para o sistema de 1ª ordem (r', v'). """
    a1 = acc_total(r, t)
    k1_r, k1_v = v, a1

    a2 = acc_total(r + 0.5*h*k1_r, t + 0.5*h)
    k2_r, k2_v = v + 0.5*h*k1_v, a2

    a3 = acc_total(r + 0.5*h*k2_r, t + 0.5*h)
    k3_r, k3_v = v + 0.5*h*k2_v, a3

    a4 = acc_total(r + h*k3_r, t + h)
    k4_r, k4_v = v + h*k3_v, a4

    r_next = r + (h/6.0)*(k1_r + 2*k2_r + 2*k3_r + k4_r)
    v_next = v + (h/6.0)*(k1_v + 2*k2_v + 2*k3_v + k4_v)
    return r_next, v_next

# ======================== Integração ========================
N = int((t_final - t0)/dt)
t = t0

registro = {
    "tempo_dias": [],
    "x_AU": [], "y_AU": [],
    "dist_sol_AU": [],
    "dist_jupiter_Mkm": [],
    "dist_europa_Mkm": [],
    "vel_kms": [],
    "x_jup_AU": [], "y_jup_AU": [],
    "x_eur_AU": [], "y_eur_AU": [],
}

min_dist_jup = float("inf")
min_dist_eur = float("inf")
t_min_jup = None
t_min_eur = None

for i in range(N+1):
    if i % amostra == 0 or i == N:
        rj = pos_jupiter(t)
        re = pos_europa(t)

        dist_sol = np.linalg.norm(r_sc)
        dist_jup = np.linalg.norm(r_sc - rj)
        dist_eur = np.linalg.norm(r_sc - re)
        vel = np.linalg.norm(v_sc)

        if dist_jup < min_dist_jup:
            min_dist_jup = dist_jup
            t_min_jup = t
        if dist_eur < min_dist_eur:
            min_dist_eur = dist_eur
            t_min_eur = t

        registro["tempo_dias"].append(t/DIA)
        registro["x_AU"].append(r_sc[0]/AU)
        registro["y_AU"].append(r_sc[1]/AU)
        registro["dist_sol_AU"].append(dist_sol/AU)
        registro["dist_jupiter_Mkm"].append(dist_jup/KM/1000.0)
        registro["dist_europa_Mkm"].append(dist_eur/KM/1000.0)
        registro["vel_kms"].append(vel/KM)
        registro["x_jup_AU"].append(rj[0]/AU)
        registro["y_jup_AU"].append(rj[1]/AU)
        registro["x_eur_AU"].append(re[0]/AU)
        registro["y_eur_AU"].append(re[1]/AU)

    r_sc, v_sc = rk4_step(r_sc, v_sc, t, dt)
    t += dt

df = pd.DataFrame(registro)

# ======================== Resumo e salvamento ========================
resumo = {
    "delta_v1_m_s": round(delta_v1, 6),
    "delta_v2_circularizar_m_s": round(delta_v2, 6),
    "t_voo_teorico_dias": round(t_voo/DIA, 6),
    "dist_min_jupiter_Mkm": round(min_dist_jup/KM/1000.0, 6),
    "dist_min_europa_Mkm": round(min_dist_eur/KM/1000.0, 6),
    "t_dist_min_jupiter_dias": round((t_min_jup or float("nan"))/DIA, 6),
    "t_dist_min_europa_dias": round((t_min_eur or float("nan"))/DIA, 6),
}
resumo_df = pd.DataFrame([resumo])

out_dir = Path(".")
csv_path = out_dir / "simulacao_foguete_europa.csv"
df.to_csv(csv_path, index=False)

print("\n=== RESUMO NUMÉRICO ===")
print(resumo_df.to_string(index=False))
print(f"\nCSV salvo em: {csv_path.resolve()}")

print("\nPrévia da tabela (10 primeiras linhas):")
print(df.head(10).to_string(index=False))

# ======================== Gráficos ========================
# 1) Distância ao Sol
plt.figure(figsize=(10,5))
plt.plot(df["tempo_dias"], df["dist_sol_AU"])
plt.title("Distância ao Sol (AU)")
plt.xlabel("Tempo (dias)")
plt.ylabel("Distância (AU)")
plt.tight_layout()
plt.show()

# 2) Distância a Júpiter
plt.figure(figsize=(10,5))
plt.plot(df["tempo_dias"], df["dist_jupiter_Mkm"])
plt.title("Distância a Júpiter (milhões de km)")
plt.xlabel("Tempo (dias)")
plt.ylabel("Distância (Mkm)")
plt.tight_layout()
plt.show()

# 3) Distância a Europa
plt.figure(figsize=(10,5))
plt.plot(df["tempo_dias"], df["dist_europa_Mkm"])
plt.title("Distância a Europa (milhões de km)")
plt.xlabel("Tempo (dias)")
plt.ylabel("Distância (Mkm)")
plt.tight_layout()
plt.show()

# 4) Trajetória 2D inercial (AU)
plt.figure(figsize=(6,6))
plt.plot(df["x_AU"], df["y_AU"], linewidth=1, label="Sonda")
plt.plot(df["x_jup_AU"], df["y_jup_AU"], linewidth=1, label="Júpiter")
plt.plot(df["x_eur_AU"], df["y_eur_AU"], linewidth=0.8, label="Europa")
plt.scatter([0],[0], s=30, label="Sol")
plt.title("Trajetórias 2D: Sonda, Júpiter e Europa (AU)")
plt.xlabel("x (AU)"); plt.ylabel("y (AU)")
plt.axis("equal"); plt.legend()
plt.tight_layout()
plt.show()
