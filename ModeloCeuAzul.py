# ============================================
# Espalhamento de Rayleigh 
# Autor: Luiz Tiago Wilcke (LT)
# ============================================

import numpy as np
import matplotlib.pyplot as plt

H = 8000.0
n0 = 2.547e25
fator_king = 1.05
indice_ref = 1.0003
I0_constante = 1.0

lambdas_nm = np.linspace(380, 780, 401)
lambdas = lambdas_nm * 1e-9

def sigma_rayleigh(lmbda_m):
    sigma_verde = 5e-31
    return sigma_verde * (550e-9 / lmbda_m)**4 * fator_king

def densidade_ar(z):
    return n0 * np.exp(-z / H)

def beta_rayleigh(z, lmbda_m):
    return densidade_ar(z) * sigma_rayleigh(lmbda_m)

def tau_total(lmbda_m, z_max=80000.0):
    z = np.linspace(0.0, z_max, 2000)
    return np.trapz(beta_rayleigh(z, lmbda_m), z)

def P_rayleigh(psi_rad):
    return 0.75 * (1.0 + np.cos(psi_rad)**2)

def radiancia_ceu_espectral(lmbda_m, theta_obs_rad, theta_sol_rad, psi_rad, z_max=80000.0):
    mu = np.cos(theta_obs_rad)
    mu_sol = np.cos(theta_sol_rad)
    tau = tau_total(lmbda_m, z_max=z_max)
    I_sol = I0_constante * np.exp(-tau / max(mu_sol, 1e-6))
    z = np.linspace(0.0, z_max, 2000)
    beta = beta_rayleigh(z, lmbda_m)
    tau_acima = np.flip(np.cumsum(np.flip(beta)) * (z[1]-z[0]))
    atenuacao_total = np.exp(-tau_acima / max(mu, 1e-6)) * np.exp(-tau_acima / max(mu_sol, 1e-6))
    fase = P_rayleigh(psi_rad) / (4.0 * np.pi)
    integrando = beta * fase * atenuacao_total
    return I_sol * np.trapz(integrando, z)

def respostas_rgb(lmbdas_nm):
    R = np.exp(-0.5*((lmbdas_nm-610)/40)**2)
    G = np.exp(-0.5*((lmbdas_nm-550)/30)**2)
    B = np.exp(-0.5*((lmbdas_nm-450)/25)**2)
    return R, G, B

def espectro_para_rgb(lmbdas_nm, S):
    Rf, Gf, Bf = respostas_rgb(lmbdas_nm)
    R = np.trapz(S * Rf, lmbdas_nm)
    G = np.trapz(S * Gf, lmbdas_nm)
    B = np.trapz(S * Bf, lmbdas_nm)
    V = np.array([R, G, B])
    V = V / (V.max() + 1e-12)
    def to_srgb(x): return (1.055*(x**(1/2.4)) - 0.055) if x > 0.0031308 else 12.92*x
    return np.array([to_srgb(v) for v in V]).clip(0,1)

def simular_ceu(theta_obs_graus=60.0, elevacao_solar_graus=45.0, separacao_sol_graus=90.0):
    theta_obs = np.deg2rad(theta_obs_graus)
    theta_sol = np.deg2rad(90 - elevacao_solar_graus)
    psi = np.deg2rad(separacao_sol_graus)
    S = []
    for lmbda_m in lambdas:
        S.append(radiancia_ceu_espectral(lmbda_m, theta_obs, theta_sol, psi))
    S = np.array(S)
    S_rel = S / (S.max() + 1e-30)
    cor = espectro_para_rgb(lambdas_nm, S_rel)
    plt.figure(figsize=(8,4))
    plt.plot(lambdas_nm, S_rel, label='Radiância do céu (rel.)')
    lei = (lambdas_nm**-4)
    lei = lei / lei.max()
    plt.plot(lambdas_nm, lei, '--', label='Proporcional a λ⁻⁴ (escala)')
    plt.xlabel('Comprimento de onda (nm)')
    plt.ylabel('Intensidade relativa')
    plt.title('Espectro relativo do céu — espalhamento único de Rayleigh')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.figure(figsize=(3,2))
    plt.axis('off')
    plt.imshow(np.ones((50,100,3))*cor.reshape(1,1,3))
    plt.title('Cor aproximada do céu')
    plt.show()

if __name__ == "__main__":
    simular_ceu(theta_obs_graus=60.0, elevacao_solar_graus=45.0, separacao_sol_graus=90.0)
    simular_ceu(theta_obs_graus=60.0, elevacao_solar_graus=5.0, separacao_sol_graus=90.0)
