# -*- coding: utf-8 -*-
# ============================================================
# Modelo Matemático do Aquecimento Global — PDE (EBM) + Carbono (3 caixas)
# - Latitude 1D (μ = sin φ) com difusão implícita (Crank–Nicolson)
# - Ciclo do carbono integrado por BDF (solve_ivp)
# - Forçante: 5.35 ln(C/280) + termo "outros" suave (aerossóis/gases)
# - Impressões com 5 casas e gráficos via matplotlib (sem estilos/cores fixas)
# Autor: (Luiz Tiago Wilcke)
# ============================================================

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.sparse import diags, identity
from scipy.sparse.linalg import spsolve
from scipy.integrate import solve_ivp


# ---------------------------
# 0) Parâmetros físicos/numéricos
# ---------------------------

@dataclass
class ParametrosFisicos:
    capacidade_termica: float = 2.1e8   # J m^-2 K^-1
    A: float = None                     # será calibrado p/ equilíbrio de referência
    B: float = 1.3                      # W m^-2 K^-1 (feedback estabilizador)
    D: float = 0.6                      # W m^-2 K^-1 (difusividade meridional efetiva)
    albedo0: float = 0.30               # albedo médio de referência
    gamma_albedo: float = 0.002         # 1/K (feedback de albedo linearizado)
    co2_pre_industrial: float = 280.0   # ppm


@dataclass
class Malha:
    N: int = 64
    dt_anos: float = 0.25
    anos_total: float = 200.0

    @property
    def dt_seg(self) -> float:
        return self.dt_anos * 365.0 * 24.0 * 3600.0


@dataclass
class ParamCarbono:
    # taxas 1/ano e volumes relativos (adimensionais, exceto atmosfera em ppm)
    k_atm_sup: float = 0.12
    k_sup_prof: float = 0.03
    k_bio_atm: float = 0.05
    vol_atm: float = 1.0
    vol_sup: float = 50.0
    vol_prof: float = 1000.0
    vol_bio: float = 2.0
    gtco2_por_ppm: float = 7.81  # 1 ppm ≈ 7.81 GtCO2


# ---------------------------
# 1) Malha e operadores (μ = sin φ)
# ---------------------------

def construir_malha(m: Malha):
    phi = np.linspace(-np.pi/2, np.pi/2, m.N)  # rad
    mu = np.sin(phi)
    dmu = mu[1] - mu[0]
    # pesos de quadratura em μ (trapézios)
    pesos = np.ones_like(mu)
    pesos[[0, -1]] *= 0.5
    return phi, mu, dmu, pesos


def operador_difusao(m: Malha, p: ParametrosFisicos, mu: np.ndarray, dmu: float):
    """L[T] = ∂/∂μ [ D(1-μ^2) ∂T/∂μ ] com CC de Neumann (fluxo=0) nos pólos."""
    N = len(mu)
    kcoef = p.D * (1.0 - mu**2)
    k_meio = 0.5 * (kcoef[1:] + kcoef[:-1])

    diag_p = np.zeros(N)
    diag_s = np.zeros(N-1)
    diag_i = np.zeros(N-1)

    for i in range(N):
        if i == 0:
            diag_p[i] = -(k_meio[i])/(dmu**2)
            diag_s[i] =  (k_meio[i])/(dmu**2)
        elif i == N-1:
            diag_p[i]   = -(k_meio[i-1])/(dmu**2)
            diag_i[i-1] =  (k_meio[i-1])/(dmu**2)
        else:
            diag_p[i]   = -(k_meio[i] + k_meio[i-1])/(dmu**2)
            diag_s[i]   =  (k_meio[i])/(dmu**2)
            diag_i[i-1] =  (k_meio[i-1])/(dmu**2)

    L = diags([diag_i, diag_p, diag_s], offsets=[-1, 0, 1], format="csr")
    return L


# ---------------------------
# 2) Forçantes e funções auxiliares
# ---------------------------

def insolacao_media_lat(phi: np.ndarray) -> np.ndarray:
    """Insolação anual média simplificada, normalizada p/ média global ~340 W m^-2."""
    S0 = 340.0
    padrao = 1.0 - 0.48 * np.sin(phi)**2
    padrao *= S0 / padrao.mean()
    return padrao


def forca_radiativa_co2(conc_ppm: float, p: ParametrosFisicos) -> float:
    return 5.35 * math.log(conc_ppm / p.co2_pre_industrial)


def forca_radiativa_outros(t_anos: float) -> float:
    """Termo 'outros' = (gases não-CO2 - aerossóis) com dinâmica suave."""
    f_pos = 1.8 / (1.0 + math.exp(-(t_anos - 80.0)/15.0))
    f_neg = 1.2 / (1.0 + math.exp(-(t_anos - 30.0)/8.0))
    return f_pos - f_neg


def emissoes_sinteticas(t_anos: float) -> float:
    """GtCO2/ano: sino assimétrico (sobe até ~60-100 anos, cai depois)."""
    pico = 40.0
    subida = 1.0 / (1.0 + math.exp(-(t_anos - 60.0)/12.0))
    descida = 1.0 / (1.0 + math.exp((t_anos - 120.0)/18.0))
    return pico * subida * descida


# ---------------------------
# 3) Ciclo do carbono (EDS 3 caixas)
# ---------------------------

def ciclo_carbono_rhs(t, y, pars: ParamCarbono, anom_T_global: float):
    """y = [C_atm_ppm, C_sup, C_prof, C_bio]."""
    C_atm, C_sup, C_prof, C_bio = y
    E_ppm = emissoes_sinteticas(t) / pars.gtco2_por_ppm
    # feedback térmico: sumidouros menos eficientes com aquecimento
    fator = 1.0 / (1.0 + 0.05 * max(anom_T_global, 0.0))
    k_as = pars.k_atm_sup * fator
    k_sp = pars.k_sup_prof * fator
    k_ba = pars.k_bio_atm * fator

    dC_atm = E_ppm - k_as*(C_atm - C_sup/pars.vol_sup) - k_ba*(C_atm - C_bio/pars.vol_bio)
    dC_sup =  k_as*(C_atm - C_sup/pars.vol_sup) - k_sp*(C_sup/pars.vol_sup - C_prof/pars.vol_prof)
    dC_prof = k_sp*(C_sup/pars.vol_sup - C_prof/pars.vol_prof)
    dC_bio =  k_ba*(C_atm - C_bio/pars.vol_bio)
    return np.array([dC_atm, dC_sup, dC_prof, dC_bio])


# ---------------------------
# 4) Simulação acoplada PDE + EDOs
# ---------------------------

def simular(
    pf: ParametrosFisicos = ParametrosFisicos(),
    pc: ParamCarbono = ParamCarbono(),
    m: Malha = Malha(),
    exportar_csv: bool = False,
    caminho_csv: str = "series_simulacao.csv"
) -> Dict[str, np.ndarray]:

    phi, mu, dmu, pesos = construir_malha(m)
    L = operador_difusao(m, pf, mu, dmu)
    N = m.N
    S_phi = insolacao_media_lat(phi)

    # Calibração de A (referência): média global S(1-α0)
    A_ref = float(np.sum(S_phi*(1.0 - pf.albedo0) * pesos) / np.sum(pesos))
    pf.A = A_ref

    # Estado inicial
    T = np.zeros(N)  # anomalia latitudinal inicial
    yC = np.array([pf.co2_pre_industrial,
                   pf.co2_pre_industrial*pc.vol_sup,
                   pf.co2_pre_industrial*pc.vol_prof,
                   pf.co2_pre_industrial*pc.vol_bio], dtype=float)

    # Crank–Nicolson
    Ceff = pf.capacidade_termica
    dt = m.dt_seg
    I = identity(N, format="csr")
    A_cn = (Ceff/dt) * I - 0.5 * L
    B_cn = (Ceff/dt) * I + 0.5 * L

    # Séries
    tempos = np.arange(0.0, m.anos_total + m.dt_anos, m.dt_anos)
    Tglob = np.zeros_like(tempos)
    CO2 = np.zeros_like(tempos)
    Ftot = np.zeros_like(tempos)
    NM = np.zeros_like(tempos)     # nível do mar (proxy)
    Emis = np.zeros_like(tempos)

    coef_expansao = 0.8  # mm/ano por K

    for k, t in enumerate(tempos):
        # Força radiativa global (uniforme na latitude nesta formulação)
        F = forca_radiativa_co2(yC[0], pf) + forca_radiativa_outros(t)

        # Termo fonte linearizado: (S*γ - B)T + F(t)
        fonte = (S_phi * pf.gamma_albedo - pf.B) * T + F

        # Passo Crank–Nicolson
        rhs = B_cn.dot(T) + fonte
        T = spsolve(A_cn, rhs)

        # Integração do ciclo do carbono (BDF) dentro de Δt_anos
        def rhsC(t_local, y_local):
            T_global_local = float(np.sum(T * pesos) / np.sum(pesos))
            return ciclo_carbono_rhs(t + t_local, y_local, pc, T_global_local)

        sol = solve_ivp(rhsC, (0.0, m.dt_anos), yC, method="BDF",
                        rtol=1e-7, atol=1e-9, max_step=m.dt_anos/4.0)
        yC = sol.y[:, -1]

        # Médias globais e séries
        T_global = float(np.sum(T * pesos) / np.sum(pesos))
        Tglob[k] = T_global
        CO2[k] = yC[0]
        Ftot[k] = F
        Emis[k] = emissoes_sinteticas(t)
        if k > 0:
            NM[k] = NM[k-1] + coef_expansao * max(T_global, 0.0) * m.dt_anos

    # Impressões com 5 casas
    print(f"A (referência, W/m²): {pf.A:.5f}")
    print(f"gamma_albedo efetivo: {pf.gamma_albedo:.5f} 1/K")
    print(f"T_global final (K):   {float(Tglob[-1]):.5f}")
    print(f"CO₂ final (ppm):      {float(CO2[-1]):.5f}")
    print(f"Força final (W/m²):  {float(Ftot[-1]):.5f}")
    print(f"Nível mar (mm):       {float(NM[-1]):.5f}")

    # Gráficos (um por figura; sem estilos/cores fixas)
    plt.figure(figsize=(8, 4.5))
    plt.plot(tempos, Tglob)
    plt.xlabel("Tempo (anos)"); plt.ylabel("Anomalia Global (K)")
    plt.title("Anomalia Global de Temperatura"); plt.grid(True); plt.show()

    plt.figure(figsize=(8, 4.5))
    plt.plot(tempos, CO2)
    plt.xlabel("Tempo (anos)"); plt.ylabel("CO₂ (ppm)")
    plt.title("CO₂ Atmosférico"); plt.grid(True); plt.show()

    plt.figure(figsize=(8, 4.5))
    plt.plot(tempos, Ftot)
    plt.xlabel("Tempo (anos)"); plt.ylabel("Força (W/m²)")
    plt.title("Força Radiativa Total"); plt.grid(True); plt.show()

    plt.figure(figsize=(8, 4.5))
    plt.plot(tempos, NM)
    plt.xlabel("Tempo (anos)"); plt.ylabel("Nível do mar (mm)")
    plt.title("Expansão Térmica (proxy)"); plt.grid(True); plt.show()

    plt.figure(figsize=(8, 4.5))
    plt.plot(np.degrees(phi), T)
    plt.xlabel("Latitude (graus)"); plt.ylabel("Anomalia (K)")
    plt.title("Perfil Latitudinal — estado final"); plt.grid(True); plt.show()

    if exportar_csv:
        df = pd.DataFrame({
            "ano": tempos,
            "T_global_K": Tglob,
            "CO2_ppm": CO2,
            "forca_Wm2": Ftot,
            "nivel_mar_mm": NM,
            "emissoes_GtCO2ano": Emis
        })
        df.to_csv(caminho_csv, index=False)
        print(f"Séries exportadas em: {caminho_csv}")

    return {
        "phi": phi, "T_lat_final": T, "tempos": tempos, "T_global": Tglob,
        "CO2_ppm": CO2, "forca_Wm2": Ftot, "nivel_mar_mm": NM, "emissoes_GtCO2ano": Emis
    }



if __name__ == "__main__":
    
    _ = simular(
        pf=ParametrosFisicos(),
        pc=ParamCarbono(),
        m=Malha(N=64, dt_anos=0.25, anos_total=200.0),
        exportar_csv=False
    )
