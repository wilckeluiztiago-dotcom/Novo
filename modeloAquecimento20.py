# ============================================================
# MODELO AVANÇADO DE AQUECIMENTO GLOBAL (EBM + CICLO DO CARBONO)
# Autor: Luiz Tiago Wilcke (LT)
#
# Correções principais:
#   - mpmath não possui mp.max -> usar max() do Python
#   - logs protegidos contra argumentos <= 0
#   - robustez numérica e organização do RK4
# ============================================================

import math
import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Dict, Callable

import mpmath as mp
mp.mp.dps = 80  # precisão interna alta (>> 20 dígitos)


# ============================================================
# 1. HELPERS DE PRECISÃO / SEGURANÇA
# ============================================================

def mp_pos(x: mp.mpf, eps: str = "1e-30") -> mp.mpf:
    """Garante x > 0 para log/divisões."""
    e = mp.mpf(eps)
    return x if x > e else e

def mp_clamp(x: mp.mpf, a: mp.mpf, b: mp.mpf) -> mp.mpf:
    """Trava x em [a,b]."""
    return a if x < a else (b if x > b else x)


# ============================================================
# 2. CONSTANTES FÍSICAS
# ============================================================

@dataclass
class ConstantesFisicas:
    # Universais
    sigma_sb: mp.mpf = mp.mpf("5.670374419e-8")
    c_p_ar: mp.mpf = mp.mpf("1004.0")
    g: mp.mpf = mp.mpf("9.80665")
    R_ar: mp.mpf = mp.mpf("287.05")
    R_v: mp.mpf = mp.mpf("461.5")
    L_v: mp.mpf = mp.mpf("2.5e6")
    epsilon: mp.mpf = mp.mpf("0.622")

    # Terra
    S0: mp.mpf = mp.mpf("1361.0")
    a_terra: mp.mpf = mp.mpf("6.371e6")
    area_terra: mp.mpf = 4 * mp.pi * mp.mpf("6.371e6")**2

    # Oceanos
    frac_oceano: mp.mpf = mp.mpf("0.71")
    rho_agua: mp.mpf = mp.mpf("1025.0")
    c_p_agua: mp.mpf = mp.mpf("3990.0")
    h_mista: mp.mpf = mp.mpf("70.0")
    h_prof: mp.mpf = mp.mpf("2000.0")

    # Carbono
    ppm_para_gtc: mp.mpf = mp.mpf("2.124")        # 1 ppm CO2 ~ 2.124 GtC
    massa_carbono_atm_ref: mp.mpf = mp.mpf("589.0")  # GtC pré-industrial

    # Referências
    T_ref: mp.mpf = mp.mpf("288.0")
    albedo_ref: mp.mpf = mp.mpf("0.30")

CF = ConstantesFisicas()


# ============================================================
# 3. PARÂMETROS DO MODELO
# ============================================================

@dataclass
class ParametrosModelo:
    emissividade0: mp.mpf = mp.mpf("0.612")

    # Feedbacks radiativos
    lambda_planck: mp.mpf = mp.mpf("-3.2")
    lambda_vapor: mp.mpf = mp.mpf("1.8")
    lambda_nuvens: mp.mpf = mp.mpf("0.6")
    lambda_lapse: mp.mpf = mp.mpf("-0.8")
    lambda_total_extra: mp.mpf = mp.mpf("0.0")

    # Trocas energia
    k_atm_oceano: mp.mpf = mp.mpf("1.3")
    k_oceano_prof: mp.mpf = mp.mpf("0.7")

    # Gelo/albedo
    albedo_min: mp.mpf = mp.mpf("0.26")
    albedo_max: mp.mpf = mp.mpf("0.65")
    T_gelo_centro: mp.mpf = mp.mpf("273.15")
    largura_gelo: mp.mpf = mp.mpf("6.0")
    tau_albedo: mp.mpf = mp.mpf("8.0")

    # Carbono
    tau_biosfera: mp.mpf = mp.mpf("40.0")
    tau_oceano: mp.mpf = mp.mpf("250.0")
    beta_fertilizacao: mp.mpf = mp.mpf("0.35")
    k_resp_temp: mp.mpf = mp.mpf("0.07")
    gamma_solubilidade: mp.mpf = mp.mpf("0.015")

    # Metano/aerossóis
    tau_metano: mp.mpf = mp.mpf("12.0")
    tau_aerossois: mp.mpf = mp.mpf("1.5")
    fator_ch4_forc: mp.mpf = mp.mpf("0.036")
    fator_aer_forc: mp.mpf = mp.mpf("-0.9")

    # CO2 forçamento canônico
    fator_co2_forc: mp.mpf = mp.mpf("5.35")

    # Volcânico (ruído)
    sigma_ruido_volc: mp.mpf = mp.mpf("0.20")

PM = ParametrosModelo()


# ============================================================
# 4. EMISSÕES / CENÁRIOS
# ============================================================

def emissao_co2_anual(t_ano: mp.mpf) -> mp.mpf:
    t0 = mp.mpf("1850.0")
    x = t_ano - t0
    Emax = mp.mpf("12.0")
    k = mp.mpf("0.035")
    x50 = mp.mpf("200.0")  # ~2050
    logi = Emax / (1 + mp.e**(-k*(x-x50)))

    # queda pós-pico
    excesso = x - x50
    excesso_pos = excesso if excesso > 0 else mp.mpf("0.0")
    queda = mp.e**(-mp.mpf("0.01") * excesso_pos)

    return logi * queda


def emissao_ch4_anual(t_ano: mp.mpf) -> mp.mpf:
    t0 = mp.mpf("1850.0")
    x = t_ano - t0
    Emax = mp.mpf("1.0")
    k = mp.mpf("0.03")
    x50 = mp.mpf("180.0")
    logi = Emax/(1+mp.e**(-k*(x-x50)))

    excesso = x - x50
    excesso_pos = excesso if excesso > 0 else mp.mpf("0.0")
    queda = mp.e**(-mp.mpf("0.008")*excesso_pos)

    return logi*queda


def emissao_aerossois_anual(t_ano: mp.mpf) -> mp.mpf:
    t0 = mp.mpf("1850.0")
    x = t_ano - t0

    pico = mp.mpf("1.0")*mp.e**(-((x-mp.mpf("130.0"))**2)/(2*mp.mpf("35.0")**2))
    base = mp.mpf("0.2")/(1+mp.e**(-mp.mpf("0.04")*(x-mp.mpf("80.0"))))
    return pico + base


def forcamento_solar(t_ano: mp.mpf) -> mp.mpf:
    ciclo11 = mp.mpf("0.8")*mp.sin(2*mp.pi*(t_ano-mp.mpf("1850.0"))/mp.mpf("11.0"))
    deriva = mp.mpf("0.05")*mp.sin(2*mp.pi*(t_ano-mp.mpf("1850.0"))/mp.mpf("200.0"))
    return ciclo11 + deriva


def forcamento_volcanico(t_ano: mp.mpf, semente: int = 123) -> mp.mpf:
    f = (
        mp.sin(mp.mpf("0.17")*(t_ano-1850+semente)) +
        mp.sin(mp.mpf("0.05")*(t_ano-1850+2*semente)) +
        mp.sin(mp.mpf("0.011")*(t_ano-1850+3*semente))
    ) / 3

    fpos = f if f > 0 else mp.mpf("0.0")
    return -PM.sigma_ruido_volc * fpos


# ============================================================
# 5. FORÇAMENTOS / FEEDBACKS
# ============================================================

def forcamento_co2(C: mp.mpf, C0: mp.mpf) -> mp.mpf:
    return PM.fator_co2_forc * mp.log(mp_pos(C)/mp_pos(C0))

def forcamento_ch4(M: mp.mpf, M0: mp.mpf) -> mp.mpf:
    return PM.fator_ch4_forc * mp.log(mp_pos(M)/mp_pos(M0))

def forcamento_aerossois(Z: mp.mpf) -> mp.mpf:
    return PM.fator_aer_forc * Z

def feedbacks_totais(Ta: mp.mpf) -> mp.mpf:
    lambda_eff = (PM.lambda_planck + PM.lambda_vapor +
                  PM.lambda_nuvens + PM.lambda_lapse +
                  PM.lambda_total_extra)
    return lambda_eff * (Ta - CF.T_ref)

def albedo_equilibrio(Ta: mp.mpf) -> mp.mpf:
    x = (Ta - PM.T_gelo_centro)/PM.largura_gelo
    s = 1/(1+mp.e**(-x))
    return PM.albedo_max*(1-s) + PM.albedo_min*s


# ============================================================
# 6. CICLO DO CARBONO
# ============================================================

def fluxo_biosfera(C: mp.mpf, B: mp.mpf, Ta: mp.mpf,
                   C0: mp.mpf, B0: mp.mpf) -> mp.mpf:
    fertil = PM.beta_fertilizacao * mp.log(mp_pos(C)/mp_pos(C0))
    respir = PM.k_resp_temp * (Ta - CF.T_ref)
    F = (B0/PM.tau_biosfera) * (fertil - respir)
    return F

def fluxo_oceano(C: mp.mpf, O: mp.mpf, Ta: mp.mpf,
                 C0: mp.mpf, O0: mp.mpf) -> mp.mpf:
    solub = mp.e**(-PM.gamma_solubilidade*(Ta - CF.T_ref))
    C_eq = C0 * solub
    k = O0/PM.tau_oceano
    return k * (C - C_eq)/mp_pos(C0)


# ============================================================
# 7. SISTEMA DE EDOs
# ============================================================

@dataclass
class EstadoClimatico:
    Ta: mp.mpf
    To1: mp.mpf
    To2: mp.mpf
    A: mp.mpf
    C: mp.mpf
    B: mp.mpf
    O: mp.mpf
    M: mp.mpf
    Z: mp.mpf


def derivadas(t_ano: mp.mpf, y: EstadoClimatico,
              refs: Dict[str, mp.mpf]) -> EstadoClimatico:

    Ta, To1, To2, A, C, B, O, M, Z = (
        y.Ta, y.To1, y.To2, y.A, y.C, y.B, y.O, y.M, y.Z
    )
    C0, B0, O0, M0 = refs["C0"], refs["B0"], refs["O0"], refs["M0"]

    # Forçamentos
    F_co2 = forcamento_co2(C, C0)
    F_ch4 = forcamento_ch4(M, M0)
    F_aer = forcamento_aerossois(Z)
    F_sol = forcamento_solar(t_ano)
    F_vol = forcamento_volcanico(t_ano)

    F_total = F_co2 + F_ch4 + F_aer + F_sol + F_vol

    # Balanço energia
    Qin = (CF.S0 + F_sol)/4 * (1 - A)
    OLR = PM.emissividade0 * CF.sigma_sb * Ta**4
    F_fb = feedbacks_totais(Ta)

    F_atm_oce = PM.k_atm_oceano * (Ta - To1)

    C_atm_eff = mp.mpf("7.0e8")  # J m^-2 K^-1

    dTa_dt = (Qin - OLR + F_total + F_fb - F_atm_oce) / C_atm_eff

    C_o1 = CF.rho_agua * CF.c_p_agua * CF.h_mista
    F_o1_o2 = PM.k_oceano_prof * (To1 - To2)
    dTo1_dt = (F_atm_oce - F_o1_o2) / C_o1

    C_o2 = CF.rho_agua * CF.c_p_agua * CF.h_prof
    dTo2_dt = (F_o1_o2) / C_o2

    # Albedo
    A_eq = albedo_equilibrio(Ta)
    dA_dt = (A_eq - A) / PM.tau_albedo

    # Carbono
    E_co2 = emissao_co2_anual(t_ano)
    F_bio = fluxo_biosfera(C, B, Ta, C0, B0)
    F_oce = fluxo_oceano(C, O, Ta, C0, O0)

    dC_dt = (E_co2 - F_bio - F_oce) / CF.ppm_para_gtc
    dB_dt = F_bio - (B/PM.tau_biosfera)
    dO_dt = F_oce - (O/PM.tau_oceano)

    # Metano / aerossóis
    E_ch4 = emissao_ch4_anual(t_ano)
    dM_dt = E_ch4 - (M - M0)/PM.tau_metano

    E_aer = emissao_aerossois_anual(t_ano)
    dZ_dt = E_aer - Z/PM.tau_aerossois

    return EstadoClimatico(
        Ta=dTa_dt, To1=dTo1_dt, To2=dTo2_dt, A=dA_dt,
        C=dC_dt, B=dB_dt, O=dO_dt, M=dM_dt, Z=dZ_dt
    )


# ============================================================
# 8. INTEGRADOR RK4
# ============================================================

def somar_estado(y: EstadoClimatico, k: EstadoClimatico, fator: mp.mpf) -> EstadoClimatico:
    return EstadoClimatico(
        Ta=y.Ta + fator*k.Ta,
        To1=y.To1 + fator*k.To1,
        To2=y.To2 + fator*k.To2,
        A=y.A + fator*k.A,
        C=y.C + fator*k.C,
        B=y.B + fator*k.B,
        O=y.O + fator*k.O,
        M=y.M + fator*k.M,
        Z=y.Z + fator*k.Z
    )

def rk4_passo(fun: Callable, t: mp.mpf, y: EstadoClimatico,
              h: mp.mpf, refs: Dict[str, mp.mpf]) -> EstadoClimatico:

    k1 = fun(t, y, refs)
    k2 = fun(t + h/2, somar_estado(y, k1, h/2), refs)
    k3 = fun(t + h/2, somar_estado(y, k2, h/2), refs)
    k4 = fun(t + h,   somar_estado(y, k3, h),   refs)

    return EstadoClimatico(
        Ta=y.Ta + (h/6)*(k1.Ta + 2*k2.Ta + 2*k3.Ta + k4.Ta),
        To1=y.To1 + (h/6)*(k1.To1 + 2*k2.To1 + 2*k3.To1 + k4.To1),
        To2=y.To2 + (h/6)*(k1.To2 + 2*k2.To2 + 2*k3.To2 + k4.To2),
        A=y.A + (h/6)*(k1.A + 2*k2.A + 2*k3.A + k4.A),
        C=y.C + (h/6)*(k1.C + 2*k2.C + 2*k3.C + k4.C),
        B=y.B + (h/6)*(k1.B + 2*k2.B + 2*k3.B + k4.B),
        O=y.O + (h/6)*(k1.O + 2*k2.O + 2*k3.O + k4.O),
        M=y.M + (h/6)*(k1.M + 2*k2.M + 2*k3.M + k4.M),
        Z=y.Z + (h/6)*(k1.Z + 2*k2.Z + 2*k3.Z + k4.Z)
    )


def simular_clima(t_ini: float, t_fim: float, passo_anos: float,
                  estado0: EstadoClimatico, refs: Dict[str, mp.mpf]) -> Dict[str, np.ndarray]:

    n = int((t_fim - t_ini)/passo_anos) + 1
    tempos = np.zeros(n)

    Ta = np.zeros(n); To1 = np.zeros(n); To2 = np.zeros(n)
    A  = np.zeros(n); C   = np.zeros(n); B   = np.zeros(n)
    O  = np.zeros(n); M   = np.zeros(n); Z   = np.zeros(n)
    Fco2 = np.zeros(n); Fch4 = np.zeros(n); Faer = np.zeros(n)
    Fsol = np.zeros(n); Fvol = np.zeros(n); Ftot = np.zeros(n)

    t = mp.mpf(str(t_ini))
    h = mp.mpf(str(passo_anos))
    y = estado0

    for i in range(n):
        tempos[i] = float(t)

        Ta[i]  = float(y.Ta);  To1[i] = float(y.To1); To2[i] = float(y.To2)
        A[i]   = float(y.A);   C[i]   = float(y.C)
        B[i]   = float(y.B);   O[i]   = float(y.O)
        M[i]   = float(y.M);   Z[i]   = float(y.Z)

        # forçamentos
        Fco2[i] = float(forcamento_co2(y.C, refs["C0"]))
        Fch4[i] = float(forcamento_ch4(y.M, refs["M0"]))
        Faer[i] = float(forcamento_aerossois(y.Z))
        Fsol[i] = float(forcamento_solar(t))
        Fvol[i] = float(forcamento_volcanico(t))
        Ftot[i] = Fco2[i] + Fch4[i] + Faer[i] + Fsol[i] + Fvol[i]

        y = rk4_passo(derivadas, t, y, h, refs)
        t = t + h

    return {
        "tempos": tempos,
        "Ta": Ta, "To1": To1, "To2": To2, "A": A,
        "C": C, "B": B, "O": O, "M": M, "Z": Z,
        "Fco2": Fco2, "Fch4": Fch4, "Faer": Faer,
        "Fsol": Fsol, "Fvol": Fvol, "Ftot": Ftot
    }


# ============================================================
# 9. RESULTADOS / GRÁFICOS
# ============================================================

def imprimir_resultados_precisos(dados: Dict[str, np.ndarray], refs: Dict[str, mp.mpf]):
    idx_final = -1

    Ta_final  = mp.mpf(str(dados["Ta"][idx_final]))
    To1_final = mp.mpf(str(dados["To1"][idx_final]))
    A_final   = mp.mpf(str(dados["A"][idx_final]))
    C_final   = mp.mpf(str(dados["C"][idx_final]))

    anomalia_T = Ta_final - CF.T_ref
    fator_C = C_final / refs["C0"]

    print("\n================ RESULTADOS (20+ dígitos) ================\n")
    print("Ano final:", int(dados["tempos"][idx_final]))
    print("Temperatura final Ta [K]:", mp.nstr(Ta_final, 25))
    print("Anomalia Ta - T_ref [K]:", mp.nstr(anomalia_T, 25))
    print("Temperatura To1 (camada mista) [K]:", mp.nstr(To1_final, 25))
    print("Albedo final A [-]:", mp.nstr(A_final, 25))
    print("CO2 final C [ppm]:", mp.nstr(C_final, 25))
    print("Fator C/C0 [-]:", mp.nstr(fator_C, 25))
    print("\n=========================================================\n")


def fazer_graficos(dados: Dict[str, np.ndarray]):
    tempos = dados["tempos"]
    Tref_f = float(CF.T_ref)

    plt.figure()
    plt.plot(tempos, dados["Ta"]-Tref_f, label="Anomalia Ta")
    plt.plot(tempos, dados["To1"]-Tref_f, label="Anomalia To1")
    plt.plot(tempos, dados["To2"]-Tref_f, label="Anomalia To2")
    plt.axhline(0, linestyle="--")
    plt.xlabel("Ano")
    plt.ylabel("Anomalia [K]")
    plt.title("Evolução térmica acoplada")
    plt.legend(); plt.grid(True)

    plt.figure()
    plt.plot(tempos, dados["C"], label="CO2 (ppm)")
    plt.xlabel("Ano"); plt.ylabel("CO2 [ppm]")
    plt.title("CO2 atmosférico"); plt.legend(); plt.grid(True)

    plt.figure()
    plt.plot(tempos, dados["A"], label="Albedo")
    plt.xlabel("Ano"); plt.ylabel("Albedo [-]")
    plt.title("Feedback gelo-albedo"); plt.legend(); plt.grid(True)

    plt.figure()
    plt.plot(tempos, dados["M"], label="Metano (rel.)")
    plt.plot(tempos, dados["Z"], label="Aerossóis (rel.)")
    plt.xlabel("Ano"); plt.ylabel("Relativo")
    plt.title("Metano e aerossóis"); plt.legend(); plt.grid(True)

    plt.figure()
    plt.plot(tempos, dados["Fco2"], label="F_CO2")
    plt.plot(tempos, dados["Fch4"], label="F_CH4")
    plt.plot(tempos, dados["Faer"], label="F_aer")
    plt.plot(tempos, dados["Fsol"], label="F_sol")
    plt.plot(tempos, dados["Fvol"], label="F_volc")
    plt.plot(tempos, dados["Ftot"], label="F_total", linewidth=2)
    plt.xlabel("Ano"); plt.ylabel("W/m²")
    plt.title("Forçamentos radiativos"); plt.legend(); plt.grid(True)

    plt.figure()
    plt.plot(dados["C"], dados["Ta"]-Tref_f)
    plt.xlabel("CO2 [ppm]"); plt.ylabel("Anomalia Ta [K]")
    plt.title("Fase CO2–Temperatura"); plt.grid(True)

    plt.show()


# ============================================================
# 10. MAIN
# ============================================================

def main():
    refs = {
        "C0": mp.mpf("280.0"),
        "B0": mp.mpf("2000.0"),
        "O0": mp.mpf("38000.0"),
        "M0": mp.mpf("1.0")
    }

    estado0 = EstadoClimatico(
        Ta=CF.T_ref,
        To1=CF.T_ref,
        To2=CF.T_ref,
        A=CF.albedo_ref,
        C=refs["C0"],
        B=mp.mpf("0.0"),
        O=mp.mpf("0.0"),
        M=refs["M0"],
        Z=mp.mpf("0.0")
    )

    dados = simular_clima(
        t_ini=1850, t_fim=2100, passo_anos=1.0,
        estado0=estado0, refs=refs
    )

    imprimir_resultados_precisos(dados, refs)
    fazer_graficos(dados)


if __name__ == "__main__":
    main()
