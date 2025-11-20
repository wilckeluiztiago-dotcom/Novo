"""
============================================================
SUPER MODELO MATEMÁTICO PARA PREVISÃO DE RESSACAS (SURGE + ONDAS)
Autor: Luiz Tiago Wilcke (LT)
============================================================

Ideia do modelo:
- Modelar ressacas costeiras como combinação de:
    (1) ELEVAÇÃO DO NÍVEL DO MAR (storm surge) -> Equações de Águas Rasas linearizadas (1D).
    (2) ENERGIA / ALTURA DE ONDAS -> Equação de balanço de energia de ondas (simplificada).
    (3) ACOPLAMENTO surge + waves -> setup costeiro e critério de ressaca.

- Usar:
    • Integração numérica explícita (RK4 + esquema de diferenças finitas).
    • Calibração automática de parâmetros por mínimos quadrados.
    • Previsão por ensemble (Monte Carlo em forçantes).
    • Gráficos completos de diagnóstico e previsão.

============================================================
"""

import os
import math
import json
import time
import warnings
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import welch
from scipy.optimize import least_squares

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURAÇÕES / PARÂMETROS
# ============================================================

@dataclass
class ConfiguracoesModelo:
    # Tempo
    dt_seg: float = 60.0                 # passo de tempo (s)
    duracao_dias: float = 20.0           # duração total simulação (dias)
    janela_calibracao_dias: float = 10.0 # janela usada para calibração
    horizonte_previsao_dias: float = 3.0 # quanto prever

    # Grade espacial 1D (costa)
    comprimento_costa_km: float = 200.0
    n_pontos_espaciais: int = 200

    # Parâmetros físicos
    gravidade: float = 9.81
    densidade_agua: float = 1025.0
    densidade_ar: float = 1.225

    # Coeficientes iniciais (serão calibrados)
    coef_boto_vento: float = 1.2e-3     # coef vento no surge
    coef_atrito_fundo: float = 2.5e-3   # atrito linear no surge
    coef_dispersao_lateral: float = 50.0 # difusão hidrodinâmica (m^2/s)

    # Onda (energia)
    coef_input_vento_onda: float = 2.0e-4
    coef_dissipacao_quebra: float = 1.8e-4
    coef_atrito_onda: float = 1.0e-5
    profundidade_media_m: float = 30.0

    # Setup costeiro (acoplamento)
    coef_setup: float = 0.15  # fração de altura significativa que vira setup

    # Critério de ressaca
    limiar_indice_ressaca: float = 1.0
    limiar_altura_onda_m: float = 2.5
    limiar_surge_m: float = 0.6

    # Ensemble
    n_ensembles: int = 50
    ruido_vento_std: float = 2.0       # m/s
    ruido_pressao_std: float = 1.5     # hPa

    # Saída
    pasta_saida: str = "saida_ressaca"
    salvar_figuras: bool = True


# ============================================================
# 2. FERRAMENTAS NUMÉRICAS
# ============================================================

def rk4_passo(f, y, t, dt, *args):
    """Um passo de Runge-Kutta 4ª ordem."""
    k1 = f(y, t, *args)
    k2 = f(y + 0.5*dt*k1, t + 0.5*dt, *args)
    k3 = f(y + 0.5*dt*k2, t + 0.5*dt, *args)
    k4 = f(y + dt*k3, t + dt, *args)
    return y + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)


def derivada_centrada(u, dx):
    """Derivada 1ª centrada com fronteiras Neumann simples."""
    du = np.zeros_like(u)
    du[1:-1] = (u[2:] - u[:-2])/(2*dx)
    du[0] = (u[1] - u[0])/dx
    du[-1] = (u[-1] - u[-2])/dx
    return du


def derivada_segunda(u, dx):
    """Derivada 2ª centrada (Laplaciano 1D) com Neumann nas pontas."""
    d2u = np.zeros_like(u)
    d2u[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2])/(dx*dx)
    d2u[0] = d2u[1]
    d2u[-1] = d2u[-2]
    return d2u


# ============================================================
# 3. GERAÇÃO/SIMULAÇÃO DE FORÇANTES METEOROLÓGICAS
# ============================================================

def gerar_forcantes_sinteticas(cfg: ConfiguracoesModelo, semente: int = 42) -> pd.DataFrame:
    """
    Gera vento (m/s), direção (rad), pressão (hPa) e maré astronômica (m).
    Estrutura com ciclones aleatórios para criar eventos de ressaca.
    """
    rng = np.random.default_rng(semente)
    n_passos = int(cfg.duracao_dias*24*3600/cfg.dt_seg)
    t = np.arange(n_passos)*cfg.dt_seg
    dias = t/86400.0

    # Vento base com periodicidade sinótica + ruído
    vento_base = 6.0 + 2.5*np.sin(2*np.pi*dias/5.0) + rng.normal(0, 0.8, n_passos)

    # Direção variando lentamente (em radianos)
    direcao_vento = np.cumsum(rng.normal(0, 0.005, n_passos))
    direcao_vento = np.mod(direcao_vento, 2*np.pi)

    # Pressão atmosférica com queda em ciclones
    pressao = 1013.0 + 4*np.sin(2*np.pi*dias/7.0) + rng.normal(0, 0.6, n_passos)

    # Inserir ciclones (quedas de pressão + aumento de vento)
    n_eventos = 4
    for _ in range(n_eventos):
        centro = rng.integers(int(0.1*n_passos), int(0.9*n_passos))
        largura = rng.integers(int(0.02*n_passos), int(0.06*n_passos))
        amplitude_pressao = rng.uniform(10, 25)  # queda
        amplitude_vento = rng.uniform(6, 12)     # aumento

        janela = np.arange(n_passos)
        gauss = np.exp(-0.5*((janela - centro)/largura)**2)

        pressao -= amplitude_pressao*gauss
        vento_base += amplitude_vento*gauss

        # Direção tende a ficar mais perpendicular na costa nos eventos
        direcao_vento += 0.6*gauss

    vento = np.clip(vento_base, 0.0, None)

    # Maré astronômica simples (duas componentes)
    mare = 0.6*np.sin(2*np.pi*dias/0.52) + 0.3*np.sin(2*np.pi*dias/1.04)

    df = pd.DataFrame({
        "tempo_seg": t,
        "dias": dias,
        "vento_m_s": vento,
        "direcao_rad": direcao_vento,
        "pressao_hpa": pressao,
        "mare_m": mare
    })
    return df


# ============================================================
# 4. MODELO HIDRODINÂMICO (SURGE) — ÁGUAS RASAS 1D
# ============================================================

def surge_rhs(estado: np.ndarray, t_seg: float, forcante_local: Dict[str, float],
              cfg: ConfiguracoesModelo, dx: float, profundidade: np.ndarray) -> np.ndarray:
    """
    Águas rasas linearizadas:
        ∂η/∂t + ∂u/∂x = 0
        ∂u/∂t + g ∂η/∂x = (τ_vento)/(ρ*H) - r u + K ∂²u/∂x² - (1/ρ) ∂P/∂x

    onde:
        η(x,t) = elevação
        u(x,t) = velocidade
        τ_vento = ρ_ar C_d |V| V_paralelo
        r = atrito linear
        K = difusão lateral

    Estado é vetor concatenado [η0..ηN-1, u0..uN-1].
    """
    n = profundidade.size
    eta = estado[:n]
    u = estado[n:]

    g = cfg.gravidade
    rho = cfg.densidade_agua
    rho_ar = cfg.densidade_ar
    Cd = cfg.coef_boto_vento
    r = cfg.coef_atrito_fundo
    K = cfg.coef_dispersao_lateral

    vento = forcante_local["vento_m_s"]
    direcao = forcante_local["direcao_rad"]
    pressao = forcante_local["pressao_hpa"]

    # Componente do vento ao longo da costa (simplificado)
    vento_paralelo = vento*np.cos(direcao)

    # Tensão do vento
    tau_vento = rho_ar*Cd*np.abs(vento_paralelo)*vento_paralelo

    # Gradiente de pressão ao longo da costa (simplificado com decaimento)
    # aprox: pressão baixa puxa água -> termo de inclinação barométrica
    grad_p = np.gradient(np.full(n, pressao), dx)  # zero na prática; mantemos estrutura
    termo_pressao = -(1.0/rho)*grad_p

    # Derivadas espaciais
    deta_dx = derivada_centrada(eta, dx)
    du_dx = derivada_centrada(u, dx)
    d2u_dx2 = derivada_segunda(u, dx)

    # Equações
    deta_dt = -du_dx
    du_dt = -g*deta_dx + (tau_vento/(rho*profundidade)) - r*u + K*d2u_dx2 + termo_pressao

    return np.concatenate([deta_dt, du_dt])


def simular_surge(df_forcantes: pd.DataFrame, cfg: ConfiguracoesModelo,
                  parametros_surge: Optional[Dict[str, float]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simula o campo de surge ao longo da costa.
    Retorna:
        tempos_seg, eta(t,x), u(t,x)
    """
    if parametros_surge is not None:
        cfg.coef_boto_vento = parametros_surge.get("coef_boto_vento", cfg.coef_boto_vento)
        cfg.coef_atrito_fundo = parametros_surge.get("coef_atrito_fundo", cfg.coef_atrito_fundo)
        cfg.coef_dispersao_lateral = parametros_surge.get("coef_dispersao_lateral", cfg.coef_dispersao_lateral)

    n_passos = len(df_forcantes)
    L = cfg.comprimento_costa_km*1000.0
    n = cfg.n_pontos_espaciais
    x = np.linspace(0, L, n)
    dx = x[1] - x[0]

    # Profundidade variando suavemente (bathymetria simples)
    profundidade = cfg.profundidade_media_m*(1.0 + 0.3*np.sin(2*np.pi*x/L))
    profundidade = np.clip(profundidade, 5.0, None)

    # Estado inicial
    eta = np.zeros(n)
    u = np.zeros(n)
    estado = np.concatenate([eta, u])

    eta_hist = np.zeros((n_passos, n))
    u_hist = np.zeros((n_passos, n))

    tempos = df_forcantes["tempo_seg"].values

    for k in range(n_passos):
        forcante = {
            "vento_m_s": float(df_forcantes.iloc[k]["vento_m_s"]),
            "direcao_rad": float(df_forcantes.iloc[k]["direcao_rad"]),
            "pressao_hpa": float(df_forcantes.iloc[k]["pressao_hpa"]),
        }

        eta_hist[k] = estado[:n]
        u_hist[k] = estado[n:]

        # Passo RK4
        estado = rk4_passo(surge_rhs, estado, tempos[k], cfg.dt_seg,
                           forcante, cfg, dx, profundidade)

        # estabilidade/limites
        estado[:n] = np.clip(estado[:n], -3.0, 3.0)
        estado[n:] = np.clip(estado[n:], -5.0, 5.0)

    return tempos, eta_hist, u_hist


# ============================================================
# 5. MODELO DE ONDAS (ENERGIA + ALTURA SIGNIFICATIVA)
# ============================================================

def ondas_rhs(E: float, t_seg: float, vento: float, cfg: ConfiguracoesModelo) -> float:
    """
    Equação simplificada de energia de ondas:
        dE/dt = a U^2 - b E^(3/2) - c E
    onde:
        E ~ energia média espectral
        U = velocidade do vento

    Altura significativa:
        Hs = 4 sqrt(E)
    """
    a = cfg.coef_input_vento_onda
    b = cfg.coef_dissipacao_quebra
    c = cfg.coef_atrito_onda

    dE_dt = a*(vento**2) - b*(E**1.5) - c*E
    return dE_dt


def simular_ondas(df_forcantes: pd.DataFrame, cfg: ConfiguracoesModelo,
                  parametros_ondas: Optional[Dict[str, float]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simula energia E(t) e altura significativa Hs(t).
    """
    if parametros_ondas is not None:
        cfg.coef_input_vento_onda = parametros_ondas.get("coef_input_vento_onda", cfg.coef_input_vento_onda)
        cfg.coef_dissipacao_quebra = parametros_ondas.get("coef_dissipacao_quebra", cfg.coef_dissipacao_quebra)
        cfg.coef_atrito_onda = parametros_ondas.get("coef_atrito_onda", cfg.coef_atrito_onda)

    n_passos = len(df_forcantes)
    tempos = df_forcantes["tempo_seg"].values
    vento = df_forcantes["vento_m_s"].values

    E = 0.1
    E_hist = np.zeros(n_passos)
    Hs_hist = np.zeros(n_passos)

    for k in range(n_passos):
        E_hist[k] = E
        Hs_hist[k] = 4.0*np.sqrt(max(E, 0.0))

        # RK4 para escalar
        def f_local(E_local, t_local):
            return ondas_rhs(E_local, t_local, float(vento[k]), cfg)

        E = rk4_passo(lambda e, t: f_local(e, t), E, tempos[k], cfg.dt_seg)
        E = max(E, 0.0)

    return tempos, E_hist, Hs_hist


# ============================================================
# 6. ACOPLAMENTO: ÍNDICE DE RESSACA
# ============================================================

def calcular_indice_ressaca(eta_hist: np.ndarray, Hs_hist: np.ndarray, cfg: ConfiguracoesModelo) -> pd.DataFrame:
    """
    índice = surge_costeiro_normalizado + onda_normalizada + setup
    """
    # surge costeiro (tomamos o ponto x=0 como costa)
    surge_costeiro = eta_hist[:, 0]
    setup = cfg.coef_setup*Hs_hist
    nivel_total = surge_costeiro + setup

    surge_norm = surge_costeiro / cfg.limiar_surge_m
    onda_norm = Hs_hist / cfg.limiar_altura_onda_m
    indice = surge_norm + onda_norm

    return pd.DataFrame({
        "surge_costeiro_m": surge_costeiro,
        "Hs_m": Hs_hist,
        "setup_m": setup,
        "nivel_total_m": nivel_total,
        "indice_ressaca": indice,
        "evento_ressaca": (indice >= cfg.limiar_indice_ressaca).astype(int)
    })


# ============================================================
# 7. GERAÇÃO DE "OBSERVAÇÕES" SINTÉTICAS
# ============================================================

def gerar_observacoes_sinteticas(df_indice: pd.DataFrame, cfg: ConfiguracoesModelo,
                                 semente: int = 123) -> pd.DataFrame:
    """
    Cria uma base observada (com ruído) para calibrar.
    """
    rng = np.random.default_rng(semente)
    obs_surge = df_indice["surge_costeiro_m"].values + rng.normal(0, 0.05, len(df_indice))
    obs_Hs = df_indice["Hs_m"].values + rng.normal(0, 0.12, len(df_indice))
    obs_indice = (obs_surge/cfg.limiar_surge_m) + (obs_Hs/cfg.limiar_altura_onda_m)

    return pd.DataFrame({
        "obs_surge_m": obs_surge,
        "obs_Hs_m": obs_Hs,
        "obs_indice": obs_indice
    })


# ============================================================
# 8. CALIBRAÇÃO DE PARÂMETROS (MÍNIMOS QUADRADOS)
# ============================================================

def calibrar_parametros(df_forcantes: pd.DataFrame, df_obs: pd.DataFrame, cfg: ConfiguracoesModelo) -> Dict[str, float]:
    """
    Ajusta parâmetros de surge e ondas para minimizar erro observado.
    """
    n_passos_cal = int(cfg.janela_calibracao_dias*24*3600/cfg.dt_seg)
    df_forc_cal = df_forcantes.iloc[:n_passos_cal].reset_index(drop=True)
    df_obs_cal = df_obs.iloc[:n_passos_cal].reset_index(drop=True)

    obs_surge = df_obs_cal["obs_surge_m"].values
    obs_Hs = df_obs_cal["obs_Hs_m"].values

    def residuals(theta):
        # theta = [Cd, r, K, a, b, c]
        Cd, r, K, a, b, c = theta

        params_surge = {"coef_boto_vento": Cd, "coef_atrito_fundo": r, "coef_dispersao_lateral": K}
        params_ondas = {"coef_input_vento_onda": a, "coef_dissipacao_quebra": b, "coef_atrito_onda": c}

        _, eta_hist, _ = simular_surge(df_forc_cal, cfg, params_surge)
        _, _, Hs_hist = simular_ondas(df_forc_cal, cfg, params_ondas)

        surge_costeiro = eta_hist[:, 0]

        res1 = surge_costeiro - obs_surge
        res2 = Hs_hist - obs_Hs
        return np.concatenate([res1, res2])

    theta0 = np.array([
        cfg.coef_boto_vento,
        cfg.coef_atrito_fundo,
        cfg.coef_dispersao_lateral,
        cfg.coef_input_vento_onda,
        cfg.coef_dissipacao_quebra,
        cfg.coef_atrito_onda
    ])

    limites_inf = np.array([1e-4, 1e-5, 1.0, 1e-6, 1e-6, 1e-7])
    limites_sup = np.array([5e-3, 1e-2, 800.0, 5e-3, 5e-3, 5e-4])

    resultado = least_squares(residuals, theta0, bounds=(limites_inf, limites_sup), max_nfev=20)

    Cd, r, K, a, b, c = resultado.x

    return {
        "coef_boto_vento": float(Cd),
        "coef_atrito_fundo": float(r),
        "coef_dispersao_lateral": float(K),
        "coef_input_vento_onda": float(a),
        "coef_dissipacao_quebra": float(b),
        "coef_atrito_onda": float(c)
    }


# ============================================================
# 9. PREVISÃO ENSEMBLE (MONTE CARLO)
# ============================================================

def gerar_forcantes_perturbadas(df_forcantes: pd.DataFrame, cfg: ConfiguracoesModelo,
                                rng: np.random.Generator) -> pd.DataFrame:
    """
    Cria uma realização perturbada das forçantes para ensemble.
    """
    df_p = df_forcantes.copy()
    df_p["vento_m_s"] += rng.normal(0, cfg.ruido_vento_std, len(df_p))
    df_p["pressao_hpa"] += rng.normal(0, cfg.ruido_pressao_std, len(df_p))
    df_p["vento_m_s"] = np.clip(df_p["vento_m_s"], 0.0, None)
    return df_p


def rodar_ensemble_previsao(df_forcantes: pd.DataFrame, cfg: ConfiguracoesModelo,
                            parametros: Dict[str, float]) -> Dict[str, np.ndarray]:
    """
    Roda ensemble no horizonte futuro.
    """
    n_total = len(df_forcantes)
    n_cal = int(cfg.janela_calibracao_dias*24*3600/cfg.dt_seg)
    n_prev = int(cfg.horizonte_previsao_dias*24*3600/cfg.dt_seg)

    inicio_prev = n_cal
    fim_prev = min(n_cal + n_prev, n_total)
    df_prev = df_forcantes.iloc[inicio_prev:fim_prev].reset_index(drop=True)

    rng = np.random.default_rng(999)
    indice_ens = np.zeros((cfg.n_ensembles, len(df_prev)))
    surge_ens = np.zeros_like(indice_ens)
    Hs_ens = np.zeros_like(indice_ens)

    for e in range(cfg.n_ensembles):
        df_pert = gerar_forcantes_perturbadas(df_prev, cfg, rng)
        _, eta_hist, _ = simular_surge(df_pert, cfg, parametros)
        _, _, Hs_hist = simular_ondas(df_pert, cfg, parametros)

        df_ind = calcular_indice_ressaca(eta_hist, Hs_hist, cfg)
        indice_ens[e] = df_ind["indice_ressaca"].values
        surge_ens[e] = df_ind["surge_costeiro_m"].values
        Hs_ens[e] = df_ind["Hs_m"].values

    return {
        "indice_ens": indice_ens,
        "surge_ens": surge_ens,
        "Hs_ens": Hs_ens,
        "tempo_prev_seg": df_prev["tempo_seg"].values
    }


# ============================================================
# 10. MÉTRICAS
# ============================================================

def metricas_regressao(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    erro = y_pred - y_true
    mse = float(np.mean(erro**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(erro)))
    mape = float(np.mean(np.abs(erro/(y_true + 1e-9)))*100)
    r2 = float(1 - np.sum(erro**2)/np.sum((y_true - y_true.mean())**2))
    return {"RMSE": rmse, "MAE": mae, "MAPE_%": mape, "R2": r2}


# ============================================================
# 11. GRÁFICOS
# ============================================================

def plotar_forcantes(df_forcantes: pd.DataFrame, cfg: ConfiguracoesModelo, caminho: Optional[str]=None):
    t_dias = df_forcantes["dias"]
    fig = plt.figure(figsize=(12, 7))

    ax1 = fig.add_subplot(3,1,1)
    ax1.plot(t_dias, df_forcantes["vento_m_s"])
    ax1.set_title("Vento (m/s)")
    ax1.set_ylabel("m/s")

    ax2 = fig.add_subplot(3,1,2)
    ax2.plot(t_dias, df_forcantes["pressao_hpa"])
    ax2.set_title("Pressão atmosférica (hPa)")
    ax2.set_ylabel("hPa")

    ax3 = fig.add_subplot(3,1,3)
    ax3.plot(t_dias, df_forcantes["mare_m"])
    ax3.set_title("Maré astronômica (m)")
    ax3.set_ylabel("m")
    ax3.set_xlabel("Dias")

    fig.tight_layout()
    if caminho:
        fig.savefig(caminho, dpi=160)
    plt.show()


def plotar_series(df_forcantes: pd.DataFrame, df_indice: pd.DataFrame, df_obs: pd.DataFrame,
                  cfg: ConfiguracoesModelo, caminho: Optional[str]=None):
    t_dias = df_forcantes["dias"]
    fig = plt.figure(figsize=(13, 8))

    ax1 = fig.add_subplot(3,1,1)
    ax1.plot(t_dias, df_indice["surge_costeiro_m"], label="modelo")
    ax1.plot(t_dias, df_obs["obs_surge_m"], label="observado", alpha=0.7)
    ax1.axhline(cfg.limiar_surge_m, linestyle="--")
    ax1.set_title("Storm surge costeiro (m)")
    ax1.set_ylabel("m")
    ax1.legend()

    ax2 = fig.add_subplot(3,1,2)
    ax2.plot(t_dias, df_indice["Hs_m"], label="modelo")
    ax2.plot(t_dias, df_obs["obs_Hs_m"], label="observado", alpha=0.7)
    ax2.axhline(cfg.limiar_altura_onda_m, linestyle="--")
    ax2.set_title("Altura significativa Hs (m)")
    ax2.set_ylabel("m")
    ax2.legend()

    ax3 = fig.add_subplot(3,1,3)
    ax3.plot(t_dias, df_indice["indice_ressaca"], label="índice modelo")
    ax3.plot(t_dias, df_obs["obs_indice"], label="índice observado", alpha=0.7)
    ax3.axhline(cfg.limiar_indice_ressaca, linestyle="--")
    ax3.set_title("Índice de ressaca")
    ax3.set_ylabel("adimensional")
    ax3.set_xlabel("Dias")
    ax3.legend()

    fig.tight_layout()
    if caminho:
        fig.savefig(caminho, dpi=160)
    plt.show()


def plotar_espectro(df_forcantes: pd.DataFrame, df_indice: pd.DataFrame, cfg: ConfiguracoesModelo,
                    caminho: Optional[str]=None):
    fs = 1.0/cfg.dt_seg  # Hz
    surge = df_indice["surge_costeiro_m"].values
    Hs = df_indice["Hs_m"].values
    indice = df_indice["indice_ressaca"].values

    f1, P1 = welch(surge, fs=fs, nperseg=2048)
    f2, P2 = welch(Hs, fs=fs, nperseg=2048)
    f3, P3 = welch(indice, fs=fs, nperseg=2048)

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(1,1,1)
    ax.semilogy(f1, P1, label="surge")
    ax.semilogy(f2, P2, label="Hs")
    ax.semilogy(f3, P3, label="indice")
    ax.set_title("Espectro de potência (Welch)")
    ax.set_xlabel("Frequência (Hz)")
    ax.set_ylabel("Potência")
    ax.legend()
    fig.tight_layout()

    if caminho:
        fig.savefig(caminho, dpi=160)
    plt.show()


def plotar_ensemble_previsao(df_forcantes: pd.DataFrame, df_indice: pd.DataFrame,
                             ens: Dict[str, np.ndarray], cfg: ConfiguracoesModelo,
                             caminho: Optional[str]=None):
    n_cal = int(cfg.janela_calibracao_dias*24*3600/cfg.dt_seg)
    t_dias = df_forcantes["dias"].values
    t_prev = ens["tempo_prev_seg"]/86400.0 + t_dias[n_cal]

    indice_ens = ens["indice_ens"]
    p05 = np.percentile(indice_ens, 5, axis=0)
    p50 = np.percentile(indice_ens, 50, axis=0)
    p95 = np.percentile(indice_ens, 95, axis=0)

    fig = plt.figure(figsize=(13, 6))
    ax = fig.add_subplot(1,1,1)

    ax.plot(t_dias, df_indice["indice_ressaca"], label="histórico")
    for e in range(min(cfg.n_ensembles, 15)):
        ax.plot(t_prev, indice_ens[e], alpha=0.25)

    ax.plot(t_prev, p50, label="mediana previsão")
    ax.fill_between(t_prev, p05, p95, alpha=0.25, label="faixa 5-95%")
    ax.axhline(cfg.limiar_indice_ressaca, linestyle="--")

    ax.set_title("Previsão de ressaca — ensemble")
    ax.set_xlabel("Dias")
    ax.set_ylabel("Índice")
    ax.legend()

    fig.tight_layout()
    if caminho:
        fig.savefig(caminho, dpi=160)
    plt.show()


# ============================================================
# 12. PIPELINE PRINCIPAL
# ============================================================

def preparar_pasta_saida(cfg: ConfiguracoesModelo):
    os.makedirs(cfg.pasta_saida, exist_ok=True)


def salvar_json(dados: Dict, caminho: str):
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(dados, f, indent=2, ensure_ascii=False)


def pipeline_completa(cfg: ConfiguracoesModelo):
    preparar_pasta_saida(cfg)

    # 1) Forçantes
    df_forcantes = gerar_forcantes_sinteticas(cfg, semente=42)
    if cfg.salvar_figuras:
        plotar_forcantes(df_forcantes, cfg, os.path.join(cfg.pasta_saida, "forcantes.png"))
    else:
        plotar_forcantes(df_forcantes, cfg)

    # 2) Simular "verdade" inicial para gerar observações (base sintética)
    tempos, eta_hist_inicial, _ = simular_surge(df_forcantes, cfg)
    tempos, E_hist_inicial, Hs_hist_inicial = simular_ondas(df_forcantes, cfg)
    df_indice_inicial = calcular_indice_ressaca(eta_hist_inicial, Hs_hist_inicial, cfg)

    df_obs = gerar_observacoes_sinteticas(df_indice_inicial, cfg, semente=123)

    # 3) Calibração
    print("\nCalibrando parâmetros físicos...")
    parametros_calibrados = calibrar_parametros(df_forcantes, df_obs, cfg)
    print("Parâmetros calibrados:")
    for k, v in parametros_calibrados.items():
        print(f"  {k:25s} = {v:.6e}")

    salvar_json(parametros_calibrados, os.path.join(cfg.pasta_saida, "parametros_calibrados.json"))

    # 4) Rodar modelo calibrado completo
    print("\nRodando modelo calibrado completo...")
    tempos, eta_hist, _ = simular_surge(df_forcantes, cfg, parametros_calibrados)
    tempos, _, Hs_hist = simular_ondas(df_forcantes, cfg, parametros_calibrados)
    df_indice = calcular_indice_ressaca(eta_hist, Hs_hist, cfg)

    # 5) Métricas na janela de calibração
    n_cal = int(cfg.janela_calibracao_dias*24*3600/cfg.dt_seg)
    met_surge = metricas_regressao(df_obs["obs_surge_m"].iloc[:n_cal], df_indice["surge_costeiro_m"].iloc[:n_cal])
    met_Hs = metricas_regressao(df_obs["obs_Hs_m"].iloc[:n_cal], df_indice["Hs_m"].iloc[:n_cal])

    print("\nMétricas (janela calibração):")
    print("  Surge:", met_surge)
    print("  Hs   :", met_Hs)

    salvar_json({"surge": met_surge, "Hs": met_Hs}, os.path.join(cfg.pasta_saida, "metricas.json"))

    # 6) Gráficos principais
    if cfg.salvar_figuras:
        plotar_series(df_forcantes, df_indice, df_obs, cfg, os.path.join(cfg.pasta_saida, "series.png"))
        plotar_espectro(df_forcantes, df_indice, cfg, os.path.join(cfg.pasta_saida, "espectro.png"))
    else:
        plotar_series(df_forcantes, df_indice, df_obs, cfg)
        plotar_espectro(df_forcantes, df_indice, cfg)

    # 7) Previsão ensemble no futuro
    print("\nRodando ensemble de previsão...")
    ens = rodar_ensemble_previsao(df_forcantes, cfg, parametros_calibrados)

    if cfg.salvar_figuras:
        plotar_ensemble_previsao(df_forcantes, df_indice, ens, cfg,
                                 os.path.join(cfg.pasta_saida, "previsao_ensemble.png"))
    else:
        plotar_ensemble_previsao(df_forcantes, df_indice, ens, cfg)

    # 8) Probabilidade futura de ressaca
    indice_ens = ens["indice_ens"]
    prob_ressaca = np.mean(indice_ens >= cfg.limiar_indice_ressaca, axis=0)
    t_prev = ens["tempo_prev_seg"]/86400.0 + df_forcantes["dias"].iloc[n_cal]

    fig = plt.figure(figsize=(12,5))
    ax = fig.add_subplot(1,1,1)
    ax.plot(t_prev, prob_ressaca)
    ax.set_ylim(0, 1)
    ax.set_title("Probabilidade de ressaca no horizonte de previsão")
    ax.set_xlabel("Dias")
    ax.set_ylabel("Probabilidade")
    fig.tight_layout()

    if cfg.salvar_figuras:
        fig.savefig(os.path.join(cfg.pasta_saida, "probabilidade_ressaca.png"), dpi=160)
    plt.show()

    df_prob = pd.DataFrame({"dias_prev": t_prev, "prob_ressaca": prob_ressaca})
    df_prob.to_csv(os.path.join(cfg.pasta_saida, "probabilidade_ressaca.csv"), index=False)

    # 9) Log final
    print("\n✅ Pipeline finalizada.")
    print(f"Arquivos salvos em: {cfg.pasta_saida}/")


# ============================================================
# 13. EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    cfg = ConfiguracoesModelo(
        dt_seg=60.0,
        duracao_dias=20.0,
        janela_calibracao_dias=10.0,
        horizonte_previsao_dias=3.0,
        n_pontos_espaciais=200,
        n_ensembles=60,
        salvar_figuras=True
    )
    pipeline_completa(cfg)

