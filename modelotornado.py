# ============================================================
# MODELO ESPAÇO-TEMPORAL DE RISCO DE TORNADO NO PARANÁ
# SPDE + RUÍDO DE LÉVY α-ESTÁVEL (CAUDA LONGA)
# Autor: Luiz Tiago Wilcke (LT)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional


# ============================================================
# 1) PARÂMETROS DO MODELO
# ============================================================

@dataclass
class ParametrosEspacoTemporal:
    nx: int = 50           # pontos em x (longitude)
    ny: int = 40           # pontos em y (latitude)
    dx_km: float = 10.0    # resolução espacial em km
    dy_km: float = 10.0
    dt_horas: float = 0.25 # passo de tempo em horas
    passos_tempo: int = 240  # 60 horas de simulação


@dataclass
class ParametrosLevy:
    alpha: float = 1.5     # índice de estabilidade (1<alpha<2 para cauda pesada)
    beta: float = 0.0      # simétrico
    sigma_ruido: float = 0.8  # intensidade do ruído de Lévy
    escala_base: float = 1.0  # escala base dos saltos


@dataclass
class ParametrosSPDE:
    difusao_km2_h: float = 6.0    # difusão moderada
    decaimento: float = 0.12      # puxa I para baixo
    forca_media: float = 0.4      # empurra I para cima
    max_instabilidade: float = 20.0  # saturação física da instabilidade


@dataclass
class ParametrosRisco:
    limiar_tornado: float = 9.0      # nível de instabilidade "severa"
    inclinacao_logistica: float = 0.8  # suavidade da transição risco vs instabilidade


# ============================================================
# 2) GERADOR DE RUIDO LÉVY α-ESTÁVEL (Chambers–Mallows–Stuck)
# ============================================================

def levy_estavel_rvs(alpha: float, beta: float, escala: float, tamanho):
    """
    Gera variáveis aleatórias S(alpha, beta, escala, 0) (estável α-espalhado)
    usando o algoritmo de Chambers–Mallows–Stuck.
    Aqui consideramos alpha != 1.

    alpha in (0,2], beta in [-1,1], escala > 0.
    """
    U = np.random.uniform(-np.pi/2, np.pi/2, size=tamanho)
    W = np.random.exponential(1.0, size=tamanho)

    if alpha == 1.0:
        # Caso especial alpha=1 não vai ser usado aqui; deixamos simples
        raise ValueError("alpha=1 não implementado nesta versão.")

    # Para alpha ≠ 1
    # Fórmula geral
    phi = beta * np.tan(np.pi * alpha / 2.0)

    # Evita cos(U) muito próximo de zero (overflow numérico)
    cosU = np.cos(U)
    eps = 1e-6
    cosU = np.sign(cosU) * np.maximum(np.abs(cosU), eps)

    parte1 = np.sin(alpha * (U + np.arctan(phi)/alpha)) / (cosU ** (1.0/alpha))

    numerador = np.cos(U - alpha * (U + np.arctan(phi)/alpha))
    numerador = np.sign(numerador) * np.maximum(np.abs(numerador), eps)

    parte2 = (numerador / W) ** ((1.0 - alpha) / alpha)

    X = parte1 * parte2
    return escala * X.astype(np.float32)


# ============================================================
# 3) MODELO ESPAÇO-TEMPORAL (SPDE COM LÉVY)
# ============================================================

def inicializar_campo_instabilidade(par_espaco: ParametrosEspacoTemporal) -> np.ndarray:
    """
    Campo inicial I(s,0): mais instável no noroeste do Paraná, com ruído gaussiano leve.
    """
    ny, nx = par_espaco.ny, par_espaco.nx
    y_grid, x_grid = np.meshgrid(
        np.linspace(0.0, 1.0, ny),
        np.linspace(0.0, 1.0, nx),
        indexing="ij"
    )

    base = 3.0 + 3.0 * (1.0 - x_grid) * (1.0 - y_grid)
    ruido = np.random.normal(0.0, 0.5, size=(ny, nx))
    campo0 = base + ruido
    campo0[campo0 < 0.0] = 0.0
    return campo0.astype(np.float32)


def forcamento_espacial(par_espaco: ParametrosEspacoTemporal) -> np.ndarray:
    """
    Forçamento espacial F(x,y): faixa de instabilidade (corredor de tornados).
    """
    ny, nx = par_espaco.ny, par_espaco.nx
    y_grid, x_grid = np.meshgrid(
        np.linspace(0.0, 1.0, ny),
        np.linspace(0.0, 1.0, nx),
        indexing="ij"
    )

    faixa = np.exp(-((y_grid - 0.35) ** 2) / 0.02)  # faixa horizontal
    gradx = 0.6 + 0.4 * (1.0 - x_grid)             # ligeiro aumento para oeste
    F = faixa * gradx
    return F.astype(np.float32)


def laplaciano_2d(campo: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Laplaciano 2D com borda de Neumann (derivada normal ~ 0).
    """
    ext = np.pad(campo, 1, mode="edge")
    d2dx2 = (ext[1:-1, 2:] - 2.0 * ext[1:-1, 1:-1] + ext[1:-1, :-2]) / (dx ** 2)
    d2dy2 = (ext[2:, 1:-1] - 2.0 * ext[1:-1, 1:-1] + ext[:-2, 1:-1]) / (dy ** 2)
    return d2dx2 + d2dy2


def passo_spde_levy(
    campo_atual: np.ndarray,
    par_espaco: ParametrosEspacoTemporal,
    par_spde: ParametrosSPDE,
    par_levy: ParametrosLevy,
    mapa_forcamento: np.ndarray
) -> np.ndarray:
    """
    Um passo de Euler para a SPDE:

        dI = [D ∇²I - k I + μ F(x,y)] dt + σ dL_t^(α)

    onde dL_t^(α) são incrementos α-estáveis, escala ~ dt^{1/α}.
    """
    dt = par_espaco.dt_horas
    dx = par_espaco.dx_km
    dy = par_espaco.dy_km

    I = campo_atual

    lap = laplaciano_2d(I, dx, dy)
    termo_difusao = par_spde.difusao_km2_h * lap
    termo_drift = -par_spde.decaimento * I + par_spde.forca_media * mapa_forcamento

    # Incrementos de Lévy (heavy-tailed)
    escala_ruido = par_levy.sigma_ruido * (dt ** (1.0 / par_levy.alpha)) * par_levy.escala_base
    ruido_levy = levy_estavel_rvs(
        alpha=par_levy.alpha,
        beta=par_levy.beta,
        escala=escala_ruido,
        tamanho=I.shape
    )

    I_novo = I + dt * (termo_difusao + termo_drift) + ruido_levy

    # Saturação física: instabilidade não negativa e limitada
    I_novo = np.clip(I_novo, 0.0, par_spde.max_instabilidade).astype(np.float32)
    return I_novo


def simular_instabilidade_levy(
    par_espaco: ParametrosEspacoTemporal,
    par_spde: ParametrosSPDE,
    par_levy: ParametrosLevy,
    seed: Optional[int] = 42
) -> np.ndarray:
    """
    Simula I(t,x,y) com ruído de Lévy, retornando um array [T, ny, nx].
    """
    if seed is not None:
        np.random.seed(seed)

    T = par_espaco.passos_tempo
    ny, nx = par_espaco.ny, par_espaco.nx

    campos = np.zeros((T, ny, nx), dtype=np.float32)
    campos[0] = inicializar_campo_instabilidade(par_espaco)

    mapa_forcamento = forcamento_espacial(par_espaco)

    for t in range(1, T):
        campos[t] = passo_spde_levy(
            campos[t - 1],
            par_espaco,
            par_spde,
            par_levy,
            mapa_forcamento
        )

    return campos


# ============================================================
# 4) MAPA DE RISCO E ANÁLISE DE CAUDA LONGA
# ============================================================

def mapa_risco_tornado(
    campo_instabilidade: np.ndarray,
    par_risco: ParametrosRisco
) -> np.ndarray:
    """
    Converte instabilidade I(s) em probabilidade de tornado via função logística:

        risco(s) = 1 / (1 + exp(-a (I(s) - u)))
    """
    I = campo_instabilidade
    a = par_risco.inclinacao_logistica
    u = par_risco.limiar_tornado

    logits = a * (I - u)
    risco = 1.0 / (1.0 + np.exp(-logits))
    return risco.astype(np.float32)


def analisar_cauda_longa(campo_temporal: np.ndarray):
    """
    Faz análise simples de cauda longa da série temporal de um ponto:

      - Série I(t)
      - Gráfico log–log de P(X >= x) vs x (cauda empírica)
    """
    serie = campo_temporal.astype(float)

    # Série temporal já será plottada externamente; aqui preparamos cauda.
    # Cauda empírica: P(X >= x) = rank / N
    valores = np.sort(serie)
    N = len(valores)
    # Usamos apenas parte superior (cauda) – top 20%
    idx_ini = int(0.8 * N)
    cauda = valores[idx_ini:]
    if cauda.size < 5:
        return valores, None, None

    # Probabilidade de exceder: P(X >= x_i) ~ (N - i + 1)/N
    ranks = np.arange(idx_ini, N)
    probs = (N - ranks) / N

    return valores, cauda, probs


# ============================================================
# 5) GRÁFICOS
# ============================================================

def gerar_graficos(
    campos: np.ndarray,
    risco: np.ndarray,
    par_espaco: ParametrosEspacoTemporal,
    indice_ponto: Optional[tuple] = None
):
    """
    Gera vários gráficos:
      - instabilidade final
      - série temporal da instabilidade média
      - série temporal num ponto específico
      - cauda log–log num ponto
      - mapa de risco de tornado
    """
    T, ny, nx = campos.shape
    if indice_ponto is None:
        # por padrão, pega um ponto no "corredor de instabilidade"
        indice_ponto = (int(0.35 * ny), int(0.2 * nx))

    j0, i0 = indice_ponto

    # 1) Instabilidade final
    campo_final = campos[-1]
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    im1 = ax1.imshow(campo_final, origin="lower")
    ax1.plot(i0, j0, "ro", markersize=4, label="Ponto analisado")
    ax1.legend(loc="upper right")
    ax1.set_title("Instabilidade (Lévy) — Último Passo")
    ax1.set_xlabel("Longitude (índice)")
    ax1.set_ylabel("Latitude (índice)")
    plt.colorbar(im1, ax=ax1, label="Instabilidade")
    plt.tight_layout()
    fig1.savefig("levy_instabilidade_final.png", dpi=160)
    plt.close(fig1)

    # 2) Série temporal da média espacial
    media_tempo = campos.mean(axis=(1, 2))
    tempos_horas = np.arange(T) * par_espaco.dt_horas

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(tempos_horas, media_tempo, linewidth=1.5)
    ax2.set_title("Instabilidade Média no Paraná (Lévy)")
    ax2.set_xlabel("Tempo (horas)")
    ax2.set_ylabel("Instabilidade média")
    ax2.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig2.savefig("levy_serie_instabilidade_media.png", dpi=160)
    plt.close(fig2)

    # 3) Série temporal num ponto
    serie_ponto = campos[:, j0, i0]
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    ax3.plot(tempos_horas, serie_ponto, linewidth=1.5)
    ax3.set_title(f"Série I(t) no ponto ({j0}, {i0}) — Cauda Longa")
    ax3.set_xlabel("Tempo (horas)")
    ax3.set_ylabel("Instabilidade")
    ax3.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig3.savefig("levy_serie_ponto_extremo.png", dpi=160)
    plt.close(fig3)

    # 4) Cauda empírica log–log
    valores, cauda, probs = analisar_cauda_longa(serie_ponto)
    if cauda is not None:
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        ax4.loglog(cauda, probs[-len(cauda):], "o", markersize=4)
        ax4.set_title("Cauda Empírica da Série (Log–Log)")
        ax4.set_xlabel("x (instabilidade)")
        ax4.set_ylabel("P(I >= x)")
        ax4.grid(True, which="both", linestyle="--", alpha=0.4)
        plt.tight_layout()
        fig4.savefig("levy_cauda_loglog_ponto.png", dpi=160)
        plt.close(fig4)

    # 5) Mapa de risco de tornado
    fig5, ax5 = plt.subplots(figsize=(6, 4))
    im5 = ax5.imshow(risco, origin="lower", vmin=0.0, vmax=1.0)
    ax5.set_title("Mapa de Risco de Tornado (Lévy + Cauda Longa)")
    ax5.set_xlabel("Longitude (índice)")
    ax5.set_ylabel("Latitude (índice)")
    plt.colorbar(im5, ax=ax5, label="Probabilidade de tornado")
    plt.tight_layout()
    fig5.savefig("levy_mapa_risco_tornado.png", dpi=160)
    plt.close(fig5)


# ============================================================
# 6) MAIN
# ============================================================

def main():
    par_espaco = ParametrosEspacoTemporal()
    par_spde = ParametrosSPDE()
    par_levy = ParametrosLevy()
    par_risco = ParametrosRisco()

    # 1) Simular campo de instabilidade com ruído de Lévy
    campos = simular_instabilidade_levy(par_espaco, par_spde, par_levy, seed=123)
    campo_final = campos[-1]

    print("Simulação (Lévy) concluída.")
    print(f"Instabilidade média final: {campo_final.mean():.3f}")
    print(f"Instabilidade máxima final: {campo_final.max():.3f}")

    # 2) Mapa de risco
    risco = mapa_risco_tornado(campo_final, par_risco)
    print(f"Risco médio de tornado: {risco.mean():.3f}")
    print(f"Risco máximo de tornado: {risco.max():.3f}")

    # 3) Gráficos
    gerar_graficos(campos, risco, par_espaco)
    print("Figuras salvas:")
    print("  - levy_instabilidade_final.png")
    print("  - levy_serie_instabilidade_media.png")
    print("  - levy_serie_ponto_extremo.png")
    print("  - levy_cauda_loglog_ponto.png")
    print("  - levy_mapa_risco_tornado.png")


if __name__ == "__main__":
    main()
