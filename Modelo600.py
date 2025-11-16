# ============================================================
# MODELO FLUIDO-FINANCEIRO — Burgers + Preço de Ativo
# Autor: Luiz Tiago Wilcke (LT)
# ============================================================
# Ideia:
#   - Campo de "liquidez/pressão de ordens" u(x,t) obedece
#     à equação de Burgers viscosa 1D:
#
#       ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²
#
#   - Parâmetros financeiros (drift μ(t) e volatilidade σ(t))
#     são funções do campo u(x,t).
#   - O preço S_t segue:
#
#       S_{t+Δt} = S_t * exp( (μ_t - 0.5 σ_t²)Δt + σ_t sqrt(Δt) Z_t )
#
#     com Z_t ~ N(0,1).
#
# Saída:
#   - Mapa de calor do campo fluido u(x,t)
#   - Caminhos simulados de preço S_t
#   - Histograma do preço terminal S_T
#   - Série temporal de μ(t) e σ(t)
#   - Preço médio e desvio dos preços, com 6 casas decimais
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple


# ------------------------------------------------------------
# 1) Parâmetros do modelo
# ------------------------------------------------------------

@dataclass
class ParametrosFluido:
    comprimento_espaco: float = 1.0      # domínio em x: [0, L]
    num_pontos_espaco: int = 101         # número de pontos de malha em x
    tempo_final: float = 1.0             # horizonte temporal (anos)
    num_passos_tempo: int = 1000         # número de passos no tempo
    viscosidade: float = 0.01            # ν (controla dissipação do "fluido")

    # tipo de condição inicial: "pulso", "seno" ou "gauss"
    tipo_condicao_inicial: str = "gauss"


@dataclass
class ParametrosFinanceiros:
    preco_inicial: float = 100.0         # S_0
    num_caminhos: int = 8000             # número de trajetórias de preço
    semente: int = 123                   # reprodutibilidade

    # acoplamentos entre fluido e preço
    fator_drift: float = 0.6             # μ(t) ≈ fator_drift * média(u(t))
    fator_volatilidade: float = 1.2      # σ(t) ≈ fator_vol * desvio(u(t))
    volatilidade_minima: float = 0.05    # piso para σ(t)

    taxa_juros_constante: float = 0.03   # r para desconto de payoffs
    strike: float = 100.0                # strike de uma call europeia (exemplo)


# ------------------------------------------------------------
# 2) Construção da malha e condição inicial da PDE
# ------------------------------------------------------------

def construir_malha_fluido(param_fl: ParametrosFluido) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Cria a malha em x e t e devolve:
        x (array), t (array), dx, dt.
    """
    L = param_fl.comprimento_espaco
    Nx = param_fl.num_pontos_espaco
    T = param_fl.tempo_final
    Nt = param_fl.num_passos_tempo

    x = np.linspace(0.0, L, Nx)
    t = np.linspace(0.0, T, Nt + 1)

    dx = x[1] - x[0]
    dt = t[1] - t[0]

    return x, t, dx, dt


def condicao_inicial_u(x: np.ndarray, param_fl: ParametrosFluido) -> np.ndarray:
    """
    Define a condição inicial u(x,0) para o campo fluido.
    """
    if param_fl.tipo_condicao_inicial.lower() == "seno":
        # onda senoidal (fluxo oscilatório de liquidez)
        u0 = 0.8 * np.sin(2.0 * np.pi * x) + 0.5
    elif param_fl.tipo_condicao_inicial.lower() == "pulso":
        # pulso tipo degrau (choque de liquidez em uma região)
        u0 = np.zeros_like(x)
        centro = 0.4
        largura = 0.2
        u0[(x > centro - largura/2) & (x < centro + largura/2)] = 1.5
    else:
        # "gauss": bolha de liquidez concentrada
        centro = 0.5
        largura = 0.1
        u0 = 1.2 * np.exp(-((x - centro) ** 2) / (2 * largura ** 2)) + 0.2

    return u0


# ------------------------------------------------------------
# 3) Solver da equação de Burgers viscosa 1D
# ------------------------------------------------------------

def simular_burgers(param_fl: ParametrosFluido) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resolve a equação de Burgers viscosa:
        ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²
    com condições de contorno periódicas.

    Retorna:
        campo_u: array (Nt+1, Nx)
        x: malha espacial
        t: malha temporal
    """
    x, t, dx, dt = construir_malha_fluido(param_fl)
    Nx = len(x)
    Nt = len(t) - 1
    nu = param_fl.viscosidade

    # Checagem simples de estabilidade (CFL aproximado)
    # O termo convectivo exige dt <= dx / max|u|,
    # o difusivo exige dt <= dx^2 / (2ν).
    # Aqui usamos uma condição razoavelmente segura.
    u0 = condicao_inicial_u(x, param_fl)
    max_u0 = np.max(np.abs(u0))
    if max_u0 == 0:
        max_u0 = 1.0

    dt_conv_max = dx / max_u0
    dt_diff_max = dx * dx / (2.0 * nu) if nu > 0 else dt_conv_max

    dt_limite = min(dt_conv_max, dt_diff_max)
    if dt > dt_limite:
        print("AVISO: passo de tempo dt pode ser grande demais para estabilidade.")
        print(f"dt = {dt:.6f}, dt_limite ≈ {dt_limite:.6f}")

    # Campo u(x,t)
    campo_u = np.zeros((Nt + 1, Nx))
    campo_u[0, :] = u0

    # Loop no tempo
    for n in range(Nt):
        u = campo_u[n, :]

        # Derivadas espaciais com condições periódicas via np.roll
        u_x = (np.roll(u, -1) - np.roll(u, 1)) / (2.0 * dx)
        u_xx = (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / (dx * dx)

        # Equação de Burgers explícita
        du_dt = -u * u_x + nu * u_xx
        campo_u[n + 1, :] = u + dt * du_dt

    return campo_u, x, t


# ------------------------------------------------------------
# 4) Acoplamento fluido -> finanças (μ(t), σ(t))
# ------------------------------------------------------------

def gerar_parametros_dinamicos_financeiros(
    campo_u: np.ndarray,
    param_fl: ParametrosFluido,
    param_fin: ParametrosFinanceiros
) -> Tuple[np.ndarray, np.ndarray]:
    """
    A partir do campo fluido u(x,t), gera:
        - serie_mu_t: drift μ(t) ao longo do tempo
        - serie_sigma_t: volatilidade σ(t) ao longo do tempo
    Usamos média e desvio padrão espacial de u(x,t).
    """
    # campo_u tem shape (Nt+1, Nx)
    media_u = np.mean(campo_u, axis=1)           # (Nt+1,)
    desvio_u = np.std(campo_u, axis=1)           # (Nt+1,)

    # Drift μ(t): proporcional à média do "fluxo de ordens"
    serie_mu_t = param_fin.fator_drift * media_u  # ex.: 0.6 * média(u)

    # Volatilidade σ(t): proporcional ao desvio do "campo de liquidez"
    serie_sigma_t = param_fin.fator_volatilidade * desvio_u
    serie_sigma_t = np.maximum(serie_sigma_t, param_fin.volatilidade_minima)

    return serie_mu_t, serie_sigma_t


# ------------------------------------------------------------
# 5) Simulação estocástica dos preços acoplados ao fluido
# ------------------------------------------------------------

def simular_precos(
    param_fl: ParametrosFluido,
    param_fin: ParametrosFinanceiros,
    campo_u: np.ndarray,
    t: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simula caminhos de preço S_t usando μ(t) e σ(t) acoplados ao campo fluido.
    Retorna:
        caminhos_precos: array (num_caminhos, Nt+1)
        serie_mu_t: (Nt+1,)
        serie_sigma_t: (Nt+1,)
    """
    dt = t[1] - t[0]
    Nt = len(t) - 1
    num_caminhos = param_fin.num_caminhos

    serie_mu_t, serie_sigma_t = gerar_parametros_dinamicos_financeiros(
        campo_u, param_fl, param_fin
    )

    rng = np.random.default_rng(param_fin.semente)

    # Matriz de caminhos: cada linha é uma trajetória S_t
    caminhos_precos = np.zeros((num_caminhos, Nt + 1))
    caminhos_precos[:, 0] = param_fin.preco_inicial

    # Simulação passo a passo
    for n in range(Nt):
        S_t = caminhos_precos[:, n]
        mu_t = serie_mu_t[n]
        sigma_t = serie_sigma_t[n]

        # Ruído gaussiano
        Z = rng.normal(size=num_caminhos)

        # Dinâmica tipo GBM acoplada ao fluido
        incremento = (mu_t - 0.5 * sigma_t ** 2) * dt + sigma_t * np.sqrt(dt) * Z
        caminhos_precos[:, n + 1] = S_t * np.exp(incremento)

    return caminhos_precos, serie_mu_t, serie_sigma_t


# ------------------------------------------------------------
# 6) Métricas e resultados numéricos
# ------------------------------------------------------------

def calcular_metricas_precos(
    caminhos_precos: np.ndarray,
    param_fin: ParametrosFinanceiros,
    t: np.ndarray
) -> None:
    """
    Calcula métricas básicas do preço terminal e da call europeia.
    Imprime com 6 casas decimais.
    """
    T = t[-1]
    S_T = caminhos_precos[:, -1]
    media_S_T = float(np.mean(S_T))
    desvio_S_T = float(np.std(S_T, ddof=1))

    # Call europeia simples com desconto a taxa constante r
    payoff_call = np.maximum(S_T - param_fin.strike, 0.0)
    desconto = np.exp(-param_fin.taxa_juros_constante * T)
    preco_call = float(np.mean(payoff_call) * desconto)
    desvio_call = float(np.std(payoff_call * desconto, ddof=1))

    print("===================================================")
    print("Resultados numéricos (Monte Carlo acoplado ao fluido)")
    print("===================================================")
    print(f"Média de S_T           : {media_S_T:.6f}")
    print(f"Desvio padrão de S_T   : {desvio_S_T:.6f}")
    print(f"Preço da CALL (K={param_fin.strike:.2f}): {preco_call:.6f}")
    print(f"Desvio da CALL         : {desvio_call:.6f}")


# ------------------------------------------------------------
# 7) Gráficos
# ------------------------------------------------------------

def plotar_mapa_fluido(campo_u: np.ndarray, x: np.ndarray, t: np.ndarray):
    """
    Mapa de calor do campo u(x,t) ao longo do tempo.
    """
    plt.figure(figsize=(8, 4))
    # imshow: eixo x = posição, eixo y = tempo
    extent = [x[0], x[-1], t[0], t[-1]]
    plt.imshow(
        campo_u,
        extent=extent,
        origin="lower",
        aspect="auto"
    )
    plt.colorbar(label="u(x,t) — campo de liquidez/fluxo")
    plt.xlabel("Posição no 'espaço de liquidez' x")
    plt.ylabel("Tempo (anos)")
    plt.title("Campo fluido u(x,t) — Dinâmica dos 'fluxos de ordens'")


def plotar_caminhos_precos(caminhos_precos: np.ndarray, t: np.ndarray, num_plotar: int = 20):
    """
    Plota alguns caminhos de preço S_t.
    """
    plt.figure(figsize=(8, 4))
    num_plotar = min(num_plotar, caminhos_precos.shape[0])
    for i in range(num_plotar):
        plt.plot(t, caminhos_precos[i, :], alpha=0.7)
    plt.xlabel("Tempo (anos)")
    plt.ylabel("Preço S_t")
    plt.title(f"{num_plotar} caminhos simulados de preço S_t (acoplados ao fluido)")
    plt.grid(True)


def plotar_histograma_precos_terminais(caminhos_precos: np.ndarray):
    """
    Histograma do preço terminal S_T.
    """
    S_T = caminhos_precos[:, -1]
    plt.figure(figsize=(6, 4))
    plt.hist(S_T, bins=50, density=True)
    plt.xlabel("Preço terminal S_T")
    plt.ylabel("Densidade")
    plt.title("Distribuição simulada de S_T")
    plt.grid(True)


def plotar_mu_sigma(serie_mu_t: np.ndarray, serie_sigma_t: np.ndarray, t: np.ndarray):
    """
    Plota μ(t) e σ(t) ao longo do tempo.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(t, serie_mu_t, label="Drift μ(t)")
    plt.plot(t, serie_sigma_t, label="Volatilidade σ(t)")
    plt.xlabel("Tempo (anos)")
    plt.ylabel("Valor")
    plt.title("Parâmetros financeiros dinamicamente acoplados ao fluido")
    plt.grid(True)
    plt.legend()


# ------------------------------------------------------------
# 8) Execução completa do modelo
# ------------------------------------------------------------

def executar_modelo():
    # Parâmetros do fluido (PDE)
    param_fl = ParametrosFluido(
        comprimento_espaco=1.0,
        num_pontos_espaco=151,
        tempo_final=1.0,
        num_passos_tempo=1200,
        viscosidade=0.01,
        tipo_condicao_inicial="gauss"   # "gauss", "seno" ou "pulso"
    )

    # Parâmetros financeiros
    param_fin = ParametrosFinanceiros(
        preco_inicial=100.0,
        num_caminhos=8000,
        semente=42,
        fator_drift=0.6,
        fator_volatilidade=1.2,
        volatilidade_minima=0.05,
        taxa_juros_constante=0.03,
        strike=100.0
    )

    print("===============================================")
    print("Simulação do modelo fluido-financeiro completo")
    print("Autor: Luiz Tiago Wilcke (LT)")
    print("===============================================")

    # 1) Resolver PDE de Burgers (campo de liquidez)
    campo_u, x, t = simular_burgers(param_fl)

    # 2) Simular preços acoplados
    caminhos_precos, serie_mu_t, serie_sigma_t = simular_precos(
        param_fl, param_fin, campo_u, t
    )

    # 3) Resultados numéricos
    calcular_metricas_precos(caminhos_precos, param_fin, t)

    # 4) Gráficos
    plotar_mapa_fluido(campo_u, x, t)
    plotar_caminhos_precos(caminhos_precos, t, num_plotar=25)
    plotar_histograma_precos_terminais(caminhos_precos)
    plotar_mu_sigma(serie_mu_t, serie_sigma_t, t)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    executar_modelo()
