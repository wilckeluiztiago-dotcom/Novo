# ============================================================
# Fokker–Planck para Bolsa Brasileira (B3)
# Autor: Luiz Tiago Wilcke 
# ============================================================
#
# EDE em log-preço:
#   dX_t = mu * dt + sigma * dW_t
#
# Equação de Fokker–Planck associada:
#   ∂p/∂t = - ∂/∂x [ a(x,t) p ] + 1/2 ∂²/∂x² [ b(x,t)² p ]
#
# Para a(x,t) = mu (constante) e b(x,t) = sigma (constante):
#   ∂p/∂t = - mu * ∂p/∂x + (sigma²/2) * ∂²p/∂x²
#
# Aqui:
#   - X_t = log(preço_t)
#   - A solução analítica é Normal:
#       X_t ~ Normal( x0 + mu * t , sigma² * t )
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# ------------------------------------------------------------
# 1) Configurações do mercado e da simulação
# ------------------------------------------------------------
@dataclass
class ConfiguracaoMercado:
    nome_ativo: str = "PETR4"
    indice_referencia: str = "IBOVESPA"
    moeda: str = "BRL"

    preco_inicial: float = 35.0     # preço inicial em BRL
    taxa_drift_log: float = 0.12    # drift do log-preço (ao ano)
    volatilidade_log: float = 0.35  # volatilidade do log-preço (ao ano)

    horizonte_dias: int = 60        # horizonte em dias úteis de bolsa
    passos_tempo: int = 120         # número de passos de tempo na malha

    numero_caminhos: int = 20000    # número de caminhos Monte Carlo
    semente: int = 42               # semente para reprodutibilidade

    # Grade espacial para a Fokker–Planck (em log-preço)
    multiplicador_desvio: float = 4.0   # quantos sigmas em torno de x0
    pontos_espaco: int = 201           # número de pontos da malha em x


# ------------------------------------------------------------
# 2) Utilitários matemáticos
# ------------------------------------------------------------
def densidade_normal(x: np.ndarray, media: float, variancia: float) -> np.ndarray:
    """Densidade Normal unidimensional."""
    desvio = np.sqrt(variancia)
    coef = 1.0 / (np.sqrt(2.0 * np.pi) * desvio)
    expo = -0.5 * ((x - media) / desvio) ** 2
    return coef * np.exp(expo)


# ------------------------------------------------------------
# 3) Simulação Monte Carlo da EDE em log-preço
# ------------------------------------------------------------
def simular_caminhos_log_preco(cfg: ConfiguracaoMercado):
    """
    Simula caminhos para o log-preço X_t usando:
        dX_t = mu dt + sigma dW_t
    e retorna tempos_em_dias, caminhos_preco, caminhos_log.
    """
    np.random.seed(cfg.semente)

    # Tempo em anos para o horizonte total
    horizonte_anos = cfg.horizonte_dias / 252.0
    n_passos = cfg.passos_tempo + 1
    dt_anos = horizonte_anos / (n_passos - 1)

    tempos_em_dias = np.linspace(0.0, cfg.horizonte_dias, n_passos)

    x0 = np.log(cfg.preco_inicial)
    caminhos_log = np.zeros((cfg.numero_caminhos, n_passos))
    caminhos_log[:, 0] = x0

    for k in range(n_passos - 1):
        ruido = np.random.normal(loc=0.0, scale=1.0, size=cfg.numero_caminhos)
        caminhos_log[:, k + 1] = (
            caminhos_log[:, k]
            + cfg.taxa_drift_log * dt_anos
            + cfg.volatilidade_log * np.sqrt(dt_anos) * ruido
        )

    caminhos_preco = np.exp(caminhos_log)
    return tempos_em_dias, caminhos_preco, caminhos_log


# ------------------------------------------------------------
# 4) Montagem da grade espacial (log-preço)
# ------------------------------------------------------------
def construir_grade_espacial(cfg: ConfiguracaoMercado, horizonte_anos: float):
    """
    Constrói grade em log-preço em torno de x0 com largura baseada em sigma*sqrt(T).
    """
    x0 = np.log(cfg.preco_inicial)
    largura = cfg.multiplicador_desvio * cfg.volatilidade_log * np.sqrt(horizonte_anos)
    x_min = x0 - largura
    x_max = x0 + largura

    grade_x = np.linspace(x_min, x_max, cfg.pontos_espaco)
    return grade_x


# ------------------------------------------------------------
# 5) Operador de Fokker–Planck (matriz L)
# ------------------------------------------------------------
def construir_operador_fokker_planck(
    grade_x: np.ndarray, mu: float, sigma: float
) -> np.ndarray:
    """
    Constrói a matriz L tal que:
        L p ≈ -mu * ∂p/∂x + (sigma²/2) * ∂²p/∂x²
    usando diferenças finitas centradas (interior) e
    condições de fronteira de derivada nula (reflexivas).
    """
    n = len(grade_x)
    dx = grade_x[1] - grade_x[0]
    L = np.zeros((n, n), dtype=float)

    # Coeficientes
    difusao = 0.5 * sigma ** 2

    # Pontos interiores
    for i in range(1, n - 1):
        L[i, i - 1] = (mu / (2.0 * dx)) + (difusao / dx ** 2)
        L[i, i] = (-2.0 * difusao / dx ** 2)
        L[i, i + 1] = (-mu / (2.0 * dx)) + (difusao / dx ** 2)

    # Fronteiras com derivada nula dp/dx = 0
    # Aproximação: usa apenas termo de difusão (drift com derivada zero)
    # Esquerda (i = 0)
    L[0, 0] = -difusao / dx ** 2 * 2.0
    L[0, 1] = difusao / dx ** 2 * 2.0

    # Direita (i = n-1)
    L[n - 1, n - 2] = difusao / dx ** 2 * 2.0
    L[n - 1, n - 1] = -difusao / dx ** 2 * 2.0

    return L


# ------------------------------------------------------------
# 6) Resolução da Fokker–Planck via Crank–Nicolson
# ------------------------------------------------------------
def resolver_fokker_planck(cfg: ConfiguracaoMercado):
    """
    Resolve a Fokker–Planck usando esquema de Crank–Nicolson:
        (I - dt/2 L) p^{n+1} = (I + dt/2 L) p^n
    Retorna:
        tempos_anos, grade_x, matriz_densidades
    """
    horizonte_anos = cfg.horizonte_dias / 252.0
    n_passos = cfg.passos_tempo + 1
    dt = horizonte_anos / (n_passos - 1)

    grade_x = construir_grade_espacial(cfg, horizonte_anos)
    n_x = len(grade_x)

    L = construir_operador_fokker_planck(
        grade_x=grade_x, mu=cfg.taxa_drift_log, sigma=cfg.volatilidade_log
    )

    I = np.eye(n_x)
    A = I - 0.5 * dt * L
    B = I + 0.5 * dt * L

    # Condição inicial: densidade aproximadamente delta em x0 (Normal com variância pequena)
    x0 = np.log(cfg.preco_inicial)
    variancia_inicial = (0.01) ** 2
    densidade_inicial = densidade_normal(grade_x, x0, variancia_inicial)

    dx = grade_x[1] - grade_x[0]
    densidade_inicial /= np.sum(densidade_inicial) * dx  # normalização

    matriz_densidades = np.zeros((n_passos, n_x))
    matriz_densidades[0, :] = densidade_inicial

    from numpy.linalg import solve

    for n in range(n_passos - 1):
        rhs = B @ matriz_densidades[n, :]
        prox = solve(A, rhs)

        # Impõe não negatividade numérica e renormaliza
        prox = np.maximum(prox, 0.0)
        massa = np.sum(prox) * dx
        if massa > 0:
            prox /= massa

        matriz_densidades[n + 1, :] = prox

    tempos_anos = np.linspace(0.0, horizonte_anos, n_passos)
    return tempos_anos, grade_x, matriz_densidades


# ------------------------------------------------------------
# 7) Estatísticas a partir da densidade de Fokker–Planck
# ------------------------------------------------------------
def estatisticas_a_partir_da_densidade(grade_x, densidade):
    """
    Calcula média e variância de X e de S = exp(X) a partir da densidade p(x).
    """
    dx = grade_x[1] - grade_x[0]

    media_x = np.sum(grade_x * densidade) * dx
    media_x2 = np.sum((grade_x ** 2) * densidade) * dx
    variancia_x = media_x2 - media_x ** 2

    valores_s = np.exp(grade_x)
    media_s = np.sum(valores_s * densidade) * dx
    media_s2 = np.sum((valores_s ** 2) * densidade) * dx
    variancia_s = media_s2 - media_s ** 2

    return media_x, variancia_x, media_s, variancia_s


# ------------------------------------------------------------
# 8) Geração de gráficos
# ------------------------------------------------------------
def plotar_caminhos(cfg, tempos_em_dias, caminhos_preco):
    """
    Plota alguns caminhos simulados de preço.
    """
    plt.figure(figsize=(10, 5))
    n_mostrar = min(30, cfg.numero_caminhos)
    for i in range(n_mostrar):
        plt.plot(tempos_em_dias, caminhos_preco[i, :], alpha=0.5)

    plt.title(f"Caminhos simulados — {cfg.nome_ativo} ({cfg.indice_referencia})")
    plt.xlabel("Tempo (dias úteis)")
    plt.ylabel(f"Preço ({cfg.moeda})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def plotar_mapa_calor_densidade(tempos_anos, grade_x, matriz_densidades):
    """
    Mapa de calor da densidade p(x,t) ao longo do tempo.
    """
    plt.figure(figsize=(10, 5))

    # Converte tempo em dias para eixo
    tempos_dias = tempos_anos * 252.0

    # Imagem: eixo x = log-preço, eixo y = tempo em dias
    # Usamos origin='lower' para tempo crescente para cima
    plt.imshow(
        matriz_densidades,
        aspect="auto",
        origin="lower",
        extent=[
            grade_x[0],
            grade_x[-1],
            tempos_dias[0],
            tempos_dias[-1],
        ],
    )
    plt.colorbar(label="Densidade p(x, t)")
    plt.xlabel("Log-preço (x)")
    plt.ylabel("Tempo (dias úteis)")
    plt.title("Fokker–Planck — Evolução da densidade de log-preço")
    plt.tight_layout()


def plotar_comparacao_densidades(
    cfg,
    tempos_anos,
    grade_x,
    matriz_densidades,
    tempos_em_dias_mc,
    caminhos_log,
    dia_alvo: float,
):
    """
    Compara densidade:
        - Fokker–Planck numérica
        - Monte Carlo (histograma)
        - Analítica Normal
    em um tempo alvo (dia_alvo).
    """
    horizonte_anos = cfg.horizonte_dias / 252.0

    # Índice de tempo alvo na malha de Fokker-Planck
    tempos_dias_fp = tempos_anos * 252.0
    indice_fp = np.argmin(np.abs(tempos_dias_fp - dia_alvo))

    dens_fp = matriz_densidades[indice_fp, :]

    # Monte Carlo: escolhe mesmo índice aproximado de tempo
    indice_mc = np.argmin(np.abs(tempos_em_dias_mc - dia_alvo))
    amostra_log = caminhos_log[:, indice_mc]

    # Analítica: X_t ~ Normal(x0 + mu t, sigma² t)
    x0 = np.log(cfg.preco_inicial)
    t_anos = tempos_anos[indice_fp]
    media_anal = x0 + cfg.taxa_drift_log * t_anos
    variancia_anal = (cfg.volatilidade_log ** 2) * t_anos
    dens_analitica = densidade_normal(grade_x, media_anal, variancia_anal)

    # Histograma Monte Carlo (densidade)
    contagens, bordas = np.histogram(
        amostra_log, bins=60, density=True
    )
    centros = 0.5 * (bordas[:-1] + bordas[1:])

    plt.figure(figsize=(10, 5))
    plt.plot(
        grade_x,
        dens_fp,
        label="Fokker–Planck (numérica)",
        linewidth=2,
    )
    plt.plot(
        grade_x,
        dens_analitica,
        label="Solução analítica Normal",
        linestyle="--",
        linewidth=2,
    )
    plt.step(
        centros,
        contagens,
        where="mid",
        label="Monte Carlo (histograma)",
        alpha=0.6,
    )

    plt.xlabel("Log-preço (x)")
    plt.ylabel("Densidade")
    plt.title(f"Densidades no dia {dia_alvo:.1f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


# ------------------------------------------------------------
# 9) Rotina principal — executa tudo e imprime resultados
# ------------------------------------------------------------
def executar_experimento():
    cfg = ConfiguracaoMercado()

    print("====================================================")
    print("  FOKKER–PLANCK / EDE — MERCADO BRASILEIRO (B3)")
    print("  Ativo:       ", cfg.nome_ativo)
    print("  Índice ref.: ", cfg.indice_referencia)
    print("  Moeda:       ", cfg.moeda)
    print("====================================================")

    # ---------------- Simulação de caminhos ----------------
    tempos_em_dias, caminhos_preco, caminhos_log = simular_caminhos_log_preco(cfg)

    # ---------------- Fokker–Planck ----------------
    tempos_anos_fp, grade_x, matriz_densidades = resolver_fokker_planck(cfg)

    # ---------------- Estatísticas finais ----------------
    horizonte_anos = cfg.horizonte_dias / 252.0
    x0 = np.log(cfg.preco_inicial)

    # Teórica (log-preço)
    media_x_teor = x0 + cfg.taxa_drift_log * horizonte_anos
    variancia_x_teor = (cfg.volatilidade_log ** 2) * horizonte_anos

    # Teórica (preço) — lognormal
    media_s_teor = np.exp(media_x_teor + 0.5 * variancia_x_teor)
    variancia_s_teor = (np.exp(variancia_x_teor) - 1.0) * np.exp(
        2.0 * media_x_teor + variancia_x_teor
    )

    # Monte Carlo (preço)
    precos_finais_mc = caminhos_preco[:, -1]
    media_s_mc = np.mean(precos_finais_mc)
    variancia_s_mc = np.var(precos_finais_mc)
    desvio_s_mc = np.std(precos_finais_mc)
    quantis_mc = np.quantile(precos_finais_mc, [0.025, 0.5, 0.975])

    # Fokker–Planck (preço) a partir da densidade no tempo final
    dens_final_fp = matriz_densidades[-1, :]
    media_x_fp, variancia_x_fp, media_s_fp, variancia_s_fp = (
        estatisticas_a_partir_da_densidade(grade_x, dens_final_fp)
    )
    desvio_s_fp = np.sqrt(variancia_s_fp)

    print("\n==================== RESULTADOS NUMÉRICOS ====================")
    print("Parâmetros (log-preço):")
    print(f"  Drift anual (mu)     = {cfg.taxa_drift_log:.6f}")
    print(f"  Volatilidade anual   = {cfg.volatilidade_log:.6f}")
    print(f"  Horizonte (dias)     = {cfg.horizonte_dias:d}")
    print(f"  Horizonte (anos)     = {horizonte_anos:.6f}")
    print(f"  Preço inicial (BRL)  = {cfg.preco_inicial:.6f}")

    print("\nTeoria (X_t = log(S_t)) — Normal:")
    print(f"  E[X_T] (teórica)     = {media_x_teor:.6f}")
    print(f"  Var[X_T] (teórica)   = {variancia_x_teor:.6f}")

    print("\nPreço S_T — Teoria (Lognormal):")
    print(f"  E[S_T] (teórica)     = {media_s_teor:.6f}")
    print(f"  Var[S_T] (teórica)   = {variancia_s_teor:.6f}")
    print(f"  Desvio(S_T) (teórico)= {np.sqrt(variancia_s_teor):.6f}")

    print("\nPreço S_T — Monte Carlo (EDE):")
    print(f"  E[S_T] (MC)          = {media_s_mc:.6f}")
    print(f"  Var[S_T] (MC)        = {variancia_s_mc:.6f}")
    print(f"  Desvio(S_T) (MC)     = {desvio_s_mc:.6f}")
    print(
        f"  Quantis 2.5% / 50% / 97.5% = "
        f"{quantis_mc[0]:.6f} / {quantis_mc[1]:.6f} / {quantis_mc[2]:.6f}"
    )

    print("\nPreço S_T — Fokker–Planck (densidade numérica):")
    print(f"  E[S_T] (FP)          = {media_s_fp:.6f}")
    print(f"  Var[S_T] (FP)        = {variancia_s_fp:.6f}")
    print(f"  Desvio(S_T) (FP)     = {desvio_s_fp:.6f}")

    print("\nX_T — Fokker–Planck (para conferência):")
    print(f"  E[X_T] (FP)          = {media_x_fp:.6f}")
    print(f"  Var[X_T] (FP)        = {variancia_x_fp:.6f}")

    print("====================================================")
    print("Obs.: As três abordagens (Teoria, EDE, Fokker–Planck)")
    print("      devem ficar próximas se a malha estiver bem resolvida.")
    print("====================================================\n")

    # ---------------- Gráficos ----------------
    # 1) Caminhos de preço
    plotar_caminhos(cfg, tempos_em_dias, caminhos_preco)

    # 2) Mapa de calor da densidade p(x,t)
    plotar_mapa_calor_densidade(tempos_anos_fp, grade_x, matriz_densidades)

    # 3) Comparação de densidades em um tempo intermediário
    dia_alvo = cfg.horizonte_dias / 2.0
    plotar_comparacao_densidades(
        cfg,
        tempos_anos_fp,
        grade_x,
        matriz_densidades,
        tempos_em_dias,
        caminhos_log,
        dia_alvo=dia_alvo,
    )

    plt.show()


# ------------------------------------------------------------
# 10) Execução
# ------------------------------------------------------------
if __name__ == "__main__":
    executar_experimento()
