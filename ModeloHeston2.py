# ============================================================
# MODELO DE HESTON — VOLATILIDADE ESTOCÁSTICA (
# Autor: Luiz Tiago Wilcke (LT)
# ============================================================
# Sistema de EDEs (sob medida neutra ao risco):
#
#   dS_t = r S_t dt + sqrt(v_t) S_t dW_t^(1)
#   dv_t = kappa (theta - v_t) dt + sigma sqrt(v_t) dW_t^(2)
#
#   com corr(dW_t^(1), dW_t^(2)) = rho
#
# - S_t   : preço do ativo
# - v_t   : variância instantânea
# - r     : taxa livre de risco
# - kappa : velocidade de reversão da variância
# - theta : nível de longo prazo da variância
# - sigma : vol da vol (volatilidade da variância)
# - rho   : correlação entre choques em preço e variância
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Optional


# ------------------------------------------------------------
# 1) Parâmetros do modelo de Heston
# ------------------------------------------------------------
@dataclass
class ParametrosHeston:
    preco_inicial: float = 100.0          # S_0
    variancia_inicial: float = 0.04       # v_0 (ex: 20% de vol² -> 0.2^2 = 0.04)
    taxa_juros: float = 0.05              # r (contínuo, a.a.)
    kappa: float = 2.0                    # velocidade de reversão
    theta: float = 0.04                   # nível de longo prazo da variância
    sigma_vol: float = 0.5                # vol da vol (σ)
    rho: float = -0.7                     # correlação entre choques
    maturidade_anos: float = 1.0          # T em anos
    passos_tempo: int = 252               # N (ex: 252 dias úteis)
    n_caminhos: int = 50_000              # número de trajetórias para Monte Carlo

    # Controle numérico
    variancia_minima: float = 1e-8        # piso numérico para v_t (evita sqrt negativa)


# ------------------------------------------------------------
# 2) Ruído browniano correlacionado (dW1, dW2)
# ------------------------------------------------------------
def gerar_ruidos_correlacionados(
    rho: float,
    passos_tempo: int,
    n_caminhos: int,
    dt: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gera incrementos brownianos correlacionados (dW1, dW2) usando decomposição de Cholesky:
        dW1 = sqrt(dt) * Z1
        dW2 = sqrt(dt) * (rho*Z1 + sqrt(1-rho^2)*Z2)
    """
    z1 = np.random.normal(size=(n_caminhos, passos_tempo))
    z2 = np.random.normal(size=(n_caminhos, passos_tempo))
    dW1 = np.sqrt(dt) * z1
    dW2 = np.sqrt(dt) * (rho * z1 + np.sqrt(1.0 - rho ** 2) * z2)
    return dW1, dW2


# ------------------------------------------------------------
# 3) Simulação do Modelo de Heston (Euler com truncamento total)
# ------------------------------------------------------------
def simular_caminhos_heston(
    par: ParametrosHeston,
    retornar_grade_tempo: bool = False
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Simula trajetórias (S_t, v_t) sob o modelo de Heston
    usando um esquema de Euler com truncamento total na variância.
    Retorna:
        S: matriz (n_caminhos, passos_tempo+1)
        v: matriz (n_caminhos, passos_tempo+1)
        t: vetor de tempos (se solicitado)
    """
    dt = par.maturidade_anos / par.passos_tempo
    nT = par.passos_tempo
    nC = par.n_caminhos

    # Matrizes de preço e variância
    S = np.zeros((nC, nT + 1))
    v = np.zeros((nC, nT + 1))

    # Condições iniciais
    S[:, 0] = par.preco_inicial
    v[:, 0] = par.variancia_inicial

    # Ruídos correlacionados
    dW1, dW2 = gerar_ruidos_correlacionados(par.rho, nT, nC, dt)

    # Loop no tempo
    for t in range(nT):
        v_t = np.maximum(v[:, t], par.variancia_minima)

        # Atualização da variância (CIR-like com truncamento total)
        dv = par.kappa * (par.theta - v_t) * dt + par.sigma_vol * np.sqrt(v_t) * dW2[:, t]
        v[:, t + 1] = np.maximum(v_t + dv, par.variancia_minima)

        # Atualização do preço (sob medida neutra ao risco)
        # dS_t = r S_t dt + sqrt(v_t) S_t dW1
        S_t = S[:, t]
        dS = par.taxa_juros * S_t * dt + np.sqrt(v_t) * S_t * dW1[:, t]
        S[:, t + 1] = S_t + dS

    if retornar_grade_tempo:
        tempos = np.linspace(0.0, par.maturidade_anos, nT + 1)
        return S, v, tempos
    else:
        return S, v, None


# ------------------------------------------------------------
# 4) Precificação de Call Europeia via Monte Carlo
# ------------------------------------------------------------
def precificar_call_europeia_heston_mc(
    par: ParametrosHeston,
    preco_exercicio: float,
    antitetico: bool = True
) -> Tuple[float, float]:
    """
    Precificação de uma opção de compra (call) europeia sob Heston
    usando Monte Carlo padrão (com opção de variáveis antitéticas).

    Retorna:
        preco_estimado, erro_padrao
    """
    if not antitetico:
        # Simples: simula diretamente n_caminhos
        S, _, _ = simular_caminhos_heston(par, retornar_grade_tempo=False)
        ST = S[:, -1]
        payoff = np.maximum(ST - preco_exercicio, 0.0)
        desconto = np.exp(-par.taxa_juros * par.maturidade_anos)
        valor = desconto * np.mean(payoff)
        erro = desconto * np.std(payoff) / np.sqrt(par.n_caminhos)
        return valor, erro
    else:
        # Antitético: gera metade dos caminhos, espelha os ruídos
        metade = par.n_caminhos // 2
        par_half = ParametrosHeston(
            preco_inicial=par.preco_inicial,
            variancia_inicial=par.variancia_inicial,
            taxa_juros=par.taxa_juros,
            kappa=par.kappa,
            theta=par.theta,
            sigma_vol=par.sigma_vol,
            rho=par.rho,
            maturidade_anos=par.maturidade_anos,
            passos_tempo=par.passos_tempo,
            n_caminhos=metade,
            variancia_minima=par.variancia_minima
        )

        # Simula ruidos explicitamente para antitético
        dt = par_half.maturidade_anos / par_half.passos_tempo
        nT = par_half.passos_tempo
        nC = par_half.n_caminhos

        z1 = np.random.normal(size=(nC, nT))
        z2 = np.random.normal(size=(nC, nT))

        # Original
        dW1 = np.sqrt(dt) * z1
        dW2 = np.sqrt(dt) * (par_half.rho * z1 + np.sqrt(1 - par_half.rho ** 2) * z2)

        # Antitético
        dW1_ant = -dW1
        dW2_ant = -dW2

        # Função interna para evoluir S,v dado (dW1,dW2)
        def evoluir(dW1_loc: np.ndarray, dW2_loc: np.ndarray) -> np.ndarray:
            S = np.zeros((nC, nT + 1))
            v = np.zeros((nC, nT + 1))
            S[:, 0] = par_half.preco_inicial
            v[:, 0] = par_half.variancia_inicial

            for t in range(nT):
                v_t = np.maximum(v[:, t], par_half.variancia_minima)
                dv = (
                    par_half.kappa * (par_half.theta - v_t) * dt
                    + par_half.sigma_vol * np.sqrt(v_t) * dW2_loc[:, t]
                )
                v[:, t + 1] = np.maximum(v_t + dv, par_half.variancia_minima)

                S_t = S[:, t]
                dS = par_half.taxa_juros * S_t * dt + np.sqrt(v_t) * S_t * dW1_loc[:, t]
                S[:, t + 1] = S_t + dS

            return S[:, -1]

        ST = evoluir(dW1, dW2)
        ST_ant = evoluir(dW1_ant, dW2_ant)

        payoff = 0.5 * (np.maximum(ST - preco_exercicio, 0.0) +
                        np.maximum(ST_ant - preco_exercicio, 0.0))

        desconto = np.exp(-par_half.taxa_juros * par_half.maturidade_anos)
        valor = desconto * np.mean(payoff)
        erro = desconto * np.std(payoff) / np.sqrt(2 * nC)  # 2*nC amostras efetivas

        return valor, erro


# ------------------------------------------------------------
# 5) Utilidades de visualização
# ------------------------------------------------------------
def plotar_caminhos_heston(
    S: np.ndarray,
    v: np.ndarray,
    tempos: np.ndarray,
    n_exibir: int = 10
) -> None:
    """
    Plota algumas trajetórias de preço e variância.
    """
    n_exibir = min(n_exibir, S.shape[0])

    plt.figure(figsize=(12, 5))

    # Preços
    plt.subplot(1, 2, 1)
    for i in range(n_exibir):
        plt.plot(tempos, S[i, :])
    plt.title("Caminhos de Preço (Heston)")
    plt.xlabel("Tempo (anos)")
    plt.ylabel("Preço S_t")

    # Variâncias
    plt.subplot(1, 2, 2)
    for i in range(n_exibir):
        plt.plot(tempos, v[i, :])
    plt.title("Caminhos de Variância v_t")
    plt.xlabel("Tempo (anos)")
    plt.ylabel("Variância")

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 6) Exemplo de uso
# ------------------------------------------------------------
if __name__ == "__main__":
    # 1) Define parâmetros do modelo
    parametros = ParametrosHeston(
        preco_inicial=100.0,
        variancia_inicial=0.04,   # 20% de vol → 0.2^2
        taxa_juros=0.05,
        kappa=2.0,
        theta=0.04,
        sigma_vol=0.6,
        rho=-0.7,
        maturidade_anos=1.0,
        passos_tempo=252,
        n_caminhos=40_000
    )

    preco_exercicio = 100.0

    # 2) Simula alguns caminhos para visualização
    S_paths, v_paths, t_grid = simular_caminhos_heston(parametros, retornar_grade_tempo=True)
    plotar_caminhos_heston(S_paths, v_paths, t_grid, n_exibir=8)

    # 3) Precifica a call europeia via Monte Carlo
    preco_call, erro_call = precificar_call_europeia_heston_mc(
        parametros,
        preco_exercicio=preco_exercicio,
        antitetico=True
    )

    print("==============================================")
    print(" MODELO DE HESTON — CALL EUROPEIA (MONTE CARLO)")
    print("==============================================")
    print(f"Preço inicial S0        = {parametros.preco_inicial:.4f}")
    print(f"Strike K                = {preco_exercicio:.4f}")
    print(f"Taxa livre de risco r   = {parametros.taxa_juros:.4f}")
    print(f"Vol² inicial v0         = {parametros.variancia_inicial:.4f}")
    print(f"kappa                   = {parametros.kappa:.4f}")
    print(f"theta                   = {parametros.theta:.4f}")
    print(f"sigma_vol (vol da vol)  = {parametros.sigma_vol:.4f}")
    print(f"rho (correlação)        = {parametros.rho:.4f}")
    print("----------------------------------------------")
    print(f"Preço da call (MC)      = {preco_call:.4f}")
    print(f"Erro padrão do MC       = {erro_call:.4f}")
    print("==============================================")
