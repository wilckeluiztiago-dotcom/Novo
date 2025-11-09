# ============================================================
# PDE de Vasicek (juros estocásticos) — Crank–Nicolson + Validação Analítica
# Autor: Luiz Tiago Wilcke (LT)
# Descrição:
#   - Resolve a PDE de Vasicek para zero-coupon bond com diferenças finitas (CN).
#   - Usa condições de contorno a partir da solução analítica de Vasicek (estável).
#   - Compara com a solução fechada, reporta erros e plota curvas Preço × r.
# Dependências: numpy, matplotlib
# ============================================================

import math
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Dict

# ---------------------------
# 1) Parâmetros do modelo
# ---------------------------
@dataclass
class VasicekParams:
    a: float = 0.8      # velocidade de reversão (kappa)
    b: float = 0.05     # nível de longo prazo (theta)
    sigma: float = 0.02 # volatilidade dos juros

# -------------------------------------------
# 2) Fórmulas analíticas (A(τ), B(τ), P(τ))
# -------------------------------------------
def B_vasicek(tau: float, a: float) -> float:
    """B(τ) = (1 - e^{-a τ})/a."""
    if a == 0:
        return tau
    return (1.0 - math.exp(-a * tau)) / a

def A_vasicek(tau: float, a: float, b: float, sigma: float) -> float:
    """
    A(τ) = (b - σ²/(2a²)) * (B(τ) - τ) - (σ²/(4a)) * B(τ)²
    (Forma padrão; cheque sinais ao comparar com referências.)
    """
    B = B_vasicek(tau, a)
    term1 = (b - (sigma**2) / (2.0 * a**2)) * (B - tau)
    term2 = (sigma**2) * (B**2) / (4.0 * a)
    return term1 - term2

def preco_analitico(r: np.ndarray, tau: float, p: VasicekParams) -> np.ndarray:
    """Preço analítico do zero-coupon: P = exp( A(τ) - B(τ)*r )."""
    A = A_vasicek(tau, p.a, p.b, p.sigma)
    B = B_vasicek(tau, p.a)
    return np.exp(A - B * r)

# ------------------------------------------------
# 3) Solver tridiagonal (Thomas) para CN 1D
# ------------------------------------------------
def solve_tridiagonal(a_sub: np.ndarray, a_diag: np.ndarray, a_sup: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    Resolve Ax=d para matriz tridiagonal A.
    a_sub: subdiagonal (len n-1)
    a_diag: diagonal (len n)
    a_sup: superdiagonal (len n-1)
    d: vetor RHS (len n)
    """
    n = len(a_diag)
    c_prime = np.zeros(n-1)
    d_prime = np.zeros(n)

    # Forward
    c_prime[0] = a_sup[0] / a_diag[0]
    d_prime[0] = d[0] / a_diag[0]
    for i in range(1, n-1):
        denom = a_diag[i] - a_sub[i-1] * c_prime[i-1]
        c_prime[i] = a_sup[i] / denom
        d_prime[i] = (d[i] - a_sub[i-1] * d_prime[i-1]) / denom
    denom = a_diag[n-1] - a_sub[n-2] * c_prime[n-2]
    d_prime[n-1] = (d[n-1] - a_sub[n-2] * d_prime[n-2]) / denom

    # Backward
    x = np.zeros(n)
    x[-1] = d_prime[-1]
    for i in reversed(range(0, n-1)):
        x[i] = d_prime[i] - c_prime[i] * x[i+1]
    return x

# ---------------------------------------------------------------------
# 4) Discretização da PDE de Vasicek (Crank–Nicolson, backward em t)
# ---------------------------------------------------------------------
def vasicek_pde_crank_nicolson(
    p: VasicekParams,
    T: float,
    r_min: float = -0.05,
    r_max: float = 0.20,
    Nr: int = 400,
    Nt: int = 800
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resolve a PDE para um título que paga 1 em T (zero-coupon).
    Retorna:
      - r_grid: vetor de taxas
      - t_grid: vetor de tempos (0..T)
      - V_0: solução no tempo t=0 (preço para cada r em r_grid)
    """
    # Malhas
    r_grid = np.linspace(r_min, r_max, Nr)
    dr = r_grid[1] - r_grid[0]
    t_grid = np.linspace(0.0, T, Nt+1)  # t_0=0 ... t_N=T
    dt = t_grid[1] - t_grid[0]

    # Terminal: no tempo T, preço = 1 para todo r
    V_next = np.ones(Nr)

    # Pré-compute coeficientes espaciais (dependem de r_i)
    # PDE: V_t + mu(r)*V_r + 0.5*sigma^2 * V_rr - r * V = 0
    sigma2 = p.sigma**2

    # Matrizes tridiagonais para CN:
    # (I - dt/2 * L) V^n = (I + dt/2 * L) V^{n+1}
    # Onde L opera em r: alpha_i * V_{i-1} + beta_i * V_i + gamma_i * V_{i+1}
    # com:
    # alpha_i = 0.5*(sigma^2/dr^2 - mu_i/dr)
    # beta_i  = - (sigma^2/dr^2) - r_i
    # gamma_i = 0.5*(sigma^2/dr^2 + mu_i/dr)
    # (em cada passo, r no termo -r*V é avaliado no tempo corrente — constante no CN)

    for n in reversed(range(Nt)):  # de T-dt até 0
        t_n = t_grid[n]
        tau = T - t_n  # tempo até o vencimento (para contorno analítico)

        mu = p.a * (p.b - r_grid)

        alpha = 0.5 * (sigma2 / dr**2 - mu / dr)
        beta  = - (sigma2 / dr**2) - r_grid
        gamma = 0.5 * (sigma2 / dr**2 + mu / dr)

        # LHS: I - dt/2 * L  => tridiagonal (a_sub_L, a_diag_L, a_sup_L)
        a_sub_L = - (dt/2.0) * alpha[1:]               # abaixo da diagonal
        a_diag_L = 1.0 - (dt/2.0) * beta               # diagonal
        a_sup_L = - (dt/2.0) * gamma[:-1]              # acima da diagonal

        # RHS: (I + dt/2 * L) V_next  => também tridiagonal atuando em V_next
        b = np.empty_like(V_next)
        # contribuição central
        b[:] = (1.0 + (dt/2.0) * beta) * V_next
        # vizinhos
        b[1:] += (dt/2.0) * alpha[1:] * V_next[:-1]
        b[:-1] += (dt/2.0) * gamma[:-1] * V_next[1:]

        # Condições de contorno por solução analítica:
        # Em r_min e r_max (nós de bordo), impomos o valor fechado P_ana.
        P_left  = preco_analitico(np.array([r_grid[0]]), tau, p)[0]
        P_right = preco_analitico(np.array([r_grid[-1]]), tau, p)[0]

        # Ajuste RHS para condições de contorno (CN padrão para Dirichlet)
        b[0]  = P_left
        b[-1] = P_right

        # Impor Dirichlet nas linhas da LHS
        a_diag_L[0] = 1.0
        a_sup_L[0] = 0.0
        if len(a_sub_L) > 0:
            a_sub_L[0] = 0.0
        a_diag_L[-1] = 1.0
        if len(a_sup_L) > 0:
            a_sup_L[-1] = 0.0
        if len(a_sub_L) > 0:
            a_sub_L[-1] = 0.0

        # Resolver sistema tridiagonal
        V_curr = solve_tridiagonal(a_sub_L, a_diag_L, a_sup_L, b)

        V_next = V_curr  # para o próximo passo (mais atrás no tempo)

    # Solução em t=0
    return r_grid, t_grid, V_next

# --------------------------------------------------------------------
# 5) Experimento: múltiplos vencimentos, erros vs. analítico, gráficos
# --------------------------------------------------------------------
def rodar_experimento():
    params = VasicekParams(a=0.8, b=0.05, sigma=0.02)

    # Conjunto de vencimentos (anos)
    vencimentos = [0.5, 1.0, 2.0, 5.0, 10.0]

    # Malha r (mesma para todos os T)
    r_min, r_max, Nr = -0.02, 0.18, 350
    r_grid_comum = np.linspace(r_min, r_max, Nr)

    resultados: Dict[float, Dict[str, np.ndarray]] = {}

    print("=== Rodando PDE de Vasicek (Crank–Nicolson) ===")
    for T in vencimentos:
        r_grid, t_grid, V0 = vasicek_pde_crank_nicolson(
            p=params, T=T,
            r_min=r_min, r_max=r_max,
            Nr=Nr, Nt=800
        )

        assert np.allclose(r_grid, r_grid_comum)
        # Solução analítica para comparação no tempo t=0 (tau = T)
        P_ana = preco_analitico(r_grid, T, params)

        # Métricas de erro
        erro = V0 - P_ana
        mae = float(np.mean(np.abs(erro)))
        maxe = float(np.max(np.abs(erro)))
        medae = float(np.median(np.abs(erro)))

        resultados[T] = {
            "r": r_grid,
            "V_pde": V0,
            "V_ana": P_ana,
            "erro": erro,
            "MAE": mae,
            "MAXE": maxe,
            "MEDAE": medae
        }
        print(f"T = {T:>4.1f}y | MAE = {mae:.3e} | MAXE = {maxe:.3e} | MEDAE = {medae:.3e}")

    # ---------------------------
    # Plots: Preço × r por T
    # (um gráfico por figura)
    # ---------------------------
    for T in vencimentos:
        r = resultados[T]["r"]
        Vp = resultados[T]["V_pde"]
        Va = resultados[T]["V_ana"]

        plt.figure(figsize=(7,4.5))
        plt.plot(r, Va, label=f"Analítico (T={T}y)", linewidth=2)
        plt.plot(r, Vp, "--", label=f"PDE CN (T={T}y)", linewidth=1.5)
        plt.xlabel("Taxa curta r")
        plt.ylabel("Preço do zero-coupon")
        plt.title(f"Vasicek — Zero-Coupon | Comparação Analítico vs PDE | T={T} ano(s)")
        plt.legend()
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.show()

    # (Opcional) Plot de erro absoluto para o maior T
    T_ref = max(vencimentos)
    r = resultados[T_ref]["r"]
    err_abs = np.abs(resultados[T_ref]["erro"])
    plt.figure(figsize=(7,4.0))
    plt.plot(r, err_abs, linewidth=1.8)
    plt.xlabel("Taxa curta r")
    plt.ylabel("|Erro| (PDE − Analítico)")
    plt.title(f"Erro absoluto — T={T_ref} ano(s)")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    rodar_experimento()
