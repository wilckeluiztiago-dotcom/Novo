# Estimação da distribuição Gama (MLE + Bootstrap) e gráficos

import numpy as np
import matplotlib.pyplot as plt
import math
from math import lgamma
from pathlib import Path

# ============ Utilidades de funções especiais (numéricas) ============
def digama_num(x: float, h: float = 1e-6) -> float:
    """Aproxima psi(x) = d/dx ln Γ(x) por diferença central de lgamma."""
    if x <= 0:
        raise ValueError("digama definida para x>0 neste contexto.")
    h = h * (1.0 + abs(x))  # passo adaptativo
    return (lgamma(x + h) - lgamma(x - h)) / (2.0 * h)

def trigama_num(x: float, h: float = 1e-4) -> float:
    """Aproxima psi'(x) = d^2/dx^2 ln Γ(x) por segunda diferença central de lgamma."""
    if x <= 0:
        raise ValueError("trigama definida para x>0 neste contexto.")
    h = h * (1.0 + abs(x))
    return (lgamma(x + h) - 2.0 * lgamma(x) + lgamma(x - h)) / (h * h)

# ============ Log-verossimilhança e MLE ============
def log_verossimilhanca_gamma(alpha: float, theta: float, x: np.ndarray) -> float:
    """log L(alpha, theta | x) para Gama(shape alpha, scale theta)."""
    if alpha <= 0 or theta <= 0:
        return -np.inf
    n = x.size
    soma_logx = float(np.sum(np.log(x)))
    soma_x = float(np.sum(x))
    return n * (-lgamma(alpha) - alpha * math.log(theta)) + (alpha - 1.0) * soma_logx - (soma_x / theta)

def estimar_alpha_por_equacao_escore(x: np.ndarray, tol: float = 1e-8, max_iter: int = 200) -> float:
    """Resolve log(alpha) - psi(alpha) = log(mean(x)) - mean(log(x)) via Newton."""
    media = float(np.mean(x))
    e_log = float(np.mean(np.log(x)))
    alvo = math.log(media) - e_log  # RHS

    var = float(np.var(x, ddof=1))
    alpha = max(1e-3, media**2 / var) if var > 0 else 10.0  # chute por momentos

    for _ in range(max_iter):
        psi = digama_num(alpha)
        eq = math.log(alpha) - psi - alvo
        if abs(eq) < tol:
            break
        psi1 = trigama_num(alpha)
        der = 1.0/alpha - psi1
        passo = eq / der
        alpha_novo = alpha - passo
        if alpha_novo <= 0 or not np.isfinite(alpha_novo):
            alpha_novo = max(1e-3, alpha/2.0)
        if abs(alpha_novo - alpha) <= tol * (1 + abs(alpha)):
            alpha = alpha_novo
            break
        alpha = alpha_novo
    return float(max(alpha, 1e-8))

def estimar_mle_gamma(x: np.ndarray):
    """Retorna alpha_hat, theta_hat, logL_hat, Hessiano numérico e covariância aproximada."""
    alpha_hat = estimar_alpha_por_equacao_escore(x)
    theta_hat = float(np.mean(x)) / alpha_hat

    def f(a, t): 
        return log_verossimilhanca_gamma(a, t, x)

    a0, t0 = alpha_hat, theta_hat
    ha = 1e-3 * (1 + abs(a0))
    ht = 1e-3 * (1 + abs(t0))

    f_00 = f(a0, t0)
    f_ap = f(a0 + ha, t0); f_am = f(a0 - ha, t0)
    f_tp = f(a0, t0 + ht); f_tm = f(a0, t0 - ht)
    f_ap_tp = f(a0 + ha, t0 + ht); f_ap_tm = f(a0 + ha, t0 - ht)
    f_am_tp = f(a0 - ha, t0 + ht); f_am_tm = f(a0 - ha, t0 - ht)

    d2_aa = (f_ap - 2*f_00 + f_am) / (ha*ha)
    d2_tt = (f_tp - 2*f_00 + f_tm) / (ht*ht)
    d2_at = (f_ap_tp - f_ap_tm - f_am_tp + f_am_tm) / (4*ha*ht)

    H = np.array([[d2_aa, d2_at],
                  [d2_at, d2_tt]], dtype=float)

    info_obs = -H
    try:
        cov = np.linalg.inv(info_obs)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(info_obs)

    return alpha_hat, theta_hat, f_00, H, cov

# ============ CDF e quantis por simulação (sem SciPy) ============
def cdf_gama_mc(z: float, alpha: float, theta: float, rng: np.random.Generator, n_mc: int = 200_000) -> float:
    """Aproxima F_Z(z) por simulação (para ECDF vs CDF)."""
    if z <= 0:
        return 0.0
    amostra = rng.gamma(shape=alpha, scale=theta, size=n_mc)
    return float(np.mean(amostra <= z))

def quantis_teoricos_gama(p: np.ndarray, alpha: float, theta: float, rng: np.random.Generator, n_mc: int = 200_000) -> np.ndarray:
    """Quantis teóricos por simulação (ordena amostras MC e interpola)."""
    amostra = np.sort(rng.gamma(shape=alpha, scale=theta, size=n_mc))
    i = (p * (n_mc - 1)).clip(0, n_mc - 1)
    i0 = np.floor(i).astype(int)
    i1 = np.ceil(i).astype(int)
    w = i - i0
    return (1 - w) * amostra[i0] + w * amostra[i1]

# ============ Bootstrap Paramétrico ============
def bootstrap_parametrico_gamma(alpha: float, theta: float, n: int, B: int = 500, rng: np.random.Generator = None):
    if rng is None:
        rng = np.random.default_rng(123)
    alfas, thetas = [], []
    for _ in range(B):
        xb = rng.gamma(shape=alpha, scale=theta, size=n)
        ah, th, _, _, _ = estimar_mle_gamma(xb)
        alfas.append(ah); thetas.append(th)
    return np.array(alfas), np.array(thetas)

# ============ Execução principal ============
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Parâmetros verdadeiros da geração
    alpha_verdadeiro = 4.75
    theta_verdadeiro = 2.10
    n_amostra = 800

    # Gerar amostra
    amostra = rng.gamma(shape=alpha_verdadeiro, scale=theta_verdadeiro, size=n_amostra)

    # Estimar MLE
    alpha_hat, theta_hat, logL_hat, H_hat, cov_hat = estimar_mle_gamma(amostra)
    erro_padrao_alpha = float(np.sqrt(abs(cov_hat[0,0])))
    erro_padrao_theta = float(np.sqrt(abs(cov_hat[1,1])))

    # IC assintóticos 95%
    z975 = 1.959963984540054
    ic_alpha_assint = (alpha_hat - z975*erro_padrao_alpha, alpha_hat + z975*erro_padrao_alpha)
    ic_theta_assint = (theta_hat - z975*erro_padrao_theta, theta_hat + z975*erro_padrao_theta)

    # Bootstrap paramétrico
    B = 300  # aumente se quiser maior precisão
    alfas_b, thetas_b = bootstrap_parametrico_gamma(alpha_hat, theta_hat, n_amostra, B=B, rng=rng)
    ic_alpha_boot = (float(np.quantile(alfas_b, 0.025)), float(np.quantile(alfas_b, 0.975)))
    ic_theta_boot = (float(np.quantile(thetas_b, 0.025)), float(np.quantile(thetas_b, 0.975)))

    # --------- Gráfico 1: Histograma + PDF ajustada ---------
    xs = np.linspace(1e-6, max(amostra)*1.05, 800)
    pdf_aj = (xs**(alpha_hat-1.0)) * np.exp(-xs/theta_hat) / (math.exp(lgamma(alpha_hat)) * (theta_hat**alpha_hat))

    plt.figure(figsize=(7,5))
    plt.hist(amostra, bins=40, density=True, alpha=0.6)
    plt.plot(xs, pdf_aj, linewidth=2)
    plt.title("Histograma da amostra e PDF Gama ajustada (MLE)")
    plt.xlabel("x"); plt.ylabel("densidade")
    plt.tight_layout()
    plt.show()

    # --------- Gráfico 2: QQ-Plot Gama (MC) ---------
    amostra_ord = np.sort(amostra)
    probs = (np.arange(1, n_amostra+1) - 0.5) / n_amostra
    qt = quantis_teoricos_gama(probs, alpha_hat, theta_hat, rng)

    plt.figure(figsize=(6,6))
    plt.scatter(qt, amostra_ord, s=12)
    minv = float(min(qt.min(), amostra_ord.min()))
    maxv = float(max(qt.max(), amostra_ord.max()))
    plt.plot([minv, maxv], [minv, maxv], linestyle="--")
    plt.title("QQ-Plot: quantis teóricos Gama vs amostra")
    plt.xlabel("Quantis teóricos (Gama ajustada)")
    plt.ylabel("Quantis amostrais")
    plt.tight_layout()
    plt.show()

    # --------- Gráfico 3: ECDF vs CDF ajustada (MC) ---------
    ecdf_x = np.sort(amostra)
    ecdf_y = np.arange(1, n_amostra+1) / n_amostra
    idx = np.linspace(0, n_amostra-1, 2000).astype(int)
    ecdf_x_sel = ecdf_x[idx]
    cdf_mc_sel = np.array([cdf_gama_mc(z, alpha_hat, theta_hat, rng, n_mc=80_000) for z in ecdf_x_sel])

    plt.figure(figsize=(7,5))
    plt.step(ecdf_x, ecdf_y, where="post", linewidth=1.5, label="ECDF")
    plt.plot(ecdf_x_sel, cdf_mc_sel, linewidth=2, label="CDF Gama (MC)")
    plt.title("CDF empírica vs CDF Gama ajustada")
    plt.xlabel("x"); plt.ylabel("F(x)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --------- Gráfico 4: Perfil de log-verossimilhança para alpha ---------
    alphas_grid = np.linspace(max(1e-2, 0.3*alpha_hat), 2.5*alpha_hat, 150)
    media_x = float(np.mean(amostra))
    logLs = []
    for a in alphas_grid:
        t = media_x / a  # θ̂(a)
        logLs.append(log_verossimilhanca_gamma(a, t, amostra))
    logLs = np.array(logLs)
    logLs_norm = logLs - logLs.max()

    plt.figure(figsize=(7,5))
    plt.plot(alphas_grid, logLs_norm, linewidth=2)
    plt.axvline(alpha_hat, linestyle="--")
    plt.title("Perfil de log-verossimilhança normalizado para α")
    plt.xlabel("α"); plt.ylabel("logL(α, θ̂(α)) - max logL")
    plt.tight_layout()
    plt.show()

    # --------- Saídas em arquivos (opcionais) ---------
    Path("saidas").mkdir(exist_ok=True)
    np.savetxt("saidas/amostra_gama.csv", amostra, delimiter=",")
    import pandas as pd
    resumo = pd.DataFrame({
        "parâmetro": ["alpha", "theta"],
        "verdadeiro": [alpha_verdadeiro, theta_verdadeiro],
        "estimativa_MLE": [alpha_hat, theta_hat],
        "erro_padrão_assint.": [float(np.sqrt(abs(cov_hat[0,0]))), float(np.sqrt(abs(cov_hat[1,1])))],
        "IC95% assint. (inf)": [ic_alpha_assint[0], ic_theta_assint[0]],
        "IC95% assint. (sup)": [ic_alpha_assint[1], ic_theta_assint[1]],
        "IC95% bootstrap (inf)": [ic_alpha_boot[0], ic_theta_boot[0]],
        "IC95% bootstrap (sup)": [ic_alpha_boot[1], ic_theta_boot[1]],
    })
    resumo.to_csv("saidas/resumo_estimacao_gama.csv", index=False)
    print(resumo)
    print("\nArquivos salvos em ./saidas/")
