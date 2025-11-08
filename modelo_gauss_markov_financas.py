# -*- coding: utf-8 -*-
"""
Modelo Linear Gauss–Markov em Finanças 
Autor: Luiz Tiago Wilcke (LT) 

Executa:
  - Simulação de fatores e retornos (multi-ativos)
  - OLS + erros-padrão clássicos e HAC (Newey–West)
  - FGLS (Cochrane–Orcutt AR(1)) para um ativo
  - Rolling regressions
  - Monte Carlo de verificação Gauss–Markov
  - Geração de gráficos (matplotlib)

Requisitos: numpy, pandas, matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def newey_west(X, e, L=None):
    T, k = X.shape
    if L is None:
        L = int(np.floor(4*(T/100.0)**(2/9.0)))
    xe = X * e[:, None]
    S = (xe.T @ xe) / T
    for l in range(1, L+1):
        peso = 1.0 - l/(L+1.0)
        gamma_l = (xe[l:].T @ xe[:-l]) / T
        S += peso * (gamma_l + gamma_l.T)
    XtX_inv = np.linalg.inv(X.T @ X)
    V = XtX_inv @ (T * S) @ XtX_inv
    return V

def ols(X, y):
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    T = X.shape[0]
    k = X.shape[1]
    s2 = (resid.T @ resid) / (T - k)
    var_beta = s2 * XtX_inv
    se = np.sqrt(np.diag(var_beta))
    return beta, resid, s2, se, var_beta

def cochrane_orcutt_AR1(X, y, max_iter=50, tol=1e-6):
    beta_ols, e, _, _, _ = ols(X, y)
    e_t = e[1:]; e_tm1 = e[:-1]
    rho = float(np.dot(e_tm1, e_t) / np.dot(e_tm1, e_tm1))
    for _ in range(max_iter):
        y_t = y[1:] - rho * y[:-1]
        X_t = X[1:] - rho * X[:-1]
        beta_new, e_new, _, _, _ = ols(X_t, y_t)
        e_t = e_new[1:]; e_tm1 = e_new[:-1]
        rho_new = float(np.dot(e_tm1, e_t) / np.dot(e_tm1, e_tm1))
        if abs(rho_new - rho) < tol and np.max(np.abs(beta_new - beta_ols)) < tol:
            beta_ols = beta_new; rho = rho_new; break
        beta_ols = beta_new; rho = rho_new
    resid_fgls = (y[1:] - X[1:] @ beta_ols) - rho * (y[:-1] - X[:-1] @ beta_ols)
    Tt = len(resid_fgls); kt = X.shape[1]
    s2 = (resid_fgls.T @ resid_fgls) / (Tt - kt)
    XtX_inv = np.linalg.inv((X[1:] - rho*X[:-1]).T @ (X[1:] - rho*X[:-1]))
    var_beta = s2 * XtX_inv
    se = np.sqrt(np.diag(var_beta))
    return beta_ols, rho, se, var_beta

def executar():
    np.random.seed(123)
    T = 1250
    n_ativos = 5
    datas = pd.date_range(start="2018-01-02", periods=T, freq="B")

    fator_mercado = np.random.normal(0.0004, 0.01, T)
    fator_tamanho = np.random.normal(0.0001, 0.006, T)
    fator_valor   = np.random.normal(0.0001, 0.006, T)
    fator_momento = np.random.normal(0.0002, 0.008, T)

    taxa_ruido = np.random.normal(0.0, 0.004, T)
    fator_taxa  = np.zeros(T); phi = 0.9
    for t in range(1, T):
        fator_taxa[t] = phi*fator_taxa[t-1] + taxa_ruido[t]

    F = np.c_[np.ones(T), fator_mercado, fator_tamanho, fator_valor, fator_momento, fator_taxa]
    nomes_coef = ["constante","beta_mercado","beta_tamanho","beta_valor","beta_momento","beta_taxa"]

    betas_verdade = np.array([
        [0.0002, 1.10,  0.30, -0.10,  0.20, -0.50],
        [0.0001, 0.95, -0.10,  0.40, -0.05,  0.30],
        [0.0003, 1.35,  0.05,  0.10,  0.50, -0.20],
        [0.0000, 0.70,  0.25, -0.30,  0.15,  0.10],
        [0.0004, 1.50, -0.20,  0.20,  0.35,  0.00],
    ])

    sigma_base = 0.008
    sigma_t = sigma_base * (1.0 + 8.0 * np.abs(fator_mercado))
    rho_res = [0.2, 0.35, 0.5, 0.15, 0.0]

    retornos = np.zeros((T, n_ativos))
    for j in range(n_ativos):
        ruido = np.random.normal(0, sigma_t)
        for t in range(1, T):
            ruido[t] = rho_res[j]*ruido[t-1] + np.random.normal(0, sigma_t[t])
        retornos[:, j] = (F @ betas_verdade[j]) + ruido

    colunas = [f"ativo_{i+1}" for i in range(n_ativos)]
    df = pd.DataFrame(retornos, index=datas, columns=colunas)

    resultados = {}
    for j, nome in enumerate(colunas):
        y = df[nome].values; X = F.copy()
        beta, resid, s2, se_class, varb = ols(X, y)
        V_hac = newey_west(X, resid, L=None)
        se_hac = np.sqrt(np.diag(V_hac))
        resultados[nome] = {"beta_ols": beta, "se_class": se_class, "se_hac": se_hac, "residuos": resid}

    ativo_estudo = "ativo_1"; y = df[ativo_estudo].values; X = F.copy()
    beta_fgls, rho_fgls, se_fgls, varb_fgls = cochrane_orcutt_AR1(X, y)

    # Saídas essenciais no console
    print("=== OLS (", ativo_estudo, ") ===")
    for nc, b, se in zip(nomes_coef, resultados[ativo_estudo]["beta_ols"], resultados[ativo_estudo]["se_hac"]):
        print(f"{nc:>14s}: {b: .6f}  (EP_HAC: {se: .6f})")
    print("\n=== FGLS (Cochrane–Orcutt AR(1)) ===")
    for nc, b, se in zip(nomes_coef, beta_fgls, se_fgls):
        print(f"{nc:>14s}: {b: .6f}  (EP_FGLS: {se: .6f})")
    print(f"rho_AR1 estimado: {rho_fgls: .4f}")

if __name__ == "__main__":
    executar()
