#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regressão Gama GLM (link log) — Pipeline Completo 
Autor: Luiz Tiago Wilcke (LT)

Recursos:
- Leitura de CSV (fallback: gera dados sintéticos realistas)
- Pré-processamento: checagens, winsorização de outliers, padronização opcional
- Engenharia de variáveis: splines B-spline, interações, dummies
- Ajuste GLM Gamma (link log) + covariância robusta (HC3)
- Diagnósticos: resíduos (deviance/Pearson), sobre/ subdispersão, influência, alavancagem, Cook
- VIF (fator de inflação de variância) a partir do desenho do modelo
- Teste de especificação do link (Link Test/RESET para GLM)
- Validação cruzada K-Fold com métrica: Deviance e D^2 (pseudo-R²)
- Comparação com Tweedie (p≈1.5) e Lognormal (OLS em log)
- Predição com intervalos por bootstrap paramétrico
- Gráficos de diagnóstico e relatório CSV

Uso:
    1) Coloque um arquivo 'dados.csv' no mesmo diretório, com colunas:
       - resposta_positiva (alvo > 0)
       - e alguns preditores contínuos/categóricos: x1, x2, cat, ...
    2) Ajuste a fórmula_em_texto mais abaixo conforme suas variáveis.
    3) Rode:  python3 regressao_gama_pt.py

Obs.: variáveis e comentários em português. Bibliotecas: numpy, pandas, patsy, statsmodels, scipy, matplotlib.
"""

from __future__ import annotations
import os, sys, math, warnings, json
import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Gamma, Tweedie
from statsmodels.genmod.families.links import log as link_log
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from dataclasses import dataclass
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 100)

# ---------------------- Utilidades ----------------------

def winsorizar_coluna(s: pd.Series, p_inf: float = 0.01, p_sup: float = 0.99) -> pd.Series:
    lo, hi = s.quantile([p_inf, p_sup])
    return s.clip(lo, hi)


def d2_pseudo_r2(y_obs: np.ndarray, mu_hat: np.ndarray, familia: str = 'gamma') -> float:
    """Calcula D^2 = 1 - (DevianceModelo/DevianceNulo)."""
    if familia.lower() == 'gamma':
        # Deviance da família Gamma (McCullagh & Nelder)
        def dev(y, mu):
            eps = 1e-12
            y = np.clip(y, eps, None)
            mu = np.clip(mu, eps, None)
            return 2.0 * ( (y - mu)/mu - np.log(y/mu) )
    else:
        # genérico Tweedie approx via statsmodels
        raise NotImplementedError

    dev_modelo = np.sum(dev(y_obs, mu_hat))
    mu_nulo = np.repeat(np.mean(y_obs), len(y_obs))
    dev_nulo = np.sum(dev(y_obs, mu_nulo))
    return float(1.0 - dev_modelo / dev_nulo)


def calcular_vif(matriz_design: np.ndarray, nomes_cols: List[str]) -> pd.DataFrame:
    vifs = []
    for i in range(matriz_design.shape[1]):
        try:
            vifs.append(variance_inflation_factor(matriz_design, i))
        except Exception:
            vifs.append(np.nan)
    return pd.DataFrame({"variavel": nomes_cols, "VIF": vifs}).sort_values("VIF", ascending=False)


def link_test_glm(modelo, y, X_df: pd.DataFrame) -> pd.DataFrame:
    """Teste de especificação do link: regressa y em eta_chapeu e eta_chapeu^2; coef do termo quadrático ≈ 0 sugere boa especificação.
    Para GLM com link log, fazemos GLM Gamma novamente para manter comparabilidade.
    """
    eta = modelo.fittedvalues
    # Atenção: fittedvalues do GLM são mu, não eta; então obtemos eta via linear_predictor
    eta = modelo.linear_predictor
    df_aux = X_df.copy()
    df_aux["eta"] = eta
    df_aux["eta2"] = eta ** 2
    # Monta fórmula: y ~ eta + eta2
    df_aux["y"] = y
    mod = sm.GLM(df_aux["y"], sm.add_constant(df_aux[["eta", "eta2"]]), family=Gamma(link=link_log()))
    res = mod.fit()
    return pd.DataFrame({
        "coef": res.params,
        "erro_padrao": res.bse,
        "z": res.tvalues,
        "p_valor": res.pvalues
    })


def resumo_influencia_glm(result) -> pd.DataFrame:
    infl = result.get_influence(observed=True)
    alav = infl.hat_matrix_diag
    cooks = infl.cooks_distance[0]
    resid_pearson = result.resid_pearson
    df = pd.DataFrame({
        "alavancagem": alav,
        "dist_cook": cooks,
        "resid_pearson": resid_pearson
    })
    df["pontos_influentes"] = (df["dist_cook"] > 4/len(df)) | (df["alavancagem"] > 2*df["alavancagem"].mean())
    return df


def kfold_deviance(modelo_formula: str, df: pd.DataFrame, k: int = 5) -> Dict[str, float]:
    rng = np.random.default_rng(123)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    devs = []
    for f in folds:
        teste = df.iloc[f]
        treino = df.drop(df.index[f])
        mod = smf.glm(modelo_formula, data=treino, family=Gamma(link=link_log()))
        res = mod.fit()
        mu_hat = res.predict(teste)
        y = np.asarray(teste[res.model.endog_names])
        eps = 1e-12
        y = np.clip(y, eps, None)
        mu_hat = np.clip(mu_hat, eps, None)
        dev = np.sum(2 * ((y - mu_hat)/mu_hat - np.log(y/mu_hat)))
        devs.append(dev / len(teste))
    return {"deviance_media_por_obs": float(np.mean(devs)), "desvio_padrao": float(np.std(devs))}


def bootstrap_predicoes(result, X_novo: pd.DataFrame, B: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    params_hat = result.params.values
    cov = result.cov_params().values
    # Amostra parâmetros via Normal multivariada (aprox assintótica)
    amostras = rng.multivariate_normal(params_hat, cov, size=B)
    preds = []
    X = result.model.exog if X_novo is None else patsy.dmatrix(result.model.data.design_info.builder, X_novo)
    for b in range(B):
        eta_b = X @ amostras[b]
        mu_b = np.exp(eta_b)  # link log
        preds.append(mu_b)
    preds = np.vstack(preds)
    ic_inf = np.percentile(preds, 2.5, axis=0)
    ic_sup = np.percentile(preds, 97.5, axis=0)
    mu_med = preds.mean(axis=0)
    return pd.DataFrame({"mu_media": mu_med, "ic95_inf": ic_inf, "ic95_sup": ic_sup})

# ---------------------- Carregamento de dados ----------------------

ARQ = "dados.csv"
if os.path.exists(ARQ):
    dados = pd.read_csv(ARQ)
else:
    # Gera dados sintéticos (positivos) com heterogeneidade e não-linearidade
    rng = np.random.default_rng(123)
    n = 800
    x1 = rng.gamma(2.0, 2.0, size=n)           # contínua positiva
    x2 = rng.normal(0, 1, size=n)               # contínua simétrica
    x3 = rng.uniform(0, 1, size=n)              # contínua [0,1]
    cat = rng.choice(["A","B","C"], size=n, p=[0.4,0.4,0.2])
    # Efeito não-linear de x1 via raiz e log; interação com categoria
    eta = 0.5 + 0.2*np.sqrt(x1) + 0.3*np.log(x1+1) + 0.8*(x2>0) + 0.6*x3 \
          + 0.3*(cat=="B") + 0.6*(cat=="C") + 0.25*(x1*(cat=="C")) - 0.15*(x2*x3)
    mu = np.exp(eta)
    # Geração Gamma: y ~ Gamma(shape=k, scale=mu/k)
    k = 2.5  # forma (inverso da dispersão)
    y = rng.gamma(shape=k, scale=mu/k, size=n)
    dados = pd.DataFrame({
        "resposta_positiva": y,
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "cat": cat
    })

# Garante positividade do alvo
if (dados["resposta_positiva"] <= 0).any():
    raise ValueError("A coluna 'resposta_positiva' deve ser estritamente positiva para regressão Gama.")

# Winsorização suave nas contínuas (exceto alvo)
conts = [c for c in dados.select_dtypes(include=[np.number]).columns if c != "resposta_positiva"]
for c in conts:
    dados[c] = winsorizar_coluna(dados[c], 0.01, 0.99)

# Dummies da categoria
if "cat" in dados.columns and not np.issubdtype(dados["cat"].dtype, np.number):
    dados["cat"] = dados["cat"].astype("category")

# Cria algumas transformações úteis
if "x1" in dados.columns:
    dados["raiz_x1"] = np.sqrt(dados["x1"].clip(lower=1e-12))
    dados["log_x1"] = np.log(dados["x1"].clip(lower=1e-12))
if "x2" in dados.columns:
    dados["x2_pos"] = (dados["x2"] > 0).astype(int)

# ---------------------- Fórmula do modelo ----------------------
# Ajuste aqui conforme suas variáveis. Exemplos de recursos:
# - splines: bs(var, df=4)
# - interações: var1:var2, ou var1*var2 (= var1 + var2 + var1:var2)
# - C(cat): variáveis indicadoras para categorias

formula_em_texto = (
    "resposta_positiva ~ bs(x1, df=4) + log_x1 + raiz_x1 + x2 + x2_pos + x3 + C(cat) + x1:C(cat) + x2:x3"
)

# ---------------------- Ajuste GLM Gamma ----------------------

modelo_gama = smf.glm(formula_em_texto, data=dados, family=Gamma(link=link_log()))
resultado = modelo_gama.fit()
print("\n===== SUMÁRIO GLM Gama (HC3 robust) =====")
print(resultado.summary())
print("\nCovariância robusta HC3:")
print(resultado.get_robustcov_results(cov_type='HC3').summary())

# Métricas
mu_hat = resultado.fittedvalues.values
D2 = d2_pseudo_r2(dados["resposta_positiva"].values, mu_hat, familia='gamma')
print(f"\nD^2 (pseudo-R²) ~ {D2:.4f}")
print(f"AIC: {resultado.aic:.2f} | BIC: {resultado.bic:.2f}")

# Dispersão (phi) estimada: em Gamma, 1/alpha
phi = resultado.scale
print(f"Dispersão (phi) estimada: {phi:.4f}")

# ---------------------- VIF ----------------------

y, X = patsy.dmatrices(formula_em_texto, data=dados, return_type='dataframe')
X_sem_const = X.drop(columns=[c for c in X.columns if c.lower().startswith('intercept')], errors='ignore')

df_vif = calcular_vif(X_sem_const.values, list(X_sem_const.columns))
print("\n===== VIF =====")
print(df_vif)

# ---------------------- Teste de Link (Link Test) ----------------------

df_link = link_test_glm(resultado, y=np.asarray(dados["resposta_positiva"]), X_df=X_sem_const)
print("\n===== Link Test (eta, eta^2) =====")
print(df_link)

# ---------------------- Influência ----------------------

df_infl = resumo_influencia_glm(resultado)
print("\n===== Influência (top 10 por dist_Cook) =====")
print(df_infl.sort_values("dist_cook", ascending=False).head(10))

# ---------------------- Validação Cruzada ----------------------

kfold_res = kfold_deviance(formula_em_texto, dados, k=5)
print("\n===== K-Fold (deviance média por obs) =====")
print(kfold_res)

# ---------------------- Modelos de Comparação ----------------------

# Tweedie com p ~ 1.5 (entre Poisson e Gamma; admite massa em zero se p<=1, aqui >1)
try:
    modelo_tw = smf.glm(formula_em_texto, data=dados, family=Tweedie(var_power=1.5, link=link_log()))
    res_tw = modelo_tw.fit()
    print("\nAIC Tweedie p=1.5:", res_tw.aic)
except Exception as e:
    print("Tweedie falhou:", e)

# Lognormal (OLS em log y) — somente comparação bruta
try:
    dados["log_y"] = np.log(dados["resposta_positiva"].values)
    res_ln = smf.ols("log_y ~ " + formula_em_texto.split("~")[1], data=dados).fit()
    print("AIC Lognormal (OLS em log y):", res_ln.aic)
except Exception as e:
    print("Lognormal falhou:", e)

# ---------------------- Predição com IC por Bootstrap ----------------------

# Exemplo: usa 5 primeiras observações como "novas" (substitua por seu novo DataFrame)
X_novo = dados.head(5).copy()
# Remova a coluna alvo se existir
if "resposta_positiva" in X_novo.columns:
    X_novo = X_novo.drop(columns=["resposta_positiva", "log_y"], errors='ignore')

preds = resultado.get_prediction(X_novo)
summary_frame = preds.summary_frame(alpha=0.05)  # mean, mean_ci_lower, mean_ci_upper
summary_frame = summary_frame.rename(columns={
    'mean': 'mu_previsto',
    'mean_ci_lower': 'ic95_inf_media',
    'mean_ci_upper': 'ic95_sup_media'
})

boot_ic = bootstrap_predicoes(resultado, X_novo, B=500, seed=123)

df_pred = pd.concat([X_novo.reset_index(drop=True), summary_frame.reset_index(drop=True), boot_ic], axis=1)
print("\n===== Predições (primeiros 5 casos) =====")
print(df_pred)

# ---------------------- Gráficos ----------------------

os.makedirs("resultados_graficos", exist_ok=True)

# 1) Resíduo de Deviance vs Ajustado
res_dev = resultado.resid_deviance
plt.figure()
plt.scatter(mu_hat, res_dev, s=12, alpha=0.7)
plt.axhline(0, linestyle='--')
plt.xlabel('Ajustado (mu)')
plt.ylabel('Resíduo de Deviance')
plt.title('Resíduo de Deviance vs Ajustado — GLM Gama')
plt.tight_layout()
plt.savefig("resultados_graficos/residuo_deviance_vs_ajustado.png", dpi=150)

# 2) QQ-plot dos resíduos de Pearson (aprox)
plt.figure()
sm.ProbPlot(resultado.resid_pearson).qqplot(line='s')
plt.title('QQ-plot Resíduos de Pearson (aprox)')
plt.tight_layout()
plt.savefig("resultados_graficos/qq_pearson.png", dpi=150)

# 3) Influência: Cook vs índice
plt.figure()
plt.stem(np.arange(len(df_infl)), df_infl['dist_cook'], use_line_collection=True)
plt.xlabel('Índice')
plt.ylabel("Distância de Cook")
plt.title('Influência — Distância de Cook')
plt.tight_layout()
plt.savefig("resultados_graficos/cook_dist.png", dpi=150)

# 4) Alavancagem: histograma
plt.figure()
plt.hist(df_infl['alavancagem'], bins=30)
plt.xlabel('Alavancagem')
plt.ylabel('Frequência')
plt.title('Distribuição de Alavancagem')
plt.tight_layout()
plt.savefig("resultados_graficos/alavancagem_hist.png", dpi=150)

# ---------------------- Relatórios ----------------------

os.makedirs("resultados", exist_ok=True)
dados.to_csv("resultados/dados_processados.csv", index=False)
df_vif.to_csv("resultados/vif.csv", index=False)
df_infl.to_csv("resultados/influencia.csv", index=False)
df_pred.to_csv("resultados/predicoes_exemplo.csv", index=False)

with open("resultados/metricas.json", "w", encoding="utf-8") as f:
    json.dump({
        "D2_pseudo_R2": D2,
        "AIC_gama": float(resultado.aic),
        "BIC_gama": float(resultado.bic),
        "phi_dispersao": float(phi),
        "kfold": kfold_res
    }, f, ensure_ascii=False, indent=2)

print("\nArquivos salvos em 'resultados/' e figuras em 'resultados_graficos/'.")
print("Concluído.")
