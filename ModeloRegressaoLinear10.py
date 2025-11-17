# ============================================================
# Modelo de Regressão Linear Avançada em Finanças (Brasil/B3)
# Problema: Prêmio de risco de uma ação vs fatores macroeconômicos
# Autor: Luiz Tiago Wilcke (LT)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Tuple, Dict, Any

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.stattools import adfuller

from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error

# ------------------------------------------------------------
# 1) Configurações gerais
# ------------------------------------------------------------
@dataclass
class Configuracoes:
    n_observacoes: int = 240     # ~20 anos mensais
    seed: int = 42
    mostra_graficos: bool = True


cfg = Configuracoes()
np.random.seed(cfg.seed)

# ------------------------------------------------------------
# 2) Gerar dados sintéticos de finanças em português
# ------------------------------------------------------------
def gerar_dados_sinteticos(cfg: Configuracoes) -> pd.DataFrame:
    # freq="ME" para evitar FutureWarning do pandas
    datas = pd.date_range(start="2005-01-01", periods=cfg.n_observacoes, freq="ME")

    # Fator de mercado: retorno excedente do IBOV (sobre CDI, por ex.)
    retorno_excesso_mercado = np.random.normal(0.007, 0.04, cfg.n_observacoes)

    # Inflação IPCA mensal (em decimal)
    inflacao_ipca = np.random.normal(0.004, 0.002, cfg.n_observacoes)
    inflacao_ipca = np.clip(inflacao_ipca, 0.0, None)

    # Câmbio BRL/USD (log-nível com leve tendência)
    tendencia_cambio = np.linspace(0, 0.25, cfg.n_observacoes)
    ruido_cambio = np.random.normal(0.0, 0.05, cfg.n_observacoes)
    nivel_cambio_log = 1.6 + tendencia_cambio + ruido_cambio
    cambio_brlusd = np.exp(nivel_cambio_log)

    # Taxa Selic (ao ano, em decimal) com ciclos
    t = np.arange(cfg.n_observacoes)
    selic_anual = (
        0.11
        + 0.03 * np.sin(2 * np.pi * t / 60)
        + np.random.normal(0.0, 0.005, cfg.n_observacoes)
    )
    selic_anual = np.clip(selic_anual, 0.03, 0.20)

    # Dummy de crise (ex: 2008-2009 + 2014-2016)
    dummy_crise = np.zeros(cfg.n_observacoes)
    dummy_crise[(datas.year >= 2008) & (datas.year <= 2009)] = 1
    dummy_crise[(datas.year >= 2014) & (datas.year <= 2016)] = 1

    # --------------------------------------------------------
    # Gerar prêmio de risco da ação:
    #   retorno_excesso_acao = β0
    #                          + β1 * retorno_excesso_mercado
    #                          + β2 * inflacao_ipca
    #                          + β3 * variacao_cambio
    #                          + β4 * selic_centralizada
    #                          + β5 * dummy_crise
    #                          + β6 * retorno_excesso_mercado * dummy_crise
    #                          + erro
    # --------------------------------------------------------

    # Variação mensal do câmbio (retorno log)
    variacao_cambio = np.diff(np.log(cambio_brlusd), prepend=np.log(cambio_brlusd[0]))

    beta0 = 0.002
    beta1 = 1.25
    beta2 = -0.8
    beta3 = -0.4
    beta4 = -0.5
    beta5 = -0.03
    beta6 = 0.7

    erro = np.random.normal(0.0, 0.02, cfg.n_observacoes)

    selic_centralizada = selic_anual - np.mean(selic_anual)

    retorno_excesso_acao = (
        beta0
        + beta1 * retorno_excesso_mercado
        + beta2 * inflacao_ipca
        + beta3 * variacao_cambio
        + beta4 * selic_centralizada
        + beta5 * dummy_crise
        + beta6 * retorno_excesso_mercado * dummy_crise
        + erro
    )

    df = pd.DataFrame(
        {
            "data": datas,
            "retorno_excesso_acao": retorno_excesso_acao,
            "retorno_excesso_mercado": retorno_excesso_mercado,
            "inflacao_ipca": inflacao_ipca,
            "cambio_brlusd": cambio_brlusd,
            "variacao_cambio": variacao_cambio,
            "selic_anual": selic_anual,
            "dummy_crise": dummy_crise,
        }
    )
    df.set_index("data", inplace=True)
    return df


# ------------------------------------------------------------
# 3) Preparar matriz de design para regressão
# ------------------------------------------------------------
def preparar_regressores(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    # Variável dependente: prêmio de risco da ação
    y = df["retorno_excesso_acao"]

    # Regressoras:
    #   - retorno_excesso_mercado
    #   - inflacao_ipca
    #   - variacao_cambio
    #   - selic_anual centralizada
    #   - dummy_crise
    #   - interação retorno_excesso_mercado * dummy_crise
    df_reg = pd.DataFrame()
    df_reg["retorno_excesso_mercado"] = df["retorno_excesso_mercado"]
    df_reg["inflacao_ipca"] = df["inflacao_ipca"]
    df_reg["variacao_cambio"] = df["variacao_cambio"]
    df_reg["selic_centralizada"] = df["selic_anual"] - df["selic_anual"].mean()
    df_reg["dummy_crise"] = df["dummy_crise"]
    df_reg["interacao_mercado_crise"] = (
        df["retorno_excesso_mercado"] * df["dummy_crise"]
    )

    return df_reg, y


# ------------------------------------------------------------
# 4) Ajustar regressão OLS com erros robustos
# ------------------------------------------------------------
def ajustar_regressao_ols(df_reg: pd.DataFrame, y: pd.Series):
    X = sm.add_constant(df_reg)
    modelo = sm.OLS(y, X)
    resultado = modelo.fit(cov_type="HC3")  # erros robustos à heterocedasticidade
    return resultado


# ------------------------------------------------------------
# 5) Calcular VIF (Multicolinearidade)
# ------------------------------------------------------------
def calcular_vif(df_reg: pd.DataFrame) -> pd.DataFrame:
    X_vif = sm.add_constant(df_reg)
    vif_dados = []
    for i in range(X_vif.shape[1]):
        nome = X_vif.columns[i]
        vif_valor = variance_inflation_factor(X_vif.values, i)
        vif_dados.append((nome, vif_valor))
    df_vif = pd.DataFrame(vif_dados, columns=["variavel", "VIF"])
    return df_vif


# ------------------------------------------------------------
# 6) Testes de diagnóstico
# ------------------------------------------------------------
def testes_diagnostico(resultado, df_reg: pd.DataFrame, y: pd.Series):
    residuos = resultado.resid
    X = sm.add_constant(df_reg)

    print("\n====== Teste Jarque-Bera (normalidade dos resíduos) ======")
    jb_stat, jb_p, skew, kurt = jarque_bera(residuos)
    print(f"JB estatística = {jb_stat:.4f}, p-valor = {jb_p:.4f}")
    print(f"Assimetria = {skew:.4f}, Curtose = {kurt:.4f}")

    print("\n====== Teste Breusch-Pagan (heterocedasticidade) ======")
    bp_stat, bp_p, _, _ = het_breuschpagan(residuos, X)
    print(f"BP estatística = {bp_stat:.4f}, p-valor = {bp_p:.4f}")

    print("\n====== Teste ADF (raiz unitária em resíduos) ======")
    adf_stat, adf_p, _, _, crit, _ = adfuller(residuos)
    print(f"ADF estatística = {adf_stat:.4f}, p-valor = {adf_p:.4f}")
    print(f"Valores críticos: {crit}")


# ------------------------------------------------------------
# 7) Regressão com Ridge e Lasso (regularização)
# ------------------------------------------------------------
def ajustar_ridge_lasso(
    df_reg: pd.DataFrame,
    y: pd.Series,
    alpha_ridge: float = 10.0,
    alpha_lasso: float = 0.01,
) -> Dict[str, Dict[str, Any]]:
    X = df_reg.values
    y_val = y.values

    ridge = Ridge(alpha=alpha_ridge)
    ridge.fit(X, y_val)
    y_pred_ridge = ridge.predict(X)

    lasso = Lasso(alpha=alpha_lasso, max_iter=10000)
    lasso.fit(X, y_val)
    y_pred_lasso = lasso.predict(X)

    resultados = {
        "ridge": {
            "modelo": ridge,
            "r2": r2_score(y_val, y_pred_ridge),
            "rmse": np.sqrt(mean_squared_error(y_val, y_pred_ridge)),
            "coeficientes": ridge.coef_,
            "intercepto": ridge.intercept_,
        },
        "lasso": {
            "modelo": lasso,
            "r2": r2_score(y_val, y_pred_lasso),
            "rmse": np.sqrt(mean_squared_error(y_val, y_pred_lasso)),
            "coeficientes": lasso.coef_,
            "intercepto": lasso.intercept_,
        },
    }
    return resultados


# ------------------------------------------------------------
# 8) Gráficos básicos (corrigido para não dar erro de dimensões)
# ------------------------------------------------------------
def plotar_graficos(df: pd.DataFrame, df_reg: pd.DataFrame, resultado):
    # Relação entre retorno_excesso_acao e retorno_excesso_mercado
    plt.figure()
    plt.scatter(
        df_reg["retorno_excesso_mercado"],
        df["retorno_excesso_acao"],
        alpha=0.6,
        label="Observações",
    )

    x_grid = np.linspace(
        df_reg["retorno_excesso_mercado"].min(),
        df_reg["retorno_excesso_mercado"].max(),
        100,
    )

    # Prever com os demais regressores fixados na média
    df_aux = pd.DataFrame({
        "retorno_excesso_mercado": x_grid,
        "inflacao_ipca": df_reg["inflacao_ipca"].mean(),
        "variacao_cambio": df_reg["variacao_cambio"].mean(),
        "selic_centralizada": df_reg["selic_centralizada"].mean(),
        "dummy_crise": 0.0,
        "interacao_mercado_crise": 0.0,
    })

    # Garantir MESMA ordem de colunas que no modelo
    nomes_exog = resultado.model.exog_names  # ['const', 'retorno_excesso_mercado', ...]
    colunas_sem_const = [nome for nome in nomes_exog if nome != "const"]
    df_aux = df_aux[colunas_sem_const]

    # Forçar constante explícita
    X_aux = sm.add_constant(df_aux, has_constant="add")

    # Agora as dimensões batem: (100,7) x (7,)
    y_pred = resultado.predict(X_aux)

    plt.plot(x_grid, y_pred, linewidth=2, label="Linha de tendência (controle)")
    plt.xlabel("Retorno excedente do mercado")
    plt.ylabel("Retorno excedente da ação")
    plt.title("Relação ação x mercado (controle para demais fatores)")
    plt.legend()
    plt.grid(True)

    # Série temporal do prêmio de risco e valores ajustados
    plt.figure()
    plt.plot(df.index, df["retorno_excesso_acao"], label="Real")
    plt.plot(df.index, resultado.fittedvalues, label="Ajustado")
    plt.xlabel("Tempo")
    plt.ylabel("Retorno excedente da ação")
    plt.title("Série temporal: real vs ajustado (modelo OLS robusto)")
    plt.legend()
    plt.grid(True)

    # Resíduos ao longo do tempo
    plt.figure()
    plt.plot(df.index, resultado.resid)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Tempo")
    plt.ylabel("Resíduo")
    plt.title("Resíduos do modelo ao longo do tempo")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 9) Rotina principal
# ------------------------------------------------------------
def main():
    print("Gerando dados sintéticos de finanças...")
    df = gerar_dados_sinteticos(cfg)

    print("\nPreparando regressores...")
    df_reg, y = preparar_regressores(df)

    print("\nAjustando regressão OLS com erros robustos (HC3)...")
    resultado = ajustar_regressao_ols(df_reg, y)
    print(resultado.summary())

    print("\nCalculando VIF (multicolinearidade)...")
    df_vif = calcular_vif(df_reg)
    print(df_vif)

    print("\nExecutando testes de diagnóstico...")
    testes_diagnostico(resultado, df_reg, y)

    print("\nAjustando modelos Ridge e Lasso (regularização)...")
    resultados_reg = ajustar_ridge_lasso(df_reg, y)
    print("\n===== Ridge =====")
    print("R²:", resultados_reg["ridge"]["r2"])
    print("RMSE:", resultados_reg["ridge"]["rmse"])
    print("Intercepto:", resultados_reg["ridge"]["intercepto"])
    print("Coeficientes (ordem das colunas):", list(df_reg.columns))
    print(resultados_reg["ridge"]["coeficientes"])

    print("\n===== Lasso =====")
    print("R²:", resultados_reg["lasso"]["r2"])
    print("RMSE:", resultados_reg["lasso"]["rmse"])
    print("Intercepto:", resultados_reg["lasso"]["intercepto"])
    print("Coeficientes (ordem das colunas):", list(df_reg.columns))
    print(resultados_reg["lasso"]["coeficientes"])

    if cfg.mostra_graficos:
        plotar_graficos(df, df_reg, resultado)


if __name__ == "__main__":
    main()
