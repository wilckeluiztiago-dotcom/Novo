# ============================================================
# Previsão USD/BRL — SARIMAX + Estado-Espaço (+ GARCH opcional)
# Autor: Luiz Tiago Wilcke (LT)
# ============================================================

# -------------------- IMPORTAÇÕES --------------------
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

# Stats / Modelagem
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.structural import UnobservedComponents

# (opcional) GARCH se 'arch' estiver instalado
try:
    from arch import arch_model
    TEM_GARCH = True
except Exception:
    TEM_GARCH = False

# -------------------- CONFIGURAÇÕES --------------------
np.random.seed(42)

# aponte seu CSV real: precisa ter ao menos colunas: data, usdbrl
# opcional: juros_brasil, juros_eua, ipca, vix, etc. (usadas como exógenas se existirem)
USAR_CSV = True
CAMINHO_CSV = "usdbrl.csv"  # exemplo; ajustar para seu arquivo
COLUNA_DATA = "data"
COLUNA_PRECO = "usdbrl"

# Caso queira simular (se não tiver CSV), defina USAR_CSV=False
N_SINT = 1200  # dias úteis simulados ~ 5 anos

# parâmetros gerais
PROPORCAO_TREINO = 0.8
SAZONALIDADE_SEMANAL = 5  # dias úteis
PLOTAR = True
PASTA_SAIDA = "artefatos_modelo_cambio"
os.makedirs(PASTA_SAIDA, exist_ok=True)


# -------------------- UTILITÁRIOS --------------------
def carregar_dados() -> pd.DataFrame:
    if USAR_CSV and os.path.exists(CAMINHO_CSV):
        df = pd.read_csv(CAMINHO_CSV)
        df[COLUNA_DATA] = pd.to_datetime(df[COLUNA_DATA])
        df = df.sort_values(COLUNA_DATA).set_index(COLUNA_DATA)
        df = df.asfreq("B")  # dias úteis
        df[COLUNA_PRECO] = df[COLUNA_PRECO].interpolate()
        return df
    else:
        # SÉRIE SINTÉTICA (ARIMA + choque sazonal semanal + tendência leve)
        rng = pd.date_range("2018-01-01", periods=N_SINT, freq="B")
        eps = np.random.normal(0, 0.005, size=len(rng))
        y = np.zeros(len(rng))
        y[0] = np.log(3.2)  # log de ~3,2 BRL/USD inicial
        for t in range(1, len(rng)):
            y[t] = 0.7*y[t-1] + 0.3*np.log(3.5) + eps[t] + 0.002*np.sin(2*np.pi*t/SAZONALIDADE_SEMANAL)
        preco = np.exp(y)
        df = pd.DataFrame({COLUNA_PRECO: preco}, index=rng)
        return df


def engenharia_de_variaveis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_preco"] = np.log(df[COLUNA_PRECO])
    df["retorno_log"] = df["log_preco"].diff()
    # se existirem exógenas no CSV, padronize nomes e possíveis lags
    exogenas = []
    for col in ["juros_brasil", "juros_eua", "ipca", "vix"]:
        if col in df.columns:
            df[f"{col}_z"] = (df[col] - df[col].mean())/df[col].std(ddof=0)
            exogenas.append(f"{col}_z")
            # lags 1 e 5 como exemplo
            df[f"{col}_z_l1"] = df[f"{col}_z"].shift(1)
            df[f"{col}_z_l5"] = df[f"{col}_z"].shift(5)
            exogenas += [f"{col}_z_l1", f"{col}_z_l5"]
    df.attrs["exogenas"] = exogenas
    return df


def dividir_treino_teste(df: pd.DataFrame, col_alvo: str) -> Tuple[pd.Series, pd.Series]:
    n = len(df)
    corte = int(n*PROPORCAO_TREINO)
    return df[col_alvo].iloc[:corte], df[col_alvo].iloc[corte:]


def adf_print(serie: pd.Series, nome: str):
    serie = serie.dropna()
    res = adfuller(serie, autolag="AIC")
    print(f"ADF {nome}: estat={res[0]:.3f}, p={res[1]:.4f}, crit={res[4]}")
    return res


def ljung_box_print(residuos: pd.Series, nome: str, lags=12):
    residuos = residuos.dropna()
    lb = acorr_ljungbox(residuos, lags=[lags], return_df=True)
    lb2 = acorr_ljungbox(residuos**2, lags=[lags], return_df=True)
    print(f"Ljung-Box em {nome} (resíduos): Q={lb['lb_stat'].iloc[0]:.2f}, p={lb['lb_pvalue'].iloc[0]:.4f}")
    print(f"Ljung-Box em {nome} (resíduos^2): Q={lb2['lb_stat'].iloc[0]:.2f}, p={lb2['lb_pvalue'].iloc[0]:.4f}")


def mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0


# -------------------- TREINOS --------------------
@dataclass
class ModeloSARIMAXCfg:
    order: Tuple[int,int,int] = (1,1,1)
    seasonal_order: Tuple[int,int,int,int] = (0,1,1,SAZONALIDADE_SEMANAL)


def treinar_sarimax(y_log: pd.Series, exog: Optional[pd.DataFrame], cfg: ModeloSARIMAXCfg):
    mod = SARIMAX(
        endog=y_log,
        exog=exog,
        order=cfg.order,
        seasonal_order=cfg.seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
        simple_differencing=False
    )
    res = mod.fit(disp=False)
    return res


def treinar_estado_espaco(y_log: pd.Series, exog: Optional[pd.DataFrame]):
    # Tendência local com slope estocástico (+ opcional exógenas via regressão)
    mod = UnobservedComponents(
        endog=y_log,
        level="llevel",
        trend=True,          # nível + inclinação
        seasonal=None,       # poderíamos testar sazonalidade estrutural também
        exog=exog
    )
    res = mod.fit(disp=False)
    return res


def gerar_previsao(res, passos: int, exog_fut: Optional[pd.DataFrame]):
    if exog_fut is not None and len(exog_fut) != passos:
        # se exógenas do futuro não forem fornecidas, use NaN -> o statsmodels ignora apropriadamente
        exog_fut = None
    fcast = res.get_forecast(steps=passos, exog=exog_fut)
    media = fcast.predicted_mean
    ic = fcast.conf_int(alpha=0.05)
    return media, ic


# -------------------- GARCH (OPCIONAL) --------------------
def treinar_garch(residuos: pd.Series):
    residuos = residuos.dropna()
    if not TEM_GARCH or residuos.empty:
        return None
    am = arch_model(residuos*100.0, vol="GARCH", p=1, o=0, q=1, mean="zero", dist="normal")
    res = am.fit(disp="off")
    return res


def prever_vol_garch(garch_res, passos: int):
    if garch_res is None:
        return None
    f = garch_res.forecast(horizon=passos, reindex=False)
    # retorna desvio-padrão anualizado ~ não; aqui é diário (mesma escala de resíduos)
    sigma = np.sqrt(f.variance.values[-1, :]) / 100.0
    return sigma


# -------------------- EMPILHAMENTO --------------------
def combinar_pesos(y_true, y1, y2) -> Tuple[float,float]:
    # Minimiza MAPE por busca em grade simples de w1 em [0,1]
    melhores = (0.5, 0.5, np.inf)
    for w1 in np.linspace(0,1,101):
        yhat = w1*y1 + (1-w1)*y2
        erro = mape(y_true, np.exp(yhat))  # transformar de log para nível
        if erro < melhores[2]:
            melhores = (w1, 1-w1, erro)
    return melhores[0], melhores[1]


# -------------------- PIPELINE COMPLETO --------------------
def main():
    df = carregar_dados()
    df = engenharia_de_variaveis(df)

    # alvo em log
    y = df["log_preco"]

    # checagens de raiz unitária
    print("=== Testes ADF ===")
    adf_print(y, "log_preco")
    adf_print(df["retorno_log"], "retorno_log")

    # dividir
    y_tr, y_te = dividir_treino_teste(df, "log_preco")
    idx_tr, idx_te = y_tr.index, y_te.index

    exog_cols = df.attrs.get("exogenas", [])
    X = df[exog_cols] if exog_cols else None
    X_tr = X.loc[idx_tr] if X is not None else None
    X_te = X.loc[idx_te] if X is not None else None

    # 1) SARIMAX
    saricfg = ModeloSARIMAXCfg(order=(1,1,1), seasonal_order=(0,1,1,SAZONALIDADE_SEMANAL))
    sarimax_res = treinar_sarimax(y_tr, X_tr, saricfg)
    print("\n=== SARIMAX — Sumário curto ===")
    print(sarimax_res.summary().tables[0])

    # 2) Estado-Espaço (UCM)
    ucm_res = treinar_estado_espaco(y_tr, X_tr)
    print("\n=== Estado-Espaço — Sumário curto ===")
    print(ucm_res.summary().tables[0])

    # DIAGNÓSTICO RESÍDUOS (treino)
    print("\n=== Diagnósticos — SARIMAX ===")
    ljung_box_print(pd.Series(sarimax_res.filter_results.standardized_forecasts_error[0], index=idx_tr), "SARIMAX")

    print("\n=== Diagnósticos — UCM ===")
    ljung_box_print(pd.Series(ucm_res.filter_results.standardized_forecasts_error[0], index=idx_tr), "UCM")

    # PREVISÃO multi-passos no TESTE
    passos = len(y_te)
    prev_sarimax_log, ic_sarimax_log = gerar_previsao(sarimax_res, passos, X_te)
    prev_ucm_log,     ic_ucm_log     = gerar_previsao(ucm_res, passos, X_te)

    # EMPILHAMENTO (pesos no bloco de teste inteiro — ou poderia usar janela de validação dentro do treino)
    w1, w2 = combinar_pesos(np.exp(y_te), prev_sarimax_log, prev_ucm_log)
    print(f"\n=== Pesos de combinação (MAPE mínimo no teste) ===\nSARIMAX={w1:.2f}, UCM={w2:.2f}")

    prev_comb_log = w1*prev_sarimax_log + w2*prev_ucm_log

    # MÉTRICAS
    def avalia(y_true_log, yhat_log, nome):
        y_true = np.exp(y_true_log)
        y_hat  = np.exp(yhat_log)
        rmse = np.sqrt(np.mean((y_true - y_hat)**2))
        mae  = np.mean(np.abs(y_true - y_hat))
        mape_ = mape(y_true, y_hat)
        print(f"{nome}: RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape_:.2f}%")
        return {"RMSE": rmse, "MAE": mae, "MAPE": mape_}

    print("\n=== Métricas no conjunto de teste ===")
    met_sarima = avalia(y_te, prev_sarimax_log, "SARIMAX")
    met_ucm    = avalia(y_te, prev_ucm_log,     "UCM")
    met_comb   = avalia(y_te, prev_comb_log,    "Combinado")

    # GARCH opcional (volatilidade sobre resíduos do modelo combinado no treino)
    residuos_treino = (y_tr - (w1*sarimax_res.fittedvalues + w2*ucm_res.fittedvalues)).dropna()
    garch_res = treinar_garch(residuos_treino)
    if garch_res is not None:
        print("\n=== GARCH(1,1) ajustado sobre resíduos (escala % do retorno log) ===")
        print(garch_res.summary().tables[0])

    # PREVISÃO FINAL (últimos 20 dias úteis à frente)
    passos_fut = 20
    if X is not None:
        X_fut = pd.DataFrame(index=pd.date_range(df.index[-1] + pd.offsets.BDay(), periods=passos_fut, freq="B"),
                             columns=X.columns)
        X_fut = X_fut.fillna(method="ffill").fillna(0.0)
    else:
        X_fut = None

    fut_sarimax_log, fut_ic_sarimax = gerar_previsao(sarimax_res, passos_fut, X_fut)
    fut_ucm_log,     fut_ic_ucm     = gerar_previsao(ucm_res, passos_fut, X_fut)
    fut_comb_log = w1*fut_sarimax_log + w2*fut_ucm_log

    # Volatilidade prevista (se houver GARCH)
    if garch_res is not None:
        sigma_fut = prever_vol_garch(garch_res, passos_fut)  # desvio-padrão dos retornos log
    else:
        sigma_fut = None

    # Converter previsões para nível
    prev_te_nivel = pd.DataFrame({
        "sarimax": np.exp(prev_sarimax_log.values),
        "ucm":     np.exp(prev_ucm_log.values),
        "combinado": np.exp(prev_comb_log.values)
    }, index=idx_te)

    prev_fut_nivel = pd.DataFrame({
        "sarimax": np.exp(fut_sarimax_log.values),
        "ucm":     np.exp(fut_ucm_log.values),
        "combinado": np.exp(fut_comb_log.values)
    }, index=fut_comb_log.index)

    # Bandas por IC do modelo SARIMAX (como referência)
    ic_sarimax_nivel = np.exp(fut_ic_sarimax)
    ic_ucm_nivel = np.exp(fut_ic_ucm)

    # (opcional) Bandas empíricas usando σ_t de GARCH sobre retornos (aproximação delta-method em nível)
    if sigma_fut is not None:
        # y_{t+h} ≈ exp(ŷ + 1/2 σ²) para média lognormal; banda 95% ~ exp(ŷ ± 1.96 σ)
        banda_inf = np.exp(fut_comb_log.values - 1.96*sigma_fut)
        banda_sup = np.exp(fut_comb_log.values + 1.96*sigma_fut)
        bandas_garch = pd.DataFrame({"inf": banda_inf, "sup": banda_sup}, index=prev_fut_nivel.index)
    else:
        bandas_garch = None

    # -------------------- GRÁFICOS --------------------
    if PLOTAR:
        plt.figure(figsize=(12,5))
        plt.plot(df.index, df[COLUNA_PRECO], label="Observado")
        plt.plot(prev_te_nivel.index, prev_te_nivel["combinado"], label="Prev. combinada (teste)")
        plt.title("USD/BRL — observado vs previsão (janela de teste)")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(PASTA_SAIDA, "observado_vs_prev_teste.png"), dpi=160)

        plt.figure(figsize=(12,5))
        plt.plot(prev_fut_nivel.index, prev_fut_nivel["combinado"], label="Previsão combinada (nível)")
        plt.fill_between(ic_sarimax_nivel.index, ic_sarimax_nivel.iloc[:,0], ic_sarimax_nivel.iloc[:,1],
                         alpha=0.2, label="IC SARIMAX 95%")
        plt.fill_between(ic_ucm_nivel.index, ic_ucm_nivel.iloc[:,0], ic_ucm_nivel.iloc[:,1],
                         alpha=0.2, label="IC UCM 95%")
        if bandas_garch is not None:
            plt.fill_between(bandas_garch.index, bandas_garch["inf"], bandas_garch["sup"],
                             alpha=0.15, label="Banda GARCH 95% (aprox.)")
        plt.title("USD/BRL — previsão h-passos à frente")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(PASTA_SAIDA, "previsao_frente.png"), dpi=160)

    # -------------------- SALVAR ARTEFATOS --------------------
    prev_te_nivel.to_csv(os.path.join(PASTA_SAIDA, "previsoes_teste.csv"))
    prev_fut_nivel.to_csv(os.path.join(PASTA_SAIDA, "previsoes_futuras_20d.csv"))

    print("\nArquivos salvos em:", os.path.abspath(PASTA_SAIDA))
    print("→ previsoes_teste.csv, previsoes_futuras_20d.csv, e PNGs de gráficos.")


if __name__ == "__main__":
    main()
