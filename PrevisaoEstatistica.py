# ============================================================
# Previsão Avançada de Câmbio — REWRITE v2 (robusto)
# SARIMAX + Markov (robusto) + UCM + Stacking
#  - Corrige: freq explícita, FutureWarning 'Q'→'QE', fragmentação pandas,
#    e falha do Markov (SVD / steady-state) com *fallbacks* estáveis.
# Autor: Luiz Tiago Wilcke (LT)
# ============================================================

import os
import math
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

# Modelagem estatística (statsmodels)
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.tsa.statespace.structural import UnobservedComponents

# ============================
# 0) Configurações gerais
# ============================
@dataclass
class Config:
    usar_csv: bool = False
    caminho_csv: Optional[str] = None    # CSV com colunas: data, cambio, (exógenas opcionais)
    frequencia: str = "M"                # "D" diário, "M" mensal, "Q" trimestral
    coluna_data: str = "data"
    coluna_alvo: str = "cambio"
    colunas_exogenas: Optional[List[str]] = None  # ex.: ["ipca","selic","dxy","commodities"]
    prop_treino: float = 0.75
    horizonte_passos: int = 12
    usar_log: bool = False               # log do alvo (para estabilizar variância)
    normalizar_exog: bool = True         # padronizar exógenas (z-score)
    semente: int = 42
    dir_saida: str = "resultados_cambio"
    grid_sarimax: List[Tuple[int,int,int,int]] = None  # (p,d,q,s)
    maxiter_mle: int = 300

    def __post_init__(self):
        np.random.seed(self.semente)
        if self.grid_sarimax is None:
            saz = 12 if self.frequencia.upper()=="M" else (5 if self.frequencia.upper()=="D" else 4)
            self.grid_sarimax = [
                (1,0,1,saz), (1,1,1,saz), (2,0,1,saz), (2,1,1,saz), (0,1,1,saz)
            ]

cfg = Config(
    usar_csv=False,
    caminho_csv=None,
    frequencia="M",
    coluna_data="data",
    coluna_alvo="cambio",
    colunas_exogenas=["ipca","selic","dxy","commodities"],
    prop_treino=0.75,
    horizonte_passos=12,
    usar_log=False,
    normalizar_exog=True,
)

os.makedirs(cfg.dir_saida, exist_ok=True)

# ============================
# Helpers de frequência e dados
# ============================

def _freq_alias(freq: str) -> str:
    """Normaliza aliases para evitar FutureWarning (ex.: 'Q' -> 'QE')."""
    f = freq.upper()
    if f == "Q":
        return "QE"  # QuarterEnd
    return f  # 'M', 'D', etc.


def gerar_dados_sinteticos(n=180, freq="M", sem_exog=False, semente=42):
    np.random.seed(semente)
    freq = _freq_alias(freq)
    idx = pd.date_range("2010-01-01", periods=n, freq=freq)
    nivel = np.cumsum(np.random.normal(0, 0.02, size=n))
    saz = 0.5*np.sin(2*np.pi*np.arange(n)/12.0)
    regime = (np.random.rand(n) > 0.8).astype(float)
    choque_regime = np.cumsum(regime*np.random.normal(0, 0.05, size=n))
    cambio = 4.5 + nivel + 0.6*saz + 0.3*choque_regime + np.random.normal(0,0.05,size=n)
    df = pd.DataFrame({"data": idx, "cambio": cambio})
    if not sem_exog:
        ipca = 0.2*np.roll(cambio,1) + np.random.normal(0,0.05, size=n) + 0.2
        selic = 10 + 0.5*np.roll(cambio,3) + np.random.normal(0,0.2,size=n)
        dxy = 100 + 5*np.sin(2*np.pi*np.arange(n)/12.0) + np.random.normal(0,1.0,size=n)
        commodities = 80 + 0.7*np.roll(cambio,2) + np.random.normal(0,1.5,size=n)
        df = pd.concat([df,
                        pd.DataFrame({
                            "ipca": ipca,
                            "selic": selic,
                            "dxy": dxy,
                            "commodities": commodities
                        })], axis=1)
    return df


def carregar_dados(cfg: Config) -> pd.DataFrame:
    if cfg.usar_csv and cfg.caminho_csv and os.path.exists(cfg.caminho_csv):
        df = pd.read_csv(cfg.caminho_csv)
        assert cfg.coluna_data in df.columns and cfg.coluna_alvo in df.columns, \
            "CSV deve conter colunas 'data' e 'cambio' (ou ajuste cfg)."
        df[cfg.coluna_data] = pd.to_datetime(df[cfg.coluna_data])
        df = df.sort_values(cfg.coluna_data).reset_index(drop=True)
        return df
    else:
        return gerar_dados_sinteticos(n=180, freq=cfg.frequencia, sem_exog=False, semente=cfg.semente)


dados = carregar_dados(cfg)

# ============================
# 1) Engenharia de features (sem fragmentação)
# ============================

def engenharia_features(df_in: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df_in.copy()
    df[cfg.coluna_data] = pd.to_datetime(df[cfg.coluna_data])
    df = df.sort_values(cfg.coluna_data).reset_index(drop=True)
    df = df.set_index(cfg.coluna_data)

    # Batch-create de diferenças e retornos para todas as colunas numéricas
    cols_num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    diffs = {f"{c}_diff": df[c].diff() for c in cols_num}
    rets  = {f"{c}_ret": df[c].pct_change() for c in cols_num}

    # Sazonalidade trigonométrica (cria em lote para evitar fragmentação)
    mes = df.index.month
    saz = pd.DataFrame({
        "mes": mes,
        "sen": np.sin(2*np.pi*mes/12.0),
        "cos": np.cos(2*np.pi*mes/12.0)
    }, index=df.index)

    df_out = pd.concat([df, pd.DataFrame(diffs), pd.DataFrame(rets), saz], axis=1)
    return df_out.copy()  # de-fragmenta


df_all = engenharia_features(dados, cfg)

# ============================
# 2) Target/Exógenas + frequência consistente
# ============================

def preparar_matriz(df_all: pd.DataFrame, cfg: Config):
    y = df_all[[cfg.coluna_alvo]].copy()
    if cfg.usar_log:
        y = np.log(y)
        y.columns = ["alvo_log"]
    else:
        y.columns = ["alvo"]

    X = None
    if cfg.colunas_exogenas:
        exogs = [c for c in cfg.colunas_exogenas if c in df_all.columns]
        if exogs:
            X = df_all[exogs].copy()
            if cfg.normalizar_exog:
                X = (X - X.mean())/X.std(ddof=0)

    # Impõe frequência para evitar ValueWarning de statsmodels
    freq = _freq_alias(cfg.frequencia)
    y = y.asfreq(freq)
    if X is not None:
        X = X.asfreq(freq)

    # Drop NaNs iniciais vindos de diffs/retornos
    y = y.dropna()
    if X is not None:
        X = X.loc[y.index].dropna()
        y = y.loc[X.index]

    return y, X


y, X = preparar_matriz(df_all, cfg)

# Split temporal
n = len(y)
cut = int(n*cfg.prop_treino)
y_tr, y_te = y.iloc[:cut], y.iloc[cut:]
X_tr = X.iloc[:cut] if X is not None else None
X_te = X.iloc[cut:] if X is not None else None

# ============================
# 3) Ajuste dos modelos
# ============================

def ajustar_sarimax(y: pd.Series, X: Optional[pd.DataFrame], grid, maxiter=300):
    melhor_aic = np.inf
    melhor = None
    melhor_ordem = None
    for (p,d,q,s) in grid:
        try:
            saz = (p,d,q,s) if s>0 else (0,0,0,0)
            mod = SARIMAX(
                y,
                exog=X,
                order=(p,d,q),
                seasonal_order=saz,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            res = mod.fit(disp=False, maxiter=maxiter)
            if res.aic < melhor_aic:
                melhor_aic = res.aic
                melhor = res
                melhor_ordem = (p,d,q,s)
        except Exception:
            continue
    return melhor, melhor_ordem, melhor_aic


def ajustar_markov_robusto(y: pd.Series, k_regimes=2, order=1, maxiter=300):
    """Tenta MarkovAutoregression e faz *fallback* progressivo em caso de
    SVD/steady-state/EM failure: (1) MAR full → (2) MAR sem var. comutante →
    (3) MarkovRegression (sem AR) → (4) AR(1) simples como último recurso.
    Retorna (tipo_modelo, results_obj, dicionario_parametros_para_previsao).
    """
    ys = y.squeeze()

    # 1) MAR completo
    try:
        modelo = MarkovAutoregression(
            ys,
            k_regimes=k_regimes,
            order=order,
            trend='c',
            switching_ar=True,
            switching_variance=True
        )
        res = modelo.fit(disp=False, maxiter=maxiter, em_iter=10, search_reps=20)
        return "MAR", res, {}
    except Exception:
        pass

    # 2) MAR sem variância comutante (mais estável)
    try:
        modelo = MarkovAutoregression(
            ys,
            k_regimes=k_regimes,
            order=order,
            trend='c',
            switching_ar=True,
            switching_variance=False
        )
        res = modelo.fit(disp=False, maxiter=maxiter, em_iter=10, search_reps=20)
        return "MAR_novar", res, {}
    except Exception:
        pass

    # 3) MarkovRegression (sem AR explícito, média por regime)
    try:
        modelo = MarkovRegression(
            ys,
            k_regimes=k_regimes,
            trend='c',
            switching_variance=True
        )
        res = modelo.fit(disp=False, maxiter=maxiter, em_iter=10, search_reps=20)
        return "MR", res, {}
    except Exception:
        pass

    # 4) Último recurso: AR(1) simples (sem regime)
    # Usa SARIMAX(1,0,0) para extrair (c, phi)
    sar = SARIMAX(ys, order=(1,0,0), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    phi = float(sar.params.get('ar.L1', 0.8))
    c = float(sar.params.get('const', np.mean(ys)))
    return "AR1_fallback", sar, {"phi": phi, "c": c}


res_sarimax, ordem_sarimax, aic_sarimax = ajustar_sarimax(y_tr.squeeze(), X_tr, cfg.grid_sarimax, maxiter=cfg.maxiter_mle)
mar_tipo, res_mar, mar_params = ajustar_markov_robusto(y_tr.squeeze(), k_regimes=2, order=1, maxiter=cfg.maxiter_mle)
res_ucm = UnobservedComponents(y_tr.squeeze(), level='local level', trend=True, seasonal=(12 if _freq_alias(cfg.frequencia)=="M" else 0), autoregressive=1).fit(disp=False, maxiter=cfg.maxiter_mle)

# ============================
# 4) Previsões (one-step e multi-step)
# ============================

def previsoes_onestep_sarimax(res, y_te, X_te):
    preds = []
    for t in range(len(y_te)):
        ex = X_te.iloc[[t]] if X_te is not None else None
        fc = res.get_forecast(steps=1, exog=ex)
        preds.append(fc.predicted_mean.values[0])
    return pd.Series(preds, index=y_te.index)


def _extrai_c_phi_de_mar(res) -> Tuple[float,float]:
    # tenta extrair interceptos/AR(1); se não houver, devolve defaults estáveis
    phi, c = None, None
    try:
        if hasattr(res, 'params'):
            # tenta AR
            try:
                phi = float(np.mean(res.params.filter(like='ar.L1', axis=0).values))
            except Exception:
                phi = None
            # tenta interceptos
            try:
                cands = []
                for k in range(5):
                    try:
                        val = res.params.loc[f'regime{k}.intercept']
                        cands.append(val)
                    except Exception:
                        pass
                if cands:
                    c = float(np.mean(cands))
            except Exception:
                c = None
    except Exception:
        pass
    if phi is None:
        phi = 0.8
    if c is None:
        c = 0.0
    return c, phi


def previsoes_onestep_mar(mar_tipo: str, res, y_tr, y_te, mar_params):
    # Estratégia robusta: se houver predict do statsmodels, usa; senão, usa AR(1) aproximado.
    try:
        pred = res.predict(start=len(y_tr), end=len(y_tr)+len(y_te)-1)
        # garante índice
        return pd.Series(np.asarray(pred), index=y_te.index)
    except Exception:
        pass
    # fallback AR(1)
    if mar_tipo == "AR1_fallback" and {"phi","c"}.issubset(mar_params.keys()):
        phi = mar_params["phi"]; c = mar_params["c"]
    else:
        c, phi = _extrai_c_phi_de_mar(res)
    preds = []
    y_hist = y_tr.squeeze().values
    for _ in range(len(y_te)):
        y_prev = c + phi * (y_hist[-1] if len(y_hist)>0 else y_tr.squeeze().iloc[-1])
        preds.append(float(y_prev))
        y_hist = np.append(y_hist, y_prev)  # one-step puro
    return pd.Series(preds, index=y_te.index)


def previsoes_multistep_sarimax(res, y_te, X_te, horizonte, freq):
    ex_fut = X_te.iloc[:horizonte] if (X_te is not None and len(X_te)>=horizonte) else None
    fc = res.get_forecast(steps=horizonte, exog=ex_fut)
    idx_fut = pd.date_range(y_te.index[-1], periods=horizonte+1, freq=_freq_alias(freq))[1:]
    return pd.Series(fc.predicted_mean, index=idx_fut)


def previsoes_multistep_mar(mar_tipo, res, y_tr, y_te, horizonte, freq, mar_params):
    # tenta predict se disponível
    try:
        pred = res.predict(start=len(y_tr)+len(y_te), end=len(y_tr)+len(y_te)+horizonte-1)
        idx_fut = pd.date_range(y_te.index[-1], periods=horizonte+1, freq=_freq_alias(freq))[1:]
        return pd.Series(np.asarray(pred), index=idx_fut)
    except Exception:
        pass
    # fallback AR(1)
    if mar_tipo == "AR1_fallback" and {"phi","c"}.issubset(mar_params.keys()):
        phi = mar_params["phi"]; c = mar_params["c"]
    else:
        c, phi = _extrai_c_phi_de_mar(res)
    preds = []
    base = y_te.squeeze().iloc[-1]
    for _ in range(horizonte):
        base = c + phi * base
        preds.append(float(base))
    idx_fut = pd.date_range(y_te.index[-1], periods=horizonte+1, freq=_freq_alias(cfg.frequencia))[1:]
    return pd.Series(preds, index=idx_fut)


def previsoes_onestep_ucm(res, y_te):
    preds = []
    for _ in range(len(y_te)):
        fc = res.get_forecast(steps=1)
        preds.append(fc.predicted_mean.values[0])
    return pd.Series(preds, index=y_te.index)


def previsoes_multistep_ucm(res, y_te, horizonte, freq):
    fc = res.get_forecast(steps=horizonte)
    idx_fut = pd.date_range(y_te.index[-1], periods=horizonte+1, freq=_freq_alias(freq))[1:]
    return pd.Series(fc.predicted_mean, index=idx_fut)


# One-step
prev_sarimax_1s = previsoes_onestep_sarimax(res_sarimax, y_te, X_te)
prev_mar_1s     = previsoes_onestep_mar(mar_tipo, res_mar, y_tr, y_te, mar_params)
prev_ucm_1s     = previsoes_onestep_ucm(res_ucm, y_te)

# Multi-step futuro
prev_sarimax_ms = previsoes_multistep_sarimax(res_sarimax, y_te, X_te, cfg.horizonte_passos, cfg.frequencia)
prev_mar_ms     = previsoes_multistep_mar(mar_tipo, res_mar, y_tr, y_te, cfg.horizonte_passos, cfg.frequencia, mar_params)
prev_ucm_ms     = previsoes_multistep_ucm(res_ucm, y_te, cfg.horizonte_passos, cfg.frequencia)

# ============================
# 5) Stacking com restrições (>=0, soma=1)
# ============================

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true)-np.asarray(y_pred))**2)))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true)-np.asarray(y_pred))))

def mape(y_true, y_pred):
    yt = np.asarray(y_true); yp = np.asarray(y_pred)
    return float(np.mean(np.abs((yt-yp)/(np.abs(yt)+1e-8)))*100)


def otimizar_pesos(y_true: pd.Series, candidatos: Dict[str, pd.Series], passo=0.02):
    nomes = list(candidatos.keys())
    Y = y_true.reindex(candidatos[nomes[0]].index)
    melhor_rmse = np.inf
    melhor_w = None
    for w1 in np.arange(0,1+1e-12, passo):
        for w2 in np.arange(0,1-w1+1e-12, passo):
            w3 = 1.0 - w1 - w2
            comb = w1*candidatos[nomes[0]] + w2*candidatos[nomes[1]] + w3*candidatos[nomes[2]]
            r = rmse(Y.values, comb.values)
            if r < melhor_rmse:
                melhor_rmse = r
                melhor_w = (w1,w2,w3)
    return melhor_w, melhor_rmse, nomes


cands_test = {"sarimax": prev_sarimax_1s, "mar": prev_mar_1s, "ucm": prev_ucm_1s}
pesos, rmse_stack, ordem = otimizar_pesos(y_te.squeeze(), cands_test, passo=0.02)
ws, wm, wu = pesos
print(f"[Stacking] Pesos: SARIMAX={ws:.2f}, MAR={wm:.2f}, UCM={wu:.2f} | RMSE={rmse_stack:.6f}")

# Multi-step empilhado
pred_ms_dict = {"sarimax": prev_sarimax_ms, "mar": prev_mar_ms, "ucm": prev_ucm_ms}
pred_stack_ms = ws*pred_ms_dict["sarimax"] + wm*pred_ms_dict["mar"] + wu*pred_ms_dict["ucm"]

# ============================
# 6) Métricas e IC empíricos
# ============================
metricas = {}
for nome, prev in cands_test.items():
    metricas[nome] = {"RMSE": rmse(y_te.squeeze(), prev), "MAE": mae(y_te.squeeze(), prev), "MAPE%": mape(y_te.squeeze(), prev)}
metricas["stack"] = {
    "RMSE": rmse(y_te.squeeze(), (ws*cands_test["sarimax"] + wm*cands_test["mar"] + wu*cands_test["ucm"])),
    "MAE":  mae(y_te.squeeze(), (ws*cands_test["sarimax"] + wm*cands_test["mar"] + wu*cands_test["ucm"])),
    "MAPE%": mape(y_te.squeeze(), (ws*cands_test["sarimax"] + wm*cands_test["mar"] + wu*cands_test["ucm"]))
}

df_metricas = pd.DataFrame(metricas).T.round(6)
df_metricas.to_csv(os.path.join(cfg.dir_saida, "metricas_teste.csv"))
print("\nMétricas em teste:\n", df_metricas)

# PI empírico a partir do resíduo do stack
residuos = y_te.squeeze() - (ws*cands_test["sarimax"] + wm*cands_test["mar"] + wu*cands_test["ucm"])
resid_std = residuos.std(ddof=1)
ic95 = 1.96*resid_std
pi_inf = pred_stack_ms - ic95
pi_sup = pred_stack_ms + ic95

# ============================
# 7) Gráficos
# ============================
plt.figure(figsize=(11,5))
plt.plot(y_tr.index, y_tr.squeeze(), label="Treino", linewidth=1.8)
plt.plot(y_te.index, y_te.squeeze(), label="Teste (real)", linewidth=1.8)
plt.plot(prev_sarimax_1s.index, prev_sarimax_1s.values, label="SARIMAX 1-step", alpha=0.85)
plt.plot(prev_mar_1s.index, prev_mar_1s.values, label=f"{mar_tipo} 1-step", alpha=0.85)
plt.plot(prev_ucm_1s.index, prev_ucm_1s.values, label="UCM 1-step", alpha=0.85)
plt.plot(prev_ucm_1s.index, (ws*prev_sarimax_1s + wm*prev_mar_1s + wu*prev_ucm_1s), label="Stack (1-step)", linewidth=2.4)
plt.title("Câmbio — Ajuste em Teste (One-step Ahead)")
plt.legend(); plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(cfg.dir_saida, "ajuste_teste_onestep.png"), dpi=160)

plt.figure(figsize=(11,5))
plt.plot(y.index, y.squeeze(), label="Histórico", linewidth=1.8)
plt.plot(pred_stack_ms.index, pred_stack_ms.values, label="Stack — Previsão Multi-step", linewidth=2.2)
plt.fill_between(pred_stack_ms.index, pi_inf.values, pi_sup.values, alpha=0.2, label="PI ~95% (empírico)")
plt.title(f"Câmbio — Previsão {cfg.horizonte_passos} passos")
plt.legend(); plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(cfg.dir_saida, "previsao_multistep.png"), dpi=160)

plt.figure(figsize=(11,4))
plt.plot(residuos.index, residuos.values, label="Resíduo (Stack)")
plt.axhline(0, color="k", linewidth=1)
plt.title("Resíduos em Teste — Stack")
plt.legend(); plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(cfg.dir_saida, "residuos_stack.png"), dpi=160)

# ============================
# 8) Saídas
# ============================

df_teste = pd.DataFrame({
    "real": y_te.squeeze(),
    "sarimax": prev_sarimax_1s,
    "mar": prev_mar_1s,
    "ucm": prev_ucm_1s,
    "stack": (ws*prev_sarimax_1s + wm*prev_mar_1s + wu*prev_ucm_1s)
})
df_teste.to_csv(os.path.join(cfg.dir_saida, "previsoes_teste_onestep.csv"))

# Multi-step futuro
pd.DataFrame({
    "stack_pred": pred_stack_ms,
    "pi_inf": pi_inf,
    "pi_sup": pi_sup,
}).to_csv(os.path.join(cfg.dir_saida, "previsoes_multistep_futuro.csv"))

with open(os.path.join(cfg.dir_saida, "resumo_modelos.txt"), "w") as f:
    f.write(f"Ordem SARIMAX escolhida (p,d,q,s): {ordem_sarimax} | AIC: {aic_sarimax:.3f}\n")
    f.write(f"Modelo de regime usado: {mar_tipo}\n")
    f.write(f"Pesos do Stacking (teste): SARIMAX={ws:.3f}, MAR={wm:.3f}, UCM={wu:.3f}\n")
    f.write("Métricas em teste:\n")
    f.write(df_metricas.to_string())

print("\nArquivos gerados em:", os.path.abspath(cfg.dir_saida))

# ============================
# 9) PATCH rápido p/ taxaInflacao.py (evitar fragmentação)
# ============================
# Substituir inserções repetidas por batch-assign:
#
# cols_num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
# diffs = {f"{c}_diff": df[c].diff() for c in cols_num}
# rets  = {f"{c}_ret": df[c].pct_change() for c in cols_num}
# saz = pd.DataFrame({
#     "mes": df.index.month,
#     "sen": np.sin(2*np.pi*df.index.month/12.0),
#     "cos": np.cos(2*np.pi*df.index.month/12.0)
# }, index=df.index)
# df = pd.concat([df, pd.DataFrame(diffs), pd.DataFrame(rets), saz], axis=1).copy()
