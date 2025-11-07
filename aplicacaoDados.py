# =============================================================================
#  Plataforma DS Varejo — Forecast + Qualidade + Monitoramento (Streamlit)
#  Autor: Luiz Tiago Wilcke (LT)
# -----------------------------------------------------------------------------
#  Recursos:
#   - Ingestão: upload CSV (fallback: dados sintéticos realistas)
#   - Qualidade: checagens (completude, duplicatas, faixas), winsorização opcional
#   - Feature store rápida: calendário, lags, médias móveis, preço relativo, promoções
#   - Pipeline sklearn (numérico + categórico) com ColumnTransformer
#   - Modelos: GradientBoostingRegressor e ElasticNet (escolha no menu)
#   - Métricas: MAE, RMSE, MAPE, R² + gráficos (real vs previsto, resíduos)
#   - Explicabilidade: importância por permutação + Partial Dependence (PDP)
#   - Monitoramento: PSI/KS para drift + alerta simples
#   - Governança: “registro” local do modelo (JSON), versão, hash do dataset,
#                 hiperparâmetros, métricas e data/hora
#   - Artefatos: salvos em ./artefatos (modelo.pkl, preproc.pkl, registro.json, baseline.parquet)
# =============================================================================

import os, io, json, hashlib, warnings, datetime as dt
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

import joblib

warnings.filterwarnings("ignore")

# ------------------------ Configuração / Diretórios ---------------------------
ARTEFATOS_DIR = "artefatos"
os.makedirs(ARTEFATOS_DIR, exist_ok=True)

@dataclass
class Configuracao:
    alvo: str = "vendas"
    data_col: str = "data"
    id_loja: str = "loja"
    id_produto: str = "produto"
    colunas_categoricas: Tuple[str, ...] = ("loja", "produto", "promo", "dia_semana", "mes")
    colunas_numericas: Tuple[str, ...] = (
        "preco", "preco_relativo", "estoque", "lag_7", "lag_14", "mm_7", "mm_14",
        "sazonal_mes", "feriado"
    )
    winsorizar: bool = True
    winsor_quantis: Tuple[float, float] = (0.01, 0.99)
    modelo: str = "GradientBoosting"  # ou "ElasticNet"

CFG = Configuracao()

# ----------------------------- Utilidades ------------------------------------
def hash_dataframe(df: pd.DataFrame) -> str:
    # hash estável do conteúdo para governança
    b = pd.util.hash_pandas_object(df.sort_index(axis=1), index=True).values.tobytes()
    return hashlib.sha256(b).hexdigest()[:16]

def gerar_dados_sinteticos(n_lojas=10, n_produtos=30, dias=540, semente=42) -> pd.DataFrame:
    rng = np.random.default_rng(semente)
    datas = pd.date_range("2022-01-01", periods=dias, freq="D")
    linhas = []
    for loja in range(1, n_lojas+1):
        for prod in range(1, n_produtos+1):
            base = 50 + 15*np.sin(2*np.pi*(np.arange(dias)/7)) + 10*np.sin(2*np.pi*(np.arange(dias)/365))
            preco = rng.normal(20, 3, size=dias) * (1 + 0.05*(prod%5))
            promo = rng.choice([0,1], size=dias, p=[0.85,0.15])
            choque = rng.normal(0, 5, size=dias)
            vendas = np.maximum(0,
                base + 5*promo - 0.7*preco + choque + rng.poisson(2, size=dias)
            )
            estoque = np.maximum(0, rng.normal(300, 60, size=dias) - vendas*0.5)
            linhas.extend([{
                "data": datas[i],
                "loja": f"L{loja:02d}",
                "produto": f"P{prod:03d}",
                "preco": float(preco[i]),
                "promo": int(promo[i]),
                "estoque": float(estoque[i]),
                "vendas": float(vendas[i])
            } for i in range(dias)])
    df = pd.DataFrame(linhas)
    return df

def validar_dados(df: pd.DataFrame, cfg: Configuracao) -> Dict[str, Dict[str, float]]:
    rel = {}

    # completude
    faltantes = df.isna().mean().to_dict()
    rel["completude"] = {k: float(v) for k,v in faltantes.items()}

    # duplicatas (chave: data+loja+produto)
    chave = [cfg.data_col, cfg.id_loja, cfg.id_produto]
    rel["duplicatas_pct"] = {
        "pct": float(100 * (df.duplicated(chave).mean() if set(chave).issubset(df.columns) else 0.0))
    }

    # faixas simples para variáveis críticas
    faixas = {}
    for c in ["preco", "estoque", "vendas"]:
        if c in df.columns:
            s = df[c].dropna()
            faixas[c] = {"min": float(s.min()), "p01": float(s.quantile(0.01)),
                         "p99": float(s.quantile(0.99)), "max": float(s.max())}
    rel["faixas"] = faixas

    return rel

def winsorizar(df: pd.DataFrame, cols: List[str], quantis=(0.01, 0.99)) -> pd.DataFrame:
    dfx = df.copy()
    q1, q2 = quantis
    for c in cols:
        if c in dfx.columns:
            a, b = dfx[c].quantile(q1), dfx[c].quantile(q2)
            dfx[c] = dfx[c].clip(lower=a, upper=b)
    return dfx

def engenharia_variaveis(df: pd.DataFrame, cfg: Configuracao) -> pd.DataFrame:
    dfx = df.copy()
    # calendário
    dfx["data"] = pd.to_datetime(dfx[cfg.data_col])
    dfx["dia_semana"] = dfx["data"].dt.dayofweek.astype("category")
    dfx["mes"] = dfx["data"].dt.month.astype("category")
    dfx["sazonal_mes"] = np.sin(2*np.pi*(dfx["data"].dt.dayofyear/365.0))

    # feriados simples (ex.: maiores picos por mês)
    dfx["feriado"] = dfx["mes"].isin([1,5,9,12]).astype(int)

    # preço relativo por produto (desvio do preço médio do produto)
    if {"produto","preco"}.issubset(dfx.columns):
        preco_med = dfx.groupby("produto")["preco"].transform("mean")
        dfx["preco_relativo"] = (dfx["preco"] - preco_med) / (preco_med + 1e-6)
    else:
        dfx["preco_relativo"] = 0.0

    # lags e médias móveis por loja-produto
    dfx = dfx.sort_values(["loja","produto","data"])
    for k in [7,14]:
        dfx[f"lag_{k}"] = dfx.groupby(["loja","produto"])["vendas"].shift(k)
        dfx[f"mm_{k}"]  = dfx.groupby(["loja","produto"])["vendas"].shift(1).rolling(k).mean().reset_index(0,drop=True)
    dfx = dfx.dropna(subset=["lag_7","lag_14","mm_7","mm_14"])

    if CFG.winsorizar:
        dfx = winsorizar(dfx, ["preco","estoque","vendas","lag_7","lag_14","mm_7","mm_14"], CFG.winsor_quantis)

    return dfx

def preparar_conjuntos(dfx: pd.DataFrame, cfg: Configuracao, proporcao_treino=0.8):
    dfx = dfx.sort_values("data")
    corte = int(len(dfx)*proporcao_treino)
    treino, teste = dfx.iloc[:corte].copy(), dfx.iloc[corte:].copy()

    X_cols = list(cfg.colunas_categoricas) + list(cfg.colunas_numericas)
    X_train, y_train = treino[X_cols], treino[cfg.alvo]
    X_test, y_test   = teste[X_cols],  teste[cfg.alvo]

    cat_cols = [c for c in cfg.colunas_categoricas if c in X_train.columns]
    num_cols = [c for c in cfg.colunas_numericas  if c in X_train.columns]

    preproc = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), cat_cols)
    ])
    return X_train, y_train, X_test, y_test, preproc

def criar_modelo(nome: str, params: Dict):
    if nome == "GradientBoosting":
        return GradientBoostingRegressor(**params)
    elif nome == "ElasticNet":
        return ElasticNet(**params, max_iter=10000)
    else:
        raise ValueError("Modelo não suportado.")

def avaliar(y_true, y_pred) -> Dict[str, float]:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100)
    r2   = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "MAPE%": mape, "R2": r2}

def calcular_psi(a: np.ndarray, b: np.ndarray, bins=10) -> float:
    # Population Stability Index — simples
    qa = np.quantile(a, np.linspace(0,1,bins+1))
    qa[0], qa[-1] = -np.inf, np.inf
    ca, _ = np.histogram(a, bins=qa); cb, _ = np.histogram(b, bins=qa)
    pa = ca/ca.sum(); pb = cb/cb.sum()
    pa, pb = np.clip(pa, 1e-6, None), np.clip(pb, 1e-6, None)
    return float(np.sum((pb - pa)*np.log(pb/pa)))

def monitorar_drift(baseline: pd.DataFrame, atual: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    linhas = []
    for c in cols:
        if c not in baseline.columns or c not in atual.columns: 
            continue
        a, b = pd.to_numeric(baseline[c], errors="coerce").dropna(), pd.to_numeric(atual[c], errors="coerce").dropna()
        if len(a) < 50 or len(b) < 50: 
            continue
        psi = calcular_psi(a.values, b.values, bins=10)
        ks  = stats.ks_2samp(a.values, b.values).statistic
        linhas.append({"variavel": c, "PSI": psi, "KS": float(ks),
                       "alerta": "ALTO" if psi>0.25 or ks>0.2 else ("MÉDIO" if psi>0.1 or ks>0.1 else "OK")})
    return pd.DataFrame(linhas).sort_values(["alerta","PSI"], ascending=[True, False])

def salvar_registro(info: Dict, caminho=os.path.join(ARTEFATOS_DIR, "registro.json")):
    if os.path.exists(caminho):
        antigo = json.load(open(caminho,"r"))
    else:
        antigo = {"modelos": []}
    antigo["modelos"].append(info)
    with open(caminho, "w") as f:
        json.dump(antigo, f, indent=2, ensure_ascii=False)

# ------------------------------- UI -----------------------------------------
st.set_page_config(page_title="Plataforma DS Varejo — LT", layout="wide")
st.title("🛒 Plataforma DS Varejo — Forecast + Qualidade + Monitoramento (LT)")

with st.sidebar:
    st.header("⚙️ Configurações")
    modelo_escolhido = st.selectbox("Modelo", ["GradientBoosting", "ElasticNet"], index=0)
    if modelo_escolhido == "GradientBoosting":
        p_n_est   = st.slider("n_estimators", 50, 800, 300, step=25)
        p_lr      = st.slider("learning_rate", 0.01, 0.5, 0.05, step=0.01)
        p_depth   = st.slider("max_depth", 2, 6, 3, step=1)
        params_modelo = {"n_estimators": p_n_est, "learning_rate": p_lr, "max_depth": p_depth}
    else:
        p_alpha   = st.slider("alpha (L1+L2)", 0.0001, 1.0, 0.01, step=0.01)
        p_l1ratio = st.slider("l1_ratio", 0.0, 1.0, 0.5, step=0.05)
        params_modelo = {"alpha": p_alpha, "l1_ratio": p_l1ratio}

    prop_treino = st.slider("Proporção treino", 0.5, 0.95, 0.8, step=0.05)

    st.markdown("---")
    st.caption("📥 Envie um CSV com colunas: data, loja, produto, preco, promo, estoque, vendas")
    arquivo = st.file_uploader("CSV (opcional)", type=["csv"])

# --------------------------- Ingestão & Qualidade ---------------------------
if arquivo is not None:
    df_raw = pd.read_csv(arquivo)
    st.success("CSV carregado!")
else:
    df_raw = gerar_dados_sinteticos()
    st.info("Sem CSV — usando dados sintéticos realistas.")

rel_qualidade = validar_dados(df_raw, CFG)
aba1, aba2, aba3, aba4, aba5 = st.tabs(["📊 Dados", "🧪 Qualidade", "🤖 Modelagem", "🔍 Explicabilidade", "📈 Monitoramento"])

with aba1:
    st.subheader("Amostra de dados")
    st.dataframe(df_raw.head(20), use_container_width=True)
    fig = px.line(df_raw.groupby("data", as_index=False)["vendas"].sum(), x="data", y="vendas",
                  title="Série agregada de vendas (todas as lojas/produtos)")
    st.plotly_chart(fig, use_container_width=True)

with aba2:
    st.subheader("Relatório de qualidade (alto nível)")
    st.json(rel_qualidade)
    if st.checkbox("Aplicar winsorização suave (1%—99%)", value=CFG.winsorizar):
        CFG.winsorizar = True
    else:
        CFG.winsorizar = False

# ----------------------- Engenharia de Variáveis ----------------------------
df_feat = engenharia_variaveis(df_raw, CFG)
X_train, y_train, X_test, y_test, preproc = preparar_conjuntos(df_feat, CFG, prop_treino)

# ---------------------------- Treino / Avaliação ----------------------------
with aba3:
    st.subheader("Treino e avaliação")
    modelo = criar_modelo(modelo_escolhido, params_modelo)
    pipe = Pipeline([("prep", preproc), ("mdl", modelo)])
    pipe.fit(X_train, y_train)

    yhat_tr = pipe.predict(X_train)
    yhat_ts = pipe.predict(X_test)
    m_tr = avaliar(y_train, yhat_tr)
    m_ts = avaliar(y_test,  yhat_ts)

    colA, colB = st.columns(2)
    with colA: st.metric("Treino — RMSE", f"{m_tr['RMSE']:.2f}")
    with colB: st.metric("Teste  — RMSE", f"{m_ts['RMSE']:.2f}")
    st.write("Métricas detalhadas (Teste):", m_ts)

    # gráficos
    fig1 = px.scatter(x=y_test, y=yhat_ts, labels={"x":"Real", "y":"Previsto"},
                      title="Real vs Previsto (Teste)", trendline="ols")
    st.plotly_chart(fig1, use_container_width=True)

    resid = y_test - yhat_ts
    fig2 = px.histogram(resid, nbins=60, title="Distribuição de resíduos (Teste)")
    st.plotly_chart(fig2, use_container_width=True)

    # Salvar artefatos + governo
    if st.button("💾 Salvar modelo e artefatos"):
        versao = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        caminho_modelo = os.path.join(ARTEFATOS_DIR, f"modelo_{versao}.pkl")
        caminho_pre    = os.path.join(ARTEFATOS_DIR, f"preproc_{versao}.pkl")
        joblib.dump(pipe, caminho_modelo)
        joblib.dump(preproc, caminho_pre)

        # baseline para monitoramento
        baseline = X_test.copy()
        baseline[CFG.alvo] = y_test.values
        baseline.to_parquet(os.path.join(ARTEFATOS_DIR, f"baseline_{versao}.parquet"), index=False)

        registro = {
            "versao": versao,
            "modelo": modelo_escolhido,
            "hiperparametros": params_modelo,
            "metricas_teste": m_ts,
            "linhas_treino": int(len(X_train)),
            "linhas_teste": int(len(X_test)),
            "hash_dataset": hash_dataframe(df_raw),
            "data_registro": dt.datetime.now().isoformat()
        }
        salvar_registro(registro)
        st.success(f"Artefatos salvos! Versão: {versao}")

# ---------------------------- Explicabilidade -------------------------------
with aba4:
    st.subheader("Importância por permutação (Teste)")
    try:
        imp = permutation_importance(pipe, X_test, y_test, n_repeats=10, random_state=42)
        feat_names = []
        # tentar recuperar nomes expandidos (OneHot)
        try:
            oh = pipe.named_steps["prep"].named_transformers_["cat"].named_steps["oh"]
            cat_cols = pipe.named_steps["prep"].transformers_[1][2]
            num_cols = pipe.named_steps["prep"].transformers_[0][2]
            feat_names = list(num_cols) + list(oh.get_feature_names_out(cat_cols))
        except Exception:
            feat_names = list(X_test.columns)

        imp_df = pd.DataFrame({"variavel": feat_names, "importancia": imp.importances_mean})
        imp_df = imp_df.sort_values("importancia", ascending=False).head(25)
        st.dataframe(imp_df, use_container_width=True)
        st.plotly_chart(px.bar(imp_df, x="variavel", y="importancia", title="Top 25 — Importância (Permutação)"),
                        use_container_width=True)
    except Exception as e:
        st.warning(f"Falha na importância por permutação: {e}")

    st.subheader("Partial Dependence (PDP)")
    var_pdp = st.selectbox("Variável numérica para PDP", [c for c in CFG.colunas_numericas if c in X_test.columns])
    if st.button("Gerar PDP"):
        fig, ax = plt.subplots(figsize=(7,4))
        PartialDependenceDisplay.from_estimator(pipe, X_test, [var_pdp], ax=ax)
        st.pyplot(fig)

# ----------------------------- Monitoramento --------------------------------
with aba5:
    st.subheader("Monitoramento de Drift")
    # baseline mais recente
    arquivos_base = sorted([f for f in os.listdir(ARTEFATOS_DIR) if f.startswith("baseline_")])
    if arquivos_base:
        arquivo_baseline = os.path.join(ARTEFATOS_DIR, arquivos_base[-1])
        baseline = pd.read_parquet(arquivo_baseline)
        st.caption(f"Baseline: {os.path.basename(arquivo_baseline)} — {len(baseline)} linhas")
    else:
        st.info("Nenhum baseline salvo ainda. Salve o modelo na aba de Modelagem.")
        baseline = None

    st.markdown("### Lote atual (para comparação)")
    st.caption("Envie um CSV *novo* (mesmas colunas) para comparar com o baseline. Sem CSV, usa o bloco de teste atual.")
    arquivo_atual = st.file_uploader("CSV lote atual (opcional)", type=["csv"], key="drift")
    if arquivo_atual is not None:
        df_atual_raw = pd.read_csv(arquivo_atual)
        df_atual = engenharia_variaveis(df_atual_raw, CFG)
    else:
        df_atual = df_feat.iloc[-len(X_test):].copy()

    if baseline is not None:
        colunas_avaliar = [c for c in CFG.colunas_numericas if c in baseline.columns and c in df_atual.columns]
        rel = monitorar_drift(baseline, df_atual, colunas_avaliar)
        st.dataframe(rel, use_container_width=True)
        if (rel["alerta"]=="ALTO").any():
            st.error("⚠️ Drift ALTO detectado em pelo menos uma variável — reavaliar o modelo!")
        elif (rel["alerta"]=="MÉDIO").any():
            st.warning("🔶 Drift MÉDIO — acompanhar de perto.")
        else:
            st.success("✅ Sem sinais relevantes de drift.")
    else:
        st.info("Carregue um baseline primeiro (salvando um modelo).")

# --------------------------- Rodapé / Pitch ---------------------------------
st.markdown

