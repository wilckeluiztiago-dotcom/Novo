# ============================================================
# Previsão do Preço da Gasolina (Brasil) — RN com 20 neurônios
# Autor: Luiz Tiago Wilcke (LT)
# ============================================================

# -------------------- IMPORTAÇÕES --------------------
import os
import math
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Tuple, Dict, List

# Estatística / modelagem
from statsmodels.tsa.seasonal import STL
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Keras (TensorFlow)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Para salvar artefatos
from pathlib import Path

# -------------------- CONFIGURAÇÕES --------------------
np.random.seed(42)
tf.random.set_seed(42)

# aponte seu CSV real aqui (colunas mínimas: data, preco_gasolina)
# opcional: etanol, diesel, ipca, cambio, brent, salario_minimo, etc.
CAMINHO_CSV = None  # ex: "serie_gasolina_brasil.csv"
COLUNA_DATA = "data"
COLUNA_ALVO = "preco_gasolina"

PASTA_SAIDA = Path("./modelo_gasolina")
PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

# Hiperparâmetros principais
N_SPLITS_BACKTEST = 4
TAM_VALIDACAO_PROPORCAO = 0.2  # usado só se não for backtesting k-fold
EPOCHS = 400
BATCH_SIZE = 64
TAXA_APRENDIZADO = 1e-3
PATIENCE = 40

# Ativar Dropout Monte Carlo (para incerteza)
N_AMOSTRAS_MC = 100  # quantas amostras ao inferir com dropout ligado

# -------------------- UTILITÁRIOS --------------------
def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100.0

def criar_dados_sinteticos(n=1200, freq="W"):
    """Gera série sintética parecida com gasolina, com tendência, sazonalidade e choques."""
    idx = pd.date_range("2010-01-03", periods=n, freq=freq)
    tendencia = np.linspace(3.0, 7.5, n)  # aumento de longo prazo
    sazonal = 0.25*np.sin(2*np.pi*np.arange(n)/52) + 0.1*np.cos(2*np.pi*np.arange(n)/26)
    choques = np.random.normal(0, 0.1, n)
    preco = tendencia + sazonal + choques
    # variáveis exógenas correlacionadas
    brent = 60 + 12*np.sin(2*np.pi*np.arange(n)/52) + np.random.normal(0, 2.0, n)
    cambio = 3.5 + 0.5*np.sin(2*np.pi*np.arange(n)/104) + np.random.normal(0, 0.1, n)
    etanol = 2.5 + 0.2*np.sin(2*np.pi*np.arange(n)/52 + 1.0) + np.random.normal(0, 0.05, n)
    diesel = 3.2 + 0.15*np.sin(2*np.pi*np.arange(n)/52 - 0.5) + np.random.normal(0, 0.08, n)
    ipca = np.clip(np.random.normal(0.4, 0.1, n), 0.1, 0.8)  # inflação mensal aprox. semanalizada
    df = pd.DataFrame({
        COLUNA_DATA: idx,
        COLUNA_ALVO: preco,
        "brent": brent,
        "cambio": cambio,
        "etanol": etanol,
        "diesel": diesel,
        "ipca": ipca
    })
    return df

def carregar_base():
    if CAMINHO_CSV and os.path.exists(CAMINHO_CSV):
        df = pd.read_csv(CAMINHO_CSV)
        # normalizar nomes (opcional)
        df.columns = [c.strip().lower() for c in df.columns]
        df[COLUNA_DATA] = pd.to_datetime(df[COLUNA_DATA])
        df = df.sort_values(COLUNA_DATA).reset_index(drop=True)
        return df
    else:
        print(">> Usando base sintética (defina CAMINHO_CSV se tiver dados reais).")
        return criar_dados_sinteticos()

def adicionar_termos_fourier(df: pd.DataFrame, periodo: int, K: int, col_data: str) -> pd.DataFrame:
    df = df.copy()
    t = np.arange(len(df))
    for k in range(1, K+1):
        df[f"fourier_sin_{periodo}_{k}"] = np.sin(2*np.pi*k*t/periodo)
        df[f"fourier_cos_{periodo}_{k}"] = np.cos(2*np.pi*k*t/periodo)
    return df

def engenhar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(COLUNA_DATA).reset_index(drop=True)

    # Defasagens do alvo
    for lag in [1, 2, 3, 4, 8, 12, 26, 52]:
        df[f"{COLUNA_ALVO}_lag{lag}"] = df[COLUNA_ALVO].shift(lag)

    # Médias móveis e volatilidades
    for w in [4, 8, 12, 26, 52]:
        df[f"{COLUNA_ALVO}_mm_{w}"] = df[COLUNA_ALVO].rolling(w).mean()
        df[f"{COLUNA_ALVO}_vol_{w}"] = df[COLUNA_ALVO].rolling(w).std()

    # EWMA (média móvel exponencial)
    for alfa in [0.1, 0.2, 0.3]:
        df[f"{COLUNA_ALVO}_ewma_{str(alfa).replace('.','_')}"] = df[COLUNA_ALVO].ewm(alpha=alfa).mean()

    # STL trend (decompõe e usa tendência como feature)
    try:
        stl = STL(df[COLUNA_ALVO], period=52, robust=True)
        res = stl.fit()
        df["tendencia_stl"] = res.trend
        df["sazonal_stl"] = res.seasonal
        df["residuo_stl"] = res.resid
    except Exception:
        # fallback se série for muito curta
        df["tendencia_stl"] = df[COLUNA_ALVO].rolling(52, min_periods=1).mean()
        df["sazonal_stl"] = df[COLUNA_ALVO] - df["tendencia_stl"]
        df["residuo_stl"] = df["sazonal_stl"] - df["sazonal_stl"].rolling(52, min_periods=1).mean()

    # Termos de Fourier (sazonalidade semanal/anual aproximada)
    df = adicionar_termos_fourier(df, periodo=52, K=3, col_data=COLUNA_DATA)
    df = adicionar_termos_fourier(df, periodo=26, K=2, col_data=COLUNA_DATA)

    # Diferenças (momentum)
    df["delta_1"] = df[COLUNA_ALVO].diff(1)
    df["delta_4"] = df[COLUNA_ALVO].diff(4)
    df["delta_12"] = df[COLUNA_ALVO].diff(12)

    # Interações simples com exógenas (se existirem)
    exogenas = [c for c in df.columns if c not in [COLUNA_DATA, COLUNA_ALVO]]
    for col in exogenas:
        if col.startswith(COLUNA_ALVO):  # pular as features do alvo
            continue
        df[f"{col}_lag1"] = df[col].shift(1)
        df[f"{col}_mm_4"] = df[col].rolling(4).mean()
        df[f"{col}_mm_12"] = df[col].rolling(12).mean()

    # Remover linhas iniciais com NaN devido a janelas/defasagens
    df = df.dropna().reset_index(drop=True)
    return df

def separar_treino_teste(df: pd.DataFrame, proporcao_teste=0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    n_teste = int(np.floor(n * proporcao_teste))
    return df.iloc[:-n_teste].copy(), df.iloc[-n_teste:].copy()

# -------------------- MODELO COM 20 NEURÔNIOS --------------------
class DropoutMC(layers.Dropout):
    """Dropout que permanece ativo em inferência para Monte Carlo."""
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)

def criar_rede(input_dim: int) -> keras.Model:
    entradas = keras.Input(shape=(input_dim,), name="entradas")
    x = layers.Dense(20, activation="relu", name="oculta_20")(entradas)  # <= EXATAMENTE 20 neurônios
    x = DropoutMC(0.25, name="dropout_mc")(x)  # MC Dropout p/ incerteza
    saida = layers.Dense(1, activation="linear", name="saida")(x)
    modelo = keras.Model(inputs=entradas, outputs=saida, name="rna_20_neuronios")
    otimizador = keras.optimizers.Adam(learning_rate=TAXA_APRENDIZADO)
    modelo.compile(optimizer=otimizador, loss=keras.losses.Huber(), metrics=["mae"])
    return modelo

# -------------------- CONFORMAL PREDICTION --------------------
def conformal_intervalos(residuos_valid: np.ndarray, alpha: float = 0.1) -> float:
    """
    Calcula o quantil absoluto dos resíduos (método simples — split conformal).
    alpha=0.1 => ~90% de cobertura.
    Retorna o raio (q̂) para construir [ŷ - q̂, ŷ + q̂].
    """
    q = np.quantile(np.abs(residuos_valid), 1 - alpha)
    return float(q)

# -------------------- TREINO + BACKTEST --------------------
def executar_backtesting(df_feat: pd.DataFrame) -> Dict[str, float]:
    colunas = [c for c in df_feat.columns if c not in [COLUNA_DATA, COLUNA_ALVO]]
    X = df_feat[colunas].values
    y = df_feat[COLUNA_ALVO].values

    tscv = TimeSeriesSplit(n_splits=N_SPLITS_BACKTEST)
    metricas = {"RMSE": [], "MAE": [], "MAPE": []}

    fold = 1
    for tr_idx, va_idx in tscv.split(X):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        # Pipeline com scaler ajustado só no treino
        escalonador = StandardScaler()
        X_tr_s = escalonador.fit_transform(X_tr)
        X_va_s = escalonador.transform(X_va)

        modelo = criar_rede(input_dim=X_tr_s.shape[1])
        es = keras.callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True, monitor="val_loss")
        hist = modelo.fit(
            X_tr_s, y_tr,
            validation_data=(X_va_s, y_va),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0,
            callbacks=[es]
        )

        # Previsão determinística (média MC ~ 1 amostra)
        y_pred = modelo.predict(X_va_s, verbose=0).ravel()

        metricas["RMSE"].append(rmse(y_va, y_pred))
        metricas["MAE"].append(mean_absolute_error(y_va, y_pred))
        metricas["MAPE"].append(mape(y_va, y_pred))

        print(f"Fold {fold}: RMSE={metricas['RMSE'][-1]:.4f}  MAE={metricas['MAE'][-1]:.4f}  MAPE={metricas['MAPE'][-1]:.2f}%")
        fold += 1

    resumo = {k: float(np.mean(v)) for k, v in metricas.items()}
    return resumo

# -------------------- TREINO FINAL + INTERVALOS --------------------
def treino_final_e_intervalos(df_feat: pd.DataFrame, proporcao_teste=0.2, alpha=0.1):
    df_tr, df_te = separar_treino_teste(df_feat, proporcao_teste)
    colunas = [c for c in df_feat.columns if c not in [COLUNA_DATA, COLUNA_ALVO]]

    X_tr = df_tr[colunas].values
    y_tr = df_tr[COLUNA_ALVO].values
    X_te = df_te[colunas].values
    y_te = df_te[COLUNA_ALVO].values

    # Split interno: parte do treino vira "validação para conformal"
    n_tr = len(X_tr)
    corte = int(np.floor(n_tr * 0.85))
    X_fit, y_fit = X_tr[:corte], y_tr[:corte]
    X_val, y_val = X_tr[corte:], y_tr[corte:]

    escalonador = StandardScaler()
    X_fit_s = escalonador.fit_transform(X_fit)
    X_val_s = escalonador.transform(X_val)
    X_te_s  = escalonador.transform(X_te)

    modelo = criar_rede(input_dim=X_fit_s.shape[1])
    es = keras.callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True, monitor="val_loss")
    modelo.fit(
        X_fit_s, y_fit,
        validation_data=(X_val_s, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=[es]
    )

    # Resíduos em validação para conformal
    y_val_pred = modelo.predict(X_val_s, verbose=0).ravel()
    residuos_val = y_val - y_val_pred
    q_hat = conformal_intervalos(residuos_val, alpha=alpha)

    # Previsão no teste com incerteza via MC Dropout
    # Fazemos N_AMOSTRAS_MC forward passes mantendo dropout ativo
    preds_mc = []
    for _ in range(N_AMOSTRAS_MC):
        preds_mc.append(modelo(X_te_s, training=True).numpy().ravel())
    preds_mc = np.vstack(preds_mc)
    y_te_pred_media = preds_mc.mean(axis=0)
    y_te_pred_std   = preds_mc.std(axis=0)  # incerteza epistemica aproximada

    # Intervalo conformal (cobertura ~ 1 - alpha)
    intervalo_inf = y_te_pred_media - q_hat
    intervalo_sup = y_te_pred_media + q_hat

    # Métricas no teste
    resultados = {
        "RMSE_teste": rmse(y_te, y_te_pred_media),
        "MAE_teste": mean_absolute_error(y_te, y_te_pred_media),
        "MAPE_teste": mape(y_te, y_te_pred_media),
        "q_hat_conformal": float(q_hat)
    }

    # Salvar artefatos
    caminho_modelo = PASTA_SAIDA / "rna_20_neuronios.keras"
    modelo.save(caminho_modelo, include_optimizer=True)
    pd.DataFrame({
        "data": df_te[COLUNA_DATA].values,
        "y_real": y_te,
        "y_pred": y_te_pred_media,
        "y_pred_std_mc": y_te_pred_std,
        "pi_inf_conformal": intervalo_inf,
        "pi_sup_conformal": intervalo_sup
    }).to_csv(PASTA_SAIDA / "previsoes_teste.csv", index=False)

    # também salvar scaler
    import joblib
    joblib.dump(escalonador, PASTA_SAIDA / "scaler.joblib")

    return resultados

# -------------------- EXECUÇÃO --------------------
if __name__ == "__main__":
    # 1) Carregar dados (CSV real ou sintético)
    dados_brutos = carregar_base()

    # 2) Engenharia de variáveis
    dados_feat = engenhar_features(dados_brutos)

    # 3) Backtesting (validação temporal k-fold)
    print("\n====== Backtesting (validação temporal) ======")
    resumo_bt = executar_backtesting(dados_feat)
    print(f"\nMÉDIAS (Backtest): RMSE={resumo_bt['RMSE']:.4f}  MAE={resumo_bt['MAE']:.4f}  MAPE={resumo_bt['MAPE']:.2f}%")

    # 4) Treino final + intervalos (conformal + MC Dropout)
    print("\n====== Treino final e intervalos de predição ======")
    resultados_te = treino_final_e_intervalos(dados_feat, proporcao_teste=0.2, alpha=0.1)
    print("\nMétricas no TESTE:")
    for k, v in resultados_te.items():
        if "MAPE" in k:
            print(f"{k}: {v:.2f}%")
        else:
            print(f"{k}: {v:.4f}")

    print(f"\nArtefatos salvos em: {PASTA_SAIDA.resolve()}")
    print(" - rna_20_neuronios.keras")
    print(" - scaler.joblib")
    print(" - previsoes_teste.csv")
