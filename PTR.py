# ============================================================
# MODELO NÃO LINEAR (EDO/SDE) PARA PREVER PETR4
# Autor: Luiz Tiago Wilcke (LT)
# ============================================================

import warnings, math
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

# --- dependências opcionais
try:
    import yfinance as yf
except Exception:
    yf = None

try:
    from scipy.optimize import minimize
except Exception:
    minimize = None


# ============================================================
# 1. Configurações
# ============================================================
@dataclass
class ConfiguracoesModelo:
    ticker_acao: str = "PETR4.SA"
    ticker_petroleo: str = "BZ=F"   # Brent
    data_inicio: str = "2015-01-01"
    data_fim: Optional[str] = None
    dt: float = 1.0/252.0
    janela_memoria: int = 20
    horizonte_previsao_dias: int = 60
    n_simulacoes: int = 300
    seed: int = 42


# ============================================================
# 2. Utilidades de dados
# ============================================================
def baixar_series(ticker: str, inicio: str, fim: Optional[str]) -> pd.Series:
    if yf is None:
        raise RuntimeError("yfinance não está instalado. Rode: pip install yfinance")

    df = yf.download(
        ticker,
        start=inicio,
        end=fim,
        progress=False,
        auto_adjust=False,
        group_by="column"
    )

    if df is None or df.empty:
        raise RuntimeError(f"Sem dados para {ticker}.")

    # escolher coluna de preço de forma robusta
    if "Adj Close" in df.columns:
        s = df["Adj Close"]
    elif "Close" in df.columns:
        s = df["Close"]
    else:
        # às vezes vem MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            # tenta achar Adj Close / Close dentro do MultiIndex
            adj = [c for c in df.columns if "Adj Close" in c]
            clo = [c for c in df.columns if "Close" in c]
            col = adj[0] if adj else (clo[0] if clo else None)
            if col is None:
                raise RuntimeError(f"Coluna de preço não encontrada em {ticker}. Colunas: {df.columns}")
            s = df[col]
        else:
            raise RuntimeError(f"Coluna de preço não encontrada em {ticker}. Colunas: {df.columns}")

    # se por algum motivo virar DataFrame (multiindex), pega a 1ª coluna
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]

    if not isinstance(s, pd.Series):
        raise RuntimeError(f"Falha ao extrair série de preço para {ticker} (tipo={type(s)}).")

    s = s.dropna()
    if s.empty:
        raise RuntimeError(f"Série vazia após limpeza para {ticker}.")

    # garante índice datetime
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)

    return s


def normalizar_zscore(x: pd.Series, janela: int = 60) -> pd.Series:
    media = x.rolling(janela, min_periods=1).mean()
    desvio = x.rolling(janela, min_periods=1).std()
    z = (x - media) / (desvio + 1e-12)
    return z.fillna(0.0)


def ema(x: pd.Series, alpha: float) -> pd.Series:
    return x.ewm(alpha=alpha, adjust=False).mean()


# ============================================================
# 3. Construção de fatores exógenos / proxies
# ============================================================
def construir_fatores(df: pd.DataFrame, cfg: ConfiguracoesModelo) -> pd.DataFrame:
    janela = cfg.janela_memoria
    alpha = 2/(janela+1)

    df["ret_acao"] = np.log(df["preco_acao"]).diff().fillna(0.0)
    df["ret_petroleo"] = np.log(df["preco_petroleo"]).diff().fillna(0.0)

    df["choque_petroleo"] = ema(df["ret_petroleo"], alpha=alpha)

    df["vol_acao"] = df["ret_acao"].rolling(janela, min_periods=1).std().fillna(0.0)
    df["sentimento"] = normalizar_zscore(df["vol_acao"], janela=janela)

    divergencia = df["ret_acao"] - 0.6*df["ret_petroleo"]
    df["risco_politico_proxy"] = ema(normalizar_zscore(divergencia, janela=janela), alpha=alpha)

    vol_norm = normalizar_zscore(df["volume_acao"], janela=janela)
    fluxo = df["ret_acao"] * vol_norm
    df["fluxo_ordens_proxy"] = ema(fluxo, alpha=alpha)

    return df.fillna(0.0)


# ============================================================
# 4. Sistema Não Linear (Euler-Maruyama)
# ============================================================
def simular_sistema(
    preco_inicial: float,
    fatores: Dict[str, np.ndarray],
    parametros: Dict[str, float],
    dt: float,
    n_passos: int,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    if seed is not None:
        np.random.seed(seed)

    choque_petroleo = fatores["choque_petroleo"]
    sentimento = fatores["sentimento"]
    risco_politico = fatores["risco_politico"]
    fluxo_ordens = fatores["fluxo_ordens"]

    mu = parametros["mu"]
    kappa = parametros["kappa"]
    p_eq = parametros["p_eq"]

    a_pet, b_pet = parametros["a_pet"], parametros["b_pet"]
    a_sent, b_sent = parametros["a_sent"], parametros["b_sent"]
    a_risco, b_risco = parametros["a_risco"], parametros["b_risco"]
    a_fluxo, b_fluxo = parametros["a_fluxo"], parametros["b_fluxo"]

    sig0 = parametros["sig0"]
    sig_bar = parametros["sig_bar"]
    eta = parametros["eta"]
    a_crise = parametros["a_crise"]

    P = np.zeros(n_passos)
    V = np.zeros(n_passos)
    SIG = np.zeros(n_passos)

    P[0] = preco_inicial
    V[0] = 0.0
    SIG[0] = sig0

    for t in range(n_passos-1):
        dW = np.random.normal(0.0, math.sqrt(dt))

        crise = np.tanh(a_crise * sentimento[t])**2
        dSIG = eta*(sig_bar - SIG[t])*dt + 0.5*crise*dt
        SIG[t+1] = max(1e-6, SIG[t] + dSIG)

        forca_pet = a_pet * np.tanh(b_pet * choque_petroleo[t])
        forca_sent = a_sent * np.tanh(b_sent * sentimento[t]) * (1 + 0.5*SIG[t])
        forca_risco = -a_risco * np.tanh(b_risco * risco_politico[t]) * (P[t]/max(p_eq,1e-6))
        forca_fluxo = a_fluxo * np.tanh(b_fluxo * fluxo_ordens[t])

        dV = (mu*(p_eq - P[t]) - kappa*V[t]
              + forca_pet + forca_sent + forca_risco + forca_fluxo) * dt
        V[t+1] = V[t] + dV

        dP = V[t]*dt + SIG[t]*P[t]*dW
        P[t+1] = max(1e-6, P[t] + dP)

    return P, V, SIG


# ============================================================
# 5. Calibração
# ============================================================
def vetor_para_parametros(x: np.ndarray, preco_medio: float) -> Dict[str, float]:
    return dict(
        mu      = x[0],
        kappa   = abs(x[1]) + 1e-8,
        p_eq    = max(1.0, preco_medio*(1 + 0.1*np.tanh(x[2]))),

        a_pet   = x[3],
        b_pet   = abs(x[4]) + 1e-6,

        a_sent  = x[5],
        b_sent  = abs(x[6]) + 1e-6,

        a_risco = abs(x[7]),
        b_risco = abs(x[8]) + 1e-6,

        a_fluxo = x[9],
        b_fluxo = abs(x[10]) + 1e-6,

        sig0    = abs(x[11]) + 1e-4,
        sig_bar = abs(x[12]) + 1e-4,
        eta     = abs(x[13]) + 1e-4,
        a_crise = abs(x[14]) + 1e-4
    )


def parametros_padrao(df: pd.DataFrame) -> Dict[str, float]:
    preco_medio = df["preco_acao"].mean()
    return dict(
        mu=0.06, kappa=4.0, p_eq=preco_medio,
        a_pet=2.0, b_pet=5.0,
        a_sent=1.5, b_sent=3.0,
        a_risco=2.0, b_risco=4.0,
        a_fluxo=1.0, b_fluxo=6.0,
        sig0=0.25, sig_bar=0.20, eta=3.0, a_crise=4.0
    )


def funcao_erro_calibracao(x: np.ndarray, df: pd.DataFrame, cfg: ConfiguracoesModelo) -> float:
    preco_medio = df["preco_acao"].mean()
    parametros = vetor_para_parametros(x, preco_medio)

    fatores = dict(
        choque_petroleo = df["choque_petroleo"].values,
        sentimento      = df["sentimento"].values,
        risco_politico  = df["risco_politico_proxy"].values,
        fluxo_ordens    = df["fluxo_ordens_proxy"].values
    )

    P_sim, _, _ = simular_sistema(
        preco_inicial=df["preco_acao"].iloc[0],
        fatores=fatores,
        parametros=parametros,
        dt=cfg.dt,
        n_passos=len(df),
        seed=0
    )

    ret_sim = np.diff(np.log(P_sim + 1e-12))
    ret_real = df["ret_acao"].values[1:]
    return float(np.mean((ret_sim - ret_real)**2))


def calibrar_parametros(df: pd.DataFrame, cfg: ConfiguracoesModelo) -> Dict[str, float]:
    if minimize is None:
        print("scipy não disponível -> usando parâmetros padrão.")
        return parametros_padrao(df)

    x0 = np.array([
        0.05,  5.0,   0.0,
        2.0,   5.0,
        1.5,   3.0,
        2.0,   4.0,
        1.0,   6.0,
        0.25,  0.20,  3.0,  4.0
    ], dtype=float)

    res = minimize(
        funcao_erro_calibracao, x0,
        args=(df, cfg),
        method="Nelder-Mead",
        options=dict(maxiter=2500, disp=False)
    )

    parametros = vetor_para_parametros(res.x, df["preco_acao"].mean())
    print("\nParâmetros calibrados (resumo):")
    for k, v in parametros.items():
        print(f"  {k:10s} = {v:.6f}")
    print(f"Erro final (MSE retornos): {res.fun:.8e}\n")
    return parametros


# ============================================================
# 6. Previsão Monte Carlo
# ============================================================
def _pad_fatores(arr: np.ndarray, n: int) -> np.ndarray:
    if len(arr) >= n:
        return arr[:n]
    if len(arr) == 0:
        return np.zeros(n)
    last = arr[-1]
    pad = np.full(n - len(arr), last)
    return np.concatenate([arr, pad])


def prever_futuro_monte_carlo(df: pd.DataFrame, cfg: ConfiguracoesModelo, parametros: Dict[str, float]) -> pd.DataFrame:
    np.random.seed(cfg.seed)
    n_passos = int(cfg.horizonte_previsao_dias)
    preco0 = float(df["preco_acao"].iloc[-1])

    ultimos = df.iloc[-n_passos:].copy()

    fatores_base = dict(
        choque_petroleo = _pad_fatores(ultimos["choque_petroleo"].values, n_passos),
        sentimento      = _pad_fatores(ultimos["sentimento"].values, n_passos),
        risco_politico  = _pad_fatores(ultimos["risco_politico_proxy"].values, n_passos),
        fluxo_ordens    = _pad_fatores(ultimos["fluxo_ordens_proxy"].values, n_passos)
    )

    trilhas = np.zeros((cfg.n_simulacoes, n_passos))
    for i in range(cfg.n_simulacoes):
        P_sim, _, _ = simular_sistema(
            preco_inicial=preco0,
            fatores=fatores_base,
            parametros=parametros,
            dt=cfg.dt,
            n_passos=n_passos,
            seed=cfg.seed + i
        )
        trilhas[i, :] = P_sim

    media = trilhas.mean(axis=0)
    p05 = np.quantile(trilhas, 0.05, axis=0)
    p95 = np.quantile(trilhas, 0.95, axis=0)

    try:
        datas_fut = pd.bdate_range(df.index[-1], periods=n_passos+1, inclusive="right")
    except TypeError:
        datas_fut = pd.bdate_range(df.index[-1], periods=n_passos+1, closed="right")

    return pd.DataFrame({
        "preco_medio_previsto": media,
        "p05": p05,
        "p95": p95
    }, index=datas_fut)


# ============================================================
# 7. Pipeline completo
# ============================================================
def pipeline_modelo(cfg: ConfiguracoesModelo):
    print("Baixando séries históricas...")
    preco_acao = baixar_series(cfg.ticker_acao, cfg.data_inicio, cfg.data_fim).rename("preco_acao")
    preco_pet  = baixar_series(cfg.ticker_petroleo, cfg.data_inicio, cfg.data_fim).rename("preco_petroleo")

    # concat alinha índices e evita erro de "all scalar values"
    df = pd.concat([preco_acao, preco_pet], axis=1, join="inner").dropna()

    if df.empty:
        raise RuntimeError("Após alinhar PETR4 e Brent, não sobrou dados. Verifique tickers/datas.")

    # volume da ação
    if yf is not None:
        vol_df = yf.download(cfg.ticker_acao, start=cfg.data_inicio, end=cfg.data_fim, progress=False)
        if vol_df is not None and not vol_df.empty and "Volume" in vol_df.columns:
            df["volume_acao"] = vol_df["Volume"].reindex(df.index).ffill().fillna(0.0)
        else:
            df["volume_acao"] = 1.0
    else:
        df["volume_acao"] = 1.0

    print("Construindo fatores (proxies)...")
    df = construir_fatores(df, cfg)

    print("Calibrando parâmetros...")
    parametros = calibrar_parametros(df, cfg)

    print("Simulando ajuste histórico...")
    fatores_hist = dict(
        choque_petroleo = df["choque_petroleo"].values,
        sentimento      = df["sentimento"].values,
        risco_politico  = df["risco_politico_proxy"].values,
        fluxo_ordens    = df["fluxo_ordens_proxy"].values
    )

    P_hist, V_hist, SIG_hist = simular_sistema(
        preco_inicial=df["preco_acao"].iloc[0],
        fatores=fatores_hist,
        parametros=parametros,
        dt=cfg.dt,
        n_passos=len(df),
        seed=1
    )

    df["preco_simulado"]       = P_hist
    df["velocidade_simulada"]  = V_hist
    df["vol_simulada"]         = SIG_hist

    print("Gerando previsão futura (Monte Carlo)...")
    df_prev = prever_futuro_monte_carlo(df, cfg, parametros)

    print("\nResumo numérico:")
    print(f"Preço atual: R$ {df['preco_acao'].iloc[-1]:.4f}")
    print(f"Previsão média {cfg.horizonte_previsao_dias} dias: R$ {df_prev['preco_medio_previsto'].iloc[-1]:.4f}")
    print(f"Intervalo 90%: [R$ {df_prev['p05'].iloc[-1]:.4f}, R$ {df_prev['p95'].iloc[-1]:.4f}]\n")

    # gráficos
    plt.figure(figsize=(12,6))
    plt.plot(df.index, df["preco_acao"], label="PETR4 real")
    plt.plot(df.index, df["preco_simulado"], label="Ajuste simulado")
    plt.title("Ajuste histórico do modelo não linear")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12,6))
    plt.plot(df_prev.index, df_prev["preco_medio_previsto"], label="Previsão média")
    plt.fill_between(df_prev.index, df_prev["p05"], df_prev["p95"], alpha=0.25, label="Banda 90% (p05-p95)")
    plt.axhline(df["preco_acao"].iloc[-1], linestyle="--", label="Preço atual")
    plt.title("Previsão futura PETR4 — Modelo não linear + ruído estocástico")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12,4))
    plt.plot(df.index, df["vol_simulada"], label="Volatilidade latente SIG(t)")
    plt.title("Volatilidade dinâmica simulada")
    plt.legend()
    plt.grid(True)
    plt.show()

    return df, df_prev, parametros


# ============================================================
# 8. Execução
# ============================================================
if __name__ == "__main__":
    cfg = ConfiguracoesModelo(
        data_inicio="2018-01-01",
        horizonte_previsao_dias=90,
        n_simulacoes=400
    )
    df_hist, df_prev, parametros = pipeline_modelo(cfg)
