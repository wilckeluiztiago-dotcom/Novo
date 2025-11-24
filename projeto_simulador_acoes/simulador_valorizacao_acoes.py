# ============================================================
# SIMULADOR NEURAL-ESTOCÁSTICO DE VALORIZAÇÃO DE AÇÕES
# Autor: Luiz Tiago Wilcke (LT)
# ============================================================
# Este script implementa um "super modelo" para simular
# valorização de ações de uma empresa na bolsa:
#
#   1) Dinâmica de preço via SDE com volatilidade estocástica
#      tipo Heston + saltos de Merton (Heston-Jump).
#   2) Drift parcialmente fundamentalista (momentum + reversão
#      à média de um "valor justo" latente).
#   3) Rede neural opcional (PyTorch) para aprender correção
#      não linear do drift a partir de features históricas.
#   4) Simulação Monte Carlo com Euler-Maruyama (Heston com
#      "full truncation") e saltos Poisson.
#   5) Métricas: distribuição futura, probabilidades
#      condicionais, VaR/CVaR e gráficos.
#
# O objetivo é didático/profissional: um projeto completo para
# portfólio. Não é recomendação de investimento.
# ============================================================

import warnings, math, random, os, sys, argparse
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ------------------------------------------------------------
# 1. Configurações do modelo e simulação
# ------------------------------------------------------------
@dataclass
class ConfiguracoesModelo:
    ticker: str = "PETR4.SA"              # símbolo no Yahoo Finance (opcional)
    data_inicio: str = "2015-01-01"
    data_fim: Optional[str] = None

    # Horizonte e discretização
    horizonte_dias: int = 252            # ~1 ano útil
    passos_por_dia: int = 1              # dt = 1 dia / passos_por_dia
    num_trajetorias: int = 5000

    # Drift estrutural (real-world)
    mu_base: float = 0.10                # retorno anual base
    lambda_fund: float = 0.50            # peso do componente fundamentalista
    gamma_momentum: float = 1.50         # força do momentum
    kappa_valor: float = 1.00            # reversão ao valor justo

    # Heston (volatilidade estocástica)
    kappa_v: float = 3.0                 # reversão da variância
    theta_v: float = 0.04                # variância de longo prazo
    sigma_v: float = 0.50                # vol-of-vol
    rho: float = -0.60                   # correlação preço-vol

    v0: float = 0.04                     # variância inicial

    # Saltos de Merton (log-normal)
    lambda_j: float = 0.25               # intensidade anual de saltos
    mu_j: float = -0.02                  # média do tamanho do salto (log)
    sigma_j: float = 0.10                # desvio do tamanho do salto (log)

    # Rede neural opcional
    usar_rede_neural: bool = False
    epocas_rede: int = 30
    taxa_aprendizado: float = 1e-3
    tamanho_janela_features: int = 30

    # Semente
    semente: int = 42


# ------------------------------------------------------------
# 2. Utilidades: download/geração de dados e features
# ------------------------------------------------------------
def tentar_baixar_dados_yahoo(ticker: str, data_inicio: str, data_fim: Optional[str]) -> Optional[pd.DataFrame]:
    """
    Tenta baixar dados via yfinance se disponível.
    Retorna DataFrame com coluna 'Close' ou None se falhar.
    """
    try:
        import yfinance as yf
        df = yf.download(ticker, start=data_inicio, end=data_fim, auto_adjust=True, progress=False)
        if df is None or df.empty:
            return None
        if "Close" not in df.columns:
            # auto_adjust pode devolver 'Adj Close' dependendo do pacote
            for c in df.columns:
                if "close" in c.lower():
                    df["Close"] = df[c]
                    break
        df = df[["Close"]].dropna()
        return df
    except Exception:
        return None


def gerar_dados_sinteticos(preco_inicial: float = 50.0, dias: int = 1500, mu: float = 0.12, sigma: float = 0.30, semente: int = 42) -> pd.DataFrame:
    """
    Gera uma série sintética GBM para permitir rodar o projeto sem internet/pacotes extras.
    """
    rng = np.random.default_rng(semente)
    dt = 1/252
    precos = [preco_inicial]
    for _ in range(dias - 1):
        z = rng.normal()
        s_ant = precos[-1]
        s_novo = s_ant * math.exp((mu - 0.5*sigma**2)*dt + sigma*math.sqrt(dt)*z)
        precos.append(s_novo)
    idx = pd.date_range("2018-01-01", periods=dias, freq="B")
    return pd.DataFrame({"Close": precos}, index=idx)


def calcular_features(df: pd.DataFrame, janela: int = 30) -> pd.DataFrame:
    """
    Calcula features simples para a rede neural e drift:
    - retornos log
    - volatilidade realizada
    - momentum (retorno acumulado)
    - estimativa de "valor justo" por EMA
    """
    precos = df["Close"]
    ret_log = np.log(precos).diff()
    vol_real = ret_log.rolling(janela).std() * math.sqrt(252)
    momentum = np.log(precos / precos.shift(janela))
    ema_valor_justo = precos.ewm(span=janela*3, adjust=False).mean()

    feat = pd.DataFrame({
        "preco": precos,
        "ret_log": ret_log,
        "vol_real": vol_real,
        "momentum": momentum,
        "valor_justo": ema_valor_justo
    }, index=df.index).dropna()
    return feat


# ------------------------------------------------------------
# 3. Rede Neural (opcional) para corrigir drift
# ------------------------------------------------------------
class RedeDriftTorch:
    """
    Rede simples para aprender correção no drift:
    entrada: janela de features
    saída: delta_mu (ajuste do drift diário)
    """
    def __init__(self, num_features: int, janela: int, taxa_aprendizado: float, semente: int = 42):
        import torch
        import torch.nn as nn

        torch.manual_seed(semente)

        self.torch = torch
        self.nn = nn

        dim_entrada = num_features * janela

        self.modelo = nn.Sequential(
            nn.Linear(dim_entrada, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1)
        )

        self.otimizador = torch.optim.Adam(self.modelo.parameters(), lr=taxa_aprendizado)
        self.criterio = nn.MSELoss()

    def _preparar_dataset(self, features: np.ndarray, alvo: np.ndarray, janela: int):
        """
        Constrói janelas para série temporal.
        """
        X, y = [], []
        for i in range(janela, len(features)):
            X.append(features[i-janela:i].reshape(-1))
            y.append(alvo[i])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32).reshape(-1, 1)

    def treinar(self, df_feat: pd.DataFrame, janela: int, epocas: int = 30, batch: int = 64, verbose: bool = True):
        torch = self.torch

        feats = df_feat[["ret_log", "vol_real", "momentum"]].values
        # alvo: próximo retorno log
        alvo = df_feat["ret_log"].shift(-1).fillna(0.0).values

        X, y = self._preparar_dataset(feats, alvo, janela)

        ds = torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(y))
        dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True)

        self.modelo.train()
        for e in range(1, epocas+1):
            perda_ep = 0.0
            for xb, yb in dl:
                pred = self.modelo(xb)
                perda = self.criterio(pred, yb)
                self.otimizador.zero_grad()
                perda.backward()
                self.otimizador.step()
                perda_ep += perda.item() * len(xb)

            perda_ep /= len(ds)
            if verbose and (e % max(1, epocas//5) == 0 or e == 1):
                print(f"Época {e:02d}/{epocas} | Perda: {perda_ep:.6f}")

    def prever_delta_mu(self, janela_feat: np.ndarray) -> float:
        torch = self.torch
        self.modelo.eval()
        with torch.no_grad():
            x = torch.tensor(janela_feat.reshape(1, -1), dtype=torch.float32)
            delta = self.modelo(x).item()
        return float(delta)


# ------------------------------------------------------------
# 4. Núcleo: Simulação Heston + saltos (Heston-Jump)
# ------------------------------------------------------------
def simular_heston_jump(
    s0: float,
    v0: float,
    mu_base: float,
    kappa_v: float,
    theta_v: float,
    sigma_v: float,
    rho: float,
    lambda_j: float,
    mu_j: float,
    sigma_j: float,
    horizonte_dias: int,
    passos_por_dia: int,
    num_trajetorias: int,
    drift_fundamentalista: Optional[np.ndarray] = None,
    semente: int = 42
) -> np.ndarray:
    """
    Simula trajetórias S_t com:
        dS_t/S_t = mu_t dt + sqrt(v_t) dW_1 + (J-1) dN_t
        dv_t     = kappa_v (theta_v - v_t) dt + sigma_v sqrt(v_t) dW_2
    com corr(dW_1, dW_2) = rho.

    drift_fundamentalista: vetor (T,) com ajuste diário do drift.
    """
    rng = np.random.default_rng(semente)

    T = horizonte_dias
    dt = 1/252/ passos_por_dia
    n_passos = T * passos_por_dia

    S = np.zeros((num_trajetorias, n_passos+1), dtype=np.float64)
    v = np.zeros_like(S)

    S[:, 0] = s0
    v[:, 0] = v0

    # compensação de saltos na drift (garante E[J] incorporado)
    k_barra = math.exp(mu_j + 0.5*sigma_j**2) - 1.0

    for t in range(n_passos):
        # Brownianos correlacionados
        z1 = rng.normal(size=num_trajetorias)
        z2 = rng.normal(size=num_trajetorias)
        w1 = z1
        w2 = rho*z1 + math.sqrt(1-rho**2)*z2

        v_t = np.maximum(v[:, t], 0.0)  # full truncation

        # Heston variance update (Euler full truncation)
        dv = kappa_v*(theta_v - v_t)*dt + sigma_v*np.sqrt(v_t*dt)*w2
        v[:, t+1] = np.maximum(v_t + dv, 0.0)

        # drift diário (mu_base anual -> diário)
        mu_t = mu_base
        if drift_fundamentalista is not None:
            mu_t = mu_t + float(drift_fundamentalista[min(t//passos_por_dia, len(drift_fundamentalista)-1)])
        mu_dia = mu_t

        # Saltos Poisson
        n_saltos = rng.poisson(lambda_j*dt, size=num_trajetorias)
        # tamanho do salto multiplicativo J = exp(Y), Y~N(mu_j, sigma_j)
        Y = rng.normal(mu_j, sigma_j, size=num_trajetorias)
        J = np.exp(Y)  # se n_saltos=0, ignoramos via máscara

        # Atualização de preço
        dlogS = (mu_dia - lambda_j*k_barra - 0.5*v_t)*dt + np.sqrt(v_t*dt)*w1
        S[:, t+1] = S[:, t]*np.exp(dlogS) * np.where(n_saltos>0, J, 1.0)

    return S


# ------------------------------------------------------------
# 5. Drift fundamentalista + momentum
# ------------------------------------------------------------
def construir_drift_fundamentalista(df_feat: pd.DataFrame, cfg: ConfiguracoesModelo) -> np.ndarray:
    """
    Constrói ajuste diário de drift baseado em:
    - distância ao valor justo (reversão)
    - momentum empírico (tendência recente)
    """
    precos = df_feat["preco"].values
    valor_justo = df_feat["valor_justo"].values
    momentum = df_feat["momentum"].values
    vol_real = df_feat["vol_real"].fillna(df_feat["vol_real"].median()).values

    # escala diária
    mu_base_dia = cfg.mu_base / 252

    ajuste = []
    for i in range(len(precos)):
        dist_valor = (valor_justo[i] - precos[i]) / max(precos[i], 1e-8)
        comp_fund = cfg.kappa_valor * dist_valor
        comp_mom = cfg.gamma_momentum * momentum[i] / max(cfg.tamanho_janela_features, 1)

        # menor drift em períodos muito voláteis
        amortecimento_vol = 1.0 / (1.0 + 2.0*vol_real[i])

        delta_mu = cfg.lambda_fund * (comp_fund + comp_mom) * amortecimento_vol
        ajuste.append(delta_mu)

    return np.array(ajuste, dtype=np.float64) + mu_base_dia


# ------------------------------------------------------------
# 6. Métricas de risco/retorno
# ------------------------------------------------------------
def metricas_distribuicao(precos_finais: np.ndarray) -> Dict[str, float]:
    """
    Calcula estatísticas e risco (VaR/CVaR).
    """
    retornos = precos_finais / precos_finais.mean() - 1.0
    media = float(precos_finais.mean())
    mediana = float(np.median(precos_finais))
    desvio = float(precos_finais.std())
    p5 = float(np.quantile(precos_finais, 0.05))
    p95 = float(np.quantile(precos_finais, 0.95))

    var_95 = float(np.quantile(retornos, 0.05))
    cvar_95 = float(retornos[retornos <= var_95].mean())

    return {
        "preco_medio": media,
        "preco_mediana": mediana,
        "preco_desvio": desvio,
        "preco_p5": p5,
        "preco_p95": p95,
        "VaR_95_retorno": var_95,
        "CVaR_95_retorno": cvar_95
    }


# ------------------------------------------------------------
# 7. Pipeline principal
# ------------------------------------------------------------
def rodar_pipeline(cfg: ConfiguracoesModelo):
    np.random.seed(cfg.semente)
    random.seed(cfg.semente)

    # 7.1 dados históricos (opcional)
    df = tentar_baixar_dados_yahoo(cfg.ticker, cfg.data_inicio, cfg.data_fim)
    if df is None:
        print("Aviso: não consegui baixar dados. Usando série sintética.")
        df = gerar_dados_sinteticos(semente=cfg.semente)

    df_feat = calcular_features(df, janela=cfg.tamanho_janela_features)

    s0 = float(df_feat["preco"].iloc[-1])
    v0 = cfg.v0

    # 7.2 drift fundamentalista
    drift_fund = construir_drift_fundamentalista(df_feat, cfg)

    # 7.3 rede neural opcional: aprende correção adicional do drift
    delta_mu_neural = None
    if cfg.usar_rede_neural:
        try:
            print("\nTreinando rede neural para correção de drift...")
            rede = RedeDriftTorch(num_features=3, janela=cfg.tamanho_janela_features,
                                  taxa_aprendizado=cfg.taxa_aprendizado, semente=cfg.semente)
            rede.treinar(df_feat, janela=cfg.tamanho_janela_features, epocas=cfg.epocas_rede, verbose=True)

            # gera correção futura assumindo repetição do último padrão
            feats = df_feat[["ret_log", "vol_real", "momentum"]].values
            janela_ultima = feats[-cfg.tamanho_janela_features:].reshape(-1)
            delta = rede.prever_delta_mu(janela_ultima)  # retorno log diário previsto
            delta_mu_neural = np.full(cfg.horizonte_dias, delta, dtype=np.float64)
        except Exception as e:
            print(f"Falha ao treinar rede neural ({e}). Continuando sem ela.")
            delta_mu_neural = None

    # combina drift fundamentalista com neural (se houver)
    drift_sim = drift_fund[-cfg.horizonte_dias:].copy()
    if delta_mu_neural is not None:
        drift_sim = drift_sim + delta_mu_neural

    # 7.4 simulação
    print("\nSimulando trajetórias Monte Carlo (Heston + saltos)...")
    S = simular_heston_jump(
        s0=s0, v0=v0,
        mu_base=0.0,  # já embutido em drift_sim (diário)
        kappa_v=cfg.kappa_v, theta_v=cfg.theta_v, sigma_v=cfg.sigma_v, rho=cfg.rho,
        lambda_j=cfg.lambda_j, mu_j=cfg.mu_j, sigma_j=cfg.sigma_j,
        horizonte_dias=cfg.horizonte_dias, passos_por_dia=cfg.passos_por_dia,
        num_trajetorias=cfg.num_trajetorias,
        drift_fundamentalista=drift_sim,
        semente=cfg.semente
    )

    precos_finais = S[:, -1]
    mets = metricas_distribuicao(precos_finais)

    # 7.5 probabilidades de cenário
    alvo_up = s0 * 1.10
    alvo_down = s0 * 0.90
    prob_up = float((precos_finais >= alvo_up).mean())
    prob_down = float((precos_finais <= alvo_down).mean())

    print("\n================ RESULTADOS =================")
    print(f"Preço inicial (s0): {s0:.4f}")
    print(f"Horizonte: {cfg.horizonte_dias} dias úteis")
    print(f"Preço médio final: {mets['preco_medio']:.4f}")
    print(f"P5 / P95 final: {mets['preco_p5']:.4f} / {mets['preco_p95']:.4f}")
    print(f"Prob(>= +10%): {prob_up*100:.2f}%")
    print(f"Prob(<= -10%): {prob_down*100:.2f}%")
    print(f"VaR95 retorno: {mets['VaR_95_retorno']*100:.2f}%")
    print(f"CVaR95 retorno: {mets['CVaR_95_retorno']*100:.2f}%")
    print("============================================\n")

    # 7.6 gráficos
    plt.figure()
    # algumas trajetórias
    k = min(50, cfg.num_trajetorias)
    for i in range(k):
        plt.plot(S[i], alpha=0.25)
    plt.title(f"Trajetórias simuladas — {cfg.ticker}")
    plt.xlabel("Passos")
    plt.ylabel("Preço")
    plt.grid(True)
    plt.tight_layout()

    plt.figure()
    plt.hist(precos_finais, bins=60, density=True)
    plt.axvline(alvo_up, linestyle="--", label="+10%")
    plt.axvline(alvo_down, linestyle="--", label="-10%")
    plt.title("Distribuição do preço final")
    plt.xlabel("Preço final")
    plt.ylabel("Densidade")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.figure()
    retornos_finais = precos_finais/s0 - 1.0
    plt.hist(retornos_finais, bins=60, density=True)
    plt.axvline(np.quantile(retornos_finais, 0.05), linestyle="--", label="VaR 95%")
    plt.title("Distribuição de retorno no horizonte")
    plt.xlabel("Retorno")
    plt.ylabel("Densidade")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()

    return S, mets


# ------------------------------------------------------------
# 8. CLI
# ------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Simulador Heston-Jump para valorização de ações.")
    p.add_argument("--ticker", type=str, default="PETR4.SA", help="Ticker Yahoo Finance (ex.: PETR4.SA, AAPL).")
    p.add_argument("--horizonte", type=int, default=252, help="Horizonte em dias úteis.")
    p.add_argument("--trajetorias", type=int, default=5000, help="Número de trajetórias Monte Carlo.")
    p.add_argument("--usar_rede_neural", action="store_true", help="Ativa correção neural do drift (requer torch).")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = ConfiguracoesModelo(
        ticker=args.ticker,
        horizonte_dias=args.horizonte,
        num_trajetorias=args.trajetorias,
        usar_rede_neural=args.usar_rede_neural
    )
    rodar_pipeline(cfg)


if __name__ == "__main__":
    main()
