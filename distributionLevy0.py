# ============================================================
# MODELO AVANÇADO DE QUEDAS EM AÇÕES — DISTRIBUIÇÃO DE LÉVY
# Autor: Luiz Tiago Wilcke (LT)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Tuple

from scipy.stats import levy, norm

# ------------------------------------------------------------
# 1) Configurações e parâmetros
# ------------------------------------------------------------
@dataclass
class ParametrosMercado:
    preco_inicial: float = 100.0
    dias_ano: int = 252
    drift_anual: float = 0.08
    volatilidade_anual: float = 0.25


@dataclass
class ParametrosLevy:
    prob_evento_queda: float = 0.10    # probabilidade de "evento de crash" por dia
    loc_chute: float = 0.0            # chute inicial loc
    scale_chute: float = 0.03         # chute inicial scale


@dataclass
class ConfiguracaoModelo:
    usar_csv: bool = False
    caminho_csv: Optional[str] = None
    coluna_preco: Optional[str] = None  # se None, tenta detectar automaticamente
    dias_historico_sintetico: int = 2000
    dias_futuro: int = 252
    n_cenarios_futuros: int = 300
    quantil_cauda: float = 0.1         # usamos, por ex., os 10% piores retornos negativos
    nivel_confianca: float = 0.99      # nível de VaR/ES (99%)
    semente: int = 42


# ------------------------------------------------------------
# 2) Classe do modelo avançado
# ------------------------------------------------------------
class ModeloLevyAvancadoAcoes:
    """
    Modelo avançado de risco de queda em ações usando distribuição de Lévy:

      - Log-retornos diários r_t = ln(S_t / S_{t-1})
      - Cauda de quedas: r_t < limiar (quantil inferior)
      - Queda = -r_t (magnitude positiva da perda)
      - Queda ~ Lévy(loc, scale) ajustada por MLE

    A partir disso:
      - Estimamos VaR e ES da queda
      - Simulamos caminhos futuros GBM + choques de Lévy
      - Fazemos gráficos de diagnóstico (histograma, Q-Q, etc).
    """

    def __init__(
        self,
        parametros_mercado: ParametrosMercado,
        parametros_levy: ParametrosLevy,
        config: ConfiguracaoModelo
    ):
        self.pm = parametros_mercado
        self.pl = parametros_levy
        self.cfg = config

        np.random.seed(self.cfg.semente)

        # Objetos que serão preenchidos
        self.precos_: Optional[pd.Series] = None
        self.log_retornos_: Optional[pd.Series] = None
        self.cauda_quedas_: Optional[pd.Series] = None

        self.loc_levy_: Optional[float] = None
        self.scale_levy_: Optional[float] = None

    # --------------------------------------------------------
    # 2.1) Carregar ou simular preços
    # --------------------------------------------------------
    def carregar_ou_simular_precos(self) -> pd.Series:
        if self.cfg.usar_csv:
            return self._carregar_precos_csv()
        else:
            return self._simular_precos_gbm_com_levy()

    def _carregar_precos_csv(self) -> pd.Series:
        if self.cfg.caminho_csv is None:
            raise ValueError("caminho_csv não definido.")

        df = pd.read_csv(self.cfg.caminho_csv)

        if self.cfg.coluna_preco is not None:
            col = self.cfg.coluna_preco
        else:
            # tentativa simples de detectar coluna de fechamento
            candidatos = ["fechamento", "close", "preco", "price", "adjclose", "adj_close"]
            col = None
            for c in df.columns:
                if c.lower() in candidatos:
                    col = c
                    break
            if col is None:
                raise ValueError("Não foi possível detectar automaticamente a coluna de preços.")

        serie = pd.Series(df[col].astype(float).values, name="preco").reset_index(drop=True)
        self.precos_ = serie
        return serie

    def _simular_precos_gbm_com_levy(self) -> pd.Series:
        """
        Simula uma trajetória de preço com:
          dS/S = mu dt + sigma dW - dJ_t
        onde J_t tem incrementos Levy quando ocorre "evento de queda".
        """
        n = self.cfg.dias_historico_sintetico
        dt = 1.0 / self.pm.dias_ano
        mu = self.pm.drift_anual
        sigma = self.pm.volatilidade_anual
        p = self.pl.prob_evento_queda

        precos = np.zeros(n)
        precos[0] = self.pm.preco_inicial

        for t in range(1, n):
            z = np.random.randn()
            retorno_difusivo = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z

            # evento de Lévy (crash)
            if np.random.rand() < p:
                queda = levy.rvs(loc=self.pl.loc_chute, scale=self.pl.scale_chute)
            else:
                queda = 0.0

            log_retorno = retorno_difusivo - queda
            precos[t] = precos[t-1] * np.exp(log_retorno)

        serie = pd.Series(precos, name="preco")
        self.precos_ = serie
        return serie

    # --------------------------------------------------------
    # 2.2) Log-retornos e cauda de quedas
    # --------------------------------------------------------
    def calcular_log_retornos(self):
        if self.precos_ is None:
            raise ValueError("Precisa carregar ou simular preços primeiro.")

        lr = np.log(self.precos_ / self.precos_.shift(1))
        self.log_retornos_ = lr.dropna()

    def extrair_cauda_quedas(self):
        if self.log_retornos_ is None:
            raise ValueError("log_retornos_ ainda não calculado.")

        # pegamos apenas retornos negativos
        retornos_negativos = self.log_retornos_[self.log_retornos_ < 0.0]

        if retornos_negativos.empty:
            raise ValueError("Não há retornos negativos suficientes.")

        # definimos um limiar de cauda: quantil inferior (ex: 10% piores)
        limiar = retornos_negativos.quantile(self.cfg.quantil_cauda)
        cauda = retornos_negativos[retornos_negativos <= limiar]

        # magnitude positiva da queda
        quedas = -cauda
        self.cauda_quedas_ = quedas

    # --------------------------------------------------------
    # 2.3) Ajuste da Lévy à cauda
    # --------------------------------------------------------
    def ajustar_levy(self) -> Tuple[float, float]:
        if self.cauda_quedas_ is None or self.cauda_quedas_.empty:
            raise ValueError("cauda_quedas_ ainda não definida ou vazia.")

        dados = self.cauda_quedas_.values

        # Ajuste MLE
        loc_est, scale_est = levy.fit(dados)

        self.loc_levy_ = float(loc_est)
        self.scale_levy_ = float(scale_est)

        return self.loc_levy_, self.scale_levy_

    # --------------------------------------------------------
    # 2.4) Medidas de risco: VaR e ES (Expected Shortfall)
    # --------------------------------------------------------
    def calcular_var_es_diario(self) -> Tuple[float, float]:
        """
        Calcula VaR e ES da QUEDA (magnitude |r|) para um nível de confiança.
        VaR_alpha: quantil alpha da Lévy.
        ES_alpha: E[queda | queda > VaR_alpha] (estimado por Monte Carlo).
        """
        if self.loc_levy_ is None or self.scale_levy_ is None:
            raise ValueError("Distribuição de Lévy ainda não ajustada.")

        alpha = self.cfg.nivel_confianca

        # VaR diário em termos de magnitude de log-retorno
        var_queda = levy.ppf(alpha, loc=self.loc_levy_, scale=self.scale_levy_)

        # ES via Monte Carlo (poderia ser integral, mas MC é mais geral)
        n_mc = 200_000
        amostras = levy.rvs(loc=self.loc_levy_, scale=self.scale_levy_, size=n_mc)
        perdas_extremas = amostras[amostras > var_queda]
        es_queda = perdas_extremas.mean()

        return float(var_queda), float(es_queda)

    # --------------------------------------------------------
    # 2.5) Simulação futura de preços com Lévy calibrada
    # --------------------------------------------------------
    def simular_caminhos_futuros(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simula caminhos futuros de preço com choques de Lévy calibrados.
        """
        if self.loc_levy_ is None or self.scale_levy_ is None:
            raise ValueError("Distribuição de Lévy ainda não ajustada.")

        if self.precos_ is None:
            raise ValueError("Série de preços ainda não existe.")

        n = self.cfg.dias_futuro
        m = self.cfg.n_cenarios_futuros

        dt = 1.0 / self.pm.dias_ano
        mu = self.pm.drift_anual
        sigma = self.pm.volatilidade_anual
        p = self.pl.prob_evento_queda

        s0 = float(self.precos_.iloc[-1])

        precos_futuros = np.zeros((n + 1, m))
        precos_futuros[0, :] = s0

        for c in range(m):
            for t in range(1, n + 1):
                z = np.random.randn()
                retorno_difusivo = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z

                # choque de Lévy calibrado
                if np.random.rand() < p:
                    queda = levy.rvs(loc=self.loc_levy_, scale=self.scale_levy_)
                else:
                    queda = 0.0

                log_retorno = retorno_difusivo - queda
                precos_futuros[t, c] = precos_futuros[t-1, c] * np.exp(log_retorno)

        media = precos_futuros.mean(axis=1)
        return precos_futuros, media

    # --------------------------------------------------------
    # 2.6) Gráficos de diagnóstico
    # --------------------------------------------------------
    def plot_serie_precos(self):
        if self.precos_ is None:
            raise ValueError("Precisa de preços para plotar.")
        plt.figure()
        self.precos_.plot()
        plt.title("Série de preços da ação")
        plt.xlabel("Tempo")
        plt.ylabel("Preço")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_histograma_cauda_vs_levy(self, n_bins: int = 40):
        if self.cauda_quedas_ is None or self.cauda_quedas_.empty:
            raise ValueError("Cauda de quedas ainda não definida.")
        if self.loc_levy_ is None or self.scale_levy_ is None:
            raise ValueError("Distribuição de Lévy ainda não ajustada.")

        dados = self.cauda_quedas_.values
        x_min = dados.min()
        x_max = np.percentile(dados, 99.5)
        x = np.linspace(x_min, x_max, 400)
        pdf = levy.pdf(x, loc=self.loc_levy_, scale=self.scale_levy_)

        plt.figure()
        plt.hist(dados, bins=n_bins, density=True, alpha=0.4, label="Cauda de quedas empírica")
        plt.plot(x, pdf, linewidth=2, label="PDF Lévy ajustada")
        plt.title("Cauda de quedas vs distribuição de Lévy")
        plt.xlabel("Magnitude da queda |r_t| (log-retorno)")
        plt.ylabel("Densidade")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_qq_levy(self):
        """
        Q-Q plot da cauda de quedas vs quantis da Lévy ajustada.
        Se a Lévy for adequada, os pontos ficam próximos da reta 45º.
        """
        if self.cauda_quedas_ is None or self.cauda_quedas_.empty:
            raise ValueError("Cauda de quedas ainda não definida.")
        if self.loc_levy_ is None or self.scale_levy_ is None:
            raise ValueError("Distribuição de Lévy ainda não ajustada.")

        dados = np.sort(self.cauda_quedas_.values)
        n = len(dados)
        probs = (np.arange(1, n + 1) - 0.5) / n
        quantis_teoricos = levy.ppf(probs, loc=self.loc_levy_, scale=self.scale_levy_)

        plt.figure()
        plt.scatter(quantis_teoricos, dados, s=10, alpha=0.7)
        minimo = min(dados.min(), quantis_teoricos.min())
        maximo = max(dados.max(), quantis_teoricos.max())
        plt.plot([minimo, maximo], [minimo, maximo], linestyle="--", linewidth=1.5)
        plt.title("Q-Q plot — cauda de quedas vs Lévy ajustada")
        plt.xlabel("Quantis teóricos Lévy")
        plt.ylabel("Quantis empíricos (quedas)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_simulacoes_futuras(self, precos_futuros: np.ndarray, media: np.ndarray):
        plt.figure()
        n_cenarios = precos_futuros.shape[1]
        for c in range(min(n_cenarios, 50)):
            plt.plot(precos_futuros[:, c], alpha=0.25)
        plt.plot(media, linewidth=2, label="Média dos cenários")
        plt.title("Simulações futuras de preço (GBM + choques de Lévy)")
        plt.xlabel("Dias no futuro")
        plt.ylabel("Preço simulado")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# ------------------------------------------------------------
# 3) Exemplo de uso
# ------------------------------------------------------------
if __name__ == "__main__":
    pm = ParametrosMercado(
        preco_inicial=100.0,
        dias_ano=252,
        drift_anual=0.08,
        volatilidade_anual=0.25
    )

    pl = ParametrosLevy(
        prob_evento_queda=0.08,   # ~8% dos dias têm choque "de Lévy"
        loc_chute=0.0,
        scale_chute=0.025
    )

    cfg = ConfiguracaoModelo(
        usar_csv=False,                # mude para True para usar dados reais
        caminho_csv="meus_dados.csv",  # se usar CSV, ajuste aqui
        coluna_preco=None,
        dias_historico_sintetico=2500,
        dias_futuro=252,
        n_cenarios_futuros=400,
        quantil_cauda=0.1,
        nivel_confianca=0.99,
        semente=42
    )

    modelo = ModeloLevyAvancadoAcoes(pm, pl, cfg)

    # 1) Preços históricos
    precos = modelo.carregar_ou_simular_precos()
    print("Primeiros preços:")
    print(precos.head())

    modelo.calcular_log_retornos()
    modelo.extrair_cauda_quedas()

    print("\nResumo da cauda de quedas (|r_t|):")
    print(modelo.cauda_quedas_.describe())

    # 2) Ajuste da Lévy
    loc_est, scale_est = modelo.ajustar_levy()
    print("\nParâmetros da Lévy ajustada à cauda de quedas:")
    print(f"loc_levy   = {loc_est:.6f}")
    print(f"scale_levy = {scale_est:.6f}")

    # 3) Medidas de risco (VaR e ES da queda)
    var_queda, es_queda = modelo.calcular_var_es_diario()
    alpha = cfg.nivel_confianca
    print(f"\nNível de confiança: {alpha*100:.1f}%")
    print(f"VaR diário da queda (log-retorno)  ≈ {var_queda:.4f} (~{var_queda*100:.2f}%)")
    print(f"ES  diário da queda (log-retorno)  ≈ {es_queda:.4f} (~{es_queda*100:.2f}%)")

    # 4) Simulações futuras
    precos_futuros, media = modelo.simular_caminhos_futuros()
    print("\nPreços médios simulados (primeiros dias):")
    for d in range(6):
        print(f"Dia {d:3d}: {media[d]:.2f}")

    # 5) Gráficos
    modelo.plot_serie_precos()
    modelo.plot_histograma_cauda_vs_levy()
    modelo.plot_qq_levy()
    modelo.plot_simulacoes_futuras(precos_futuros, media)
