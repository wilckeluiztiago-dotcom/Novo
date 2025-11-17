from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from ..data.market_data import DadosMercado
from ..logging_utils import get_logger
from ..models.sde import ParametrosGBM, estimar_parametros_gbm, simular_gbm

logger = get_logger()


@dataclass
class MetricasRisco:
    var: float
    cvar: float


class MotorRisco:
    """
    Motor de risco que integra estimativa de parâmetros,
    simulação de Monte Carlo e cálculo de métricas.
    """

    def __init__(self, nivel_confianca: float = 0.99) -> None:
        self.nivel_confianca = nivel_confianca

    def calibrar_gbm(self, dados: DadosMercado) -> ParametrosGBM:
        mu, sigma = estimar_parametros_gbm(dados.retornos)
        return ParametrosGBM(
            preco_inicial=float(dados.precos[-1]),
            mu=mu,
            volatilidade=sigma,
            dt=1 / 252,
            passos=252,
            cenarios=10_000,
        )

    def simular_caminhos(self, parametros: ParametrosGBM, seed: int | None = None) -> np.ndarray:
        caminhos = simular_gbm(parametros, seed=seed)
        logger.info("Simulação concluída: %s cenários, %s passos", *caminhos.shape)
        return caminhos

    def calcular_metricas(self, caminhos: np.ndarray) -> MetricasRisco:
        # perda relativa no horizonte final
        precos_iniciais = caminhos[:, 0]
        precos_finais = caminhos[:, -1]
        retornos = precos_finais / precos_iniciais - 1.0
        perdas = -retornos  # perda = -retorno

        alpha = 1 - self.nivel_confianca
        quantil = np.quantile(perdas, self.nivel_confianca)
        cvar = perdas[perdas >= quantil].mean()

        logger.info("VaR(%.1f%%) = %.4f, CVaR = %.4f", self.nivel_confianca * 100, quantil, cvar)
        return MetricasRisco(var=float(quantil), cvar=float(cvar))

    def rodar_pipeline(self, dados: DadosMercado, seed: int | None = None) -> Dict[str, float]:
        logger.info("Rodando pipeline completo de risco.")
        parametros = self.calibrar_gbm(dados)
        caminhos = self.simular_caminhos(parametros, seed=seed)
        metricas = self.calcular_metricas(caminhos)

        return {
            "preco_atual": float(dados.precos[-1]),
            "mu_est": float(parametros.mu),
            "sigma_est": float(parametros.volatilidade),
            "var": metricas.var,
            "cvar": metricas.cvar,
        }
