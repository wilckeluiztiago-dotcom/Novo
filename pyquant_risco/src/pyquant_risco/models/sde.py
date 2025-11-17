from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..logging_utils import get_logger

logger = get_logger()


@dataclass
class ParametrosGBM:
    preco_inicial: float
    mu: float
    volatilidade: float
    dt: float = 1 / 252
    passos: int = 252
    cenarios: int = 1


@dataclass
class ParametrosOU:
    nivel_medio: float
    velocidade_reversao: float
    volatilidade: float
    x0: float
    dt: float = 1 / 252
    passos: int = 252
    cenarios: int = 1


def simular_gbm(param: ParametrosGBM, seed: int | None = None) -> np.ndarray:
    """
    Simula um Geometric Brownian Motion (modelo de Black–Scholes).
    Retorna array (cenarios, passos+1).
    """
    if seed is not None:
        np.random.seed(seed)

    logger.debug("Simulando GBM com parâmetros: %s", param)
    dt = param.dt
    passos = param.passos
    cenarios = param.cenarios

    caminhos = np.zeros((cenarios, passos + 1))
    caminhos[:, 0] = param.preco_inicial

    for t in range(1, passos + 1):
        z = np.random.normal(size=cenarios)
        caminhos[:, t] = caminhos[:, t - 1] * np.exp(
            (param.mu - 0.5 * param.volatilidade**2) * dt
            + param.volatilidade * np.sqrt(dt) * z
        )

    return caminhos


def simular_ou(param: ParametrosOU, seed: int | None = None) -> np.ndarray:
    """
    Simula um processo de Ornstein-Uhlenbeck.
    Retorna array (cenarios, passos+1).
    """
    if seed is not None:
        np.random.seed(seed)

    logger.debug("Simulando OU com parâmetros: %s", param)

    dt = param.dt
    passos = param.passos
    cenarios = param.cenarios

    caminhos = np.zeros((cenarios, passos + 1))
    caminhos[:, 0] = param.x0

    for t in range(1, passos + 1):
        z = np.random.normal(size=cenarios)
        caminhos[:, t] = (
            caminhos[:, t - 1]
            + param.velocidade_reversao
            * (param.nivel_medio - caminhos[:, t - 1])
            * dt
            + param.volatilidade * np.sqrt(dt) * z
        )

    return caminhos


def estimar_parametros_gbm(retornos_log: np.ndarray, dt: float = 1 / 252) -> tuple[float, float]:
    """
    Estima drift (mu) e volatilidade (sigma) de um GBM
    a partir de retornos logarítmicos.
    """
    media = np.mean(retornos_log)
    variancia = np.var(retornos_log, ddof=1)

    mu = media / dt + 0.5 * variancia / dt
    sigma = np.sqrt(variancia / dt)

    logger.info("Parâmetros estimados GBM: mu=%.4f, sigma=%.4f", mu, sigma)
    return mu, sigma
