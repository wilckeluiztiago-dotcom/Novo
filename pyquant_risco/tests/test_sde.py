from __future__ import annotations

import numpy as np

from pyquant_risco.models.sde import ParametrosGBM, simular_gbm, estimar_parametros_gbm


def test_simular_gbm_dimensoes():
    param = ParametrosGBM(
        preco_inicial=100.0,
        mu=0.1,
        volatilidade=0.2,
        passos=10,
        cenarios=5,
        dt=1 / 252,
    )
    caminhos = simular_gbm(param, seed=42)
    assert caminhos.shape == (5, 11)
    assert np.all(caminhos > 0)


def test_estimar_parametros_gbm():
    # gera retornos sintéticos
    mu_true = 0.1
    sigma_true = 0.3
    dt = 1 / 252
    n = 10_000
    retornos = (mu_true - 0.5 * sigma_true**2) * dt + sigma_true * np.sqrt(dt) * np.random.randn(n)

    mu_est, sigma_est = estimar_parametros_gbm(retornos, dt=dt)
    assert abs(mu_est - mu_true) < 0.05
    assert abs(sigma_est - sigma_true) < 0.05
