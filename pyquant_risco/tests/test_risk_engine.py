from __future__ import annotations

import numpy as np

from pyquant_risco.data.market_data import DadosMercado
from pyquant_risco.pipeline.risk_engine import MotorRisco


def criar_dados_sinteticos() -> DadosMercado:
    n = 1_000
    datas = np.arange(n, dtype="datetime64[D]")
    precos = 100 * np.exp(0.0005 * np.arange(n) + 0.01 * np.random.randn(n))
    retornos = np.diff(np.log(precos))
    return DadosMercado(datas=datas[1:], precos=precos[1:], retornos=retornos)


def test_motor_risco_pipeline():
    dados = criar_dados_sinteticos()
    motor = MotorRisco(nivel_confianca=0.95)
    resultado = motor.rodar_pipeline(dados, seed=123)

    assert "preco_atual" in resultado
    assert "var" in resultado
    assert "cvar" in resultado
    assert resultado["var"] >= 0
    assert resultado["cvar"] >= resultado["var"]
