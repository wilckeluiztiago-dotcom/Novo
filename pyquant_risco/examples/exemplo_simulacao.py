from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from pyquant_risco.data.market_data import DadosMercado
from pyquant_risco.models.sde import ParametrosGBM, simular_gbm, estimar_parametros_gbm
from pyquant_risco.pipeline.risk_engine import MotorRisco


def gerar_dados_sinteticos() -> DadosMercado:
    n = 1_000
    datas = np.arange(n, dtype="datetime64[D]")
    precos = 100 * np.exp(0.0003 * np.arange(n) + 0.02 * np.random.randn(n))
    retornos = np.diff(np.log(precos))
    return DadosMercado(datas=datas[1:], precos=precos[1:], retornos=retornos)


def main() -> None:
    dados = gerar_dados_sinteticos()

    mu_est, sigma_est = estimar_parametros_gbm(dados.retornos)
    print(f"Parâmetros estimados: mu={mu_est:.4f}, sigma={sigma_est:.4f}")

    motor = MotorRisco(nivel_confianca=0.99)
    resultado = motor.rodar_pipeline(dados, seed=123)
    print("Resultado de risco:")
    for k, v in resultado.items():
        print(f"  {k}: {v:.6f}")

    param = ParametrosGBM(
        preco_inicial=dados.precos[-1],
        mu=resultado["mu_est"],
        volatilidade=resultado["sigma_est"],
        passos=252,
        cenarios=50,
    )
    caminhos = simular_gbm(param, seed=123)

    plt.figure(figsize=(10, 5))
    for i in range(caminhos.shape[0]):
        plt.plot(caminhos[i], alpha=0.4)
    plt.title("Trajetórias simuladas (horizonte 1 ano)")
    plt.xlabel("Passos")
    plt.ylabel("Preço")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
