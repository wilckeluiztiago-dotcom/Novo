from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import typer
import uvicorn

from .api.server import app
from .config import obter_configuracao
from .data.market_data import CarregadorDadosMercado
from .logging_utils import get_logger
from .models.deeplearning import ConfigRedeLSTM, treinar_lstm_retornos
from .models.sde import ParametrosGBM, simular_gbm
from .pipeline.risk_engine import MotorRisco

app_cli = typer.Typer(help="PyQuant Risco — CLI")
logger = get_logger()


@app_cli.command("simular-sde")
def simular_sde_cli(
    preco_inicial: float = typer.Option(100.0),
    mu: float = typer.Option(0.08),
    volatilidade: float = typer.Option(0.25),
    passos: int = typer.Option(252),
    cenarios: int = typer.Option(1_000),
    dt: float = typer.Option(1 / 252),
    seed: Optional[int] = typer.Option(None),
    salvar_figura: Optional[str] = typer.Option(None, help="Caminho para salvar gráfico PNG"),
) -> None:
    """
    Simula GBM e mostra (ou salva) gráfico com algumas trajetórias.
    """
    param = ParametrosGBM(
        preco_inicial=preco_inicial,
        mu=mu,
        volatilidade=volatilidade,
        passos=passos,
        cenarios=cenarios,
        dt=dt,
    )
    caminhos = simular_gbm(param, seed=seed)
    logger.info("Simulação concluída: %s cenários, %s passos", *caminhos.shape)

    plt.figure(figsize=(10, 5))
    for i in range(min(20, cenarios)):
        plt.plot(caminhos[i], alpha=0.5)
    plt.title("Trajetórias simuladas (GBM)")
    plt.xlabel("Passo")
    plt.ylabel("Preço")
    plt.tight_layout()

    if salvar_figura:
        Path(salvar_figura).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(salvar_figura)
        logger.info("Figura salva em %s", salvar_figura)
    else:
        plt.show()


@app_cli.command("calcular-risco")
def calcular_risco_cli(
    caminho_csv: str = typer.Argument(..., help="CSV com colunas: data,preco"),
    nivel_confianca: float = typer.Option(0.99),
    seed: Optional[int] = typer.Option(None),
) -> None:
    """
    Roda pipeline de risco completo a partir de um CSV.
    """
    loader = CarregadorDadosMercado(caminho_csv)
    dados = loader.carregar()

    motor = MotorRisco(nivel_confianca=nivel_confianca)
    resultado = motor.rodar_pipeline(dados, seed=seed)

    typer.echo(json.dumps(resultado, indent=2))


@app_cli.command("treinar-rede")
def treinar_rede_cli(
    caminho_csv: str = typer.Argument(..., help="CSV com colunas: data,preco"),
    tamanho_janela: int = typer.Option(20),
    tamanho_oculto: int = typer.Option(32),
    epocas: int = typer.Option(10),
    usar_gpu: bool = typer.Option(False),
) -> None:
    """
    Treina LSTM para prever retornos e imprime perda final.
    """
    loader = CarregadorDadosMercado(caminho_csv)
    dados = loader.carregar()

    config = ConfigRedeLSTM(
        tamanho_janela=tamanho_janela,
        tamanho_oculto=tamanho_oculto,
        epocas=epocas,
        usar_gpu=usar_gpu,
    )

    modelo, perda = treinar_lstm_retornos(dados.retornos, config)
    if modelo is None:
        typer.echo("PyTorch não disponível. Instale 'torch' para usar este comando.")
    else:
        typer.echo(f"Treino concluído. Perda final: {perda:.6f}")


@app_cli.command("api")
def iniciar_api() -> None:
    """
    Inicializa a API FastAPI com base na configuração.
    """
    cfg = obter_configuracao()
    logger.info("Inicializando API em %s:%d", cfg.host_api, cfg.porta_api)
    uvicorn.run(app, host=cfg.host_api, port=cfg.porta_api)


app = app_cli
