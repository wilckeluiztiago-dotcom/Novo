from __future__ import annotations

from typing import Literal, Optional

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ..data.market_data import CarregadorDadosMercado
from ..logging_utils import get_logger
from ..models.sde import ParametrosGBM, ParametrosOU, simular_gbm, simular_ou
from ..pipeline.risk_engine import MetricasRisco, MotorRisco

logger = get_logger()

app = FastAPI(title="PyQuant Risco API", version="0.1.0")


class RequisicaoSimulacaoSDE(BaseModel):
    tipo: Literal["gbm", "ou"] = "gbm"
    preco_inicial: float = 100.0
    mu: float = 0.08
    volatilidade: float = 0.25
    passos: int = 252
    cenarios: int = 1_000
    dt: float = 1 / 252
    # parâmetros OU
    nivel_medio: float = 0.0
    velocidade_reversao: float = 1.0
    x0: float = 0.0
    seed: Optional[int] = None


class RespostaSimulacaoSDE(BaseModel):
    caminhos: list[list[float]]


class RequisicaoRisco(BaseModel):
    caminho_csv: str
    nivel_confianca: float = Field(0.99, ge=0.8, le=0.999)
    seed: Optional[int] = None


class RespostaRisco(BaseModel):
    preco_atual: float
    mu_est: float
    sigma_est: float
    var: float
    cvar: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/simular/sde", response_model=RespostaSimulacaoSDE)
def simular_sde(req: RequisicaoSimulacaoSDE) -> RespostaSimulacaoSDE:
    if req.tipo == "gbm":
        param = ParametrosGBM(
            preco_inicial=req.preco_inicial,
            mu=req.mu,
            volatilidade=req.volatilidade,
            passos=req.passos,
            cenarios=req.cenarios,
            dt=req.dt,
        )
        caminhos = simular_gbm(param, seed=req.seed)
    else:
        param_ou = ParametrosOU(
            x0=req.x0,
            nivel_medio=req.nivel_medio,
            velocidade_reversao=req.velocidade_reversao,
            volatilidade=req.volatilidade,
            passos=req.passos,
            cenarios=req.cenarios,
            dt=req.dt,
        )
        caminhos = simular_ou(param_ou, seed=req.seed)

    return RespostaSimulacaoSDE(caminhos=caminhos.tolist())


@app.post("/risco/var-cvar", response_model=RespostaRisco)
def calcular_risco(req: RequisicaoRisco) -> RespostaRisco:
    logger.info("Recebida requisição de risco para CSV: %s", req.caminho_csv)
    loader = CarregadorDadosMercado(req.caminho_csv)
    dados = loader.carregar()

    motor = MotorRisco(nivel_confianca=req.nivel_confianca)
    resultado = motor.rodar_pipeline(dados, seed=req.seed)

    return RespostaRisco(**resultado)
