from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..logging_utils import get_logger

logger = get_logger()


@dataclass
class DadosMercado:
    datas: np.ndarray
    precos: np.ndarray
    retornos: np.ndarray


class CarregadorDadosMercado:
    """
    Responsável por carregar e pré-processar dados de mercado a partir de CSV.
    Espera um CSV com colunas 'data' e 'preco'.
    """

    def __init__(self, caminho_csv: str | Path) -> None:
        self.caminho_csv = Path(caminho_csv)

    def carregar(self) -> DadosMercado:
        logger.info("Carregando dados de mercado de %s", self.caminho_csv)
        df = pd.read_csv(self.caminho_csv)

        if "data" not in df.columns or "preco" not in df.columns:
            raise ValueError("CSV deve conter colunas 'data' e 'preco'.")

        df["data"] = pd.to_datetime(df["data"])
        df = df.sort_values("data")

        precos = df["preco"].astype(float).to_numpy()
        datas = df["data"].to_numpy()

        retornos = np.diff(np.log(precos))
        logger.info("Dados carregados: %d observações.", len(precos))

        return DadosMercado(datas=datas[1:], precos=precos[1:], retornos=retornos)
