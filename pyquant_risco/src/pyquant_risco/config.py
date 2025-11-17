from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class ConfiguracaoGeral:
    """
    Configurações globais do projeto, carregadas de variáveis de ambiente.

    Permite customizar o comportamento sem alterar código.
    """

    nivel_log: str = os.getenv("PYQUANT_NIVEL_LOG", "INFO")
    diretorio_dados: str = os.getenv("PYQUANT_DIRETORIO_DADOS", "dados")
    host_api: str = os.getenv("PYQUANT_HOST_API", "127.0.0.1")
    porta_api: int = int(os.getenv("PYQUANT_PORTA_API", "8000"))
    usar_gpu: bool = os.getenv("PYQUANT_USAR_GPU", "false").lower() == "true"

    @classmethod
    def carregar(cls) -> "ConfiguracaoGeral":
        return cls()


config_global: Optional[ConfiguracaoGeral] = None


def obter_configuracao() -> ConfiguracaoGeral:
    global config_global
    if config_global is None:
        config_global = ConfiguracaoGeral.carregar()
    return config_global
