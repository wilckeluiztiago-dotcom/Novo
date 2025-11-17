from __future__ import annotations

import logging
import sys
from typing import Optional

from .config import obter_configuracao


def configurar_logging(nome_logger: str = "pyquant_risco") -> logging.Logger:
    cfg = obter_configuracao()
    logger = logging.getLogger(nome_logger)

    if logger.handlers:
        return logger  # já configurado

    logger.setLevel(getattr(logging, cfg.nivel_log.upper(), logging.INFO))

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.debug("Logger configurado com nível %s", cfg.nivel_log)
    return logger


logger_principal: Optional[logging.Logger] = None


def get_logger() -> logging.Logger:
    global logger_principal
    if logger_principal is None:
        logger_principal = configurar_logging()
    return logger_principal
