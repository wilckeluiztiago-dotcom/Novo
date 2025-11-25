"""
Utilitários para geração de dados e métricas.
"""

from .dados import gerar_dados_eleitorais, gerar_dados_historicos
from .metricas import calcular_metricas_regressao, calcular_metricas_classificacao

__all__ = [
    'gerar_dados_eleitorais',
    'gerar_dados_historicos',
    'calcular_metricas_regressao',
    'calcular_metricas_classificacao'
]
