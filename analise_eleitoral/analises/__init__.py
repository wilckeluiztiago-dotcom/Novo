"""
Módulos de análises eleitorais específicas.
"""

from .coligacoes import AnalisadorColigacoes
from .volatilidade import CalculadorVolatilidade
from .fragmentacao import AnalisadorFragmentacao
from .competitividade import IndiceCompetitividade

__all__ = [
    'AnalisadorColigacoes',
    'CalculadorVolatilidade',
    'AnalisadorFragmentacao',
    'IndiceCompetitividade'
]
