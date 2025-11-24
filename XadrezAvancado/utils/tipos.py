"""
Definições de Tipos e Dataclasses
"""

from dataclasses import dataclass
from typing import Tuple, Optional
from config import VAZIO

@dataclass
class Movimento:
    origem: Tuple[int, int]
    destino: Tuple[int, int]
    peca: int
    capturada: int = VAZIO
    promocao: int = VAZIO
    en_passant: bool = False
    roque: bool = False
    roque_longo: bool = False

    def __str__(self) -> str:
        colunas = "abcdefgh"
        return f"{colunas[self.origem[1]]}{8-self.origem[0]}{colunas[self.destino[1]]}{8-self.destino[0]}"

@dataclass
class Historico:
    movimento: Movimento
    capturada: int
    direitos_roque: int
    coluna_en_passant: int
    relogio_meia_jogada: int
    hash_anterior: int

@dataclass
class EntradaTT:
    hash_chave: int
    profundidade: int
    valor: int
    flag: int  # 0=exato, 1=limite_inferior, 2=limite_superior
    melhor_movimento: Optional[Movimento]
