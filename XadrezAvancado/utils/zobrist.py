"""
Hashing Zobrist para Tabela de Transposição
"""

import random

# Semente fixa para reprodutibilidade
random.seed(42)

ZOBRIST_PECA = [
    [[random.getrandbits(64) for _ in range(12)] for _ in range(8)]
    for _ in range(8)
]
ZOBRIST_LADO_A_JOGAR = random.getrandbits(64)
ZOBRIST_ROQUE = [random.getrandbits(64) for _ in range(16)]
ZOBRIST_COLUNA_EP = [random.getrandbits(64) for _ in range(8)]

def indice_peca_zobrist(peca: int) -> int:
    """Mapeia peça (-6 a 6) para índice 0-11"""
    if peca > 0:
        base = 0
    else:
        base = 6
    return base + abs(peca) - 1
