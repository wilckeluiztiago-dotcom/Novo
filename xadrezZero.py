import pygame
import sys
import time
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

# ============================================================
# XADREZ AVANÇADO — IA com Minimax + Poda Alpha-Beta + TT
# Autor: Luiz Tiago Wilcke (LT)
# ============================================================

# ----------------------------
# Configurações de Pygame
# ----------------------------
LADO_CASA = 80
TAMANHO_TABULEIRO = LADO_CASA * 8
LARGURA_PAINEL_INFO = 220
LARGURA_JANELA = TAMANHO_TABULEIRO + LARGURA_PAINEL_INFO
ALTURA_JANELA = TAMANHO_TABULEIRO

QUADROS_POR_SEGUNDO = 60

COR_TAB_CLARO = (240, 217, 181)
COR_TAB_ESCURO = (181, 136, 99)
COR_FUNDO = (15, 23, 42)
COR_DESTAQUE = (252, 211, 77)
COR_SELECIONADO = (56, 189, 248)
COR_TEXTO = (248, 250, 252)
COR_IA = (94, 234, 212)

# ----------------------------
# Representação das peças
# ----------------------------
VAZIO = 0
PEAO_BRANCO, CAVALO_BRANCO, BISPO_BRANCO, TORRE_BRANCA, DAMA_BRANCA, REI_BRANCO = 1, 2, 3, 4, 5, 6
PEAO_PRETO, CAVALO_PRETO, BISPO_PRETO, TORRE_PRETA, DAMA_PRETA, REI_PRETO = -1, -2, -3, -4, -5, -6

PECAS_UNICODE = {
    PEAO_BRANCO: "♙", CAVALO_BRANCO: "♘", BISPO_BRANCO: "♗",
    TORRE_BRANCA: "♖", DAMA_BRANCA: "♕", REI_BRANCO: "♔",
    PEAO_PRETO: "♟", CAVALO_PRETO: "♞", BISPO_PRETO: "♝",
    TORRE_PRETA: "♜", DAMA_PRETA: "♛", REI_PRETO: "♚",
}

def eh_branca(peca: int) -> bool:
    return peca > 0

def eh_preta(peca: int) -> bool:
    return peca < 0

def sinal(peca: int) -> int:
    if peca > 0:
        return 1
    if peca < 0:
        return -1
    return 0

# ----------------------------
# Tabelas de posição
# ----------------------------
TABELA_PEAO = [
    0, 0, 0, 0, 0, 0, 0, 0,
    5, 10, 10, -20, -20, 10, 10, 5,
    5, -5, -10, 0, 0, -10, -5, 5,
    0, 0, 0, 20, 20, 0, 0, 0,
    5, 5, 10, 25, 25, 10, 5, 5,
    10, 10, 20, 30, 30, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,
    0, 0, 0, 0, 0, 0, 0, 0,
]

TABELA_CAVALO = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]

TABELA_BISPO = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
]

TABELA_TORRE = [
    0, 0, 0, 5, 5, 0, 0, 0,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    5, 10, 10, 10, 10, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0,
]

TABELA_DAMA = [
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 5, 0, 0, 0, 0, -10,
    -10, 5, 5, 5, 5, 5, 0, -10,
    0, 0, 5, 5, 5, 5, 0, -5,
    -5, 0, 5, 5, 5, 5, 0, -5,
    -10, 0, 5, 0, 0, 0, 0, -10,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20,
]

TABELA_REI_MEIOJOGO = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20, 0, 0, 0, 0, 20, 20,
    20, 30, 10, 0, 0, 10, 30, 20,
]

TABELAS_POSICAO = {
    PEAO_BRANCO: TABELA_PEAO,
    CAVALO_BRANCO: TABELA_CAVALO,
    BISPO_BRANCO: TABELA_BISPO,
    TORRE_BRANCA: TABELA_TORRE,
    DAMA_BRANCA: TABELA_DAMA,
    REI_BRANCO: TABELA_REI_MEIOJOGO,
    PEAO_PRETO: TABELA_PEAO,
    CAVALO_PRETO: TABELA_CAVALO,
    BISPO_PRETO: TABELA_BISPO,
    TORRE_PRETA: TABELA_TORRE,
    DAMA_PRETA: TABELA_DAMA,
    REI_PRETO: TABELA_REI_MEIOJOGO,
}

VALOR_PECA = {
    PEAO_BRANCO: 100, CAVALO_BRANCO: 320, BISPO_BRANCO: 330,
    TORRE_BRANCA: 500, DAMA_BRANCA: 900, REI_BRANCO: 20000,
    PEAO_PRETO: -100, CAVALO_PRETO: -320, BISPO_PRETO: -330,
    TORRE_PRETA: -500, DAMA_PRETA: -900, REI_PRETO: -20000,
}

# ----------------------------
# Direitos de roque (CORRIGIDOS)
# ----------------------------
ROQUE_BRANCO_REI = 1   # roque pequeno branco
ROQUE_BRANCO_DAMA = 2  # roque grande branco
ROQUE_PRETO_REI = 4    # roque pequeno preto
ROQUE_PRETO_DAMA = 8   # roque grande preto

# ----------------------------
# Hashing Zobrist
# ----------------------------
import random as _random

_random.seed(42)
ZOBRIST_PECA = [
    [[_random.getrandbits(64) for _ in range(12)] for _ in range(8)]
    for _ in range(8)
]
ZOBRIST_LADO_A_JOGAR = _random.getrandbits(64)
ZOBRIST_ROQUE = [_random.getrandbits(64) for _ in range(16)]
ZOBRIST_COLUNA_EP = [_random.getrandbits(64) for _ in range(8)]

def indice_peca_zobrist(peca: int) -> int:
    if peca > 0:
        base = 0
    else:
        base = 6
    return base + abs(peca) - 1

# ----------------------------
# Estruturas de dados
# ----------------------------
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

def definir_bit(mascara: int, bitmask: int, valor: bool) -> int:
    """Liga/desliga um bit específico (bitmask = 1,2,4,8...)."""
    if valor:
        return mascara | bitmask
    else:
        return mascara & ~bitmask

# ----------------------------
# Estado do jogo
# ----------------------------
class Estado:
    def __init__(self):
        self.tabuleiro = [[VAZIO for _ in range(8)] for _ in range(8)]
        self.brancas_jogam = True
        self.direitos_roque = (
            ROQUE_BRANCO_REI | ROQUE_BRANCO_DAMA |
            ROQUE_PRETO_REI | ROQUE_PRETO_DAMA
        )
        self.coluna_en_passant = -1
        self.relogio_meia_jogada = 0
        self.numero_jogada_completa = 1
        self.historico: List[Historico] = []
        self.hash_atual = 0
        self._inicializar_tabuleiro()
        self._atualizar_hash_inicial()

    def _inicializar_tabuleiro(self):
        # Pretas
        self.tabuleiro[0] = [
            TORRE_PRETA, CAVALO_PRETO, BISPO_PRETO, DAMA_PRETA,
            REI_PRETO, BISPO_PRETO, CAVALO_PRETO, TORRE_PRETA
        ]
        self.tabuleiro[1] = [PEAO_PRETO] * 8
        # Casas vazias
        for linha in range(2, 6):
            self.tabuleiro[linha] = [VAZIO] * 8
        # Brancas
        self.tabuleiro[6] = [PEAO_BRANCO] * 8
        self.tabuleiro[7] = [
            TORRE_BRANCA, CAVALO_BRANCO, BISPO_BRANCO, DAMA_BRANCA,
            REI_BRANCO, BISPO_BRANCO, CAVALO_BRANCO, TORRE_BRANCA
        ]

    def _atualizar_hash_inicial(self):
        h = 0
        for linha in range(8):
            for coluna in range(8):
                peca = self.tabuleiro[linha][coluna]
                if peca != VAZIO:
                    idx = indice_peca_zobrist(peca)
                    h ^= ZOBRIST_PECA[linha][coluna][idx]
        if not self.brancas_jogam:
            h ^= ZOBRIST_LADO_A_JOGAR
        h ^= ZOBRIST_ROQUE[self.direitos_roque]
        if self.coluna_en_passant != -1:
            h ^= ZOBRIST_COLUNA_EP[self.coluna_en_passant]
        self.hash_atual = h

    def _aplicar_hash_peca(self, linha: int, coluna: int, peca: int):
        if peca == VAZIO:
            return
        idx = indice_peca_zobrist(peca)
        self.hash_atual ^= ZOBRIST_PECA[linha][coluna][idx]

    def _aplicar_hash_lado(self):
        self.hash_atual ^= ZOBRIST_LADO_A_JOGAR

    def _aplicar_hash_roque(self, antigo: int, novo: int):
        self.hash_atual ^= ZOBRIST_ROQUE[antigo]
        self.hash_atual ^= ZOBRIST_ROQUE[novo]

    def _aplicar_hash_en_passant(self, coluna_antiga: int, coluna_nova: int):
        if coluna_antiga != -1:
            self.hash_atual ^= ZOBRIST_COLUNA_EP[coluna_antiga]
        if coluna_nova != -1:
            self.hash_atual ^= ZOBRIST_COLUNA_EP[coluna_nova]

    # ----------------------------
    # Utilitários
    # ----------------------------
    def dentro(self, linha: int, coluna: int) -> bool:
        return 0 <= linha < 8 and 0 <= coluna < 8

    def em_xeque(self, brancas: bool) -> bool:
        alvo = REI_BRANCO if brancas else REI_PRETO
        linha_rei = coluna_rei = -1
        for linha in range(8):
            for coluna in range(8):
                if self.tabuleiro[linha][coluna] == alvo:
                    linha_rei, coluna_rei = linha, coluna
                    break
            if linha_rei != -1:
                break
        if linha_rei == -1:
            return False
        return self._casa_atacada(linha_rei, coluna_rei, por_brancas=not brancas)

    def _casa_atacada(self, linha: int, coluna: int, por_brancas: bool) -> bool:
        # Peões
        if por_brancas:
            delta_linha = -1
            for delta_coluna in [-1, 1]:
                lr, cr = linha + delta_linha, coluna + delta_coluna
                if self.dentro(lr, cr) and self.tabuleiro[lr][cr] == PEAO_BRANCO:
                    return True
        else:
            delta_linha = 1
            for delta_coluna in [-1, 1]:
                lr, cr = linha + delta_linha, coluna + delta_coluna
                if self.dentro(lr, cr) and self.tabuleiro[lr][cr] == PEAO_PRETO:
                    return True

        # Cavalos
        movimentos_cavalo = [
            (-2, -1), (-2, 1), (-1, -2), (-1, 2),
            (1, -2), (1, 2), (2, -1), (2, 1)
        ]
        alvo_cavalo = CAVALO_BRANCO if por_brancas else CAVALO_PRETO
        for dl, dc in movimentos_cavalo:
            lr, cr = linha + dl, coluna + dc
            if self.dentro(lr, cr) and self.tabuleiro[lr][cr] == alvo_cavalo:
                return True

        # Bispos / Damas (diagonais)
        alvo_bispo = BISPO_BRANCO if por_brancas else BISPO_PRETO
        alvo_dama = DAMA_BRANCA if por_brancas else DAMA_PRETA
        for dl, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            lr, cr = linha + dl, coluna + dc
            while self.dentro(lr, cr):
                peca = self.tabuleiro[lr][cr]
                if peca != VAZIO:
                    if peca == alvo_bispo or peca == alvo_dama:
                        return True
                    break
                lr += dl
                cr += dc

        # Torres / Damas (retas)
        alvo_torre = TORRE_BRANCA if por_brancas else TORRE_PRETA
        for dl, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            lr, cr = linha + dl, coluna + dc
            while self.dentro(lr, cr):
                peca = self.tabuleiro[lr][cr]
                if peca != VAZIO:
                    if peca == alvo_torre or peca == alvo_dama:
                        return True
                    break
                lr += dl
                cr += dc

        # Rei
        alvo_rei = REI_BRANCO if por_brancas else REI_PRETO
        for dl in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dl == 0 and dc == 0:
                    continue
                lr, cr = linha + dl, coluna + dc
                if self.dentro(lr, cr) and self.tabuleiro[lr][cr] == alvo_rei:
                    return True

        return False

    # ----------------------------
    # Geração de movimentos
    # ----------------------------
    def gerar_movimentos_legais(self) -> List[Movimento]:
        movimentos_pseudo = self._gerar_movimentos_pseudo()
        movimentos_legais: List[Movimento] = []
        for movimento in movimentos_pseudo:
            self.fazer_movimento(movimento)
            if not self.em_xeque(not self.brancas_jogam):
                movimentos_legais.append(movimento)
            self.desfazer_movimento()
        return movimentos_legais

    def _gerar_movimentos_pseudo(self) -> List[Movimento]:
        movimentos: List[Movimento] = []
        brancas_na_vez = self.brancas_jogam
        for linha in range(8):
            for coluna in range(8):
                peca = self.tabuleiro[linha][coluna]
                if peca == VAZIO:
                    continue
                if brancas_na_vez and not eh_branca(peca):
                    continue
                if (not brancas_na_vez) and not eh_preta(peca):
                    continue
                tipo = abs(peca)
                if tipo == 1:
                    self._movimentos_peao(linha, coluna, peca, movimentos)
                elif tipo == 2:
                    self._movimentos_cavalo(linha, coluna, peca, movimentos)
                elif tipo == 3:
                    self._movimentos_bispo(linha, coluna, peca, movimentos)
                elif tipo == 4:
                    self._movimentos_torre(linha, coluna, peca, movimentos)
                elif tipo == 5:
                    self._movimentos_dama(linha, coluna, peca, movimentos)
                elif tipo == 6:
                    self._movimentos_rei(linha, coluna, peca, movimentos)
        return movimentos

    def _movimentos_peao(self, linha: int, coluna: int, peca: int, movimentos: List[Movimento]):
        direcao = -1 if peca > 0 else 1
        linha_inicial = 6 if peca > 0 else 1
        linha_promocao = 0 if peca > 0 else 7

        linha_frente = linha + direcao
        if self.dentro(linha_frente, coluna) and self.tabuleiro[linha_frente][coluna] == VAZIO:
            if linha_frente == linha_promocao:
                novas_pecas = (
                    [DAMA_BRANCA, TORRE_BRANCA, BISPO_BRANCO, CAVALO_BRANCO]
                    if peca > 0 else
                    [DAMA_PRETA, TORRE_PRETA, BISPO_PRETO, CAVALO_PRETO]
                )
                for nova in novas_pecas:
                    movimentos.append(
                        Movimento((linha, coluna), (linha_frente, coluna), peca, VAZIO, nova)
                    )
            else:
                movimentos.append(Movimento((linha, coluna), (linha_frente, coluna), peca))
            if linha == linha_inicial:
                linha_dupla = linha + 2 * direcao
                if self.tabuleiro[linha_dupla][coluna] == VAZIO:
                    movimentos.append(
                        Movimento((linha, coluna), (linha_dupla, coluna), peca)
                    )

        # capturas
        for delta_coluna in [-1, 1]:
            coluna_destino = coluna + delta_coluna
            linha_destino = linha + direcao
            if not self.dentro(linha_destino, coluna_destino):
                continue
            alvo = self.tabuleiro[linha_destino][coluna_destino]
            if alvo != VAZIO and sinal(alvo) != sinal(peca):
                if linha_destino == linha_promocao:
                    novas_pecas = (
                        [DAMA_BRANCA, TORRE_BRANCA, BISPO_BRANCO, CAVALO_BRANCO]
                        if peca > 0 else
                        [DAMA_PRETA, TORRE_PRETA, BISPO_PRETO, CAVALO_PRETO]
                    )
                    for nova in novas_pecas:
                        movimentos.append(
                            Movimento((linha, coluna), (linha_destino, coluna_destino), peca, alvo, nova)
                        )
                else:
                    movimentos.append(
                        Movimento((linha, coluna), (linha_destino, coluna_destino), peca, alvo)
                    )

        # en passant
        if self.coluna_en_passant != -1 and (linha == (3 if peca > 0 else 4)):
            for delta_coluna in [-1, 1]:
                coluna_lateral = coluna + delta_coluna
                if coluna_lateral == self.coluna_en_passant:
                    alvo = self.tabuleiro[linha][coluna_lateral]
                    if alvo == (-peca):
                        movimentos.append(
                            Movimento(
                                (linha, coluna),
                                (linha + direcao, coluna_lateral),
                                peca,
                                alvo,
                                en_passant=True
                            )
                        )

    def _movimentos_cavalo(self, linha: int, coluna: int, peca: int, movimentos: List[Movimento]):
        deslocamentos = [
            (-2, -1), (-2, 1), (-1, -2), (-1, 2),
            (1, -2), (1, 2), (2, -1), (2, 1)
        ]
        for dl, dc in deslocamentos:
            lr, cr = linha + dl, coluna + dc
            if not self.dentro(lr, cr):
                continue
            alvo = self.tabuleiro[lr][cr]
            if alvo == VAZIO or sinal(alvo) != sinal(peca):
                movimentos.append(Movimento((linha, coluna), (lr, cr), peca, alvo))

    def _movimentos_bispo(self, linha: int, coluna: int, peca: int, movimentos: List[Movimento]):
        for dl, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            lr, cr = linha + dl, coluna + dc
            while self.dentro(lr, cr):
                alvo = self.tabuleiro[lr][cr]
                if alvo == VAZIO:
                    movimentos.append(Movimento((linha, coluna), (lr, cr), peca))
                else:
                    if sinal(alvo) != sinal(peca):
                        movimentos.append(Movimento((linha, coluna), (lr, cr), peca, alvo))
                    break
                lr += dl
                cr += dc

    def _movimentos_torre(self, linha: int, coluna: int, peca: int, movimentos: List[Movimento]):
        for dl, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            lr, cr = linha + dl, coluna + dc
            while self.dentro(lr, cr):
                alvo = self.tabuleiro[lr][cr]
                if alvo == VAZIO:
                    movimentos.append(Movimento((linha, coluna), (lr, cr), peca))
                else:
                    if sinal(alvo) != sinal(peca):
                        movimentos.append(Movimento((linha, coluna), (lr, cr), peca, alvo))
                    break
                lr += dl
                cr += dc

    def _movimentos_dama(self, linha: int, coluna: int, peca: int, movimentos: List[Movimento]):
        self._movimentos_bispo(linha, coluna, peca, movimentos)
        self._movimentos_torre(linha, coluna, peca, movimentos)

    def _movimentos_rei(self, linha: int, coluna: int, peca: int, movimentos: List[Movimento]):
        for dl in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dl == 0 and dc == 0:
                    continue
                lr, cr = linha + dl, coluna + dc
                if not self.dentro(lr, cr):
                    continue
                alvo = self.tabuleiro[lr][cr]
                if alvo == VAZIO or sinal(alvo) != sinal(peca):
                    movimentos.append(Movimento((linha, coluna), (lr, cr), peca, alvo))

        # roques (com direitos de roque CORRETOS)
        if peca == REI_BRANCO and linha == 7 and coluna == 4:
            # roque pequeno branco
            if (self.direitos_roque & ROQUE_BRANCO_REI) and \
               self.tabuleiro[7][5] == VAZIO and self.tabuleiro[7][6] == VAZIO:
                if not self._casa_atacada(7, 4, False) and \
                   not self._casa_atacada(7, 5, False) and \
                   not self._casa_atacada(7, 6, False):
                    movimentos.append(
                        Movimento((7, 4), (7, 6), peca, VAZIO, roque=True)
                    )
            # roque grande branco
            if (self.direitos_roque & ROQUE_BRANCO_DAMA) and \
               self.tabuleiro[7][1] == VAZIO and \
               self.tabuleiro[7][2] == VAZIO and \
               self.tabuleiro[7][3] == VAZIO:
                if not self._casa_atacada(7, 4, False) and \
                   not self._casa_atacada(7, 3, False) and \
                   not self._casa_atacada(7, 2, False):
                    movimentos.append(
                        Movimento((7, 4), (7, 2), peca, VAZIO, roque=True, roque_longo=True)
                    )

        if peca == REI_PRETO and linha == 0 and coluna == 4:
            # roque pequeno preto
            if (self.direitos_roque & ROQUE_PRETO_REI) and \
               self.tabuleiro[0][5] == VAZIO and self.tabuleiro[0][6] == VAZIO:
                if not self._casa_atacada(0, 4, True) and \
                   not self._casa_atacada(0, 5, True) and \
                   not self._casa_atacada(0, 6, True):
                    movimentos.append(
                        Movimento((0, 4), (0, 6), peca, VAZIO, roque=True)
                    )
            # roque grande preto
            if (self.direitos_roque & ROQUE_PRETO_DAMA) and \
               self.tabuleiro[0][1] == VAZIO and \
               self.tabuleiro[0][2] == VAZIO and \
               self.tabuleiro[0][3] == VAZIO:
                if not self._casa_atacada(0, 4, True) and \
                   not self._casa_atacada(0, 3, True) and \
                   not self._casa_atacada(0, 2, True):
                    movimentos.append(
                        Movimento((0, 4), (0, 2), peca, VAZIO, roque=True, roque_longo=True)
                    )

    # ----------------------------
    # Fazer / desfazer movimentos
    # ----------------------------
    def fazer_movimento(self, movimento: Movimento):
        linha_origem, coluna_origem = movimento.origem
        linha_destino, coluna_destino = movimento.destino
        peca_original = movimento.peca
        capturada = self.tabuleiro[linha_destino][coluna_destino]

        direitos_roque_antigos = self.direitos_roque
        coluna_ep_antiga = self.coluna_en_passant
        relogio_meia_jogada_antigo = self.relogio_meia_jogada
        hash_antigo = self.hash_atual

        # hash: remover peça de origem e capturada (se existir)
        self._aplicar_hash_peca(linha_origem, coluna_origem, peca_original)
        if capturada != VAZIO:
            self._aplicar_hash_peca(linha_destino, coluna_destino, capturada)

        # meia jogada
        if abs(peca_original) == 1 or capturada != VAZIO:
            self.relogio_meia_jogada = 0
        else:
            self.relogio_meia_jogada += 1

        # captura en passant
        if movimento.en_passant:
            if peca_original > 0:
                linha_peao_capturado = linha_destino + 1
            else:
                linha_peao_capturado = linha_destino - 1
            peca_cap = self.tabuleiro[linha_peao_capturado][coluna_destino]
            self._aplicar_hash_peca(linha_peao_capturado, coluna_destino, peca_cap)
            self.tabuleiro[linha_peao_capturado][coluna_destino] = VAZIO
            capturada = peca_cap

        # mover peça
        self.tabuleiro[linha_origem][coluna_origem] = VAZIO
        peca_final = movimento.promocao if movimento.promocao != VAZIO else peca_original
        self.tabuleiro[linha_destino][coluna_destino] = peca_final
        self._aplicar_hash_peca(linha_destino, coluna_destino, peca_final)

        # roques (mexer a torre certa)
        if movimento.roque:
            if peca_original == REI_BRANCO:
                linha_roque = 7
                if movimento.roque_longo:
                    # torre de a1 para d1
                    self._aplicar_hash_peca(linha_roque, 0, TORRE_BRANCA)
                    self.tabuleiro[linha_roque][0] = VAZIO
                    self.tabuleiro[linha_roque][3] = TORRE_BRANCA
                    self._aplicar_hash_peca(linha_roque, 3, TORRE_BRANCA)
                else:
                    # torre de h1 para f1
                    self._aplicar_hash_peca(linha_roque, 7, TORRE_BRANCA)
                    self.tabuleiro[linha_roque][7] = VAZIO
                    self.tabuleiro[linha_roque][5] = TORRE_BRANCA
                    self._aplicar_hash_peca(linha_roque, 5, TORRE_BRANCA)
            elif peca_original == REI_PRETO:
                linha_roque = 0
                if movimento.roque_longo:
                    # torre de a8 para d8
                    self._aplicar_hash_peca(linha_roque, 0, TORRE_PRETA)
                    self.tabuleiro[linha_roque][0] = VAZIO
                    self.tabuleiro[linha_roque][3] = TORRE_PRETA
                    self._aplicar_hash_peca(linha_roque, 3, TORRE_PRETA)
                else:
                    # torre de h8 para f8
                    self._aplicar_hash_peca(linha_roque, 7, TORRE_PRETA)
                    self.tabuleiro[linha_roque][7] = VAZIO
                    self.tabuleiro[linha_roque][5] = TORRE_PRETA
                    self._aplicar_hash_peca(linha_roque, 5, TORRE_PRETA)

        # perda de direitos de roque ao mover REI/TORRE
        if peca_original == REI_BRANCO:
            self.direitos_roque = definir_bit(self.direitos_roque, ROQUE_BRANCO_REI, False)
            self.direitos_roque = definir_bit(self.direitos_roque, ROQUE_BRANCO_DAMA, False)
        elif peca_original == REI_PRETO:
            self.direitos_roque = definir_bit(self.direitos_roque, ROQUE_PRETO_REI, False)
            self.direitos_roque = definir_bit(self.direitos_roque, ROQUE_PRETO_DAMA, False)
        elif peca_original == TORRE_BRANCA:
            if linha_origem == 7 and coluna_origem == 0:
                self.direitos_roque = definir_bit(self.direitos_roque, ROQUE_BRANCO_DAMA, False)
            elif linha_origem == 7 and coluna_origem == 7:
                self.direitos_roque = definir_bit(self.direitos_roque, ROQUE_BRANCO_REI, False)
        elif peca_original == TORRE_PRETA:
            if linha_origem == 0 and coluna_origem == 0:
                self.direitos_roque = definir_bit(self.direitos_roque, ROQUE_PRETO_DAMA, False)
            elif linha_origem == 0 and coluna_origem == 7:
                self.direitos_roque = definir_bit(self.direitos_roque, ROQUE_PRETO_REI, False)

        # se capturamos torre adversária, atualizar direitos de roque
        if capturada == TORRE_BRANCA:
            if linha_destino == 7 and coluna_destino == 0:
                self.direitos_roque = definir_bit(self.direitos_roque, ROQUE_BRANCO_DAMA, False)
            elif linha_destino == 7 and coluna_destino == 7:
                self.direitos_roque = definir_bit(self.direitos_roque, ROQUE_BRANCO_REI, False)
        elif capturada == TORRE_PRETA:
            if linha_destino == 0 and coluna_destino == 0:
                self.direitos_roque = definir_bit(self.direitos_roque, ROQUE_PRETO_DAMA, False)
            elif linha_destino == 0 and coluna_destino == 7:
                self.direitos_roque = definir_bit(self.direitos_roque, ROQUE_PRETO_REI, False)

        # hash roque
        self._aplicar_hash_roque(direitos_roque_antigos, self.direitos_roque)

        # en passant (nova coluna)
        self._aplicar_hash_en_passant(coluna_ep_antiga, -1)
        self.coluna_en_passant = -1
        if abs(peca_original) == 1 and abs(linha_destino - linha_origem) == 2:
            self.coluna_en_passant = coluna_origem
            self._aplicar_hash_en_passant(-1, self.coluna_en_passant)

        # trocar lado a jogar
        self.brancas_jogam = not self.brancas_jogam
        self._aplicar_hash_lado()

        if not self.brancas_jogam:
            self.numero_jogada_completa += 1

        self.historico.append(
            Historico(
                movimento,
                capturada,
                direitos_roque_antigos,
                coluna_ep_antiga,
                relogio_meia_jogada_antigo,
                hash_antigo
            )
        )

    def desfazer_movimento(self):
        historico = self.historico.pop()
        movimento = historico.movimento
        linha_origem, coluna_origem = movimento.origem
        linha_destino, coluna_destino = movimento.destino
        peca_original = movimento.peca

        # restaurar hash e metadados direto
        self.hash_atual = historico.hash_anterior
        self.direitos_roque = historico.direitos_roque
        self.coluna_en_passant = historico.coluna_en_passant
        self.relogio_meia_jogada = historico.relogio_meia_jogada

        # restaurar lado a jogar e número da jogada
        self.brancas_jogam = not self.brancas_jogam
        if self.brancas_jogam:
            self.numero_jogada_completa -= 1

        # desfazer roque (mover torre de volta)
        if movimento.roque:
            if peca_original == REI_BRANCO:
                linha_roque = 7
                if movimento.roque_longo:
                    self.tabuleiro[linha_roque][0] = TORRE_BRANCA
                    self.tabuleiro[linha_roque][3] = VAZIO
                else:
                    self.tabuleiro[linha_roque][7] = TORRE_BRANCA
                    self.tabuleiro[linha_roque][5] = VAZIO
            elif peca_original == REI_PRETO:
                linha_roque = 0
                if movimento.roque_longo:
                    self.tabuleiro[linha_roque][0] = TORRE_PRETA
                    self.tabuleiro[linha_roque][3] = VAZIO
                else:
                    self.tabuleiro[linha_roque][7] = TORRE_PRETA
                    self.tabuleiro[linha_roque][5] = VAZIO

        # desfazer en passant ou movimento normal
        if movimento.en_passant:
            self.tabuleiro[linha_origem][coluna_origem] = peca_original
            self.tabuleiro[linha_destino][coluna_destino] = VAZIO
            if peca_original > 0:
                self.tabuleiro[linha_destino + 1][coluna_destino] = PEAO_PRETO
            else:
                self.tabuleiro[linha_destino - 1][coluna_destino] = PEAO_BRANCO
        else:
            self.tabuleiro[linha_origem][coluna_origem] = peca_original
            self.tabuleiro[linha_destino][coluna_destino] = historico.capturada

# ----------------------------
# Avaliação
# ----------------------------
class Avaliador:
    def __init__(self):
        pass

    def avaliar(self, estado: Estado) -> int:
        pontuacao = 0
        for linha in range(8):
            for coluna in range(8):
                peca = estado.tabuleiro[linha][coluna]
                if peca == VAZIO:
                    continue
                pontuacao += VALOR_PECA[peca]
                indice = linha * 8 + coluna
                tabela = TABELAS_POSICAO[peca]
                if peca > 0:
                    pontuacao += tabela[indice]
                else:
                    pontuacao -= tabela[(7 - linha) * 8 + coluna]

        # mobilidade (brancas - pretas)
        brancas_na_vez = estado.brancas_jogam
        estado.brancas_jogam = True
        movimentos_brancas = len(estado._gerar_movimentos_pseudo())
        estado.brancas_jogam = False
        movimentos_pretas = len(estado._gerar_movimentos_pseudo())
        estado.brancas_jogam = brancas_na_vez
        pontuacao += 10 * (movimentos_brancas - movimentos_pretas)

        # bônus/penalidade simples por xeque
        if estado.em_xeque(True):
            pontuacao -= 50
        if estado.em_xeque(False):
            pontuacao += 50

        return pontuacao

# ----------------------------
# Tabela de Transposição
# ----------------------------
@dataclass
class EntradaTT:
    hash_chave: int
    profundidade: int
    valor: int
    flag: int  # 0=exato, 1=limite_inferior, 2=limite_superior
    melhor_movimento: Optional[Movimento]

# ----------------------------
# IA (busca)
# ----------------------------
class MotorIA:
    def __init__(self, avaliador: Avaliador, tempo_por_lance: float = 3.0):
        self.avaliador = avaliador
        self.tempo_limite = tempo_por_lance
        self.tabela_transposicao: Dict[int, EntradaTT] = {}
        self.inicio_busca = 0.0
        self.nos_avaliados = 0
        self.melhor_movimento_global: Optional[Movimento] = None

    def _ordenar_movimentos(self, movimentos: List[Movimento]) -> None:
        # Heurística MVV-LVA + promoção
        def score(m: Movimento) -> int:
            s = 0
            if m.capturada != VAZIO:
                s += abs(VALOR_PECA[m.capturada]) - abs(VALOR_PECA[m.peca]) // 10
            if m.promocao != VAZIO:
                s += 800
            if m.roque:
                s += 200
            return s
        movimentos.sort(key=score, reverse=True)

    def escolher_movimento(self, estado: Estado) -> Optional[Movimento]:
        self.inicio_busca = time.time()
        self.nos_avaliados = 0
        self.melhor_movimento_global = None

        profundidade_maxima = 5
        alfa_global = -10**9
        beta_global = 10**9

        for profundidade in range(1, profundidade_maxima + 1):
            valor, movimento = self._alpha_beta_raiz(estado, profundidade, alfa_global, beta_global)
            if movimento is not None:
                self.melhor_movimento_global = movimento
            if time.time() - self.inicio_busca > self.tempo_limite:
                break
        return self.melhor_movimento_global

    def _alpha_beta_raiz(self, estado: Estado, profundidade: int, alfa: int, beta: int) -> Tuple[int, Optional[Movimento]]:
        movimentos = estado.gerar_movimentos_legais()
        if not movimentos:
            if estado.em_xeque(estado.brancas_jogam):
                return (-999999 if estado.brancas_jogam else 999999), None
            else:
                return 0, None

        self._ordenar_movimentos(movimentos)

        melhor_valor = -10**9 if estado.brancas_jogam else 10**9
        melhor_movimento = None

        for movimento in movimentos:
            if time.time() - self.inicio_busca > self.tempo_limite:
                break
            estado.fazer_movimento(movimento)
            valor = self._alpha_beta(estado, profundidade - 1, alfa, beta)
            estado.desfazer_movimento()
            if estado.brancas_jogam:
                if valor > melhor_valor:
                    melhor_valor = valor
                    melhor_movimento = movimento
                alfa = max(alfa, valor)
            else:
                if valor < melhor_valor:
                    melhor_valor = valor
                    melhor_movimento = movimento
                beta = min(beta, valor)
        return melhor_valor, melhor_movimento

    def _alpha_beta(self, estado: Estado, profundidade: int, alfa: int, beta: int) -> int:
        if time.time() - self.inicio_busca > self.tempo_limite:
            return self.avaliador.avaliar(estado)
        self.nos_avaliados += 1

        entrada = self.tabela_transposicao.get(estado.hash_atual)
        if entrada is not None and entrada.profundidade >= profundidade:
            if entrada.flag == 0:
                return entrada.valor
            elif entrada.flag == 1 and entrada.valor > alfa:
                alfa = entrada.valor
            elif entrada.flag == 2 and entrada.valor < beta:
                beta = entrada.valor
            if alfa >= beta:
                return entrada.valor

        if profundidade == 0:
            return self._busca_quiescencia(estado, alfa, beta)

        movimentos = estado.gerar_movimentos_legais()
        if not movimentos:
            if estado.em_xeque(estado.brancas_jogam):
                return -999999 if estado.brancas_jogam else 999999
            return 0

        self._ordenar_movimentos(movimentos)

        melhor_valor = -10**9 if estado.brancas_jogam else 10**9
        flag = 2 if estado.brancas_jogam else 1

        for movimento in movimentos:
            estado.fazer_movimento(movimento)
            valor = self._alpha_beta(estado, profundidade - 1, alfa, beta)
            estado.desfazer_movimento()
            if estado.brancas_jogam:
                if valor > melhor_valor:
                    melhor_valor = valor
                if valor > alfa:
                    alfa = valor
                    flag = 0
                if alfa >= beta:
                    break
            else:
                if valor < melhor_valor:
                    melhor_valor = valor
                if valor < beta:
                    beta = valor
                    flag = 0
                if beta <= alfa:
                    break

        self.tabela_transposicao[estado.hash_atual] = EntradaTT(
            hash_chave=estado.hash_atual,
            profundidade=profundidade,
            valor=melhor_valor,
            flag=flag,
            melhor_movimento=None
        )
        return melhor_valor

    def _busca_quiescencia(self, estado: Estado, alfa: int, beta: int) -> int:
        avaliacao_estatica = self.avaliador.avaliar(estado)
        if estado.brancas_jogam:
            if avaliacao_estatica >= beta:
                return avaliacao_estatica
            if avaliacao_estatica > alfa:
                alfa = avaliacao_estatica
        else:
            if avaliacao_estatica <= alfa:
                return avaliacao_estatica
            if avaliacao_estatica < beta:
                beta = avaliacao_estatica

        movimentos = estado.gerar_movimentos_legais()
        movimentos_captura = [m for m in movimentos if m.capturada != VAZIO]

        self._ordenar_movimentos(movimentos_captura)

        for movimento in movimentos_captura:
            estado.fazer_movimento(movimento)
            pontuacao = self._busca_quiescencia(estado, alfa, beta)
            estado.desfazer_movimento()
            if estado.brancas_jogam:
                if pontuacao > alfa:
                    alfa = pontuacao
                if alfa >= beta:
                    return beta
            else:
                if pontuacao < beta:
                    beta = pontuacao
                if beta <= alfa:
                    return alfa
        return alfa if estado.brancas_jogam else beta

# ----------------------------
# Interface gráfica (pygame)
# ----------------------------
class JogoXadrez:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Xadrez IA Avançada — Luiz Tiago Wilcke (LT)")
        self.tela = pygame.display.set_mode((LARGURA_JANELA, ALTURA_JANELA))
        self.relogio = pygame.time.Clock()

        self.fonte_pecas = pygame.font.SysFont("DejaVu Sans", 56)
        self.fonte_info = pygame.font.SysFont("DejaVu Sans", 20)
        self.fonte_titulo = pygame.font.SysFont("DejaVu Sans", 24, bold=True)

        self.estado = Estado()
        self.avaliador = Avaliador()
        self.motor_ia = MotorIA(self.avaliador, tempo_por_lance=2.5)

        self.casa_selecionada: Optional[Tuple[int, int]] = None
        self.movimentos_legais_cache: List[Movimento] = []
        self.ultimo_movimento: Optional[Movimento] = None
        self.rodando = True

        # Por padrão: humano = brancas, IA = pretas
        self.humano_brancas = True

    def desenhar_tabuleiro(self):
        for linha in range(8):
            for coluna in range(8):
                cor_casa = COR_TAB_CLARO if (linha + coluna) % 2 == 0 else COR_TAB_ESCURO
                pygame.draw.rect(
                    self.tela,
                    cor_casa,
                    (coluna * LADO_CASA, linha * LADO_CASA, LADO_CASA, LADO_CASA)
                )

        # último movimento
        if self.ultimo_movimento:
            l1, c1 = self.ultimo_movimento.origem
            l2, c2 = self.ultimo_movimento.destino
            superficie = pygame.Surface((LADO_CASA, LADO_CASA), pygame.SRCALPHA)
            superficie.fill((252, 211, 77, 120))
            self.tela.blit(superficie, (c1 * LADO_CASA, l1 * LADO_CASA))
            self.tela.blit(superficie, (c2 * LADO_CASA, l2 * LADO_CASA))

        # seleção
        if self.casa_selecionada:
            linha, coluna = self.casa_selecionada
            pygame.draw.rect(
                self.tela,
                COR_SELECIONADO,
                (coluna * LADO_CASA, linha * LADO_CASA, LADO_CASA, LADO_CASA),
                4
            )

        # destinos possíveis
        if self.casa_selecionada:
            for movimento in self.movimentos_legais_cache:
                if movimento.origem == self.casa_selecionada:
                    lr, cr = movimento.destino
                    pygame.draw.circle(
                        self.tela,
                        (30, 64, 175),
                        (cr * LADO_CASA + LADO_CASA // 2, lr * LADO_CASA + LADO_CASA // 2),
                        10
                    )

    def desenhar_pecas(self):
        for linha in range(8):
            for coluna in range(8):
                peca = self.estado.tabuleiro[linha][coluna]
                if peca == VAZIO:
                    continue
                texto = PECAS_UNICODE.get(peca)
                if texto is None:
                    continue
                cor_peca = (15, 23, 42)
                superficie = self.fonte_pecas.render(texto, True, cor_peca)
                ret = superficie.get_rect(
                    center=(coluna * LADO_CASA + LADO_CASA // 2, linha * LADO_CASA + LADO_CASA // 2)
                )
                self.tela.blit(superficie, ret)

    def desenhar_painel(self):
        x0 = TAMANHO_TABULEIRO
        pygame.draw.rect(self.tela, COR_FUNDO, (x0, 0, LARGURA_PAINEL_INFO, ALTURA_JANELA))

        titulo = "Xadrez IA Avançada"
        superficie_titulo = self.fonte_titulo.render(titulo, True, COR_TEXTO)
        self.tela.blit(superficie_titulo, (x0 + 20, 20))

        turno = "Brancas" if self.estado.brancas_jogam else "Pretas"
        texto_turno = self.fonte_info.render(
            f"Vez: {turno}",
            True,
            COR_IA if self.estado.brancas_jogam == self.humano_brancas else COR_TEXTO
        )
        self.tela.blit(texto_turno, (x0 + 20, 60))

        pontuacao = self.avaliador.avaliar(self.estado)
        avaliacao = pontuacao / 100.0
        texto_score = self.fonte_info.render(f"Avaliação IA: {avaliacao:+.2f}", True, COR_TEXTO)
        self.tela.blit(texto_score, (x0 + 20, 90))

        texto_nos = self.fonte_info.render(
            f"Nós avaliados: {self.motor_ia.nos_avaliados}",
            True,
            COR_TEXTO
        )
        self.tela.blit(texto_nos, (x0 + 20, 120))

        linhas = [
            "Controles:",
            "- Clique numa peça",
            "  e depois no destino",
            "- ESC: sair",
            "",
            "IA: Minimax + Alpha-Beta",
            "TT + Quiescence",
        ]
        y = 160
        for linha_texto in linhas:
            superficie = self.fonte_info.render(linha_texto, True, COR_TEXTO)
            self.tela.blit(superficie, (x0 + 20, y))
            y += 22

    def posicao_mouse_para_casa(self, posicao: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        x, y = posicao
        if x >= TAMANHO_TABULEIRO:
            return None
        coluna = x // LADO_CASA
        linha = y // LADO_CASA
        return (linha, coluna)

    def loop(self):
        while self.rodando:
            self.relogio.tick(QUADROS_POR_SEGUNDO)
            for evento in pygame.event.get():
                if evento.type == pygame.QUIT:
                    self.rodando = False
                elif evento.type == pygame.KEYDOWN:
                    if evento.key == pygame.K_ESCAPE:
                        self.rodando = False
                elif evento.type == pygame.MOUSEBUTTONDOWN:
                    self.tratar_clique(evento.pos)

            # vez da IA (lado oposto ao humano)
            if (self.estado.brancas_jogam and not self.humano_brancas) or \
               ((not self.estado.brancas_jogam) and self.humano_brancas):
                movimento_ia = self.motor_ia.escolher_movimento(self.estado)
                if movimento_ia is not None:
                    self.estado.fazer_movimento(movimento_ia)
                    self.ultimo_movimento = movimento_ia

            self.tela.fill((0, 0, 0))
            self.desenhar_tabuleiro()
            self.desenhar_pecas()
            self.desenhar_painel()
            pygame.display.flip()

        pygame.quit()
        sys.exit()

    def tratar_clique(self, posicao):
        casa = self.posicao_mouse_para_casa(posicao)
        if casa is None:
            return
        if (self.estado.brancas_jogam and not self.humano_brancas) or \
           ((not self.estado.brancas_jogam) and self.humano_brancas):
            return

        linha, coluna = casa
        peca = self.estado.tabuleiro[linha][coluna]

        if self.casa_selecionada is None:
            if peca == VAZIO:
                return
            if self.estado.brancas_jogam and not eh_branca(peca):
                return
            if not self.estado.brancas_jogam and not eh_preta(peca):
                return
            self.casa_selecionada = (linha, coluna)
            self.movimentos_legais_cache = self.estado.gerar_movimentos_legais()
        else:
            if self.casa_selecionada == casa:
                self.casa_selecionada = None
                self.movimentos_legais_cache = []
                return
            movimento_escolhido = None
            for movimento in self.movimentos_legais_cache:
                if movimento.origem == self.casa_selecionada and movimento.destino == casa:
                    movimento_escolhido = movimento
                    break
            if movimento_escolhido:
                self.estado.fazer_movimento(movimento_escolhido)
                self.ultimo_movimento = movimento_escolhido
            self.casa_selecionada = None
            self.movimentos_legais_cache = []

# ----------------------------
# Execução
# ----------------------------
if __name__ == "__main__":
    jogo = JogoXadrez()
    jogo.loop()
