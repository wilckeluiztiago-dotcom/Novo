"""
Configurações Globais e Tabelas de Avaliação (PST)
"""

# ----------------------------
# Configurações de Pygame
# ----------------------------
LADO_CASA = 80
TAMANHO_TABULEIRO = LADO_CASA * 8
LARGURA_PAINEL_INFO = 260
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
COR_HUMANO = (255, 255, 255)

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

# ----------------------------
# Direitos de roque
# ----------------------------
ROQUE_BRANCO_REI = 1   # roque pequeno branco
ROQUE_BRANCO_DAMA = 2  # roque grande branco
ROQUE_PRETO_REI = 4    # roque pequeno preto
ROQUE_PRETO_DAMA = 8   # roque grande preto

# ----------------------------
# Tabelas de Posição (Piece-Square Tables)
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
