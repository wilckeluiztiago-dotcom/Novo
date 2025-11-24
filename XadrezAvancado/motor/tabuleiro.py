"""
Lógica do Tabuleiro e Regras do Jogo
"""

from typing import List, Tuple
from config import *
from utils.tipos import Movimento, Historico
from utils.zobrist import (
    ZOBRIST_PECA, ZOBRIST_LADO_A_JOGAR, ZOBRIST_ROQUE, 
    ZOBRIST_COLUNA_EP, indice_peca_zobrist
)

def eh_branca(peca: int) -> bool:
    return peca > 0

def eh_preta(peca: int) -> bool:
    return peca < 0

def sinal(peca: int) -> int:
    if peca > 0: return 1
    if peca < 0: return -1
    return 0

def definir_bit(mascara: int, bitmask: int, valor: bool) -> int:
    if valor:
        return mascara | bitmask
    else:
        return mascara & ~bitmask

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
        self.tabuleiro[0] = [
            TORRE_PRETA, CAVALO_PRETO, BISPO_PRETO, DAMA_PRETA,
            REI_PRETO, BISPO_PRETO, CAVALO_PRETO, TORRE_PRETA
        ]
        self.tabuleiro[1] = [PEAO_PRETO] * 8
        for linha in range(2, 6):
            self.tabuleiro[linha] = [VAZIO] * 8
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
        if peca == VAZIO: return
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
            if linha_rei != -1: break
        if linha_rei == -1: return False
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
        for dl, dc in [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]:
            lr, cr = linha + dl, coluna + dc
            if self.dentro(lr, cr):
                peca = self.tabuleiro[lr][cr]
                alvo = CAVALO_BRANCO if por_brancas else CAVALO_PRETO
                if peca == alvo: return True

        # Bispos/Damas e Torres/Damas
        alvo_bispo = BISPO_BRANCO if por_brancas else BISPO_PRETO
        alvo_torre = TORRE_BRANCA if por_brancas else TORRE_PRETA
        alvo_dama = DAMA_BRANCA if por_brancas else DAMA_PRETA
        
        # Diagonais
        for dl, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            lr, cr = linha + dl, coluna + dc
            while self.dentro(lr, cr):
                peca = self.tabuleiro[lr][cr]
                if peca != VAZIO:
                    if peca == alvo_bispo or peca == alvo_dama: return True
                    break
                lr += dl
                cr += dc
        
        # Retas
        for dl, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            lr, cr = linha + dl, coluna + dc
            while self.dentro(lr, cr):
                peca = self.tabuleiro[lr][cr]
                if peca != VAZIO:
                    if peca == alvo_torre or peca == alvo_dama: return True
                    break
                lr += dl
                cr += dc

        # Rei
        alvo_rei = REI_BRANCO if por_brancas else REI_PRETO
        for dl in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dl == 0 and dc == 0: continue
                lr, cr = linha + dl, coluna + dc
                if self.dentro(lr, cr) and self.tabuleiro[lr][cr] == alvo_rei:
                    return True
        return False

    def gerar_movimentos_legais(self) -> List[Movimento]:
        movimentos_pseudo = self._gerar_movimentos_pseudo()
        movimentos_legais = []
        for movimento in movimentos_pseudo:
            self.fazer_movimento(movimento)
            if not self.em_xeque(not self.brancas_jogam):
                movimentos_legais.append(movimento)
            self.desfazer_movimento()
        return movimentos_legais

    def _gerar_movimentos_pseudo(self) -> List[Movimento]:
        movimentos = []
        brancas_na_vez = self.brancas_jogam
        for linha in range(8):
            for coluna in range(8):
                peca = self.tabuleiro[linha][coluna]
                if peca == VAZIO: continue
                if brancas_na_vez and not eh_branca(peca): continue
                if (not brancas_na_vez) and not eh_preta(peca): continue
                
                tipo = abs(peca)
                if tipo == 1: self._movimentos_peao(linha, coluna, peca, movimentos)
                elif tipo == 2: self._movimentos_cavalo(linha, coluna, peca, movimentos)
                elif tipo == 3: self._movimentos_bispo(linha, coluna, peca, movimentos)
                elif tipo == 4: self._movimentos_torre(linha, coluna, peca, movimentos)
                elif tipo == 5: self._movimentos_dama(linha, coluna, peca, movimentos)
                elif tipo == 6: self._movimentos_rei(linha, coluna, peca, movimentos)
        return movimentos

    def _movimentos_peao(self, linha, coluna, peca, movimentos):
        direcao = -1 if peca > 0 else 1
        linha_promocao = 0 if peca > 0 else 7
        linha_inicial = 6 if peca > 0 else 1
        
        # Frente
        lf = linha + direcao
        if self.dentro(lf, coluna) and self.tabuleiro[lf][coluna] == VAZIO:
            if lf == linha_promocao:
                novas = [DAMA_BRANCA, TORRE_BRANCA, BISPO_BRANCO, CAVALO_BRANCO] if peca > 0 else \
                        [DAMA_PRETA, TORRE_PRETA, BISPO_PRETO, CAVALO_PRETO]
                for n in novas:
                    movimentos.append(Movimento((linha, coluna), (lf, coluna), peca, VAZIO, n))
            else:
                movimentos.append(Movimento((linha, coluna), (lf, coluna), peca))
                if linha == linha_inicial:
                    ld = linha + 2 * direcao
                    if self.tabuleiro[ld][coluna] == VAZIO:
                        movimentos.append(Movimento((linha, coluna), (ld, coluna), peca))

        # Capturas
        for dc in [-1, 1]:
            cd = coluna + dc
            ld = linha + direcao
            if self.dentro(ld, cd):
                alvo = self.tabuleiro[ld][cd]
                if alvo != VAZIO and sinal(alvo) != sinal(peca):
                    if ld == linha_promocao:
                        novas = [DAMA_BRANCA, TORRE_BRANCA, BISPO_BRANCO, CAVALO_BRANCO] if peca > 0 else \
                                [DAMA_PRETA, TORRE_PRETA, BISPO_PRETO, CAVALO_PRETO]
                        for n in novas:
                            movimentos.append(Movimento((linha, coluna), (ld, cd), peca, alvo, n))
                    else:
                        movimentos.append(Movimento((linha, coluna), (ld, cd), peca, alvo))

        # En Passant
        if self.coluna_en_passant != -1 and linha == (3 if peca > 0 else 4):
            for dc in [-1, 1]:
                cl = coluna + dc
                if cl == self.coluna_en_passant:
                    alvo = self.tabuleiro[linha][cl]
                    if alvo == -peca:
                        movimentos.append(Movimento((linha, coluna), (linha + direcao, cl), peca, alvo, en_passant=True))

    def _movimentos_cavalo(self, linha, coluna, peca, movimentos):
        for dl, dc in [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]:
            lr, cr = linha + dl, coluna + dc
            if self.dentro(lr, cr):
                alvo = self.tabuleiro[lr][cr]
                if alvo == VAZIO or sinal(alvo) != sinal(peca):
                    movimentos.append(Movimento((linha, coluna), (lr, cr), peca, alvo))

    def _movimentos_bispo(self, linha, coluna, peca, movimentos):
        for dl, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            lr, cr = linha + dl, coluna + dc
            while self.dentro(lr, cr):
                alvo = self.tabuleiro[lr][cr]
                if alvo == VAZIO:
                    movimentos.append(Movimento((linha, coluna), (lr, cr), peca))
                elif sinal(alvo) != sinal(peca):
                    movimentos.append(Movimento((linha, coluna), (lr, cr), peca, alvo))
                    break
                else: break
                lr += dl
                cr += dc

    def _movimentos_torre(self, linha, coluna, peca, movimentos):
        for dl, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            lr, cr = linha + dl, coluna + dc
            while self.dentro(lr, cr):
                alvo = self.tabuleiro[lr][cr]
                if alvo == VAZIO:
                    movimentos.append(Movimento((linha, coluna), (lr, cr), peca))
                elif sinal(alvo) != sinal(peca):
                    movimentos.append(Movimento((linha, coluna), (lr, cr), peca, alvo))
                    break
                else: break
                lr += dl
                cr += dc

    def _movimentos_dama(self, linha, coluna, peca, movimentos):
        self._movimentos_bispo(linha, coluna, peca, movimentos)
        self._movimentos_torre(linha, coluna, peca, movimentos)

    def _movimentos_rei(self, linha, coluna, peca, movimentos):
        for dl in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dl == 0 and dc == 0: continue
                lr, cr = linha + dl, coluna + dc
                if self.dentro(lr, cr):
                    alvo = self.tabuleiro[lr][cr]
                    if alvo == VAZIO or sinal(alvo) != sinal(peca):
                        movimentos.append(Movimento((linha, coluna), (lr, cr), peca, alvo))

        # Roques
        if peca == REI_BRANCO and linha == 7 and coluna == 4:
            if (self.direitos_roque & ROQUE_BRANCO_REI) and self.tabuleiro[7][5] == VAZIO and self.tabuleiro[7][6] == VAZIO:
                if not self._casa_atacada(7, 4, False) and not self._casa_atacada(7, 5, False) and not self._casa_atacada(7, 6, False):
                    movimentos.append(Movimento((7, 4), (7, 6), peca, VAZIO, roque=True))
            if (self.direitos_roque & ROQUE_BRANCO_DAMA) and self.tabuleiro[7][1] == VAZIO and self.tabuleiro[7][2] == VAZIO and self.tabuleiro[7][3] == VAZIO:
                if not self._casa_atacada(7, 4, False) and not self._casa_atacada(7, 3, False) and not self._casa_atacada(7, 2, False):
                    movimentos.append(Movimento((7, 4), (7, 2), peca, VAZIO, roque=True, roque_longo=True))

        if peca == REI_PRETO and linha == 0 and coluna == 4:
            if (self.direitos_roque & ROQUE_PRETO_REI) and self.tabuleiro[0][5] == VAZIO and self.tabuleiro[0][6] == VAZIO:
                if not self._casa_atacada(0, 4, True) and not self._casa_atacada(0, 5, True) and not self._casa_atacada(0, 6, True):
                    movimentos.append(Movimento((0, 4), (0, 6), peca, VAZIO, roque=True))
            if (self.direitos_roque & ROQUE_PRETO_DAMA) and self.tabuleiro[0][1] == VAZIO and self.tabuleiro[0][2] == VAZIO and self.tabuleiro[0][3] == VAZIO:
                if not self._casa_atacada(0, 4, True) and not self._casa_atacada(0, 3, True) and not self._casa_atacada(0, 2, True):
                    movimentos.append(Movimento((0, 4), (0, 2), peca, VAZIO, roque=True, roque_longo=True))

    def fazer_movimento(self, movimento: Movimento):
        l_orig, c_orig = movimento.origem
        l_dest, c_dest = movimento.destino
        peca = movimento.peca
        capturada = self.tabuleiro[l_dest][c_dest]

        # Salva estado anterior
        self.historico.append(Historico(
            movimento, capturada, self.direitos_roque, self.coluna_en_passant,
            self.relogio_meia_jogada, self.hash_atual
        ))

        # Atualiza Hash (remove peças antigas)
        self._aplicar_hash_peca(l_orig, c_orig, peca)
        if capturada != VAZIO:
            self._aplicar_hash_peca(l_dest, c_dest, capturada)

        # Regra 50 movimentos
        if abs(peca) == 1 or capturada != VAZIO:
            self.relogio_meia_jogada = 0
        else:
            self.relogio_meia_jogada += 1

        # En Passant
        if movimento.en_passant:
            l_cap = l_dest + 1 if peca > 0 else l_dest - 1
            peca_cap = self.tabuleiro[l_cap][c_dest]
            self._aplicar_hash_peca(l_cap, c_dest, peca_cap)
            self.tabuleiro[l_cap][c_dest] = VAZIO
            capturada = peca_cap

        # Move peça
        self.tabuleiro[l_orig][c_orig] = VAZIO
        peca_final = movimento.promocao if movimento.promocao != VAZIO else peca
        self.tabuleiro[l_dest][c_dest] = peca_final
        self._aplicar_hash_peca(l_dest, c_dest, peca_final)

        # Roque (move torre)
        if movimento.roque:
            l_roque = 7 if peca == REI_BRANCO else 0
            torre = TORRE_BRANCA if peca == REI_BRANCO else TORRE_PRETA
            if movimento.roque_longo:
                self._aplicar_hash_peca(l_roque, 0, torre)
                self.tabuleiro[l_roque][0] = VAZIO
                self.tabuleiro[l_roque][3] = torre
                self._aplicar_hash_peca(l_roque, 3, torre)
            else:
                self._aplicar_hash_peca(l_roque, 7, torre)
                self.tabuleiro[l_roque][7] = VAZIO
                self.tabuleiro[l_roque][5] = torre
                self._aplicar_hash_peca(l_roque, 5, torre)

        # Atualiza Direitos de Roque
        if peca == REI_BRANCO:
            self.direitos_roque &= ~(ROQUE_BRANCO_REI | ROQUE_BRANCO_DAMA)
        elif peca == REI_PRETO:
            self.direitos_roque &= ~(ROQUE_PRETO_REI | ROQUE_PRETO_DAMA)
        elif peca == TORRE_BRANCA:
            if l_orig == 7 and c_orig == 0: self.direitos_roque &= ~ROQUE_BRANCO_DAMA
            elif l_orig == 7 and c_orig == 7: self.direitos_roque &= ~ROQUE_BRANCO_REI
        elif peca == TORRE_PRETA:
            if l_orig == 0 and c_orig == 0: self.direitos_roque &= ~ROQUE_PRETO_DAMA
            elif l_orig == 0 and c_orig == 7: self.direitos_roque &= ~ROQUE_PRETO_REI
        
        # Se torre capturada
        if capturada == TORRE_BRANCA:
            if l_dest == 7 and c_dest == 0: self.direitos_roque &= ~ROQUE_BRANCO_DAMA
            elif l_dest == 7 and c_dest == 7: self.direitos_roque &= ~ROQUE_BRANCO_REI
        elif capturada == TORRE_PRETA:
            if l_dest == 0 and c_dest == 0: self.direitos_roque &= ~ROQUE_PRETO_DAMA
            elif l_dest == 0 and c_dest == 7: self.direitos_roque &= ~ROQUE_PRETO_REI

        self._aplicar_hash_roque(self.historico[-1].direitos_roque, self.direitos_roque)

        # Atualiza En Passant
        self._aplicar_hash_en_passant(self.historico[-1].coluna_en_passant, -1)
        self.coluna_en_passant = -1
        if abs(peca) == 1 and abs(l_dest - l_orig) == 2:
            self.coluna_en_passant = c_orig
            self._aplicar_hash_en_passant(-1, self.coluna_en_passant)

        # Troca turno
        self.brancas_jogam = not self.brancas_jogam
        self._aplicar_hash_lado()
        if not self.brancas_jogam:
            self.numero_jogada_completa += 1

    def desfazer_movimento(self):
        hist = self.historico.pop()
        mov = hist.movimento
        l_orig, c_orig = mov.origem
        l_dest, c_dest = mov.destino
        peca = mov.peca

        self.hash_atual = hist.hash_anterior
        self.direitos_roque = hist.direitos_roque
        self.coluna_en_passant = hist.coluna_en_passant
        self.relogio_meia_jogada = hist.relogio_meia_jogada

        self.brancas_jogam = not self.brancas_jogam
        if self.brancas_jogam: self.numero_jogada_completa -= 1

        # Desfaz Roque
        if mov.roque:
            l_roque = 7 if peca == REI_BRANCO else 0
            torre = TORRE_BRANCA if peca == REI_BRANCO else TORRE_PRETA
            if mov.roque_longo:
                self.tabuleiro[l_roque][0] = torre
                self.tabuleiro[l_roque][3] = VAZIO
            else:
                self.tabuleiro[l_roque][7] = torre
                self.tabuleiro[l_roque][5] = VAZIO

        # Desfaz Movimento
        if mov.en_passant:
            self.tabuleiro[l_orig][c_orig] = peca
            self.tabuleiro[l_dest][c_dest] = VAZIO
            l_cap = l_dest + 1 if peca > 0 else l_dest - 1
            self.tabuleiro[l_cap][c_dest] = -peca # Peão capturado
        else:
            self.tabuleiro[l_orig][c_orig] = peca
            self.tabuleiro[l_dest][c_dest] = hist.capturada
