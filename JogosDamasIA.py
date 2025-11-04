# Jogo de Damas 8x8 com IA Super Sofisticada (Pygame)
# Autor: Luiz Tiago Wilcke (LT)

import math
import time
import random
import pygame
from collections import defaultdict, namedtuple

# ============================= Parâmetros Gerais =============================
LADO = 8
BRANCO, BRANCO_D = 1, 2
PRETO,  PRETO_D  = -1, -2
VAZIO = 0

LARGURA_TAB = 640
ALTURA_TAB  = 640
PAINEL_INFO  = 320
LARGURA_JANELA = LARGURA_TAB + PAINEL_INFO
ALTURA_JANELA  = ALTURA_TAB
TAM_CASA = LARGURA_TAB // LADO

FPS = 60
TEMPO_IA_PADRAO = 2.0  # segundos por lance (iterative deepening)
PROFUNDIDADE_MAX = 10   # limite duro de profundidade

# Cores
COR_CLARA   = (236, 217, 185)
COR_ESCURA  = (102,  72,  43)
COR_SELEC   = ( 82, 141,  62)
COR_ALVO    = ( 50,  94, 168)
COR_TEXTO   = ( 30,  30,  30)
COR_BRANCA  = (240, 240, 240)
COR_PRETA   = ( 25,  25,  25)
COR_DESTAQ  = (255, 215,   0)

# ================================ Modelo ====================================
Jogada = namedtuple('Jogada', 'seq captura promo')
# seq: lista de casas [(lin,col), ...], captura: bool, promo: bool

class Estado:
    __slots__ = ('tab', 'turno', 'meio_lances')
    def __init__(self):
        self.tab = [[VAZIO for _ in range(LADO)] for __ in range(LADO)]
        for i in range(3):
            for j in range(LADO):
                if (i+j) & 1: self.tab[i][j] = PRETO
        for i in range(LADO-3, LADO):
            for j in range(LADO):
                if (i+j) & 1: self.tab[i][j] = BRANCO
        self.turno = 1  # brancas começam
        self.meio_lances = 0  # empate se atingir limite sem captura/coroa

    def copia(self):
        e = Estado.__new__(Estado)
        e.tab = [linha[:] for linha in self.tab]
        e.turno = self.turno
        e.meio_lances = self.meio_lances
        return e

# ============================ Utilidades Básicas ============================

def casa_valida(i,j):
    return 0 <= i < LADO and 0 <= j < LADO

def casa_escura(i,j):
    return ((i+j) & 1) == 1

def sinal(p):
    return 1 if p>0 else (-1 if p<0 else 0)

# ========================== Zobrist / TT / Heurísticas =======================
random.seed(123456789)
ZOBRIST = [[[random.getrandbits(64) for _ in range(4)] for __ in range(LADO)] for ___ in range(LADO)]
# map: PRETO->0, PRETO_D->1, BRANCO->2, BRANCO_D->3
ZOB_TURNO = random.getrandbits(64)

def idx_peca(p):
    if p == PRETO: return 0
    if p == PRETO_D: return 1
    if p == BRANCO: return 2
    if p == BRANCO_D: return 3
    return -1

def hash_estado(e: Estado) -> int:
    h = 0
    for i in range(LADO):
        li = e.tab[i]
        for j in range(LADO):
            p = li[j]
            k = idx_peca(p)
            if k >= 0:
                h ^= ZOBRIST[i][j][k]
    if e.turno < 0:
        h ^= ZOB_TURNO
    return h

class TTEntrada:
    __slots__ = ('chave','valor','prof','flag','best')
    # flag: 0 exato, 1 alpha, 2 beta | best: primeiro passo compactado ((i0,j0,i1,j1))
    def __init__(self):
        self.chave = 0
        self.valor = 0
        self.prof  = -1
        self.flag  = 0
        self.best  = None

TT = {}
HISTORY = defaultdict(int)  # history heuristic
KILLERS = defaultdict(lambda: [None, None])  # por profundidade: 2 killers

# =============================== Avaliação ==================================
VAL_HOMEM = 100
VAL_DAMA  = 175
BONUS_CENTRO = (
    (0, 8,10,12,12,10, 8,0),
    (8,12,14,16,16,14,12,8),
    (10,14,18,20,20,18,14,10),
    (12,16,20,24,24,20,16,12),
    (12,16,20,24,24,20,16,12),
    (10,14,18,20,20,18,14,10),
    (8, 12,14,16,16,14,12,8),
    (0, 8,10,12,12,10, 8,0),
)

# pés de mobilidade/avanço
BONUS_MOB = 2
BONUS_AVANCO = 3

def avalia(e: Estado) -> int:
    esc = 0
    mobB = mobP = 0
    for i in range(LADO):
        for j in range(LADO):
            p = e.tab[i][j]
            if p == VAZIO: continue
            base = VAL_HOMEM if abs(p)==1 else VAL_DAMA
            pos  = BONUS_CENTRO[i][j]
            if p == BRANCO: pos += (LADO-1-i)*BONUS_AVANCO
            elif p == PRETO: pos += i * BONUS_AVANCO
            val = base + pos
            # mobilidade local (aprox)
            if abs(p)==1:
                dirr = -1 if p==BRANCO else 1
                for dj in (-1,1):
                    i1, j1 = i+dirr, j+dj
                    if casa_valida(i1,j1) and e.tab[i1][j1]==VAZIO:
                        if p>0: mobB += 1
                        else:   mobP += 1
            else:
                for di,dj in ((-1,-1),(-1,1),(1,-1),(1,1)):
                    i1, j1 = i+di, j+dj
                    if casa_valida(i1,j1) and e.tab[i1][j1]==VAZIO:
                        if p>0: mobB += 1
                        else:   mobP += 1
            esc += val if p>0 else -val
    esc += BONUS_MOB * (mobB - mobP)
    return esc

# ============================= Geração de Jogadas ============================
# Captura obrigatória + múltiplas capturas em cadeia

def gerar(e: Estado):
    """Retorna lista de Jogada. Se houver captura em qualquer peça do turno,
    gera apenas capturas (incluindo cadeias)."""
    capturas = []
    silencios = []
    turno = e.turno
    for i in range(LADO):
        for j in range(LADO):
            p = e.tab[i][j]
            if p == VAZIO or sinal(p) != turno: continue
            if abs(p) == 1:
                # homem
                dirr = -1 if p==BRANCO else 1
                # tenta capturas
                for dj in (-1,1):
                    i1, j1 = i+dirr, j+dj
                    i2, j2 = i+2*dirr, j+2*dj
                    if casa_valida(i2,j2) and casa_escura(i2,j2):
                        if casa_valida(i1,j1) and e.tab[i1][j1]!=VAZIO and sinal(e.tab[i1][j1])==-turno and e.tab[i2][j2]==VAZIO:
                            # expandir cadeias
                            expandir_capturas(e, [(i,j),(i2,j2)], p, capturas)
                if not capturas:
                    # silenciosos se ainda não existe captura no tabuleiro
                    i1, j1 = i+dirr, j-1
                    i2, j2 = i+dirr, j+1
                    if casa_valida(i1,j1) and casa_escura(i1,j1) and e.tab[i1][j1]==VAZIO:
                        promo = (p==BRANCO and i1==0) or (p==PRETO and i1==LADO-1)
                        silencios.append(Jogada([(i,j),(i1,j1)], False, promo))
                    if casa_valida(i2,j2) and casa_escura(i2,j2) and e.tab[i2][j2]==VAZIO:
                        promo = (p==BRANCO and i2==0) or (p==PRETO and i2==LADO-1)
                        silencios.append(Jogada([(i,j),(i2,j2)], False, promo))
            else:
                # dama: anda 1 casa; captura 2 casas
                for di,dj in ((-1,-1),(-1,1),(1,-1),(1,1)):
                    i1, j1 = i+di, j+dj
                    i2, j2 = i+2*di, j+2*dj
                    if casa_valida(i2,j2) and casa_escura(i2,j2):
                        if casa_valida(i1,j1) and e.tab[i1][j1]!=VAZIO and sinal(e.tab[i1][j1])==-turno and e.tab[i2][j2]==VAZIO:
                            expandir_capturas(e, [(i,j),(i2,j2)], p, capturas)
                if not capturas:
                    for di,dj in ((-1,-1),(-1,1),(1,-1),(1,1)):
                        i1, j1 = i+di, j+dj
                        if casa_valida(i1,j1) and casa_escura(i1,j1) and e.tab[i1][j1]==VAZIO:
                            silencios.append(Jogada([(i,j),(i1,j1)], False, False))
    return capturas if capturas else silencios


def expandir_capturas(e: Estado, caminho, peca, saida):
    """Expande recursivamente múltiplas capturas a partir de um caminho de 2 nós.
    caminho: [(i0,j0),(i2,j2)] representando 1 captura já realizada.
    """
    oi, oj = caminho[0]
    ci, cj = caminho[-1]

    # aplica parcial (em cópia leve)
    t = e.copia()
    # remover origem
    t.tab[oi][oj] = VAZIO
    # remover presa intermediária
    mi, mj = (oi+ci)//2, (oj+cj)//2
    presa = t.tab[mi][mj]
    t.tab[mi][mj] = VAZIO
    np = peca
    # promoção de homem ao pousar
    if abs(peca)==1 and ((peca==BRANCO and ci==0) or (peca==PRETO and ci==LADO-1)):
        np = BRANCO_D if peca==BRANCO else PRETO_D
    t.tab[ci][cj] = np

    continuou = False
    turno = sinal(peca)
    if abs(np)==1:
        dirr = -1 if np==BRANCO else 1
        for dj in (-1,1):
            i1, j1 = ci+dirr, cj+dj
            i2, j2 = ci+2*dirr, cj+2*dj
            if casa_valida(i2,j2) and casa_escura(i2,j2):
                if casa_valida(i1,j1) and t.tab[i1][j1]!=VAZIO and sinal(t.tab[i1][j1])==-turno and t.tab[i2][j2]==VAZIO:
                    continuou = True
                    expandir_capturas(t, caminho+[(i2,j2)], np, saida)
    else:
        for di,dj in ((-1,-1),(-1,1),(1,-1),(1,1)):
            i1, j1 = ci+di, cj+dj
            i2, j2 = ci+2*di, cj+2*dj
            if casa_valida(i2,j2) and casa_escura(i2,j2):
                if casa_valida(i1,j1) and t.tab[i1][j1]!=VAZIO and sinal(t.tab[i1][j1])==-turno and t.tab[i2][j2]==VAZIO:
                    continuou = True
                    expandir_capturas(t, caminho+[(i2,j2)], np, saida)

    if not continuou:
        promo = (abs(peca)==1 and np != peca)
        saida.append(Jogada(list(caminho), True, promo))

# ============================= Aplicar / Desfazer ============================
class Undo:
    __slots__ = ('orig','orig_p','dst','dst_old','caps','coroou','meio_ant','turno_ant')
    def __init__(self):
        self.orig = None
        self.orig_p = None
        self.dst = None
        self.dst_old = None
        self.caps = []  # [(i,j,p)]
        self.coroou = False
        self.meio_ant = 0
        self.turno_ant = 1


def aplicar(e: Estado, J: Jogada) -> Undo:
    u = Undo()
    u.meio_ant = e.meio_lances
    u.turno_ant = e.turno

    (oi,oj) = J.seq[0]
    p = e.tab[oi][oj]
    u.orig = (oi,oj)
    u.orig_p = p
    e.tab[oi][oj] = VAZIO

    ci, cj = oi, oj
    for (ni,nj) in J.seq[1:]:
        if abs(ni-ci)==2 and abs(nj-cj)==2:
            mi, mj = (ni+ci)//2, (nj+cj)//2
            preso = e.tab[mi][mj]
            if preso != VAZIO:
                u.caps.append((mi,mj,preso))
            e.tab[mi][mj] = VAZIO
        ci, cj = ni, nj

    u.dst = (ci,cj)
    u.dst_old = e.tab[ci][cj]

    # promoção
    np = p
    if abs(p)==1 and ((p==BRANCO and ci==0) or (p==PRETO and ci==LADO-1)):
        np = BRANCO_D if p==BRANCO else PRETO_D
        u.coroou = True
    e.tab[ci][cj] = np

    e.turno = -e.turno
    e.meio_lances = 0 if (u.caps or u.coroou) else (e.meio_lances + 1)

    return u


def desfazer(e: Estado, u: Undo):
    (ci,cj) = u.dst
    (oi,oj) = u.orig
    e.tab[ci][cj] = u.dst_old
    e.tab[oi][oj] = u.orig_p
    for (mi,mj,preso) in u.caps:
        e.tab[mi][mj] = preso
    e.meio_lances = u.meio_ant
    e.turno = u.turno_ant

# =============================== Busca (IA) =================================
# Negamax com alpha-beta, TT, killers e history. Quiescence em capturas.

INF = 10**9

class Relogio:
    __slots__ = ('t0','limite')
    def __init__(self, limite):
        self.t0 = time.perf_counter()
        self.limite = limite
    def acabou(self):
        return (time.perf_counter() - self.t0) >= self.limite


def ordenar_jogadas(e: Estado, jogs, profundidade, melhor_hash):
    # Capturas primeiro, melhor da TT depois, killers, history
    def chave(j: Jogada):
        # base: capturas no topo
        score = 100000 if j.captura else 0
        if melhor_hash and len(j.seq) >= 2:
            if (j.seq[0][0],j.seq[0][1], j.seq[1][0],j.seq[1][1]) == melhor_hash:
                score += 90000
        k1,k2 = KILLERS.get(profundidade, [None,None])
        mv = (j.seq[0][0],j.seq[0][1], j.seq[1][0],j.seq[1][1]) if len(j.seq)>=2 else None
        if mv == k1:
            score += 80000
        elif mv == k2:
            score += 70000
        score += HISTORY[mv]
        return -score
    jogs.sort(key=chave)


def quiescence(e: Estado, alpha, beta, rel: Relogio):
    if rel.acabou():
        return 0
    stand = avalia(e) * (1 if e.turno>0 else -1)
    if stand >= beta: return beta
    if alpha < stand: alpha = stand

    jogs = [j for j in gerar(e) if j.captura]
    for j in jogs:
        if rel.acabou(): break
        u = aplicar(e, j)
        val = -quiescence(e, -beta, -alpha, rel)
        desfazer(e, u)
        if val >= beta: return beta
        if val > alpha: alpha = val
    return alpha


def negamax(e: Estado, prof, alpha, beta, rel: Relogio, depth_idx=0):
    if rel.acabou():
        return 0
    if prof == 0:
        return quiescence(e, alpha, beta, rel)
    jogs = gerar(e)
    if not jogs:
        # sem movimentos = derrota
        return -INF + depth_idx

    h = hash_estado(e)
    entrada = TT.get(h)
    melhor_hash = None
    if entrada and entrada.chave == h:
        if entrada.prof >= prof:
            if entrada.flag == 0: return entrada.valor
            elif entrada.flag == 1 and entrada.valor <= alpha: return alpha
            elif entrada.flag == 2 and entrada.valor >= beta:  return beta
        melhor_hash = entrada.best

    ordenar_jogadas(e, jogs, depth_idx, melhor_hash)

    melhor_val = -INF
    melhor_mov_compact = None

    for j in jogs:
        if rel.acabou():
            break
        u = aplicar(e, j)
        val = -negamax(e, prof-1, -beta, -alpha, rel, depth_idx+1)
        desfazer(e, u)

        if val > melhor_val:
            melhor_val = val
            if len(j.seq)>=2:
                melhor_mov_compact = (j.seq[0][0],j.seq[0][1], j.seq[1][0],j.seq[1][1])
        if val > alpha:
            alpha = val
        if alpha >= beta:
            # atualiza killers
            k = KILLERS[depth_idx]
            mv = (j.seq[0][0],j.seq[0][1], j.seq[1][0],j.seq[1][1]) if len(j.seq)>=2 else None
            if mv:
                if k[0] != mv: KILLERS[depth_idx] = [mv, k[0]]
                HISTORY[mv] += 1
            break

    # grava na TT
    ent = TTEntrada()
    ent.chave = h
    ent.valor = melhor_val
    ent.prof  = prof
    ent.best  = melhor_mov_compact
    if melhor_val <= alpha:
        # alpha original — como estamos em negamax, interpretamos flags padrão
        pass
    # flag
    if   melhor_val <= alpha: ent.flag = 1  # alpha
    if   melhor_val >= beta:  ent.flag = 2  # beta
    if not (melhor_val <= alpha or melhor_val >= beta): ent.flag = 0
    TT[h] = ent

    return melhor_val


def pensar(e: Estado, tempo_limite: float) -> Jogada:
    rel = Relogio(tempo_limite)
    melhor = None
    melhor_val = -INF
    # iterative deepening
    for prof in range(2, PROFUNDIDADE_MAX+1):
        if rel.acabou(): break
        jogs = gerar(e)
        if not jogs: break
        ordenar_jogadas(e, jogs, 0, None)
        val_local = -INF
        mov_local = jogs[0]
        for j in jogs:
            if rel.acabou(): break
            u = aplicar(e, j)
            val = -negamax(e, prof-1, -INF, INF, rel, 1)
            desfazer(e, u)
            if val > val_local:
                val_local = val
                mov_local = j
        if not rel.acabou():
            melhor = mov_local
            melhor_val = val_local
        if melhor_val > INF//4: # encontrou algo muito bom
            break
    return melhor if melhor else (gerar(e)[0] if gerar(e) else Jogada([], False, False))

# =============================== Interface (UI) ==============================
pygame.init()
pygame.display.set_caption('Damas 8x8 — IA LT')
tela = pygame.display.set_mode((LARGURA_JANELA, ALTURA_JANELA))
clock = pygame.time.Clock()
fonte = pygame.font.SysFont('arial', 18)
fonte_big = pygame.font.SysFont('arial', 24, bold=True)


def desenha_tabuleiro(e: Estado, selecionada=None, destinos_validos=None):
    # casas
    for i in range(LADO):
        for j in range(LADO):
            cor = COR_ESCURA if casa_escura(i,j) else COR_CLARA
            pygame.draw.rect(tela, cor, (j*TAM_CASA, i*TAM_CASA, TAM_CASA, TAM_CASA))
    # seleção
    if selecionada:
        si, sj = selecionada
        pygame.draw.rect(tela, COR_SELEC, (sj*TAM_CASA, si*TAM_CASA, TAM_CASA, TAM_CASA), 4)
    # destinos
    if destinos_validos:
        for (di,dj) in destinos_validos:
            cx = dj*TAM_CASA + TAM_CASA//2
            cy = di*TAM_CASA + TAM_CASA//2
            pygame.draw.circle(tela, COR_ALVO, (cx,cy), 10)

    # peças
    for i in range(LADO):
        for j in range(LADO):
            p = e.tab[i][j]
            if p == VAZIO: continue
            cx = j*TAM_CASA + TAM_CASA//2
            cy = i*TAM_CASA + TAM_CASA//2
            raio = TAM_CASA//2 - 8
            cor = COR_BRANCA if p>0 else COR_PRETA
            borda = (60,60,60)
            pygame.draw.circle(tela, cor, (cx,cy), raio)
            pygame.draw.circle(tela, borda, (cx,cy), raio, 3)
            if abs(p) == 2:
                # coroa estilizada
                pygame.draw.circle(tela, COR_DESTAQ, (cx,cy), raio//2, 3)
                pygame.draw.line(tela, COR_DESTAQ, (cx-raio//2, cy), (cx+raio//2, cy), 2)


def desenha_painel(e: Estado, msg=""):
    x0 = LARGURA_TAB
    pygame.draw.rect(tela, (245,245,245), (x0,0,PAINEL_INFO,ALTURA_JANELA))
    # Título
    texto = fonte_big.render('Damas — IA LT', True, COR_TEXTO)
    tela.blit(texto, (x0+16, 16))

    turno_txt = 'Brancas (Você)' if e.turno>0 else 'Pretas (IA)'
    t2 = fonte.render(f'Turno: {turno_txt}', True, COR_TEXTO)
    tela.blit(t2, (x0+16, 56))

    t3 = fonte.render('Controles:', True, COR_TEXTO)
    tela.blit(t3, (x0+16, 96))
    linhas = [
        '- Clique na peça e depois no destino',
        '- Captura é obrigatória',
        f'- IA ~{TEMPO_IA_PADRAO:.1f}s por lance',
        '- R: reiniciar | ESC: sair'
    ]
    y = 120
    for lin in linhas:
        tela.blit(fonte.render(lin, True, COR_TEXTO), (x0+16, y))
        y += 24

    if msg:
        tela.blit(fonte.render(msg, True, (140,0,0)), (x0+16, y+8))


def casas_de_jogada(j: Jogada):
    # para evidenciar destinos no tabuleiro a partir de uma origem
    if len(j.seq) < 2: return []
    return [j.seq[-1]]

# ================================ Game Loop =================================

def principal():
    estado = Estado()
    selecionada = None
    jogadas_legais = gerar(estado)
    destinos = set()
    msg = ''

    rodando = True
    while rodando:
        clock.tick(FPS)
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                rodando = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    rodando = False
                elif ev.key == pygame.K_r:
                    estado = Estado()
                    selecionada = None
                    jogadas_legais = gerar(estado)
                    destinos = set()
                    msg = ''

            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                mx, my = ev.pos
                if mx < LARGURA_TAB:
                    j = mx // TAM_CASA
                    i = my // TAM_CASA
                    if estado.turno > 0:  # humano joga brancas
                        if selecionada is None:
                            # selecionar peça válida
                            if estado.tab[i][j] > 0:
                                # só permite selecionar se tem algum movimento da peça
                                tem = False
                                for jog in jogadas_legais:
                                    if jog.seq[0] == (i,j):
                                        tem = True; break
                                if tem:
                                    selecionada = (i,j)
                                    destinos = set()
                                    for jog in jogadas_legais:
                                        if jog.seq[0] == selecionada:
                                            destinos.update(casas_de_jogada(jog))
                        else:
                            # tentar mover
                            if (i,j) == selecionada:
                                selecionada = None
                                destinos = set()
                            else:
                                jog_escolhida = None
                                for jog in jogadas_legais:
                                    if jog.seq[0]==selecionada and jog.seq[-1]==(i,j):
                                        jog_escolhida = jog
                                        break
                                if jog_escolhida:
                                    aplicar(estado, jog_escolhida)
                                    selecionada = None
                                    destinos = set()
                                    jogadas_legais = gerar(estado)
                                    msg = ''
                                else:
                                    msg = 'Jogada inválida (ou captura obrigatória em outra peça).'

        # turno da IA
        if rodando and estado.turno < 0:
            pygame.event.pump()
            jogs = gerar(estado)
            if not jogs:
                rodando = False
            else:
                j = pensar(estado, TEMPO_IA_PADRAO)
                aplicar(estado, j)
                jogadas_legais = gerar(estado)

        # fim de jogo
        if rodando:
            if not gerar(estado):
                rodando = False

        # desenhar
        tela.fill((0,0,0))
        desenha_tabuleiro(estado, selecionada, destinos)
        desenha_painel(estado, msg)
        pygame.display.flip()

    # mensagem final
    tela.fill((255,255,255))
    vencedor = 'Pretas (IA)' if estado.turno>0 else 'Brancas (Você)'
    fimtxt = fonte_big.render(f'Fim de jogo — Vencedor: {vencedor}', True, (10,10,10))
    tela.blit(fimtxt, (40, ALTURA_JANELA//2-20))
    pygame.display.flip()
    time.sleep(2)


if __name__ == '__main__':
    principal()
