# ============================================================
# GO 19x19 — IA MCTS estilo AlphaGo 
# Autor: Luiz Tiago Wilcke (LT)
# ============================================================
# - Tabuleiro 19x19 (Go clássico)
# - Preto = IA (MCTS com heurísticas e "prior" tipo AlphaGo)
# - Branco = Humano (mouse)
# - Regras básicas: capturas, proibição de suicídio, ko simples
# - Pontuação: território + capturas (estimativa de área)
# ============================================================

import pygame
import sys
import math
import random
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np

# ------------------------------------------------------------
# Configurações do tabuleiro e da janela
# ------------------------------------------------------------
TAMANHO_GRADE = 19
COR_VAZIO = 0
COR_PRETA = 1
COR_BRANCA = 2

LARGURA_JANELA = 900
ALTURA_JANELA = 900

# Margens específicas do tabuleiro
MARGEM_ESQUERDA = 60
MARGEM_DIREITA = 60
MARGEM_SUPERIOR_TAB = 100   # começa abaixo da legenda
MARGEM_INFERIOR_TAB = 60

COR_FUNDO = (230, 200, 140)   # madeira clara
COR_LINHA = (0, 0, 0)
COR_PRETA_PYGAME = (10, 10, 10)
COR_BRANCA_PYGAME = (240, 240, 240)
COR_TEXTO = (0, 0, 0)
COR_PAINEL = (250, 245, 210)

FPS = 30

# Estrelas do tabuleiro (hoshi) em um 19x19
HOSHI_COORDS = [
    (3, 3), (3, 9), (3, 15),
    (9, 3), (9, 9), (9, 15),
    (15, 3), (15, 9), (15, 15),
]

# ------------------------------------------------------------
# Estruturas de jogo
# ------------------------------------------------------------

@dataclass
class EstadoJogo:
    tabuleiro: np.ndarray
    jogador_atual: int
    capturas_preto: int = 0
    capturas_branco: int = 0
    historico_hash: List[int] = None
    ultimo_lance: Optional[Tuple[int, int]] = None
    passes_consecutivos: int = 0

    def copiar(self) -> 'EstadoJogo':
        return EstadoJogo(
            tabuleiro=self.tabuleiro.copy(),
            jogador_atual=self.jogador_atual,
            capturas_preto=self.capturas_preto,
            capturas_branco=self.capturas_branco,
            historico_hash=list(self.historico_hash) if self.historico_hash is not None else [],
            ultimo_lance=self.ultimo_lance,
            passes_consecutivos=self.passes_consecutivos
        )

# ------------------------------------------------------------
# Funções de regras do Go
# ------------------------------------------------------------

def criar_tabuleiro() -> np.ndarray:
    return np.zeros((TAMANHO_GRADE, TAMANHO_GRADE), dtype=np.int8)

def dentro_limites(linha: int, coluna: int) -> bool:
    return 0 <= linha < TAMANHO_GRADE and 0 <= coluna < TAMANHO_GRADE

def vizinhos(linha: int, coluna: int) -> List[Tuple[int, int]]:
    vs = []
    if linha > 0: vs.append((linha-1, coluna))
    if linha < TAMANHO_GRADE-1: vs.append((linha+1, coluna))
    if coluna > 0: vs.append((linha, coluna-1))
    if coluna < TAMANHO_GRADE-1: vs.append((linha, coluna+1))
    return vs

def hash_tabuleiro(tabuleiro: np.ndarray) -> int:
    # Hash simples (não é Zobrist, mas suficiente para ko básico)
    return hash(tabuleiro.tobytes())

def grupo_e_liberdades(tabuleiro: np.ndarray, linha: int, coluna: int) -> Tuple[List[Tuple[int, int]], set]:
    cor = tabuleiro[linha, coluna]
    assert cor != COR_VAZIO
    visitados = set()
    liberdades = set()
    pilha = [(linha, coluna)]
    while pilha:
        l, c = pilha.pop()
        if (l, c) in visitados:
            continue
        visitados.add((l, c))
        for (ln, cn) in vizinhos(l, c):
            if tabuleiro[ln, cn] == COR_VAZIO:
                liberdades.add((ln, cn))
            elif tabuleiro[ln, cn] == cor and (ln, cn) not in visitados:
                pilha.append((ln, cn))
    return list(visitados), liberdades

def remover_grupo(tabuleiro: np.ndarray, grupo: List[Tuple[int, int]]) -> int:
    removidos = 0
    for (l, c) in grupo:
        if tabuleiro[l, c] != COR_VAZIO:
            tabuleiro[l, c] = COR_VAZIO
            removidos += 1
    return removidos

def aplicar_jogada_bruta(estado: EstadoJogo, linha: Optional[int], coluna: Optional[int]) -> EstadoJogo:
    """
    Aplica jogada assumindo que já é legal (sem verificar suicídio/ko aqui).
    Retorna novo EstadoJogo.
    """
    novo = estado.copiar()
    tab = novo.tabuleiro
    jogador = novo.jogador_atual
    adversario = COR_PRETA if jogador == COR_BRANCA else COR_BRANCA

    if linha is None and coluna is None:
        # Passar
        novo.jogador_atual = adversario
        novo.passes_consecutivos += 1
        novo.ultimo_lance = None
    else:
        tab[linha, coluna] = jogador
        capturas = 0
        # capturar grupos adversários sem liberdade
        for (ln, cn) in vizinhos(linha, coluna):
            if tab[ln, cn] == adversario:
                grupo_adv, libs = grupo_e_liberdades(tab, ln, cn)
                if len(libs) == 0:
                    capturas += remover_grupo(tab, grupo_adv)

        if jogador == COR_PRETA:
            novo.capturas_preto += capturas
        else:
            novo.capturas_branco += capturas

        novo.passes_consecutivos = 0
        novo.jogador_atual = adversario
        novo.ultimo_lance = (linha, coluna)

    # Atualiza histórico de hash
    h = hash_tabuleiro(tab)
    if novo.historico_hash is None:
        novo.historico_hash = []
    novo.historico_hash.append(h)
    return novo

def jogada_legal(estado: EstadoJogo, linha: Optional[int], coluna: Optional[int]) -> bool:
    """
    Verifica se uma jogada é legal.
    linha,coluna = None,None -> passar (sempre legal).
    """
    if linha is None and coluna is None:
        return True

    tab = estado.tabuleiro
    jogador = estado.jogador_atual
    adversario = COR_PRETA if jogador == COR_BRANCA else COR_BRANCA

    if not dentro_limites(linha, coluna):
        return False
    if tab[linha, coluna] != COR_VAZIO:
        return False

    # Copia para testar
    tab_teste = tab.copy()
    tab_teste[linha, coluna] = jogador

    # Capturas de grupos adversários
    capturas = 0
    for (ln, cn) in vizinhos(linha, coluna):
        if tab_teste[ln, cn] == adversario:
            grupo_adv, libs = grupo_e_liberdades(tab_teste, ln, cn)
            if len(libs) == 0:
                capturas += len(grupo_adv)
                for (gx, gy) in grupo_adv:
                    tab_teste[gx, gy] = COR_VAZIO

    # Verifica se o próprio grupo tem liberdades (evitar suicídio)
    grupo_jogador, libs_jogador = grupo_e_liberdades(tab_teste, linha, coluna)
    if len(libs_jogador) == 0 and capturas == 0:
        return False

    # Ko simples: proíbe repetir exatamente a posição anterior
    h_teste = hash_tabuleiro(tab_teste)
    if estado.historico_hash is not None and len(estado.historico_hash) > 0:
        if h_teste == estado.historico_hash[-1]:
            return False

    return True

def jogadas_legais(estado: EstadoJogo, incluir_passar: bool = True) -> List[Tuple[Optional[int], Optional[int]]]:
    jogadas = []
    tab = estado.tabuleiro
    for l in range(TAMANHO_GRADE):
        for c in range(TAMANHO_GRADE):
            if tab[l, c] == COR_VAZIO and jogada_legal(estado, l, c):
                jogadas.append((l, c))
    if incluir_passar:
        jogadas.append((None, None))
    return jogadas

def pontuacao_territorio(estado: EstadoJogo) -> Tuple[float, float]:
    """
    Estima pontuação final: território + capturas.
    Método simples de área: grupos de vazios pertencem a uma cor
    se só tocam aquela cor.
    """
    tab = estado.tabuleiro
    visitado = np.zeros_like(tab, dtype=bool)
    territorio_preto = 0
    territorio_branco = 0

    for l in range(TAMANHO_GRADE):
        for c in range(TAMANHO_GRADE):
            if tab[l, c] == COR_VAZIO and not visitado[l, c]:
                fila = [(l, c)]
                regiao = []
                cores_vizinhas = set()
                while fila:
                    x, y = fila.pop()
                    if not dentro_limites(x, y):
                        continue
                    if visitado[x, y]:
                        continue
                    visitado[x, y] = True
                    if tab[x, y] == COR_VAZIO:
                        regiao.append((x, y))
                        for (vx, vy) in vizinhos(x, y):
                            if tab[vx, vy] == COR_VAZIO and not visitado[vx, vy]:
                                fila.append((vx, vy))
                            elif tab[vx, vy] in (COR_PRETA, COR_BRANCA):
                                cores_vizinhas.add(tab[vx, vy])
                if len(cores_vizinhas) == 1:
                    cor = list(cores_vizinhas)[0]
                    if cor == COR_PRETA:
                        territorio_preto += len(regiao)
                    else:
                        territorio_branco += len(regiao)
    # soma capturas
    territorio_preto += estado.capturas_preto
    territorio_branco += estado.capturas_branco
    return territorio_preto, territorio_branco

# ------------------------------------------------------------
# Política "prior" heurística (estilo AlphaGo simplificada)
# ------------------------------------------------------------

def politica_prior(estado: EstadoJogo) -> Dict[Tuple[Optional[int], Optional[int]], float]:
    """
    Retorna um dicionário {lance: prior} com pesos heurísticos:
    - Capturas ganham peso alto
    - Extensões de grupos próprios ganham peso
    - Proximidade de hoshi e centro no início
    - Passo ganha peso maior no final da partida
    """
    jogs = jogadas_legais(estado, incluir_passar=True)
    tab = estado.tabuleiro
    jogador = estado.jogador_atual
    adversario = COR_PRETA if jogador == COR_BRANCA else COR_BRANCA

    num_pedras = int(np.count_nonzero(tab != COR_VAZIO))
    centro = (TAMANHO_GRADE - 1) / 2.0

    # definir "fase" da partida
    if num_pedras < 60:
        fase = "inicio"
    elif num_pedras < 180:
        fase = "meio"
    else:
        fase = "fim"

    pesos: Dict[Tuple[Optional[int], Optional[int]], float] = {}

    for lance in jogs:
        l, c = lance
        if l is None and c is None:
            # passo
            if fase == "fim":
                pesos[lance] = 0.3
            else:
                pesos[lance] = 0.05
            continue

        # base
        peso = 1.0

        # distância ao centro
        dist_centro = abs(l - centro) + abs(c - centro)
        if fase == "inicio":
            peso *= 1.0 / (1.0 + 0.5 * dist_centro)
        elif fase == "meio":
            peso *= 1.0 / (1.0 + 0.7 * dist_centro)
        else:  # fim
            peso *= 1.0 / (1.0 + 1.0 * dist_centro)

        # proximidade de hoshi (valoriza pontos fortes no início)
        for (hl, hc) in HOSHI_COORDS:
            dist_hoshi = abs(l - hl) + abs(c - hc)
            if dist_hoshi <= 2:
                if fase == "inicio":
                    peso *= 1.4
                else:
                    peso *= 1.1
                break

        # vizinhança de grupos (valoriza lances que conectam/fight)
        tem_vizinho_proprio = False
        tem_vizinho_adv = False
        for (vx, vy) in vizinhos(l, c):
            if tab[vx, vy] == jogador:
                tem_vizinho_proprio = True
            elif tab[vx, vy] == adversario:
                tem_vizinho_adv = True

        if tem_vizinho_proprio:
            peso *= 1.6
        if tem_vizinho_adv:
            peso *= 1.2

        # checar capturas potenciais (rápido)
        tab_teste = tab.copy()
        tab_teste[l, c] = jogador
        capturas = 0
        for (vx, vy) in vizinhos(l, c):
            if tab_teste[vx, vy] == adversario:
                grupo_adv, libs = grupo_e_liberdades(tab_teste, vx, vy)
                if len(libs) == 0:
                    capturas += len(grupo_adv)
        if capturas > 0:
            peso *= (2.0 + 0.2 * capturas)

        # evitar jogadas muito suicidas (auto-atari grosseiro)
        grupo_j, libs_j = grupo_e_liberdades(tab_teste, l, c)
        if len(libs_j) == 1 and capturas == 0:
            peso *= 0.4

        # peso mínimo para não zerar
        peso = max(peso, 0.01)
        pesos[lance] = peso

    # normalizar para obter "probabilidade"
    soma = sum(pesos.values())
    if soma <= 0:
        # fallback totalmente uniforme
        n = len(jogs)
        return {j: 1.0 / n for j in jogs}

    for j in pesos:
        pesos[j] /= soma
    return pesos

# ------------------------------------------------------------
# IA: Monte Carlo Tree Search com heurísticas + PUCT
# ------------------------------------------------------------

class NoMCTS:
    def __init__(
        self,
        estado: EstadoJogo,
        pai: Optional['NoMCTS'] = None,
        lance: Optional[Tuple[Optional[int], Optional[int]]] = None,
        prior_por_lance: Optional[Dict[Tuple[Optional[int], Optional[int]], float]] = None
    ):
        self.estado = estado
        self.pai = pai
        self.lance = lance
        # prior de cada lance a partir deste nó
        self.prior_por_lance: Dict[Tuple[Optional[int], Optional[int]], float] = (
            prior_por_lance if prior_por_lance is not None else politica_prior(estado)
        )
        self.filhos: List[NoMCTS] = []
        self.visitas: int = 0
        self.valor_acumulado: float = 0.0
        # lista de jogadas ainda não expandidas
        self.jogadas_nao_expandidas: List[Tuple[Optional[int], Optional[int]]] = list(self.prior_por_lance.keys())

    def esta_completamente_expandido(self) -> bool:
        return len(self.jogadas_nao_expandidas) == 0

    def filho_com_melhor_puct(self, c_puct_base: float = 2.5) -> 'NoMCTS':
        """
        Seleção com fórmula estilo PUCT:
        Q + c_puct * P * sqrt(N_pai) / (1 + N_filho)
        """
        melhor_no = None
        melhor_valor = -1e9

        for filho in self.filhos:
            if filho.visitas == 0:
                q = 0.0
            else:
                q = filho.valor_acumulado / filho.visitas

            p = self.prior_por_lance.get(filho.lance, 1e-3)

            # ajuste leve de exploração pela fase da partida
            num_pedras = int(np.count_nonzero(self.estado.tabuleiro != COR_VAZIO))
            if num_pedras < 60:
                c_puct = c_puct_base * 1.1
            elif num_pedras < 180:
                c_puct = c_puct_base
            else:
                c_puct = c_puct_base * 0.8

            u = c_puct * p * math.sqrt(self.visitas + 1) / (1 + filho.visitas)
            valor = q + u

            if valor > melhor_valor:
                melhor_valor = valor
                melhor_no = filho

        return melhor_no

    def escolher_jogada_para_expandir(self) -> Tuple[Optional[int], Optional[int]]:
        """
        Escolhe uma jogada dentre as não expandidas, amostrando
        proporcionalmente ao prior heurístico.
        """
        if not self.jogadas_nao_expandidas:
            # fallback (não deveria acontecer aqui)
            return (None, None)

        pesos = []
        for lance in self.jogadas_nao_expandidas:
            pesos.append(self.prior_por_lance.get(lance, 1e-3))

        soma = sum(pesos)
        if soma <= 0:
            lance = random.choice(self.jogadas_nao_expandidas)
            self.jogadas_nao_expandidas.remove(lance)
            return lance

        r = random.random() * soma
        acum = 0.0
        for i, lance in enumerate(self.jogadas_nao_expandidas):
            acum += pesos[i]
            if r <= acum:
                self.jogadas_nao_expandidas.remove(lance)
                return lance

        lance = self.jogadas_nao_expandidas.pop()
        return lance

    def expandir(self) -> Optional['NoMCTS']:
        if not self.jogadas_nao_expandidas:
            return None

        lance = self.escolher_jogada_para_expandir()
        novo_estado = aplicar_jogada_bruta(self.estado, lance[0], lance[1])
        # prior para o próximo nó
        prior_filho = politica_prior(novo_estado)
        filho = NoMCTS(novo_estado, pai=self, lance=lance, prior_por_lance=prior_filho)
        self.filhos.append(filho)
        return filho

    def atualizar(self, resultado_para_preto: float):
        self.visitas += 1
        self.valor_acumulado += resultado_para_preto

# ------------------------------------------------------------
# Política de simulação (playout) usando priors
# ------------------------------------------------------------

def politica_simulacao_heuristica(estado: EstadoJogo) -> Tuple[Optional[int], Optional[int]]:
    """
    Política rápida: amostra um lance a partir dos priors heurísticos,
    com um pouco de ruído para diversificar simulações.
    """
    priors = politica_prior(estado)
    jogs = list(priors.keys())
    if len(jogs) == 1:
        return jogs[0]

    # misturar com ruído epsilon-greedy
    epsilon = 0.1
    if random.random() < epsilon:
        return random.choice(jogs)

    pesos = [priors[j] for j in jogs]
    soma = sum(pesos)
    if soma <= 0:
        return random.choice(jogs)

    r = random.random() * soma
    acum = 0.0
    for j, p in zip(jogs, pesos):
        acum += p
        if r <= acum:
            return j
    return random.choice(jogs)

def simulacao_aleatoria(estado: EstadoJogo, limite_lances: int = 200) -> float:
    """
    Executa uma simulação (playout) até 2 passes consecutivos ou limite de lances.
    Retorna resultado para preto: +1 vitória clara, -1 derrota clara,
    proporções intermediárias baseadas em diferença de pontos.
    """
    sim = estado.copiar()
    for _ in range(limite_lances):
        if sim.passes_consecutivos >= 2:
            break
        lance = politica_simulacao_heuristica(sim)
        if not jogada_legal(sim, lance[0], lance[1]):
            # se política sugeriu algo ilegal, escolhe jogada aleatória
            jogs = jogadas_legais(sim, incluir_passar=True)
            if not jogs:
                break
            lance = random.choice(jogs)
        sim = aplicar_jogada_bruta(sim, lance[0], lance[1])

    pts_preto, pts_branco = pontuacao_territorio(sim)
    diff = pts_preto - pts_branco
    # normaliza em [-1, 1] com "zona morta"
    if diff > 10:
        return 1.0
    elif diff < -10:
        return -1.0
    else:
        return diff / 10.0

def mcts_escolher_jogada(
    estado_raiz: EstadoJogo,
    tempo_segundos: float = 2.0,
    iteracoes_max: int = 5000
) -> Tuple[Optional[int], Optional[int]]:
    """
    Executa MCTS a partir de estado_raiz e retorna a melhor jogada para o preto.
    Sempre avalia resultado do ponto de vista do preto.
    """
    inicio = time.time()
    raiz = NoMCTS(estado_raiz)

    iteracoes = 0
    while time.time() - inicio < tempo_segundos and iteracoes < iteracoes_max:
        iteracoes += 1
        no = raiz

        # Seleção
        while no.esta_completamente_expandido() and no.filhos:
            no = no.filho_com_melhor_puct()

        # Expansão
        if not no.esta_completamente_expandido():
            no = no.expandir() or no

        # Simulação
        resultado_para_preto = simulacao_aleatoria(no.estado, limite_lances=120)

        # Retropropagação
        atual = no
        while atual is not None:
            atual.atualizar(resultado_para_preto)
            atual = atual.pai

    # Escolhe filho mais visitado
    if not raiz.filhos:
        # sem filhos: passar
        return (None, None)

    melhor_filho = max(raiz.filhos, key=lambda f: f.visitas)
    return melhor_filho.lance

# ------------------------------------------------------------
# Desenho com Pygame
# ------------------------------------------------------------

def calcular_espaco_e_origem():
    largura_util = LARGURA_JANELA - MARGEM_ESQUERDA - MARGEM_DIREITA
    altura_util = ALTURA_JANELA - MARGEM_SUPERIOR_TAB - MARGEM_INFERIOR_TAB
    espaco = min(
        largura_util / (TAMANHO_GRADE - 1),
        altura_util / (TAMANHO_GRADE - 1)
    )
    origem_x = MARGEM_ESQUERDA
    origem_y = MARGEM_SUPERIOR_TAB
    return espaco, origem_x, origem_y

def desenhar_tabuleiro(screen, fonte, fonte_menor, estado: EstadoJogo):
    screen.fill(COR_FUNDO)
    espaco, origem_x, origem_y = calcular_espaco_e_origem()

    # Painel superior (legenda) — não será cruzado por linhas
    altura_painel = 80
    pygame.draw.rect(screen, COR_PAINEL, (0, 0, LARGURA_JANELA, altura_painel))

    # Linha 1: título
    texto_titulo = "GO 19x19 — IA MCTS estilo AlphaGo (simplificada)"
    surf_titulo = fonte.render(texto_titulo, True, COR_TEXTO)
    screen.blit(surf_titulo, (10, 5))

    # Linha 2: autor
    texto_autor = "Autor: Luiz Tiago Wilcke (LT) — Preto (IA) vs Branco (Humano)"
    surf_autor = fonte_menor.render(texto_autor, True, COR_TEXTO)
    screen.blit(surf_autor, (10, 30))

    # Linha 3: placar
    texto_placar = (
        f"Capturas — Preto (IA): {estado.capturas_preto} | "
        f"Branco (Humano): {estado.capturas_branco}"
    )
    surf_placar = fonte_menor.render(texto_placar, True, COR_TEXTO)
    screen.blit(surf_placar, (10, 55))

    # Linhas do tabuleiro (começando abaixo do painel)
    for i in range(TAMANHO_GRADE):
        # horizontais
        y = origem_y + i * espaco
        x1 = origem_x
        x2 = origem_x + (TAMANHO_GRADE - 1) * espaco
        pygame.draw.line(screen, COR_LINHA, (x1, y), (x2, y), 2)

        # verticais
        x = origem_x + i * espaco
        y1 = origem_y
        y2 = origem_y + (TAMANHO_GRADE - 1) * espaco
        pygame.draw.line(screen, COR_LINHA, (x, y1), (x, y2), 2)

    # Hoshi (pontos)
    raio_hoshi = 5
    for (l, c) in HOSHI_COORDS:
        x = int(origem_x + c * espaco)
        y = int(origem_y + l * espaco)
        pygame.draw.circle(screen, COR_LINHA, (x, y), raio_hoshi)

    # Pedras
    raio_pedra = int(espaco * 0.45)
    for l in range(TAMANHO_GRADE):
        for c in range(TAMANHO_GRADE):
            if estado.tabuleiro[l, c] == COR_VAZIO:
                continue
            x = int(origem_x + c * espaco)
            y = int(origem_y + l * espaco)
            cor = COR_PRETA_PYGAME if estado.tabuleiro[l, c] == COR_PRETA else COR_BRANCA_PYGAME
            pygame.draw.circle(screen, cor, (x, y), raio_pedra)
            pygame.draw.circle(screen, COR_LINHA, (x, y), raio_pedra, 1)

    # Indicar último lance
    if estado.ultimo_lance is not None:
        l, c = estado.ultimo_lance
        x = int(origem_x + c * espaco)
        y = int(origem_y + l * espaco)
        r = int(raio_pedra * 0.5)
        pygame.draw.rect(screen, (255, 0, 0), (x - r, y - r, 2*r, 2*r), 2)

def pos_mouse_para_coordenadas(mouse_pos) -> Tuple[int, int]:
    mx, my = mouse_pos
    espaco, origem_x, origem_y = calcular_espaco_e_origem()
    c = int(round((mx - origem_x) / espaco))
    l = int(round((my - origem_y) / espaco))
    return l, c

def mostrar_mensagem(screen, fonte, msg: str):
    y = ALTURA_JANELA - 30
    pygame.draw.rect(screen, COR_PAINEL, (0, y - 5, LARGURA_JANELA, 40))
    surf = fonte.render(msg, True, COR_TEXTO)
    screen.blit(surf, (10, y))

# ------------------------------------------------------------
# Loop principal
# ------------------------------------------------------------

def jogar():
    pygame.init()
    screen = pygame.display.set_mode((LARGURA_JANELA, ALTURA_JANELA))
    pygame.display.set_caption(
        "GO 19x19 — IA MCTS (Preto) vs Humano (Branco) — Autor: Luiz Tiago Wilcke"
    )
    clock = pygame.time.Clock()
    fonte = pygame.font.SysFont("arial", 18)
    fonte_menor = pygame.font.SysFont("arial", 14)

    estado = EstadoJogo(
        tabuleiro=criar_tabuleiro(),
        jogador_atual=COR_PRETA,  # IA começa (preto)
        capturas_preto=0,
        capturas_branco=0,
        historico_hash=[],
        ultimo_lance=None,
        passes_consecutivos=0
    )
    estado.historico_hash.append(hash_tabuleiro(estado.tabuleiro))

    jogo_terminado = False
    mensagem = "Preto (IA) começa. Clique no tabuleiro para jogar como Branco. Tecla P = passar, R = reiniciar."

    while True:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if jogo_terminado:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    # reiniciar mesmo após fim
                    estado = EstadoJogo(
                        tabuleiro=criar_tabuleiro(),
                        jogador_atual=COR_PRETA,
                        capturas_preto=0,
                        capturas_branco=0,
                        historico_hash=[],
                        ultimo_lance=None,
                        passes_consecutivos=0
                    )
                    estado.historico_hash.append(hash_tabuleiro(estado.tabuleiro))
                    jogo_terminado = False
                    mensagem = "Novo jogo iniciado. Preto (IA) começa."
                continue

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p and estado.jogador_atual == COR_BRANCA:
                    # Branco passa
                    if jogada_legal(estado, None, None):
                        estado = aplicar_jogada_bruta(estado, None, None)
                        mensagem = "Branco passou. Agora é a vez da IA (Preto)."
                if event.key == pygame.K_r:
                    # Reiniciar
                    estado = EstadoJogo(
                        tabuleiro=criar_tabuleiro(),
                        jogador_atual=COR_PRETA,
                        capturas_preto=0,
                        capturas_branco=0,
                        historico_hash=[],
                        ultimo_lance=None,
                        passes_consecutivos=0
                    )
                    estado.historico_hash.append(hash_tabuleiro(estado.tabuleiro))
                    jogo_terminado = False
                    mensagem = "Novo jogo iniciado. Preto (IA) começa."

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if estado.jogador_atual == COR_BRANCA and not jogo_terminado:
                    l, c = pos_mouse_para_coordenadas(pygame.mouse.get_pos())
                    if dentro_limites(l, c) and jogada_legal(estado, l, c):
                        estado = aplicar_jogada_bruta(estado, l, c)
                        mensagem = "Lance do Branco aceito. A IA está pensando..."
                    else:
                        mensagem = "Lance ilegal para Branco."

        # Turno da IA (Preto)
        if not jogo_terminado and estado.jogador_atual == COR_PRETA:
            desenhar_tabuleiro(screen, fonte, fonte_menor, estado)
            mostrar_mensagem(screen, fonte_menor, "IA (Preto) pensando...")
            pygame.display.flip()

            lance_ia = mcts_escolher_jogada(estado, tempo_segundos=1.5, iteracoes_max=4000)
            if not jogada_legal(estado, lance_ia[0], lance_ia[1]):
                # fallback simples
                jogs = jogadas_legais(estado, incluir_passar=True)
                if jogs:
                    lance_ia = random.choice(jogs)
                else:
                    lance_ia = (None, None)

            estado = aplicar_jogada_bruta(estado, lance_ia[0], lance_ia[1])
            if lance_ia[0] is None:
                mensagem = "IA (Preto) passou. Sua vez (Branco)."
            else:
                mensagem = (
                    f"IA (Preto) jogou em ({lance_ia[0]+1}, {lance_ia[1]+1}). "
                    "Sua vez (Branco)."
                )

        # Verificar fim de jogo (dois passes)
        if not jogo_terminado and estado.passes_consecutivos >= 2:
            jogo_terminado = True
            pts_preto, pts_branco = pontuacao_territorio(estado)
            if pts_preto > pts_branco:
                mensagem = f"Fim de jogo. IA (Preto) vence por {pts_preto} a {pts_branco}."
            elif pts_branco > pts_preto:
                mensagem = f"Fim de jogo. Branco (Humano) vence por {pts_branco} a {pts_preto}."
            else:
                mensagem = f"Fim de jogo. Empate {pts_preto} a {pts_preto}."
            mensagem += " Tecle R para reiniciar."

        # Desenhar
        desenhar_tabuleiro(screen, fonte, fonte_menor, estado)
        mostrar_mensagem(screen, fonte_menor, mensagem)
        pygame.display.flip()

if __name__ == "__main__":
    jogar()
