import pygame
import numpy as np
import random
import sys
from collections import defaultdict, deque
import math
import time
import copy

# Inicialização do Pygame
pygame.init()

# Constantes
TAMANHO_TABULEIRO = 19
TAMANHO_GRADE = 30
MARGEM = 40
RAIO_PEDRA = 13
TAMANHO_JANELA = 2 * MARGEM + TAMANHO_GRADE * (TAMANHO_TABULEIRO - 1)
FPS = 60

# Cores
COR_FUNDO = (220, 179, 92)
PRETO = (0, 0, 0)
BRANCO = (255, 255, 255)
COR_GRADE = (0, 0, 0)
COR_TEXTO = (50, 50, 50)
COR_DESTAQUE = (255, 0, 0)
COR_IA = (100, 0, 100)

class JogoGo:
    def __init__(self):
        self.tabuleiro = np.zeros((TAMANHO_TABULEIRO, TAMANHO_TABULEIRO), dtype=int)
        self.jogador_atual = 1  # 1 para preto, -1 para branco
        self.jogo_acabou = False
        self.ultimo_movimento = None
        self.historico_tabuleiros = []
        self.pontos_pretas = 6.5  # Komi para brancas
        self.pontos_brancas = 0
        self.passou_vez = False
        self.contador_passes = 0
        self.movimento_numero = 0
        
    def copiar(self):
        """Cria uma cópia profunda do jogo"""
        novo_jogo = JogoGo()
        novo_jogo.tabuleiro = self.tabuleiro.copy()
        novo_jogo.jogador_atual = self.jogador_atual
        novo_jogo.jogo_acabou = self.jogo_acabou
        novo_jogo.ultimo_movimento = self.ultimo_movimento
        novo_jogo.pontos_pretas = self.pontos_pretas
        novo_jogo.pontos_brancas = self.pontos_brancas
        novo_jogo.passou_vez = self.passou_vez
        novo_jogo.contador_passes = self.contador_passes
        novo_jogo.movimento_numero = self.movimento_numero
        return novo_jogo
        
    def obter_vizinhos(self, linha, coluna):
        """Retorna as posições vizinhas válidas"""
        vizinhos = []
        for dl, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nl, nc = linha + dl, coluna + dc
            if 0 <= nl < TAMANHO_TABULEIRO and 0 <= nc < TAMANHO_TABULEIRO:
                vizinhos.append((nl, nc))
        return vizinhos
    
    def tem_liberdades(self, linha, coluna, visitados=None):
        """Verifica se um grupo de pedras tem liberdades"""
        if visitados is None:
            visitados = set()
        
        jogador = self.tabuleiro[linha, coluna]
        if jogador == 0:
            return True
        
        visitados.add((linha, coluna))
        
        for nl, nc in self.obter_vizinhos(linha, coluna):
            if (nl, nc) not in visitados:
                if self.tabuleiro[nl, nc] == 0:
                    return True
                elif self.tabuleiro[nl, nc] == jogador:
                    if self.tem_liberdades(nl, nc, visitados):
                        return True
        return False
    
    def contar_liberdades_grupo(self, linha, coluna):
        """Conta o número de liberdades de um grupo"""
        jogador = self.tabuleiro[linha, coluna]
        if jogador == 0:
            return 0
        
        visitados = set()
        fila = [(linha, coluna)]
        liberdades = set()
        
        while fila:
            l, c = fila.pop()
            if (l, c) in visitados:
                continue
            visitados.add((l, c))
            
            for nl, nc in self.obter_vizinhos(l, c):
                if self.tabuleiro[nl, nc] == 0:
                    liberdades.add((nl, nc))
                elif self.tabuleiro[nl, nc] == jogador and (nl, nc) not in visitados:
                    fila.append((nl, nc))
        
        return len(liberdades)
    
    def identificar_grupo(self, linha, coluna):
        """Identifica todas as pedras de um grupo"""
        jogador = self.tabuleiro[linha, coluna]
        if jogador == 0:
            return set()
        
        grupo = set()
        fila = [(linha, coluna)]
        
        while fila:
            l, c = fila.pop()
            if (l, c) in grupo:
                continue
            grupo.add((l, c))
            
            for nl, nc in self.obter_vizinhos(l, c):
                if self.tabuleiro[nl, nc] == jogador and (nl, nc) not in grupo:
                    fila.append((nl, nc))
        
        return grupo
    
    def remover_grupo(self, linha, coluna):
        """Remove um grupo de pedras sem liberdades"""
        jogador = self.tabuleiro[linha, coluna]
        if jogador == 0:
            return 0
        
        grupo = self.identificar_grupo(linha, coluna)
        
        # Verifica se o grupo tem liberdades
        tem_liberdade = False
        for l, c in grupo:
            for nl, nc in self.obter_vizinhos(l, c):
                if self.tabuleiro[nl, nc] == 0:
                    tem_liberdade = True
                    break
            if tem_liberdade:
                break
        
        if not tem_liberdade:
            for l, c in grupo:
                self.tabuleiro[l, c] = 0
            return len(grupo) * jogador
        
        return 0
    
    def eh_movimento_valido(self, linha, coluna, jogador):
        """Verifica se um movimento é válido"""
        # Verifica se a posição está vazia
        if self.tabuleiro[linha, coluna] != 0:
            return False
        
        # Verifica ko (posição repetida)
        for historico in self.historico_tabuleiros[-3:]:  # Verifica últimos 3 movimentos
            tabuleiro_temp = self.tabuleiro.copy()
            tabuleiro_temp[linha, coluna] = jogador
            if np.array_equal(historico, tabuleiro_temp):
                return False
        
        # Verifica suicídio
        tabuleiro_temporario = self.tabuleiro.copy()
        tabuleiro_temporario[linha, coluna] = jogador
        
        # Verifica se tem liberdades após a jogada
        tem_liberdade = False
        for nl, nc in self.obter_vizinhos(linha, coluna):
            if tabuleiro_temporario[nl, nc] == 0:
                tem_liberdade = True
                break
        
        if not tem_liberdade:
            # Verifica se captura alguma pedra adversária
            capturou = False
            for nl, nc in self.obter_vizinhos(linha, coluna):
                if tabuleiro_temporario[nl, nc] == -jogador:
                    grupo_adversario = self.identificar_grupo(nl, nc)
                    liberdades_adversario = 0
                    for l, c in grupo_adversario:
                        for vl, vc in self.obter_vizinhos(l, c):
                            if tabuleiro_temporario[vl, vc] == 0:
                                liberdades_adversario += 1
                    if liberdades_adversario == 0:
                        capturou = True
                        break
            
            if not capturou:
                return False
        
        return True
    
    def fazer_movimento(self, linha, coluna, jogador):
        """Faz um movimento no tabuleiro"""
        if not self.eh_movimento_valido(linha, coluna, jogador):
            return False
        
        self.tabuleiro[linha, coluna] = jogador
        self.ultimo_movimento = (linha, coluna)
        self.movimento_numero += 1
        
        # Remove pedras capturadas
        pedras_capturadas = 0
        for nl, nc in self.obter_vizinhos(linha, coluna):
            if self.tabuleiro[nl, nc] == -jogador:
                capturas = self.remover_grupo(nl, nc)
                if capturas > 0:
                    pedras_capturadas += abs(capturas)
        
        # Atualiza pontos
        if jogador == 1:  # Pretas
            self.pontos_pretas += pedras_capturadas
        else:  # Brancas
            self.pontos_brancas += pedras_capturadas
        
        self.historico_tabuleiros.append(self.tabuleiro.copy())
        self.passou_vez = False
        self.contador_passes = 0
        return True
    
    def passar_vez(self):
        """Passa a vez do jogador atual"""
        self.passou_vez = True
        self.jogador_atual *= -1
        self.contador_passes += 1
        self.movimento_numero += 1
        
        if self.contador_passes >= 2:
            self.jogo_acabou = True
    
    def calcular_pontuacao_final(self):
        """Calcula a pontuação final do jogo"""
        territorio_pretas = self.pontos_pretas
        territorio_brancas = self.pontos_brancas
        
        # Marca território
        visitados = set()
        for i in range(TAMANHO_TABULEIRO):
            for j in range(TAMANHO_TABULEIRO):
                if (i, j) not in visitados and self.tabuleiro[i, j] == 0:
                    area, borda_pretas, borda_brancas = self.analisar_area(i, j, visitados)
                    if borda_pretas and not borda_brancas:
                        territorio_pretas += len(area)
                    elif borda_brancas and not borda_pretas:
                        territorio_brancas += len(area)
        
        return territorio_pretas, territorio_brancas
    
    def analisar_area(self, linha, coluna, visitados):
        """Analisa uma área vazia para determinar território"""
        area = []
        fila = [(linha, coluna)]
        borda_pretas = False
        borda_brancas = False
        
        while fila:
            x, y = fila.pop()
            if (x, y) in visitados:
                continue
            visitados.add((x, y))
            area.append((x, y))
            
            for nx, ny in self.obter_vizinhos(x, y):
                if self.tabuleiro[nx, ny] == 1:
                    borda_pretas = True
                elif self.tabuleiro[nx, ny] == -1:
                    borda_brancas = True
                elif self.tabuleiro[nx, ny] == 0 and (nx, ny) not in visitados:
                    fila.append((nx, ny))
        
        return area, borda_pretas, borda_brancas

class IAGoInteligente:
    def __init__(self, nivel_dificuldade=5):
        self.nivel_dificuldade = nivel_dificuldade
        self.padroes_abertura = self.inicializar_padroes_abertura()
        self.historico_movimentos = []
        self.estilo_jogo = "balanceado"  # balanceado, agressivo, defensivo
        self.tempo_analise = 0.0
        
    def inicializar_padroes_abertura(self):
        """Inicializa padrões profissionais de abertura"""
        return {
            'shimari': [(3, 3), (3, 15), (15, 3), (15, 15)],
            'hoshi': [(3, 9), (9, 3), (9, 15), (15, 9)],
            'komoku': [(4, 4), (4, 14), (14, 4), (14, 14)],
            'takamoku': [(4, 3), (4, 15), (14, 3), (14, 15)],
            'mokuhazushi': [(3, 4), (3, 14), (15, 4), (15, 14)]
        }
    
    def fazer_movimento_ia(self, jogo):
        """IA inteligente que usa múltiplas estratégias"""
        inicio_tempo = time.time()
        
        movimentos_validos = self.encontrar_movimentos_validos(jogo)
        if not movimentos_validos:
            return None
        
        # Análise da posição atual
        analise = self.analisar_posicao(jogo)
        
        # Escolhe estratégia baseada na fase do jogo
        if jogo.movimento_numero < 20:
            movimento = self.estrategia_abertura(jogo, movimentos_validos, analise)
        elif jogo.movimento_numero < 100:
            movimento = self.estrategia_meio_jogo(jogo, movimentos_validos, analise)
        else:
            movimento = self.estrategia_final(jogo, movimentos_validos, analise)
        
        self.tempo_analise = time.time() - inicio_tempo
        self.historico_movimentos.append(movimento)
        return movimento
    
    def encontrar_movimentos_validos(self, jogo):
        """Encontra movimentos válidos com filtragem inteligente"""
        movimentos = []
        # Prioriza áreas mais interessantes primeiro
        areas_interessantes = self.identificar_areas_interessantes(jogo)
        
        for area in areas_interessantes:
            for i in range(max(0, area[0]-2), min(TAMANHO_TABULEIRO, area[0]+3)):
                for j in range(max(0, area[1]-2), min(TAMANHO_TABULEIRO, area[1]+3)):
                    if jogo.eh_movimento_valido(i, j, jogo.jogador_atual):
                        movimentos.append((i, j))
        
        # Se não encontrou nas áreas interessantes, busca em todo tabuleiro
        if not movimentos:
            for i in range(TAMANHO_TABULEIRO):
                for j in range(TAMANHO_TABULEIRO):
                    if jogo.eh_movimento_valido(i, j, jogo.jogador_atual):
                        movimentos.append((i, j))
        
        return movimentos
    
    def identificar_areas_interessantes(self, jogo):
        """Identifica áreas do tabuleiro que merecem atenção"""
        areas = []
        
        # 1. Próximo ao último movimento
        if jogo.ultimo_movimento:
            areas.append(jogo.ultimo_movimento)
        
        # 2. Cantos vazios
        for canto in [(3,3), (3,15), (15,3), (15,15)]:
            if jogo.tabuleiro[canto] == 0:
                areas.append(canto)
        
        # 3. Próximo a grupos fracos adversários
        for i in range(TAMANHO_TABULEIRO):
            for j in range(TAMANHO_TABULEIRO):
                if jogo.tabuleiro[i, j] == -jogo.jogador_atual:
                    if jogo.contar_liberdades_grupo(i, j) <= 2:
                        areas.append((i, j))
        
        # 4. Centro estratégico
        areas.append((TAMANHO_TABULEIRO//2, TAMANHO_TABULEIRO//2))
        
        return areas
    
    def analisar_posicao(self, jogo):
        """Analisa a posição atual do tabuleiro"""
        analise = {
            'controle_centro': 0,
            'grupos_fracos_oponente': [],
            'grupos_fracos_proprios': [],
            'territorio_estimado': 0,
            'iniciative': 0
        }
        
        # Analisa controle do centro
        centro_size = 5
        centro_start = (TAMANHO_TABULEIRO - centro_size) // 2
        for i in range(centro_start, centro_start + centro_size):
            for j in range(centro_start, centro_start + centro_size):
                if jogo.tabuleiro[i, j] == 1:
                    analise['controle_centro'] += 1
                elif jogo.tabuleiro[i, j] == -1:
                    analise['controle_centro'] -= 1
        
        # Identifica grupos fracos
        visitados = set()
        for i in range(TAMANHO_TABULEIRO):
            for j in range(TAMANHO_TABULEIRO):
                if (i, j) not in visitados and jogo.tabuleiro[i, j] != 0:
                    grupo = jogo.identificar_grupo(i, j)
                    visitados.update(grupo)
                    
                    liberdades = jogo.contar_liberdades_grupo(i, j)
                    if liberdades <= 2:
                        info_grupo = {
                            'tamanho': len(grupo),
                            'liberdades': liberdades,
                            'posicao': (i, j)
                        }
                        if jogo.tabuleiro[i, j] == -1:
                            analise['grupos_fracos_oponente'].append(info_grupo)
                        else:
                            analise['grupos_fracos_proprios'].append(info_grupo)
        
        return analise
    
    def estrategia_abertura(self, jogo, movimentos_validos, analise):
        """Estratégia para a fase de abertura"""
        # Tenta jogar em pontos de abertura clássicos
        for nome_padrao, pontos in self.padroes_abertura.items():
            for ponto in pontos:
                if ponto in movimentos_validos:
                    return ponto
        
        # Se não encontrou padrão clássico, escolhe baseado em influência
        return self.escolher_movimento_por_influencia(jogo, movimentos_validos)
    
    def estrategia_meio_jogo(self, jogo, movimentos_validos, analise):
        """Estratégia para o meio jogo"""
        # 1. Primeiro, verifica movimentos urgentes (ataque/defesa)
        movimento_urgente = self.buscar_movimento_urgente(jogo, movimentos_validos, analise)
        if movimento_urgente:
            return movimento_urgente
        
        # 2. Busca por movimentos que ampliam influência
        movimento_influente = self.buscar_movimento_influente(jogo, movimentos_validos)
        if movimento_influente:
            return movimento_influente
        
        # 3. Movimento padrão baseado em múltiplos critérios
        return self.escolher_movimento_estrategico(jogo, movimentos_validos)
    
    def estrategia_final(self, jogo, movimentos_validos, analise):
        """Estratégia para o final do jogo"""
        # Foca em maximizar território e minimizar riscos
        melhor_movimento = None
        melhor_pontuacao = -float('inf')
        
        for movimento in movimentos_validos:
            pontuacao = self.avaliar_movimento_final(jogo, movimento[0], movimento[1])
            if pontuacao > melhor_pontuacao:
                melhor_pontuacao = pontuacao
                melhor_movimento = movimento
        
        return melhor_movimento
    
    def buscar_movimento_urgente(self, jogo, movimentos_validos, analise):
        """Busca movimentos urgentes (ataque/defesa)"""
        # Defesa: protege grupos próprios fracos
        for grupo in analise['grupos_fracos_proprios']:
            movimento_defesa = self.encontrar_movimento_defesa(jogo, grupo, movimentos_validos)
            if movimento_defesa:
                return movimento_defesa
        
        # Ataque: ataca grupos adversários fracos
        for grupo in analise['grupos_fracos_oponente']:
            movimento_ataque = self.encontrar_movimento_ataque(jogo, grupo, movimentos_validos)
            if movimento_ataque:
                return movimento_ataque
        
        return None
    
    def encontrar_movimento_defesa(self, jogo, grupo, movimentos_validos):
        """Encontra movimento para defender grupo fraco"""
        posicao = grupo['posicao']
        
        # Procura por pontos que aumentam liberdades do grupo
        for movimento in movimentos_validos:
            dist = max(abs(movimento[0] - posicao[0]), abs(movimento[1] - posicao[1]))
            if dist <= 2:
                # Simula o movimento
                jogo_temp = jogo.copiar()
                if jogo_temp.fazer_movimento(movimento[0], movimento[1], jogo.jogador_atual):
                    novas_liberdades = jogo_temp.contar_liberdades_grupo(posicao[0], posicao[1])
                    if novas_liberdades > grupo['liberdades']:
                        return movimento
        
        return None
    
    def encontrar_movimento_ataque(self, jogo, grupo, movimentos_validos):
        """Encontra movimento para atacar grupo adversário"""
        posicao = grupo['posicao']
        
        # Procura por pontos que reduzem liberdades do grupo adversário
        for movimento in movimentos_validos:
            dist = max(abs(movimento[0] - posicao[0]), abs(movimento[1] - posicao[1]))
            if dist <= 2:
                # Verifica se ameaça o grupo
                jogo_temp = jogo.copiar()
                if jogo_temp.fazer_movimento(movimento[0], movimento[1], jogo.jogador_atual):
                    # Verifica se alguma pedra adversária ficou com 1 liberdade
                    for nl, nc in jogo_temp.obter_vizinhos(movimento[0], movimento[1]):
                        if jogo_temp.tabuleiro[nl, nc] == -jogo.jogador_atual:
                            if jogo_temp.contar_liberdades_grupo(nl, nc) == 1:
                                return movimento
        
        return None
    
    def buscar_movimento_influente(self, jogo, movimentos_validos):
        """Busca movimentos que aumentam influência"""
        melhor_movimento = None
        melhor_influencia = -1
        
        for movimento in movimentos_validos:
            influencia = self.calcular_influencia_movimento(jogo, movimento[0], movimento[1])
            if influencia > melhor_influencia:
                melhor_influencia = influencia
                melhor_movimento = movimento
        
        return melhor_movimento
    
    def calcular_influencia_movimento(self, jogo, linha, coluna):
        """Calcula a influência de um movimento"""
        influencia = 0
        raio = 4
        
        for i in range(max(0, linha-raio), min(TAMANHO_TABULEIRO, linha+raio+1)):
            for j in range(max(0, coluna-raio), min(TAMANHO_TABULEIRO, coluna+raio+1)):
                if jogo.tabuleiro[i, j] == 0:
                    distancia = max(abs(linha-i), abs(coluna-j))
                    if distancia <= raio:
                        influencia += (raio - distancia)
        
        return influencia
    
    def escolher_movimento_por_influencia(self, jogo, movimentos_validos):
        """Escolhe movimento baseado em influência e conexões"""
        melhor_movimento = None
        melhor_pontuacao = -float('inf')
        
        for movimento in movimentos_validos:
            pontuacao = self.avaliar_movimento_completo(jogo, movimento[0], movimento[1])
            if pontuacao > melhor_pontuacao:
                melhor_pontuacao = pontuacao
                melhor_movimento = movimento
        
        return melhor_movimento
    
    def escolher_movimento_estrategico(self, jogo, movimentos_validos):
        """Escolhe movimento estratégico considerando múltiplos fatores"""
        movimentos_avaliados = []
        
        for movimento in movimentos_validos:
            pontuacao = self.avaliar_movimento_estrategico(jogo, movimento[0], movimento[1])
            movimentos_avaliados.append((pontuacao, movimento))
        
        # Escolhe entre os melhores com algum elemento aleatório
        movimentos_avaliados.sort(reverse=True)
        melhores = movimentos_avaliados[:max(3, len(movimentos_avaliados)//10)]
        
        if len(melhores) > 1 and random.random() < 0.2:
            return random.choice(melhores[1:])[1]
        else:
            return melhores[0][1]
    
    def avaliar_movimento_completo(self, jogo, linha, coluna):
        """Avaliação completa de um movimento"""
        pontuacao = 0
        
        # Influência territorial
        pontuacao += self.calcular_influencia_movimento(jogo, linha, coluna) * 2
        
        # Segurança
        jogo_temp = jogo.copiar()
        if jogo_temp.fazer_movimento(linha, coluna, jogo.jogador_atual):
            if jogo_temp.contar_liberdades_grupo(linha, coluna) >= 2:
                pontuacao += 15
            else:
                pontuacao -= 10
        
        # Potencial de captura
        for nl, nc in jogo.obter_vizinhos(linha, coluna):
            if jogo.tabuleiro[nl, nc] == -jogo.jogador_atual:
                if jogo.contar_liberdades_grupo(nl, nc) == 1:
                    pontuacao += 20
        
        # Conexões com grupos próprios
        conexoes = 0
        for nl, nc in jogo.obter_vizinhos(linha, coluna):
            if jogo.tabuleiro[nl, nc] == jogo.jogador_atual:
                conexoes += 1
        pontuacao += conexoes * 8
        
        return pontuacao
    
    def avaliar_movimento_estrategico(self, jogo, linha, coluna):
        """Avaliação estratégica mais refinada"""
        return self.avaliar_movimento_completo(jogo, linha, coluna)
    
    def avaliar_movimento_final(self, jogo, linha, coluna):
        """Avaliação para fase final - foco em território"""
        pontuacao = self.calcular_influencia_movimento(jogo, linha, coluna) * 3
        
        # Bônus por fechar território
        for nl, nc in jogo.obter_vizinhos(linha, coluna):
            if jogo.tabuleiro[nl, nc] == jogo.jogador_atual:
                pontuacao += 5
        
        return pontuacao

class InterfaceGo:
    def __init__(self):
        self.tela = pygame.display.set_mode((TAMANHO_JANELA, TAMANHO_JANELA + 120))
        pygame.display.set_caption("Go 19x19 - IA Inteligente")
        self.relogio = pygame.time.Clock()
        self.fonte = pygame.font.SysFont('Arial', 14)
        self.fonte_grande = pygame.font.SysFont('Arial', 24, bold=True)
        self.fonte_pequena = pygame.font.SysFont('Arial', 12)
        
        self.jogo = JogoGo()
        self.ia = IAGoInteligente(nivel_dificuldade=5)
        self.mostrar_ajuda = True
        self.mostrar_analise = True
    
    def desenhar_tabuleiro(self):
        """Desenha o tabuleiro de Go"""
        self.tela.fill(COR_FUNDO)
        
        # Desenha as linhas da grade
        for i in range(TAMANHO_TABULEIRO):
            # Linhas horizontais
            pygame.draw.line(
                self.tela, COR_GRADE,
                (MARGEM, MARGEM + i * TAMANHO_GRADE),
                (TAMANHO_JANELA - MARGEM, MARGEM + i * TAMANHO_GRADE),
                2
            )
            # Linhas verticais
            pygame.draw.line(
                self.tela, COR_GRADE,
                (MARGEM + i * TAMANHO_GRADE, MARGEM),
                (MARGEM + i * TAMANHO_GRADE, TAMANHO_JANELA - MARGEM),
                2
            )
        
        # Desenha os pontos estrela
        pontos_estrela = [3, 9, 15]
        for i in pontos_estrela:
            for j in pontos_estrela:
                pygame.draw.circle(
                    self.tela, COR_GRADE,
                    (MARGEM + i * TAMANHO_GRADE, MARGEM + j * TAMANHO_GRADE),
                    4
                )
    
    def desenhar_pedras(self):
        """Desenha as pedras no tabuleiro"""
        for i in range(TAMANHO_TABULEIRO):
            for j in range(TAMANHO_TABULEIRO):
                if self.jogo.tabuleiro[i, j] != 0:
                    cor = PRETO if self.jogo.tabuleiro[i, j] == 1 else BRANCO
                    x = MARGEM + j * TAMANHO_GRADE
                    y = MARGEM + i * TAMANHO_GRADE
                    
                    pygame.draw.circle(self.tela, cor, (x, y), RAIO_PEDRA)
                    pygame.draw.circle(self.tela, COR_GRADE, (x, y), RAIO_PEDRA, 1)
        
        # Destaca o último movimento
        if self.jogo.ultimo_movimento:
            linha, coluna = self.jogo.ultimo_movimento
            x = MARGEM + coluna * TAMANHO_GRADE
            y = MARGEM + linha * TAMANHO_GRADE
            pygame.draw.circle(self.tela, COR_DESTAQUE, (x, y), 5, 2)
    
    def desenhar_painel_info(self):
        """Desenha o painel de informações"""
        y_base = TAMANHO_JANELA + 10
        
        # Título
        titulo = self.fonte_grande.render("GO 19x19 - IA Inteligente", True, COR_TEXTO)
        self.tela.blit(titulo, (20, y_base))
        
        # Informações do jogo
        jogador_atual = "PRETAS (IA)" if self.jogo.jogador_atual == 1 else "BRANCAS (Você)"
        info_texto = f"Jogador: {jogador_atual} | Movimento: {self.jogo.movimento_numero}"
        info = self.fonte.render(info_texto, True, COR_TEXTO)
        self.tela.blit(info, (20, y_base + 30))
        
        # Pontuação
        pontos_texto = f"Pontuação - Pretas: {self.jogo.pontos_pretas:.1f} | Brancas: {self.jogo.pontos_brancas}"
        pontos = self.fonte.render(pontos_texto, True, COR_TEXTO)
        self.tela.blit(pontos, (20, y_base + 50))
        
        # Análise da IA
        if self.mostrar_analise:
            fase = "Abertura" if self.jogo.movimento_numero < 20 else "Meio-jogo" if self.jogo.movimento_numero < 100 else "Final"
            analise_texto = f"Fase: {fase} | Tempo: {self.ia.tempo_analise:.2f}s"
            analise = self.fonte_pequena.render(analise_texto, True, COR_IA)
            self.tela.blit(analise, (20, y_base + 70))
        
        # Instruções
        if self.mostrar_ajuda:
            instrucoes = [
                "Clique: Colocar pedra | P: Passar vez | R: Reiniciar | H: Ajuda",
                "IA: Análise posicional + Estratégia adaptativa + Padrões profissionais"
            ]
            for i, texto in enumerate(instrucoes):
                texto_render = self.fonte_pequena.render(texto, True, COR_TEXTO)
                self.tela.blit(texto_render, (20, y_base + 90 + i * 16))
    
    def obter_posicao_mouse(self, pos):
        """Converte posição do mouse para coordenadas do tabuleiro"""
        x, y = pos
        if (MARGEM - RAIO_PEDRA <= x <= TAMANHO_JANELA - MARGEM + RAIO_PEDRA and
            MARGEM - RAIO_PEDRA <= y <= TAMANHO_JANELA - MARGEM + RAIO_PEDRA):
            
            coluna = round((x - MARGEM) / TAMANHO_GRADE)
            linha = round((y - MARGEM) / TAMANHO_GRADE)
            
            if 0 <= linha < TAMANHO_TABULEIRO and 0 <= coluna < TAMANHO_TABULEIRO:
                return linha, coluna
        return None
    
    def reiniciar_jogo(self):
        """Reinicia o jogo"""
        self.jogo = JogoGo()
    
    def executar(self):
        """Loop principal do jogo"""
        executando = True
        
        while executando:
            for evento in pygame.event.get():
                if evento.type == pygame.QUIT:
                    executando = False
                
                elif evento.type == pygame.KEYDOWN:
                    if evento.key == pygame.K_r:
                        self.reiniciar_jogo()
                    elif evento.key == pygame.K_p:
                        if self.jogo.jogador_atual == -1:  # Só jogador humano pode passar
                            self.jogo.passar_vez()
                    elif evento.key == pygame.K_h:
                        self.mostrar_ajuda = not self.mostrar_ajuda
                    elif evento.key == pygame.K_a:
                        self.mostrar_analise = not self.mostrar_analise
                
                elif evento.type == pygame.MOUSEBUTTONDOWN:
                    if self.jogo.jogador_atual == -1 and not self.jogo.jogo_acabou:  # Vez do jogador humano
                        pos = self.obter_posicao_mouse(evento.pos)
                        if pos:
                            linha, coluna = pos
                            if self.jogo.fazer_movimento(linha, coluna, -1):
                                self.jogo.jogador_atual = 1  # Passa para a IA
            
            # Vez da IA
            if self.jogo.jogador_atual == 1 and not self.jogo.jogo_acabou:
                movimento_ia = self.ia.fazer_movimento_ia(self.jogo)
                if movimento_ia:
                    linha, coluna = movimento_ia
                    self.jogo.fazer_movimento(linha, coluna, 1)
                    self.jogo.jogador_atual = -1
                else:
                    self.jogo.passar_vez()
            
            # Verifica se o jogo acabou
            if self.jogo.jogo_acabou:
                pontos_pretas, pontos_brancas = self.jogo.calcular_pontuacao_final()
                vencedor = "PRETAS (IA)" if pontos_pretas > pontos_brancas else "BRANCAS (Você)" if pontos_brancas > pontos_pretas else "EMPATE"
                print(f"Jogo acabou! Pretas: {pontos_pretas:.1f}, Brancas: {pontos_brancas}, Vencedor: {vencedor}")
            
            # Desenha tudo
            self.desenhar_tabuleiro()
            self.desenhar_pedras()
            self.desenhar_painel_info()
            
            pygame.display.flip()
            self.relogio.tick(FPS)
        
        pygame.quit()
        sys.exit()

# Executa o jogo
if __name__ == "__main__":
    jogo = InterfaceGo()
    jogo.executar()