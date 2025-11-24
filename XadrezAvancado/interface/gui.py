"""
Interface Gráfica Pygame
"""

import pygame
import sys
from typing import Optional, Tuple
from config import *
from motor.tabuleiro import Estado, eh_branca, eh_preta
from motor.avaliacao import Avaliador
from motor.ia import MotorIA
from utils.tipos import Movimento

class JogoXadrez:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Xadrez 2.0 — IA Avançada (PVS)")
        self.tela = pygame.display.set_mode((LARGURA_JANELA, ALTURA_JANELA))
        self.relogio = pygame.time.Clock()

        # Fontes
        self.fonte_pecas = pygame.font.SysFont("DejaVu Sans", 56)
        self.fonte_info = pygame.font.SysFont("Arial", 18)
        self.fonte_titulo = pygame.font.SysFont("Arial", 22, bold=True)

        self.estado = Estado()
        self.avaliador = Avaliador()
        self.motor_ia = MotorIA(self.avaliador, tempo_por_lance=2.0)

        self.casa_selecionada: Optional[Tuple[int, int]] = None
        self.movimentos_legais_cache: List[Movimento] = []
        self.ultimo_movimento: Optional[Movimento] = None
        self.rodando = True
        self.humano_brancas = True
        self.pensando = False

    def desenhar_tabuleiro(self):
        for linha in range(8):
            for coluna in range(8):
                cor = COR_TAB_CLARO if (linha + coluna) % 2 == 0 else COR_TAB_ESCURO
                pygame.draw.rect(self.tela, cor, (coluna * LADO_CASA, linha * LADO_CASA, LADO_CASA, LADO_CASA))

        # Último movimento
        if self.ultimo_movimento:
            l1, c1 = self.ultimo_movimento.origem
            l2, c2 = self.ultimo_movimento.destino
            s = pygame.Surface((LADO_CASA, LADO_CASA), pygame.SRCALPHA)
            s.fill((252, 211, 77, 120))
            self.tela.blit(s, (c1 * LADO_CASA, l1 * LADO_CASA))
            self.tela.blit(s, (c2 * LADO_CASA, l2 * LADO_CASA))

        # Seleção
        if self.casa_selecionada:
            l, c = self.casa_selecionada
            pygame.draw.rect(self.tela, COR_SELECIONADO, (c * LADO_CASA, l * LADO_CASA, LADO_CASA, LADO_CASA), 4)
            
            # Dicas
            for mov in self.movimentos_legais_cache:
                if mov.origem == self.casa_selecionada:
                    lr, cr = mov.destino
                    pygame.draw.circle(self.tela, (30, 64, 175), (cr * LADO_CASA + LADO_CASA//2, lr * LADO_CASA + LADO_CASA//2), 10)

    def desenhar_pecas(self):
        for l in range(8):
            for c in range(8):
                peca = self.estado.tabuleiro[l][c]
                if peca == VAZIO:
                    continue
                texto = PECAS_UNICODE.get(peca)
                if texto is None:
                    continue
                
                # Cor da peça conforme solicitado (azul escuro para todas, usando o glifo para diferenciar)
                cor_peca = (15, 23, 42)
                
                superficie = self.fonte_pecas.render(texto, True, cor_peca)
                ret = superficie.get_rect(
                    center=(c * LADO_CASA + LADO_CASA // 2, l * LADO_CASA + LADO_CASA // 2)
                )
                self.tela.blit(superficie, ret)

    def desenhar_painel(self):
        x0 = TAMANHO_TABULEIRO
        pygame.draw.rect(self.tela, COR_FUNDO, (x0, 0, LARGURA_PAINEL_INFO, ALTURA_JANELA))
        
        def texto(txt, y, cor=COR_TEXTO, tamanho=18):
            f = self.fonte_info if tamanho==18 else self.fonte_titulo
            s = f.render(txt, True, cor)
            self.tela.blit(s, (x0 + 20, y))

        texto("XADREZ 2.0", 20, COR_DESTAQUE, 24)
        
        turno = "Brancas" if self.estado.brancas_jogam else "Pretas"
        cor_turno = COR_HUMANO if self.estado.brancas_jogam == self.humano_brancas else COR_IA
        texto(f"Vez: {turno}", 60, cor_turno)
        
        if self.pensando:
            texto("IA Pensando...", 90, COR_IA)
        
        score = self.avaliador.avaliar(self.estado) / 100.0
        texto(f"Eval: {score:+.2f}", 130)
        texto(f"Nós: {self.motor_ia.nos_avaliados}", 160)
        
        texto("Controles:", 220, COR_DESTAQUE)
        texto("R: Reiniciar", 250)
        texto("F: Inverter Lado", 275)
        texto("ESC: Sair", 300)

    def loop(self):
        while self.rodando:
            self.relogio.tick(60)
            
            # IA Joga
            ia_turn = (self.estado.brancas_jogam and not self.humano_brancas) or \
                      (not self.estado.brancas_jogam and self.humano_brancas)
            
            if ia_turn and not self.pensando:
                self.pensando = True
                # Redesenha para mostrar "Pensando..."
                self.desenhar_tudo()
                pygame.display.flip()
                
                mov = self.motor_ia.escolher_movimento(self.estado)
                if mov:
                    self.estado.fazer_movimento(mov)
                    self.ultimo_movimento = mov
                else:
                    print("Game Over ou Bug na IA")
                self.pensando = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.rodando = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE: self.rodando = False
                    if event.key == pygame.K_r: self.__init__() # Reinicia
                    if event.key == pygame.K_f: self.humano_brancas = not self.humano_brancas
                elif event.type == pygame.MOUSEBUTTONDOWN and not ia_turn:
                    self.tratar_clique(event.pos)

            self.desenhar_tudo()
            pygame.display.flip()
        
        pygame.quit()
        sys.exit()

    def desenhar_tudo(self):
        self.tela.fill((0,0,0))
        self.desenhar_tabuleiro()
        self.desenhar_pecas()
        self.desenhar_painel()

    def tratar_clique(self, pos):
        x, y = pos
        if x >= TAMANHO_TABULEIRO: return
        c, l = x // LADO_CASA, y // LADO_CASA
        
        # Se clicou na mesma casa, deseleciona
        if self.casa_selecionada == (l, c):
            self.casa_selecionada = None
            self.movimentos_legais_cache = []
            return

        # Tenta mover
        if self.casa_selecionada:
            for mov in self.movimentos_legais_cache:
                if mov.origem == self.casa_selecionada and mov.destino == (l, c):
                    self.estado.fazer_movimento(mov)
                    self.ultimo_movimento = mov
                    self.casa_selecionada = None
                    self.movimentos_legais_cache = []
                    return
        
        # Seleciona nova peça
        peca = self.estado.tabuleiro[l][c]
        if peca != VAZIO:
            if (self.estado.brancas_jogam and eh_branca(peca)) or \
               (not self.estado.brancas_jogam and eh_preta(peca)):
                self.casa_selecionada = (l, c)
                self.movimentos_legais_cache = self.estado.gerar_movimentos_legais()
