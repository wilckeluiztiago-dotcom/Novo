import pygame
import numpy as np
import sys
import os

# Adicionar diretório raiz ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modelo.fisica import Foguete, solver_rk4
from modelo.parametros import ParametrosFalcon9 as P

# Configurações da Tela
LARGURA, ALTURA = 800, 600
FPS = 60

# Cores
BRANCO = (255, 255, 255)
PRETO = (0, 0, 0)
AZUL_CEU = (135, 206, 235)
AZUL_ESCURO = (0, 0, 50)
CINZA = (200, 200, 200)
VERMELHO = (255, 50, 50)
LARANJA = (255, 165, 0)

def interpolar_cor(cor1, cor2, fator):
    return tuple(int(c1 + (c2 - c1) * fator) for c1, c2 in zip(cor1, cor2))

class Visualizador:
    def __init__(self):
        pygame.init()
        self.tela = pygame.display.set_mode((LARGURA, ALTURA))
        pygame.display.set_caption("Simulador SpaceX - Falcon 9")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 16)
        self.font_grande = pygame.font.SysFont("Arial", 24)
        
        # Inicializar Simulação
        self.foguete_fisica = Foguete(carga_util=5000)
        # Pré-calcular trajetória para simplificar visualização (poderia ser tempo real)
        self.dados = solver_rk4(self.foguete_fisica, t_max=400, dt=0.1)
        self.indice_atual = 0
        self.velocidade_simulacao = 1 # 1x tempo real (considerando 60fps e dt=0.1, precisamos ajustar)
        
        # Escala: 1 pixel = X metros
        self.escala_pixels_por_metro = 0.5 
        self.offset_y = 0

    def desenhar_foguete(self, x, y, ligar_motor):
        # Corpo
        pygame.draw.rect(self.tela, CINZA, (x - 10, y - 60, 20, 60))
        # Coifa
        pygame.draw.polygon(self.tela, CINZA, [(x - 10, y - 60), (x + 10, y - 60), (x, y - 80)])
        # Aletas (decorativo)
        pygame.draw.polygon(self.tela, (150, 150, 150), [(x - 10, y), (x - 20, y + 10), (x - 10, y - 10)])
        pygame.draw.polygon(self.tela, (150, 150, 150), [(x + 10, y), (x + 20, y + 10), (x + 10, y - 10)])
        
        if ligar_motor:
            # Chama
            tamanho_chama = np.random.randint(20, 40)
            pygame.draw.polygon(self.tela, LARANJA, [(x - 8, y), (x + 8, y), (x, y + tamanho_chama)])
            pygame.draw.polygon(self.tela, VERMELHO, [(x - 5, y), (x + 5, y), (x, y + tamanho_chama * 0.6)])

    def executar(self):
        rodando = True
        while rodando:
            self.clock.tick(FPS)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    rodando = False
            
            # Atualizar Estado
            if self.indice_atual < len(self.dados['tempo']) - 1:
                # Avançar índice baseado na velocidade (dt=0.1s, FPS=60 -> 6 steps por frame para 1x?)
                # Vamos simplificar: avançar 1 step por frame = 0.1s por frame. 
                # A 60 FPS, isso é 6s de simulação por segundo real (6x acelerado).
                self.indice_atual += 1
            
            altitude = self.dados['altitude'][self.indice_atual]
            velocidade = self.dados['velocidade'][self.indice_atual]
            aceleracao = self.dados['aceleracao'][self.indice_atual]
            tempo = self.dados['tempo'][self.indice_atual]
            empuxo = self.dados['empuxo'][self.indice_atual]
            
            # Cor do Céu (Gradiente com altitude)
            # Até 100km (100000m) transição para preto
            fator_ceu = min(altitude / 50000.0, 1.0)
            cor_fundo = interpolar_cor(AZUL_CEU, PRETO, fator_ceu)
            self.tela.fill(cor_fundo)
            
            # Desenhar Chão
            chao_y = ALTURA - 50 + altitude * self.escala_pixels_por_metro # O chão desce
            if chao_y < ALTURA:
                pygame.draw.rect(self.tela, (34, 139, 34), (0, chao_y, LARGURA, ALTURA - chao_y))
            
            # Desenhar Foguete (Fixo no centro verticalmente ou subindo até certo ponto?)
            # Vamos fixar o foguete no centro e mover o mundo, até o chão sumir
            pos_foguete_y = ALTURA - 150
            if altitude < 200: # Nos primeiros metros, o foguete sobe na tela
                pos_foguete_y = (ALTURA - 50) - altitude * self.escala_pixels_por_metro - 60
                # Ajustar escala para ver a decolagem
                self.escala_pixels_por_metro = 1.0
            else:
                # Depois fixamos e mudamos a escala para dar sensação de velocidade?
                # Simplesmente fixo no meio
                pos_foguete_y = ALTURA / 2
            
            self.desenhar_foguete(LARGURA // 2, pos_foguete_y, empuxo > 0)
            
            # Telemetria (HUD)
            texto_alt = self.font.render(f"Altitude: {altitude/1000:.2f} km", True, BRANCO)
            texto_vel = self.font.render(f"Velocidade: {velocidade:.0f} m/s", True, BRANCO)
            texto_acc = self.font.render(f"Aceleração: {aceleracao/9.81:.2f} G", True, BRANCO)
            texto_tempo = self.font_grande.render(f"T+ {tempo:.1f} s", True, BRANCO)
            
            self.tela.blit(texto_tempo, (10, 10))
            self.tela.blit(texto_alt, (10, 50))
            self.tela.blit(texto_vel, (10, 70))
            self.tela.blit(texto_acc, (10, 90))
            
            pygame.display.flip()
            
        pygame.quit()

if __name__ == "__main__":
    sim = Visualizador()
    sim.executar()
