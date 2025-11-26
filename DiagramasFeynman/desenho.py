"""
Módulo de Desenho para Diagramas de Feynman
Autor: Luiz Tiago Wilcke

Funções matemáticas para desenhar linhas onduladas e encaracoladas.
"""

import math
import numpy as np

def calcular_pontos_foton(x1, y1, x2, y2, amplitude=5, frequencia=0.2):
    """
    Calcula pontos para desenhar um fóton (senoide).
    Retorna lista de tuplas (x, y).
    """
    distancia = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    angulo = math.atan2(y2 - y1, x2 - x1)
    
    pontos = []
    passos = int(distancia)
    
    for t in range(0, passos + 1, 2): # Passo de 2 pixels
        # Coordenada ao longo da linha reta
        local_x = t
        # Deslocamento perpendicular (seno)
        local_y = amplitude * math.sin(t * frequencia)
        
        # Rotacionar para a direção correta
        rot_x = local_x * math.cos(angulo) - local_y * math.sin(angulo)
        rot_y = local_x * math.sin(angulo) + local_y * math.cos(angulo)
        
        pontos.append((x1 + rot_x, y1 + rot_y))
        
    return pontos

def calcular_pontos_gluon(x1, y1, x2, y2, raio=5, densidade=0.3):
    """
    Calcula pontos para desenhar um glúon (cicloide/mola).
    Retorna lista de tuplas (x, y).
    """
    distancia = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    angulo = math.atan2(y2 - y1, x2 - x1)
    
    pontos = []
    passos = int(distancia * 2) # Mais resolução
    
    # Ajustar densidade para caber loops inteiros
    num_loops = int(distancia * densidade / (2 * math.pi))
    if num_loops == 0: num_loops = 1
    k = (2 * math.pi * num_loops) / distancia
    
    for t in range(passos + 1):
        dist_atual = t * (distancia / passos)
        
        # Equação paramétrica de uma "mola" projetada 2D (cicloide modificada)
        local_x = dist_atual + raio * math.cos(dist_atual * k + math.pi)
        local_y = raio * math.sin(dist_atual * k + math.pi)
        
        # Rotacionar
        rot_x = local_x * math.cos(angulo) - local_y * math.sin(angulo)
        rot_y = local_x * math.sin(angulo) + local_y * math.cos(angulo)
        
        pontos.append((x1 + rot_x, y1 + rot_y))
        
    return pontos

def calcular_seta(x1, y1, x2, y2, tamanho=10):
    """Calcula vértices de uma seta no meio da linha."""
    meio_x = (x1 + x2) / 2
    meio_y = (y1 + y2) / 2
    angulo = math.atan2(y2 - y1, x2 - x1)
    
    # Pontos da seta
    p1_x = meio_x - tamanho * math.cos(angulo - math.pi/6)
    p1_y = meio_y - tamanho * math.sin(angulo - math.pi/6)
    
    p2_x = meio_x - tamanho * math.cos(angulo + math.pi/6)
    p2_y = meio_y - tamanho * math.sin(angulo + math.pi/6)
    
    return [(meio_x, meio_y), (p1_x, p1_y), (p2_x, p2_y)]
