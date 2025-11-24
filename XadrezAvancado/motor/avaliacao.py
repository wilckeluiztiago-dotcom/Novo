"""
Função de Avaliação Estática
"""

from config import *
from motor.tabuleiro import Estado

class Avaliador:
    def avaliar(self, estado: Estado) -> int:
        pontuacao = 0
        
        # Material e Posição
        for linha in range(8):
            for coluna in range(8):
                peca = estado.tabuleiro[linha][coluna]
                if peca == VAZIO: continue
                
                # Material base
                pontuacao += VALOR_PECA[peca]
                
                # Tabelas de Posição (PST)
                tabela = TABELAS_POSICAO[peca]
                idx = linha * 8 + coluna
                if peca > 0:
                    pontuacao += tabela[idx]
                else:
                    # Espelha para as pretas
                    pontuacao -= tabela[(7 - linha) * 8 + coluna]

        # Mobilidade (simples)
        # Nota: Gerar movimentos é caro, usar apenas se necessário ou simplificado
        # Aqui mantemos simples para performance na busca
        
        # Bônus por Xeque (ajuda a encontrar mates)
        if estado.em_xeque(True):
            pontuacao -= 50
        if estado.em_xeque(False):
            pontuacao += 50

        return pontuacao
