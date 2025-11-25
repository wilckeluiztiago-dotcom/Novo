"""
Visualizador de genoma circular usando Pygame.
Mostra contigs, ORFs e genes em um mapa circular estilo plasmídeo.
"""

import pygame
import math
import os

class VisualizadorGenoma:
    """
    Cria visualização circular do genoma bacteriano com anotações de genes.
    """
    
    def __init__(self, largura=1000, altura=1000):
        self.largura = largura
        self.altura = altura
        self.centro_x = largura // 2
        self.centro_y = altura // 2
        self.raio_genoma = min(largura, altura) // 3
        
        # Cores
        self.COR_FUNDO = (20, 20, 30)
        self.COR_GENOMA = (70, 70, 90)
        self.COR_GENE_POSITIVO = (138, 180, 250)  # Azul
        self.COR_GENE_NEGATIVO = (166, 227, 161)  # Verde
        self.COR_ORF = (249, 226, 175)  # Amarelo claro
        self.COR_TEXTO = (205, 214, 244)
        self.COR_DESTAQUE = (250, 179, 135)  # Laranja
        
    def criar_visualizacao(self, contigs, orfs_por_contig, arquivo_saida="genoma_circular.png"):
        """
        Cria visualização circular do genoma e salva como imagem.
        """
        pygame.init()
        tela = pygame.Surface((self.largura, self.altura))
        tela.fill(self.COR_FUNDO)
        
        # Calcula tamanho total do genoma
        tamanho_total = sum(len(c) for c in contigs)
        
        if tamanho_total == 0:
            print("Erro: Genoma vazio")
            return
        
        # Desenha círculo do genoma
        pygame.draw.circle(tela, self.COR_GENOMA, (self.centro_x, self.centro_y), 
                          self.raio_genoma, 3)
        
        # Desenha contigs
        posicao_atual = 0
        for i, contig in enumerate(contigs):
            tamanho_contig = len(contig)
            angulo_inicio = (posicao_atual / tamanho_total) * 360
            angulo_fim = ((posicao_atual + tamanho_contig) / tamanho_total) * 360
            
            # Desenha arco para o contig
            self._desenhar_arco(tela, self.raio_genoma + 10, angulo_inicio, angulo_fim, 
                               (100, 100, 120), 5)
            
            # Desenha ORFs deste contig
            if i < len(orfs_por_contig):
                for orf in orfs_por_contig[i]:
                    pos_orf_inicio = posicao_atual + orf['inicio']
                    pos_orf_fim = posicao_atual + orf['fim']
                    
                    angulo_orf_inicio = (pos_orf_inicio / tamanho_total) * 360
                    angulo_orf_fim = (pos_orf_fim / tamanho_total) * 360
                    
                    # Cor baseada na fita
                    cor = self.COR_GENE_POSITIVO if orf['fita'] == '+' else self.COR_GENE_NEGATIVO
                    raio = self.raio_genoma + 30 if orf['fita'] == '+' else self.raio_genoma - 30
                    
                    # Desenha gene
                    self._desenhar_gene(tela, raio, angulo_orf_inicio, angulo_orf_fim, cor)
            
            posicao_atual += tamanho_contig
        
        # Adiciona título e informações
        self._adicionar_texto(tela, "GENOMA BACTERIANO CIRCULAR", 
                             self.centro_x, 50, 32, self.COR_TEXTO, centralizado=True)
        
        self._adicionar_texto(tela, f"Tamanho: {tamanho_total:,} bp", 
                             self.centro_x, 90, 20, self.COR_TEXTO, centralizado=True)
        
        # Legenda
        y_legenda = self.altura - 150
        self._desenhar_legenda(tela, y_legenda)
        
        # Adiciona marcadores de posição (0°, 90°, 180°, 270°)
        self._adicionar_marcadores_posicao(tela, tamanho_total)
        
        # Salva imagem
        pygame.image.save(tela, arquivo_saida)
        pygame.quit()
        
        print(f"Visualização salva em: {arquivo_saida}")
        return arquivo_saida
    
    def _desenhar_arco(self, tela, raio, angulo_inicio, angulo_fim, cor, espessura):
        """Desenha um arco no círculo."""
        # Converte ângulos para radianos (Pygame usa graus, mas começando do leste)
        # Ajusta para começar do topo (norte)
        ang_i = math.radians(angulo_inicio - 90)
        ang_f = math.radians(angulo_fim - 90)
        
        # Desenha linha do arco
        pontos = []
        num_pontos = max(2, int(abs(angulo_fim - angulo_inicio)))
        for i in range(num_pontos + 1):
            angulo = ang_i + (ang_f - ang_i) * (i / num_pontos)
            x = self.centro_x + raio * math.cos(angulo)
            y = self.centro_y + raio * math.sin(angulo)
            pontos.append((x, y))
        
        if len(pontos) > 1:
            pygame.draw.lines(tela, cor, False, pontos, espessura)
    
    def _desenhar_gene(self, tela, raio, angulo_inicio, angulo_fim, cor):
        """Desenha um gene como uma barra radial."""
        ang_i = math.radians(angulo_inicio - 90)
        ang_f = math.radians(angulo_fim - 90)
        ang_meio = (ang_i + ang_f) / 2
        
        # Pontos do retângulo radial
        raio_interno = raio - 15
        raio_externo = raio + 15
        
        pontos = []
        # Arco externo
        for i in range(5):
            angulo = ang_i + (ang_f - ang_i) * (i / 4)
            x = self.centro_x + raio_externo * math.cos(angulo)
            y = self.centro_y + raio_externo * math.sin(angulo)
            pontos.append((x, y))
        
        # Arco interno (reverso)
        for i in range(5):
            angulo = ang_f - (ang_f - ang_i) * (i / 4)
            x = self.centro_x + raio_interno * math.cos(angulo)
            y = self.centro_y + raio_interno * math.sin(angulo)
            pontos.append((x, y))
        
        pygame.draw.polygon(tela, cor, pontos)
        pygame.draw.polygon(tela, (255, 255, 255), pontos, 1)  # Borda branca
    
    def _adicionar_texto(self, tela, texto, x, y, tamanho, cor, centralizado=False):
        """Adiciona texto na tela."""
        try:
            fonte = pygame.font.Font(None, tamanho)
            superficie_texto = fonte.render(texto, True, cor)
            if centralizado:
                rect = superficie_texto.get_rect(center=(x, y))
                tela.blit(superficie_texto, rect)
            else:
                tela.blit(superficie_texto, (x, y))
        except:
            pass  # Se falhar, não mostra texto
    
    def _desenhar_legenda(self, tela, y_inicio):
        """Desenha legenda explicativa."""
        self._adicionar_texto(tela, "LEGENDA:", 50, y_inicio, 24, self.COR_TEXTO)
        
        # Fita positiva
        pygame.draw.rect(tela, self.COR_GENE_POSITIVO, (50, y_inicio + 30, 30, 15))
        self._adicionar_texto(tela, "Genes fita + (sentido horário)", 90, y_inicio + 30, 20, self.COR_TEXTO)
        
        # Fita negativa
        pygame.draw.rect(tela, self.COR_GENE_NEGATIVO, (50, y_inicio + 55, 30, 15))
        self._adicionar_texto(tela, "Genes fita - (sentido anti-horário)", 90, y_inicio + 55, 20, self.COR_TEXTO)
        
        # Contigs
        pygame.draw.rect(tela, (100, 100, 120), (50, y_inicio + 80, 30, 15))
        self._adicionar_texto(tela, "Contigs montados", 90, y_inicio + 80, 20, self.COR_TEXTO)
    
    def _adicionar_marcadores_posicao(self, tela, tamanho_total):
        """Adiciona marcadores de posição ao redor do círculo."""
        posicoes = [0, 90, 180, 270]
        labels = ["0 bp", f"{tamanho_total//4:,} bp", f"{tamanho_total//2:,} bp", f"{3*tamanho_total//4:,} bp"]
        
        for angulo, label in zip(posicoes, labels):
            ang_rad = math.radians(angulo - 90)
            x = self.centro_x + (self.raio_genoma + 60) * math.cos(ang_rad)
            y = self.centro_y + (self.raio_genoma + 60) * math.sin(ang_rad)
            
            # Desenha linha marcadora
            x_linha = self.centro_x + self.raio_genoma * math.cos(ang_rad)
            y_linha = self.centro_y + self.raio_genoma * math.sin(ang_rad)
            pygame.draw.line(tela, self.COR_TEXTO, (x_linha, y_linha), (x, y), 2)
            
            # Adiciona texto
            self._adicionar_texto(tela, label, int(x) - 30, int(y) - 10, 18, self.COR_TEXTO)
