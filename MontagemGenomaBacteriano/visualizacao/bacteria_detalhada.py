"""
Visualizador avançado de bactéria com formato realista e DNA interno.
Mostra forma da bactéria, DNA, genes e informações detalhadas.
"""

import pygame
import math
import os

class VisualizadorBacteriaAvancado:
    """
    Visualização realista de bactéria com DNA interno e informações.
    """
    
    def __init__(self, largura=1600, altura=1200):
        self.largura = largura
        self.altura = altura
        
        # Cores
        self.COR_FUNDO = (240, 248, 255)  # Alice blue
        self.COR_MEMBRANA = (100, 149, 237)  # Cornflower blue
        self.COR_CITOPLASMA = (176, 224, 230)  # Powder blue
        self.COR_DNA = (220, 20, 60)  # Crimson
        self.COR_GENE_POSITIVO = (50, 205, 50)  # Lime green
        self.COR_GENE_NEGATIVO = (255, 140, 0)  # Dark orange
        self.COR_PAREDE_GRAM_POS = (138, 43, 226)  # Blue violet
        self.COR_PAREDE_GRAM_NEG = (255, 20, 147)  # Deep pink
        self.COR_TEXTO = (25, 25, 112)  # Midnight blue
        self.COR_DESTAQUE = (255, 215, 0)  # Gold
        
    def criar_visualizacao(self, bacteria_info, contigs, orfs_por_contig, 
                          arquivo_saida="bacteria_detalhada.png"):
        """
        Cria visualização detalhada da bactéria.
        
        Args:
            bacteria_info: Dicionário com informações da bactéria
            contigs: Lista de contigs
            orfs_por_contig: Lista de ORFs por contig
            arquivo_saida: Nome do arquivo de saída
        """
        pygame.init()
        tela = pygame.Surface((self.largura, self.altura))
        tela.fill(self.COR_FUNDO)
        
        # Desenhar bactéria com DNA - layout horizontal
        # Bactéria à esquerda, painel à direita
        bacteria_x = 400  # Centralizado na metade esquerda
        bacteria_y = 550  # Mais abaixo para caber tudo
        
        forma = bacteria_info.get('forma', 'bacilo')
        gram = bacteria_info.get('gram', 'negativa')
        
        if forma == 'bacilo':
            self._desenhar_bacilo(tela, bacteria_x, bacteria_y, gram, contigs, orfs_por_contig)
        elif forma == 'coco':
            self._desenhar_coco(tela, bacteria_x, bacteria_y, gram, contigs, orfs_por_contig)
        elif forma == 'espiral':
            self._desenhar_espiral(tela, bacteria_x, bacteria_y, gram, contigs, orfs_por_contig)
        elif forma == 'cocobacilo':
            self._desenhar_cocobacilo(tela, bacteria_x, bacteria_y, gram, contigs, orfs_por_contig)
        elif forma == 'diplococo':
            self._desenhar_diplococo(tela, bacteria_x, bacteria_y, gram, contigs, orfs_por_contig)
        elif forma == 'bacilo curvo':
            self._desenhar_bacilo_curvo(tela, bacteria_x, bacteria_y, gram, contigs, orfs_por_contig)
        elif forma == 'bacilo fusiforme':
            self._desenhar_bacilo_fusiforme(tela, bacteria_x, bacteria_y, gram, contigs, orfs_por_contig)
        elif forma == 'bacilo filamentoso':
            self._desenhar_bacilo_filamentoso(tela, bacteria_x, bacteria_y, gram, contigs, orfs_por_contig)
        elif forma == 'coco em cadeia':
            self._desenhar_coco_cadeia(tela, bacteria_x, bacteria_y, gram, contigs, orfs_por_contig)
        else:
            self._desenhar_bacilo(tela, bacteria_x, bacteria_y, gram, contigs, orfs_por_contig)
        
        # Desenhar painel de informações
        self._desenhar_painel_info(tela, bacteria_info, contigs, orfs_por_contig)
        
        # Título
        self._desenhar_titulo(tela, bacteria_info)
        
        # Legenda na parte inferior
        self._desenhar_legenda(tela)
        
        # Salvar
        pygame.image.save(tela, arquivo_saida)
        pygame.quit()
        
        print(f"✅ Visualização detalhada salva em: {arquivo_saida}")
        return arquivo_saida
    
    def _desenhar_bacilo(self, tela, x, y, gram, contigs, orfs):
        """Desenha bactéria em forma de bacilo (bastão)."""
        comprimento = 400
        largura = 150
        self._desenhar_bacilo_base(tela, x, y, comprimento, largura, gram, contigs, orfs)
        
        # Flagelo (opcional)
        self._desenhar_flagelo(tela, x + comprimento//2, y)
        
        # Labels
        self._adicionar_label(tela, "Parede Celular", x - comprimento//2 - 100, y - largura//2 - 30, 11)
        self._adicionar_label(tela, "DNA Cromossômico", x, y - 70, 11)
        self._adicionar_label(tela, "Citoplasma", x, y + 70, 11)
        self._adicionar_label(tela, "Flagelo", x + comprimento//2 + 50, y, 10)
    
    def _desenhar_coco(self, tela, x, y, gram, contigs, orfs):
        """Desenha bactéria em forma de coco (esférica)."""
        raio = 130
        self._desenhar_coco_base(tela, x, y, raio, gram, contigs, orfs)
        self._adicionar_label(tela, "DNA Circular", x, y - raio - 30, 10)
    
    
    def _desenhar_cocobacilo(self, tela, x, y, gram, contigs, orfs):
        """Desenha cocobacilo (oval)."""
        # Bacilo curto e arredondado
        comprimento = 250
        largura = 180
        
        # Reutilizar lógica do bacilo com dimensões diferentes
        self._desenhar_bacilo_base(tela, x, y, comprimento, largura, gram, contigs, orfs)
        self._adicionar_label(tela, "Cocobacilo", x, y - 100, 12)

    def _desenhar_diplococo(self, tela, x, y, gram, contigs, orfs):
        """Desenha diplococo (dois cocos)."""
        raio = 100
        offset = 80
        
        # Coco 1
        self._desenhar_coco_base(tela, x - offset, y, raio, gram, contigs, orfs, desenhar_labels=False)
        # Coco 2
        self._desenhar_coco_base(tela, x + offset, y, raio, gram, contigs, orfs, desenhar_labels=False)
        
        self._adicionar_label(tela, "Diplococo", x, y - 150, 12)
        self._adicionar_label(tela, "DNA", x - offset, y - 40, 10)

    def _desenhar_bacilo_curvo(self, tela, x, y, gram, contigs, orfs):
        """Desenha bacilo curvo (vibrio)."""
        # Similar ao espiral mas apenas uma curva
        pontos = []
        raio_curva = 200
        for i in range(100):
            angulo = math.pi/4 + (i/100) * math.pi/2
            px = x + raio_curva * math.cos(angulo) - 100
            py = y + raio_curva * math.sin(angulo) - 150
            pontos.append((px, py))
            
        self._desenhar_forma_customizada(tela, pontos, 120, gram, contigs, orfs)
        self._adicionar_label(tela, "Bacilo Curvo", x, y - 100, 12)

    def _desenhar_bacilo_fusiforme(self, tela, x, y, gram, contigs, orfs):
        """Desenha bacilo fusiforme (pontas finas)."""
        # Desenhar losango alongado
        pts = [
            (x - 250, y),
            (x, y - 60),
            (x + 250, y),
            (x, y + 60)
        ]
        
        cor_parede = self.COR_PAREDE_GRAM_POS if gram == 'positiva' else self.COR_PAREDE_GRAM_NEG
        pygame.draw.polygon(tela, cor_parede, pts)
        
        # Membrana (menor)
        pts_mem = [
            (x - 240, y),
            (x, y - 50),
            (x + 240, y),
            (x, y + 50)
        ]
        pygame.draw.polygon(tela, self.COR_MEMBRANA, pts_mem)
        pygame.draw.polygon(tela, self.COR_CITOPLASMA, pts_mem) # Preenchimento
        
        # DNA linear
        pygame.draw.line(tela, self.COR_DNA, (x-150, y), (x+150, y), 4)
        
        self._adicionar_label(tela, "Bacilo Fusiforme", x, y - 80, 12)

    def _desenhar_bacilo_filamentoso(self, tela, x, y, gram, contigs, orfs):
        """Desenha bacilo filamentoso (longo)."""
        comprimento = 600
        largura = 80
        self._desenhar_bacilo_base(tela, x, y, comprimento, largura, gram, contigs, orfs)
        self._adicionar_label(tela, "Filamentoso", x, y - 60, 12)

    def _desenhar_coco_cadeia(self, tela, x, y, gram, contigs, orfs):
        """Desenha estreptococos (cadeia)."""
        raio = 60
        num_cocos = 5
        start_x = x - (num_cocos-1)*raio
        
        for i in range(num_cocos):
            cx = start_x + i * (raio*2 - 20)
            # Ondulação na cadeia
            cy = y + 20 * math.sin(i)
            self._desenhar_coco_base(tela, cx, cy, raio, gram, contigs, orfs, desenhar_labels=False)
            
        self._adicionar_label(tela, "Estreptococos", x, y - 100, 12)

    def _desenhar_bacilo_base(self, tela, x, y, comprimento, largura, gram, contigs, orfs):
        """Base para desenhar bacilos."""
        # Parede celular (Gram)
        espessura_parede = 8 if gram == 'positiva' else 4
        cor_parede = self.COR_PAREDE_GRAM_POS if gram == 'positiva' else self.COR_PAREDE_GRAM_NEG
        
        # Desenhar parede
        pygame.draw.rect(tela, cor_parede, 
                        (x - comprimento//2 - espessura_parede, 
                         y - largura//2 - espessura_parede,
                         comprimento + 2*espessura_parede, 
                         largura + 2*espessura_parede), 
                        border_radius=largura//2)
        
        # Membrana celular
        pygame.draw.rect(tela, self.COR_MEMBRANA, 
                        (x - comprimento//2, y - largura//2, comprimento, largura), 
                        border_radius=largura//2)
        
        # Citoplasma
        pygame.draw.rect(tela, self.COR_CITOPLASMA, 
                        (x - comprimento//2 + 5, y - largura//2 + 5, 
                         comprimento - 10, largura - 10), 
                        border_radius=largura//2)
        
        # DNA em espiral
        self._desenhar_dna_espiral(tela, x, y, comprimento - 40, largura - 40)
        
        # Genes (ORFs)
        self._desenhar_genes_bacilo(tela, x, y, comprimento, largura, orfs)

    def _desenhar_coco_base(self, tela, x, y, raio, gram, contigs, orfs, desenhar_labels=True):
        """Base para desenhar cocos."""
        # Parede celular
        espessura_parede = 8 if gram == 'positiva' else 4
        cor_parede = self.COR_PAREDE_GRAM_POS if gram == 'positiva' else self.COR_PAREDE_GRAM_NEG
        
        pygame.draw.circle(tela, cor_parede, (int(x), int(y)), raio + espessura_parede)
        pygame.draw.circle(tela, self.COR_MEMBRANA, (int(x), int(y)), raio)
        pygame.draw.circle(tela, self.COR_CITOPLASMA, (int(x), int(y)), raio - 5)
        
        # DNA circular
        self._desenhar_dna_circular(tela, x, y, raio - 30)
        
        # Genes
        self._desenhar_genes_circular(tela, x, y, raio - 20, orfs)

    def _desenhar_forma_customizada(self, tela, pontos, espessura, gram, contigs, orfs):
        """Desenha forma baseada em pontos centrais."""
        cor_parede = self.COR_PAREDE_GRAM_POS if gram == 'positiva' else self.COR_PAREDE_GRAM_NEG
        
        # Desenhar linhas grossas para simular o corpo
        if len(pontos) > 1:
            pygame.draw.lines(tela, cor_parede, False, pontos, espessura + 10)
            pygame.draw.lines(tela, self.COR_MEMBRANA, False, pontos, espessura)
            # DNA interno
            pygame.draw.lines(tela, self.COR_DNA, False, pontos, 4)

    def _desenhar_espiral(self, tela, x, y, gram, contigs, orfs):
        """Desenha bactéria em forma de espiral."""
        # Desenhar espiral
        pontos = []
        num_voltas = 3
        raio_max = 80
        
        for i in range(200):
            angulo = (i / 200) * num_voltas * 2 * math.pi
            raio = raio_max * (i / 200)
            px = x + raio * math.cos(angulo)
            py = y + raio * math.sin(angulo) * 0.3  # Achatado
            pontos.append((px, py))
        
        self._desenhar_forma_customizada(tela, pontos, 40, gram, contigs, orfs)
        self._adicionar_label(tela, "Espiral", x, y - 100, 12)
    
    def _desenhar_dna_espiral(self, tela, x, y, comp, larg):
        """Desenha DNA em dupla hélice."""
        num_pontos = 80  # Mais pontos para melhor resolução
        
        for i in range(num_pontos):
            t = i / num_pontos
            px = x - comp//2 + comp * t
            
            # Fita 1
            py1 = y + 20 * math.sin(t * 10 * math.pi)  # Mais ondulações
            # Fita 2
            py2 = y - 20 * math.sin(t * 10 * math.pi)
            
            pygame.draw.circle(tela, self.COR_DNA, (int(px), int(py1)), 4)  # Maior
            pygame.draw.circle(tela, self.COR_DNA, (int(px), int(py2)), 4)
            
            # Conexões entre fitas
            if i % 2 == 0:
                pygame.draw.line(tela, self.COR_DNA, (int(px), int(py1)), 
                               (int(px), int(py2)), 2)
    
    def _desenhar_dna_circular(self, tela, x, y, raio):
        """Desenha DNA circular (plasmídeo)."""
        num_pontos = 60
        
        for i in range(num_pontos):
            angulo = (i / num_pontos) * 2 * math.pi
            px = x + raio * math.cos(angulo)
            py = y + raio * math.sin(angulo)
            
            pygame.draw.circle(tela, self.COR_DNA, (int(px), int(py)), 4)
            
            # Fita dupla
            px2 = x + (raio - 10) * math.cos(angulo)
            py2 = y + (raio - 10) * math.sin(angulo)
            pygame.draw.circle(tela, self.COR_DNA, (int(px2), int(py2)), 3)
    
    def _desenhar_genes_bacilo(self, tela, x, y, comp, larg, orfs):
        """Desenha genes ao longo do bacilo."""
        if not orfs or not orfs[0]:
            return
        
        for i, orf in enumerate(orfs[0][:10]):  # Primeiros 10 genes
            pos_relativa = orf['inicio'] / max(orf['fim'], 1)
            px = x - comp//2 + comp * pos_relativa
            py = y + (20 if orf['fita'] == '+' else -20)
            
            cor = self.COR_GENE_POSITIVO if orf['fita'] == '+' else self.COR_GENE_NEGATIVO
            pygame.draw.circle(tela, cor, (int(px), int(py)), 5)
    
    def _desenhar_genes_circular(self, tela, x, y, raio, orfs):
        """Desenha genes ao redor do círculo."""
        if not orfs or not orfs[0]:
            return
        
        for i, orf in enumerate(orfs[0][:15]):
            angulo = (i / 15) * 2 * math.pi
            px = x + raio * math.cos(angulo)
            py = y + raio * math.sin(angulo)
            
            cor = self.COR_GENE_POSITIVO if orf['fita'] == '+' else self.COR_GENE_NEGATIVO
            pygame.draw.circle(tela, cor, (int(px), int(py)), 6)
    
    def _desenhar_flagelo(self, tela, x, y):
        """Desenha flagelo bacteriano."""
        pontos = []
        for i in range(30):
            px = x + i * 5
            py = y + 15 * math.sin(i * 0.5)
            pontos.append((px, py))
        
        if len(pontos) > 1:
            pygame.draw.lines(tela, (100, 100, 100), False, pontos, 2)
    
    def _desenhar_painel_info(self, tela, bacteria_info, contigs, orfs):
        """Desenha painel com informações detalhadas."""
        # Painel à direita, começando mais acima
        painel_x = 850
        painel_y = 100
        painel_w = 700
        painel_h = 1000
        
        # Fundo do painel
        pygame.draw.rect(tela, (255, 255, 255), (painel_x, painel_y, painel_w, painel_h))
        pygame.draw.rect(tela, self.COR_TEXTO, (painel_x, painel_y, painel_w, painel_h), 3)
        
        # Título do painel
        self._adicionar_texto(tela, "INFORMAÇÕES BACTERIANAS", painel_x + 20, painel_y + 15, 
                             18, self.COR_TEXTO, bold=True)
        
        y_offset = painel_y + 60
        line_height = 30  # Mais espaçamento
        
        # Informações
        infos = [
            ("Nome:", bacteria_info.get('nome', 'Desconhecida')),
            ("", ""),
            ("CLASSIFICAÇÃO:", ""),
            (f"Gram:", bacteria_info.get('gram', 'N/A').capitalize()),
            (f"Forma:", bacteria_info.get('forma', 'N/A').capitalize()),
            ("", ""),
            ("CARACTERÍSTICAS:", ""),
            (f"Tamanho genoma:", f"{bacteria_info.get('tamanho_genoma', 0):,} bp"),
            (f"Conteúdo GC:", f"{bacteria_info.get('gc', 0):.1f}%"),
            (f"Contigs:", str(len(contigs))),
            (f"ORFs detectados:", str(sum(len(o) for o in orfs if o))),
            ("", ""),
            ("PATOGENICIDADE:", ""),
            (bacteria_info.get('patogenicidade', 'Desconhecida'), ""),
            ("", ""),
            ("APLICAÇÕES:", ""),
        ]
        
        for label, valor in infos:
            if not label and not valor:
                y_offset += 10
                continue
            
            if label.isupper() and not valor:
                # Cabeçalho de seção
                self._adicionar_texto(tela, label, painel_x + 20, y_offset, 14, 
                                     self.COR_DESTAQUE, bold=True)
            elif valor:
                # Par label-valor
                self._adicionar_texto(tela, label, painel_x + 25, y_offset, 12, self.COR_TEXTO)
                self._adicionar_texto(tela, valor, painel_x + 200, y_offset, 12, 
                                     (0, 100, 0), bold=True)
            else:
                # Texto simples
                self._adicionar_texto(tela, label, painel_x + 25, y_offset, 11, self.COR_TEXTO)
            
            y_offset += line_height
        
        # Aplicações (texto longo)
        aplicacoes = bacteria_info.get('aplicacoes', 'N/A')
        self._adicionar_texto_wrap(tela, aplicacoes, painel_x + 25, y_offset, 
                                   painel_w - 50, 11, self.COR_TEXTO)
    
    def _desenhar_titulo(self, tela, bacteria_info):
        """Desenha título principal."""
        nome = bacteria_info.get('nome', 'Bactéria Desconhecida')
        self._adicionar_texto(tela, f"🦠 {nome}", self.largura // 2, 30, 
                             28, self.COR_TEXTO, centralizado=True, bold=True)
    
    def _desenhar_legenda(self, tela):
        """Desenha legenda na parte inferior."""
        legenda_y = self.altura - 100
        legenda_x = 50
        
        self._adicionar_texto(tela, "LEGENDA:", legenda_x, legenda_y, 14, self.COR_TEXTO, bold=True)
        
        # Genes
        pygame.draw.circle(tela, self.COR_GENE_POSITIVO, (legenda_x + 10, legenda_y + 30), 7)
        self._adicionar_texto(tela, "Genes fita +", legenda_x + 30, legenda_y + 25, 12, self.COR_TEXTO)
        
        pygame.draw.circle(tela, self.COR_GENE_NEGATIVO, (legenda_x + 10, legenda_y + 55), 7)
        self._adicionar_texto(tela, "Genes fita -", legenda_x + 30, legenda_y + 50, 12, self.COR_TEXTO)
        
        # DNA
        pygame.draw.line(tela, self.COR_DNA, (legenda_x + 200, legenda_y + 30), (legenda_x + 240, legenda_y + 30), 5)
        self._adicionar_texto(tela, "DNA", legenda_x + 250, legenda_y + 25, 12, self.COR_TEXTO)
        
        # Gram
        pygame.draw.rect(tela, self.COR_PAREDE_GRAM_POS, (legenda_x + 350, legenda_y + 25, 30, 15))
        self._adicionar_texto(tela, "Gram +", legenda_x + 390, legenda_y + 25, 12, self.COR_TEXTO)
        
        pygame.draw.rect(tela, self.COR_PAREDE_GRAM_NEG, (legenda_x + 350, legenda_y + 50, 30, 15))
        self._adicionar_texto(tela, "Gram -", legenda_x + 390, legenda_y + 50, 12, self.COR_TEXTO)
    
    def _adicionar_texto(self, tela, texto, x, y, tamanho, cor, centralizado=False, bold=False):
        """Adiciona texto na tela."""
        try:
            fonte = pygame.font.Font(None, tamanho)
            if bold:
                fonte.set_bold(True)
            superficie_texto = fonte.render(texto, True, cor)
            if centralizado:
                rect = superficie_texto.get_rect(center=(x, y))
                tela.blit(superficie_texto, rect)
            else:
                tela.blit(superficie_texto, (x, y))
        except:
            pass
    
    def _adicionar_texto_wrap(self, tela, texto, x, y, max_width, tamanho, cor):
        """Adiciona texto com quebra de linha."""
        palavras = texto.split()
        linha = ""
        y_offset = 0
        
        for palavra in palavras:
            teste = linha + palavra + " "
            if len(teste) * tamanho * 0.6 > max_width:
                self._adicionar_texto(tela, linha, x, y + y_offset, tamanho, cor)
                linha = palavra + " "
                y_offset += tamanho + 5
            else:
                linha = teste
        
        if linha:
            self._adicionar_texto(tela, linha, x, y + y_offset, tamanho, cor)
    
    def _adicionar_label(self, tela, texto, x, y, tamanho):
        """Adiciona label com seta."""
        self._adicionar_texto(tela, texto, x, y, tamanho, self.COR_TEXTO)
