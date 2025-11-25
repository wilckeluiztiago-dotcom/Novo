"""
Módulo de Visualização Avançada
Autor: Luiz Tiago Wilcke
Data: 2025-11-25

Visualizações interativas e 3D para o simulador de Quetiapina
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, Circle
import matplotlib.patches as mpatches
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo
plt.style.use('seaborn-v0_8-darkgrid')


class VisualizadorQuetiapina:
    """Classe para criar visualizações avançadas do simulador"""
    
    def __init__(self):
        self.cores = {
            'plasma': '#FF6B6B',
            'cerebro': '#4ECDC4',
            'tgi': '#FFE66D',
            'periferico': '#95E1D3',
            'receptor_d2': '#C44569',
            'receptor_5ht2a': '#5F27CD',
            'receptor_h1': '#00D2D3',
            'eficacia': '#2ECC71',
            'efeitos': '#E74C3C'
        }
    
    def plot_farmacocinetica_completa(self, tempo: np.ndarray, 
                                     concentracoes: np.ndarray,
                                     parametros_pk: Dict,
                                     salvar: str = None):
        """
        Plota painel completo de farmacocinética
        
        Args:
            tempo: Vetor de tempo (horas)
            concentracoes: Matrix de concentrações [tgi, plasma, cerebro, periferico]
            parametros_pk: Dicionário com parâmetros PK
            salvar: Caminho para salvar figura
        """
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Título principal
        fig.suptitle('FARMACOCINÉTICA DA QUETIAPINA - Modelo Compartimental ADME',
                    fontsize=18, fontweight='bold', y=0.98)
        
        # 1. Concentração plasmática
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(tempo, concentracoes[:, 1], linewidth=3, 
                color=self.cores['plasma'], label='Concentração Plasmática')
        ax1.fill_between(tempo, 0, concentracoes[:, 1], 
                        alpha=0.3, color=self.cores['plasma'])
        ax1.axhline(y=parametros_pk['Cmax_ng_mL'], color='red', 
                   linestyle='--', alpha=0.7, label=f"Cmax = {parametros_pk['Cmax_ng_mL']:.1f} ng/mL")
        ax1.axvline(x=parametros_pk['Tmax_horas'], color='red', 
                   linestyle='--', alpha=0.7, label=f"Tmax = {parametros_pk['Tmax_horas']:.1f} h")
        ax1.set_xlabel('Tempo (horas)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Concentração (ng/mL)', fontsize=12, fontweight='bold')
        ax1.set_title('Perfil de Concentração Plasmática', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, tempo[-1])
        
        # 2. Comparação de compartimentos
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(tempo, concentracoes[:, 1], linewidth=2.5, 
                color=self.cores['plasma'], label='Plasma')
        ax2.plot(tempo, concentracoes[:, 2], linewidth=2.5, 
                color=self.cores['cerebro'], label='Cérebro (alvo)')
        ax2.plot(tempo, concentracoes[:, 3], linewidth=2.5, 
                color=self.cores['periferico'], label='Tecidos Periféricos')
        ax2.set_xlabel('Tempo (horas)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Concentração (ng/mL ou ng/g)', fontsize=11, fontweight='bold')
        ax2.set_title('Distribuição nos Compartimentos', fontsize=13, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, tempo[-1])
        
        # 3. Absorção (TGI)
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(tempo, concentracoes[:, 0], linewidth=2.5, 
                color=self.cores['tgi'])
        ax3.fill_between(tempo, 0, concentracoes[:, 0], 
                        alpha=0.4, color=self.cores['tgi'])
        ax3.set_xlabel('Tempo (horas)', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Quantidade no TGI (mg)', fontsize=11, fontweight='bold')
        ax3.set_title('Absorção Gastrointestinal', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, tempo[-1])
        
        # 4. Tabela de parâmetros PK
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        
        # Criar tabela
        dados_tabela = [
            ['Cmax (ng/mL)', f"{parametros_pk['Cmax_ng_mL']:.2f}"],
            ['Tmax (horas)', f"{parametros_pk['Tmax_horas']:.2f}"],
            ['T½ (horas)', f"{parametros_pk['Tmeia_vida_horas']:.2f}"],
            ['AUC (ng·h/mL)', f"{parametros_pk['AUC_ng_h_mL']:.2f}"],
            ['CL (L/h)', f"{parametros_pk['Clearance_L_h']:.2f}"],
            ['Vd (L)', f"{parametros_pk['Volume_distribuicao_L']:.2f}"],
            ['Fração Livre (%)', f"{parametros_pk['Concentracao_livre_%']:.1f}"]
        ]
        
        tabela = ax4.table(cellText=dados_tabela,
                          colLabels=['Parâmetro', 'Valor'],
                          cellLoc='center',
                          loc='center',
                          colWidths=[0.3, 0.15])
        tabela.auto_set_font_size(False)
        tabela.set_fontsize(11)
        tabela.scale(1, 2.5)
        
        # Estilizar tabela
        for i in range(len(dados_tabela) + 1):
            if i == 0:
                for j in range(2):
                    tabela[(i, j)].set_facecolor('#34495E')
                    tabela[(i, j)].set_text_props(weight='bold', color='white')
            else:
                for j in range(2):
                    tabela[(i, j)].set_facecolor('#ECF0F1' if i % 2 == 0 else 'white')
        
        ax4.set_title('Parâmetros Farmacocinéticos', 
                     fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if salvar:
            plt.savefig(salvar, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_farmacodinamica(self, tempo: np.ndarray,
                            resultados_pd: Dict,
                            salvar: str = None):
        """
        Plota painel de farmacodinâmica
        
        Args:
            tempo: Vetor de tempo (horas)
            resultados_pd: Dicionário com ocupações e efeitos
            salvar: Caminho para salvar figura
        """
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)
        
        fig.suptitle('FARMACODINÂMICA DA QUETIAPINA - Ocupação de Receptores e Efeitos',
                    fontsize=18, fontweight='bold', y=0.98)
        
        # 1. Ocupação de receptores principais
        ax1 = fig.add_subplot(gs[0, :])
        ocupacoes = resultados_pd['ocupacoes']
        
        ax1.plot(tempo, ocupacoes['D2'], linewidth=3, 
                color=self.cores['receptor_d2'], label='Dopamina D2', marker='o', markersize=2)
        ax1.plot(tempo, ocupacoes['5-HT2A'], linewidth=3, 
                color=self.cores['receptor_5ht2a'], label='Serotonina 5-HT2A', marker='s', markersize=2)
        ax1.plot(tempo, ocupacoes['H1'], linewidth=3, 
                color=self.cores['receptor_h1'], label='Histamina H1', marker='^', markersize=2)
        
        # Zonas terapêuticas
        ax1.axhspan(60, 80, alpha=0.2, color='green', label='Zona D2 terapêutica')
        ax1.axhspan(80, 100, alpha=0.2, color='red', label='Zona D2 risco EPS')
        
        ax1.set_xlabel('Tempo (horas)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Ocupação do Receptor (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Ocupação dos Principais Receptores Cerebrais', 
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10, ncol=2)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, tempo[-1])
        ax1.set_ylim(0, 100)
        
        # 2. Todos os receptores
        ax2 = fig.add_subplot(gs[1, 0])
        for receptor, ocupacao in ocupacoes.items():
            ax2.plot(tempo, ocupacao, linewidth=2, label=receptor, alpha=0.8)
        ax2.set_xlabel('Tempo (horas)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Ocupação (%)', fontsize=11, fontweight='bold')
        ax2.set_title('Perfil Completo de Ocupação', fontsize=13, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, tempo[-1])
        
        # 3. Eficácia terapêutica
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(tempo, resultados_pd['eficacia'], linewidth=3, 
                color=self.cores['eficacia'])
        ax3.fill_between(tempo, 0, resultados_pd['eficacia'], 
                        alpha=0.3, color=self.cores['eficacia'])
        ax3.axhline(y=70, color='orange', linestyle='--', 
                   alpha=0.7, label='Limiar terapêutico')
        ax3.set_xlabel('Tempo (horas)', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Score de Eficácia (0-100)', fontsize=11, fontweight='bold')
        ax3.set_title('Eficácia Terapêutica ao Longo do Tempo', 
                     fontsize=13, fontweight='bold')
        ax3.legend(loc='lower right', fontsize=9)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, tempo[-1])
        ax3.set_ylim(0, 105)
        
        # 4. Efeitos colaterais
        ax4 = fig.add_subplot(gs[2, :])
        efeitos = resultados_pd['efeitos_colaterais']
        
        cores_efeitos = ['#E74C3C', '#F39C12', '#9B59B6', '#3498DB', '#1ABC9C', '#E67E22']
        for i, (efeito, valores) in enumerate(efeitos.items()):
            ax4.plot(tempo, valores, linewidth=2.5, 
                    color=cores_efeitos[i % len(cores_efeitos)], 
                    label=efeito.replace('_', ' '), alpha=0.8)
        
        ax4.set_xlabel('Tempo (horas)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Risco/Intensidade (%)', fontsize=12, fontweight='bold')
        ax4.set_title('Perfil de Efeitos Colaterais', fontsize=14, fontweight='bold')
        ax4.legend(loc='upper right', fontsize=10, ncol=3)
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(0, tempo[-1])
        ax4.set_ylim(0, 100)
        
        plt.tight_layout()
        
        if salvar:
            plt.savefig(salvar, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_diagrama_cerebro(self, ocupacoes: Dict[str, float], salvar: str = None):
        """
        Cria diagrama visual do cérebro com ocupação de receptores
        
        Args:
            ocupacoes: Dicionário com ocupações dos receptores (%)
            salvar: Caminho para salvar figura
        """
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Título
        ax.text(5, 9.5, 'MAPA DE OCUPAÇÃO DE RECEPTORES CEREBRAIS',
               ha='center', fontsize=16, fontweight='bold')
        
        # Desenhar "cérebro" (círculo central)
        cerebro = Circle((5, 5), 2.5, color='lightblue', alpha=0.3, ec='navy', linewidth=3)
        ax.add_patch(cerebro)
        ax.text(5, 5, 'CÉREBRO', ha='center', va='center', 
               fontsize=14, fontweight='bold', color='navy')
        
        # Posições dos receptores ao redor
        posicoes_receptores = {
            'D2': (2, 7.5, self.cores['receptor_d2']),
            '5-HT2A': (8, 7.5, self.cores['receptor_5ht2a']),
            'H1': (1, 5, self.cores['receptor_h1']),
            'α1': (9, 5, '#F39C12'),
            'M1': (2, 2.5, '#9B59B6'),
            '5-HT1A': (8, 2.5, '#1ABC9C')
        }
        
        for receptor, (x, y, cor) in posicoes_receptores.items():
            ocupacao = ocupacoes.get(receptor, 0)
            
            # Tamanho proporcional à ocupação
            raio = 0.3 + (ocupacao / 100) * 0.4
            
            # Círculo do receptor
            circ = Circle((x, y), raio, color=cor, alpha=0.7, ec='black', linewidth=2)
            ax.add_patch(circ)
            
            # Linha conectando ao cérebro
            ax.plot([x, 5], [y, 5], 'k--', alpha=0.3, linewidth=1)
            
            # Texto
            ax.text(x, y, f'{receptor}\n{ocupacao:.0f}%', 
                   ha='center', va='center', 
                   fontsize=9, fontweight='bold', color='white')
        
        # Legenda de cores
        y_legenda = 0.5
        ax.text(0.5, y_legenda, 'Tamanho e cor indicam nível de ocupação', 
               fontsize=10, style='italic')
        
        plt.tight_layout()
        
        if salvar:
            plt.savefig(salvar, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_doses_multiplas(self, tempo: np.ndarray, 
                            concentracoes: np.ndarray,
                            intervalo_dose: float,
                            num_doses: int,
                            salvar: str = None):
        """
        Plota regime de doses múltiplas
        
        Args:
            tempo: Vetor de tempo
            concentracoes: Concentrações ao longo do tempo
            intervalo_dose: Intervalo entre doses (horas)
            num_doses: Número de doses
            salvar: Caminho para salvar
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        fig.suptitle('REGIME DE DOSES MÚLTIPLAS - Estado de Equilíbrio',
                    fontsize=16, fontweight='bold')
        
        # Marcar momentos das doses
        for i in range(num_doses):
            tempo_dose = i * intervalo_dose
            ax1.axvline(x=tempo_dose, color='red', linestyle='--', alpha=0.5)
            ax2.axvline(x=tempo_dose, color='red', linestyle='--', alpha=0.5)
        
        # Plasma
        ax1.plot(tempo, concentracoes[:, 1], linewidth=2.5, color=self.cores['plasma'])
        ax1.fill_between(tempo, 0, concentracoes[:, 1], alpha=0.3, color=self.cores['plasma'])
        ax1.set_ylabel('Concentração Plasmática (ng/mL)', fontsize=11, fontweight='bold')
        ax1.set_title('Acúmulo no Plasma', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Cérebro
        ax2.plot(tempo, concentracoes[:, 2], linewidth=2.5, color=self.cores['cerebro'])
        ax2.fill_between(tempo, 0, concentracoes[:, 2], alpha=0.3, color=self.cores['cerebro'])
        ax2.set_xlabel('Tempo (horas)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Concentração Cerebral (ng/g)', fontsize=11, fontweight='bold')
        ax2.set_title('Acúmulo no Cérebro (Alvo Terapêutico)', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if salvar:
            plt.savefig(salvar, dpi=300, bbox_inches='tight')
        
        return fig


if __name__ == "__main__":
    print("Módulo de visualização carregado com sucesso!")
    print("Use as funções da classe VisualizadorQuetiapina para criar gráficos.")
