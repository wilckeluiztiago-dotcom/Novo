"""
Sistema de Visualização para Modelos de Desemprego

Gera gráficos de alta qualidade para análise de resultados dos modelos SDE.

Autor: Luiz Tiago Wilcke
Data: 2025-11-24
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Tuple
import pandas as pd
from matplotlib.gridspec import GridSpec

# Configuração de estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10


class Visualizador:
    """
    Classe para visualização de resultados de modelos de desemprego.
    
    Gera diversos tipos de gráficos profissionais para análise.
    """
    
    def __init__(self, estilo: str = 'seaborn-v0_8-darkgrid', dpi: int = 150):
        """
        Inicializa o visualizador.
        
        Args:
            estilo: Estilo do matplotlib
            dpi: Resolução dos gráficos
        """
        self.dpi = dpi
        try:
            plt.style.use(estilo)
        except:
            # Fallback se o estilo não estiver disponível
            sns.set_style("darkgrid")
    
    def plotar_trajetorias(
        self,
        tempos: np.ndarray,
        trajetorias: np.ndarray,
        titulo: str = "Trajetórias de Desemprego",
        xlabel: str = "Tempo (anos)",
        ylabel: str = "Taxa de Desemprego",
        max_trajetorias: int = 50,
        mostrar_estatisticas: bool = True,
        salvar: Optional[str] = None
    ) -> plt.Figure:
        """
        Plota múltiplas trajetórias de desemprego.
        
        Args:
            tempos: Array de tempos
            trajetorias: Array (num_traj, N+1) de trajetórias
            titulo: Título do gráfico
            xlabel: Rótulo eixo x
            ylabel: Rótulo eixo y
            max_trajetorias: Máximo de trajetórias individuais a plotar
            mostrar_estatisticas: Se True, mostra média e intervalos de confiança
            salvar: Caminho para salvar a figura (opcional)
            
        Returns:
            Objeto Figure
        """
        fig, ax = plt.subplots(figsize=(14, 8), dpi=self.dpi)
        
        num_traj = trajetorias.shape[0]
        
        # Plota trajetórias individuais (amostra se houver muitas)
        indices_plot = np.random.choice(
            num_traj, 
            min(max_trajetorias, num_traj), 
            replace=False
        )
        
        for i in indices_plot:
            ax.plot(tempos, trajetorias[i], alpha=0.1, color='steelblue', linewidth=0.8)
        
        # Estatísticas do ensemble
        if mostrar_estatisticas:
            media = np.mean(trajetorias, axis=0)
            q05 = np.percentile(trajetorias, 5, axis=0)
            q25 = np.percentile(trajetorias, 25, axis=0)
            q75 = np.percentile(trajetorias, 75, axis=0)
            q95 = np.percentile(trajetorias, 95, axis=0)
            
            # Plota média
            ax.plot(tempos, media, color='darkred', linewidth=2.5, 
                   label='Média', zorder=10)
            
            # Intervalos de confiança
            ax.fill_between(tempos, q05, q95, alpha=0.2, color='orange',
                           label='IC 90% (5%-95%)')
            ax.fill_between(tempos, q25, q75, alpha=0.3, color='orange',
                           label='IC 50% (25%-75%)')
        
        ax.set_xlabel(xlabel, fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(titulo, fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if salvar:
            plt.savefig(salvar, dpi=self.dpi, bbox_inches='tight')
            print(f"Gráfico salvo em: {salvar}")
        
        return fig
    
    def plotar_distribuicao(
        self,
        dados: np.ndarray,
        titulo: str = "Distribuição da Taxa de Desemprego",
        xlabel: str = "Taxa de Desemprego",
        bins: int = 50,
        ajustar_kde: bool = True,
        salvar: Optional[str] = None
    ) -> plt.Figure:
        """
        Plota histograma e densidade de probabilidade.
        
        Args:
            dados: Array com dados a plotar
            titulo: Título
            xlabel: Rótulo eixo x
            bins: Número de bins do histograma
            ajustar_kde: Se True, ajusta e plota KDE
            salvar: Caminho para salvar
            
        Returns:
            Objeto Figure
        """
        fig, ax = plt.subplots(figsize=(12, 7), dpi=self.dpi)
        
        # Histograma
        n, bins_edges, patches = ax.hist(
            dados, bins=bins, density=True, alpha=0.6, 
            color='steelblue', edgecolor='black', linewidth=0.5
        )
        
        # KDE
        if ajustar_kde:
            from scipy import stats
            kde = stats.gaussian_kde(dados)
            x_kde = np.linspace(dados.min(), dados.max(), 300)
            y_kde = kde(x_kde)
            ax.plot(x_kde, y_kde, 'r-', linewidth=2.5, label='Densidade (KDE)')
        
        # Estatísticas
        media = np.mean(dados)
        mediana = np.median(dados)
        desvio = np.std(dados)
        
        ax.axvline(media, color='darkred', linestyle='--', linewidth=2, 
                  label=f'Média: {media:.4f}')
        ax.axvline(mediana, color='darkgreen', linestyle='--', linewidth=2,
                  label=f'Mediana: {mediana:.4f}')
        
        # Texto com estatísticas
        texto_stats = f'μ = {media:.4f}\nσ = {desvio:.4f}\nN = {len(dados)}'
        ax.text(0.98, 0.97, texto_stats, transform=ax.transAxes,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               fontsize=11)
        
        ax.set_xlabel(xlabel, fontweight='bold')
        ax.set_ylabel('Densidade de Probabilidade', fontweight='bold')
        ax.set_title(titulo, fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if salvar:
            plt.savefig(salvar, dpi=self.dpi, bbox_inches='tight')
            print(f"Gráfico salvo em: {salvar}")
        
        return fig
    
    def plotar_diagrama_fase(
        self,
        trajetoria: np.ndarray,
        labels: List[str],
        titulo: str = "Diagrama de Fase",
        salvar: Optional[str] = None
    ) -> plt.Figure:
        """
        Plota diagrama de fase (espaço de estados).
        
        Args:
            trajetoria: Array (N, dim) com trajetória no espaço de estados
            labels: Lista com nomes das variáveis
            titulo: Título
            salvar: Caminho para salvar
            
        Returns:
            Objeto Figure
        """
        dim = trajetoria.shape[1]
        
        if dim == 2:
            # Caso 2D - plotagem simples
            fig, ax = plt.subplots(figsize=(10, 10), dpi=self.dpi)
            
            # Trajetória
            ax.plot(trajetoria[:, 0], trajetoria[:, 1], 'b-', alpha=0.6, linewidth=1.5)
            
            # Ponto inicial
            ax.plot(trajetoria[0, 0], trajetoria[0, 1], 'go', markersize=12, 
                   label='Início', zorder=10)
            
            # Ponto final
            ax.plot(trajetoria[-1, 0], trajetoria[-1, 1], 'ro', markersize=12,
                   label='Fim', zorder=10)
            
            # Setas para indicar direção
            step = max(1, len(trajetoria) // 20)
            for i in range(0, len(trajetoria) - step, step):
                dx = trajetoria[i + step, 0] - trajetoria[i, 0]
                dy = trajetoria[i + step, 1] - trajetoria[i, 1]
                ax.arrow(trajetoria[i, 0], trajetoria[i, 1], dx, dy,
                        head_width=0.01, head_length=0.01, fc='black', 
                        ec='black', alpha=0.3, linewidth=0.5)
            
            ax.set_xlabel(labels[0], fontweight='bold', fontsize=12)
            ax.set_ylabel(labels[1], fontweight='bold', fontsize=12)
            ax.set_title(titulo, fontsize=16, fontweight='bold', pad=20)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        elif dim == 3:
            # Caso 3D
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(12, 10), dpi=self.dpi)
            ax = fig.add_subplot(111, projection='3d')
            
            # Trajetória
            ax.plot(trajetoria[:, 0], trajetoria[:, 1], trajetoria[:, 2],
                   'b-', alpha=0.6, linewidth=1.5)
            
            # Pontos inicial e final
            ax.scatter(trajetoria[0, 0], trajetoria[0, 1], trajetoria[0, 2],
                      c='green', s=100, marker='o', label='Início')
            ax.scatter(trajetoria[-1, 0], trajetoria[-1, 1], trajetoria[-1, 2],
                      c='red', s=100, marker='o', label='Fim')
            
            ax.set_xlabel(labels[0], fontweight='bold')
            ax.set_ylabel(labels[1], fontweight='bold')
            ax.set_zlabel(labels[2], fontweight='bold')
            ax.set_title(titulo, fontsize=16, fontweight='bold', pad=20)
            ax.legend()
            
        else:
            # Caso multidimensional - matriz de gráficos de dispersão
            fig, axes = plt.subplots(dim, dim, figsize=(15, 15), dpi=self.dpi)
            
            for i in range(dim):
                for j in range(dim):
                    if i == j:
                        # Diagonal: histogramas
                        axes[i, j].hist(trajetoria[:, i], bins=30, 
                                       color='steelblue', alpha=0.7)
                        axes[i, j].set_ylabel('Frequência')
                    else:
                        # Fora da diagonal: diagramas de fase 2D
                        axes[i, j].plot(trajetoria[:, j], trajetoria[:, i],
                                       'b-', alpha=0.5, linewidth=0.8)
                        axes[i, j].plot(trajetoria[0, j], trajetoria[0, i],
                                       'go', markersize=6)
                        axes[i, j].plot(trajetoria[-1, j], trajetoria[-1, i],
                                       'ro', markersize=6)
                    
                    if i == dim - 1:
                        axes[i, j].set_xlabel(labels[j])
                    if j == 0:
                        axes[i, j].set_ylabel(labels[i])
            
            fig.suptitle(titulo, fontsize=18, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        
        if salvar:
            plt.savefig(salvar, dpi=self.dpi, bbox_inches='tight')
            print(f"Gráfico salvo em: {salvar}")
        
        return fig
    
    def plotar_convergencia(
        self,
        N_valores: np.ndarray,
        erros: np.ndarray,
        taxa_teorica: Optional[float] = None,
        titulo: str = "Análise de Convergência",
        salvar: Optional[str] = None
    ) -> plt.Figure:
        """
        Plota análise de convergência do método numérico.
        
        Args:
            N_valores: Valores de N testados
            erros: Erros correspondentes
            taxa_teorica: Taxa de convergência teórica (opcional)
            titulo: Título
            salvar: Caminho para salvar
            
        Returns:
            Objeto Figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=self.dpi)
        
        # Gráfico 1: Escala normal
        ax1.plot(N_valores, erros, 'bo-', linewidth=2, markersize=8, label='Erro observado')
        ax1.set_xlabel('Número de Passos (N)', fontweight='bold')
        ax1.set_ylabel('Erro', fontweight='bold')
        ax1.set_title('Convergência (Escala Linear)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Gráfico 2: Escala log-log
        ax2.loglog(N_valores, erros, 'bo-', linewidth=2, markersize=8, label='Erro observado')
        
        # Linha de referência
        if taxa_teorica:
            dt_ref = 1.0 / N_valores
            erro_ref = erros[0] * (dt_ref / dt_ref[0])**taxa_teorica
            ax2.loglog(N_valores, erro_ref, 'r--', linewidth=2,
                      label=f'Taxa teórica: {taxa_teorica:.2f}')
        
        # Estima taxa observada
        from simulador import AnaliseConvergencia
        taxa_obs, _ = AnaliseConvergencia.calcular_taxa_convergencia(N_valores, erros)
        
        ax2.text(0.05, 0.95, f'Taxa observada: {taxa_obs:.3f}',
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=12)
        
        ax2.set_xlabel('Número de Passos (N)', fontweight='bold')
        ax2.set_ylabel('Erro', fontweight='bold')
        ax2.set_title('Convergência (Escala Log-Log)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, which='both')
        ax2.legend()
        
        fig.suptitle(titulo, fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        if salvar:
            plt.savefig(salvar, dpi=self.dpi, bbox_inches='tight')
            print(f"Gráfico salvo em: {salvar}")
        
        return fig
    
    def plotar_comparacao_modelos(
        self,
        dados_modelos: Dict[str, pd.DataFrame],
        titulo: str = "Comparação de Modelos",
        salvar: Optional[str] = None
    ) -> plt.Figure:
        """
        Plota comparação entre diferentes modelos.
        
        Args:
            dados_modelos: Dicionário {nome_modelo: DataFrame}
            titulo: Título
            salvar: Caminho para salvar
            
        Returns:
            Objeto Figure
        """
        num_modelos = len(dados_modelos)
        
        fig = plt.figure(figsize=(16, 10), dpi=self.dpi)
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Subplot 1: Todas as trajetórias
        ax1 = fig.add_subplot(gs[0, :])
        cores = plt.cm.tab10(np.linspace(0, 1, num_modelos))
        
        for (nome, df), cor in zip(dados_modelos.items(), cores):
            ax1.plot(df['tempo'], df['desemprego'], label=nome.capitalize(),
                    linewidth=2, color=cor, alpha=0.8)
        
        ax1.set_xlabel('Tempo', fontweight='bold')
        ax1.set_ylabel('Taxa de Desemprego', fontweight='bold')
        ax1.set_title('Trajetórias de Desemprego por Modelo', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Distribuições
        ax2 = fig.add_subplot(gs[1, 0])
        dados_dist = [df['desemprego'].values for df in dados_modelos.values()]
        labels_dist = [nome.capitalize() for nome in dados_modelos.keys()]
        
        bp = ax2.boxplot(dados_dist, labels=labels_dist, patch_artist=True)
        for patch, cor in zip(bp['boxes'], cores):
            patch.set_facecolor(cor)
            patch.set_alpha(0.7)
        
        ax2.set_ylabel('Taxa de Desemprego', fontweight='bold')
        ax2.set_title('Distribuições Comparativas', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Subplot 3: Estatísticas
        ax3 = fig.add_subplot(gs[1, 1])
        
        stats_data = []
        for nome, df in dados_modelos.items():
            stats_data.append({
                'Modelo': nome.capitalize(),
                'Média': df['desemprego'].mean(),
                'Desvio': df['desemprego'].std(),
                'Mín': df['desemprego'].min(),
                'Máx': df['desemprego'].max()
            })
        
        stats_df = pd.DataFrame(stats_data)
        
        # Tabela
        ax3.axis('tight')
        ax3.axis('off')
        table = ax3.table(cellText=stats_df.values,
                         colLabels=stats_df.columns,
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Estiliza header
        for i in range(len(stats_df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Estiliza linhas
        for i in range(1, len(stats_df) + 1):
            for j in range(len(stats_df.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        ax3.set_title('Estatísticas Descritivas', fontsize=14, fontweight='bold', pad=20)
        
        fig.suptitle(titulo, fontsize=16, fontweight='bold', y=0.98)
        
        if salvar:
            plt.savefig(salvar, dpi=self.dpi, bbox_inches='tight')
            print(f"Gráfico salvo em: {salvar}")
        
        return fig
    
    def plotar_mapa_calor_correlacao(
        self,
        df: pd.DataFrame,
        titulo: str = "Matriz de Correlação",
        salvar: Optional[str] = None
    ) -> plt.Figure:
        """
        Plota mapa de calor de correlações.
        
        Args:
            df: DataFrame com variáveis
            titulo: Título
            salvar: Caminho para salvar
            
        Returns:
            Objeto Figure
        """
        # Seleciona apenas colunas numéricas
        df_numerico = df.select_dtypes(include=[np.number])
        
        # Calcula correlações
        corr = df_numerico.corr()
        
        # Cria figura
        fig, ax = plt.subplots(figsize=(12, 10), dpi=self.dpi)
        
        # Mapa de calor
        im = ax.imshow(corr, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        
        # Ticks
        ax.set_xticks(np.arange(len(corr.columns)))
        ax.set_yticks(np.arange(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha='right')
        ax.set_yticklabels(corr.columns)
        
        # Valores nas células
        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                text = ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                             ha='center', va='center', 
                             color='white' if abs(corr.iloc[i, j]) > 0.5 else 'black',
                             fontsize=9)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlação', rotation=270, labelpad=20, fontweight='bold')
        
        ax.set_title(titulo, fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if salvar:
            plt.savefig(salvar, dpi=self.dpi, bbox_inches='tight')
            print(f"Gráfico salvo em: {salvar}")
        
        return fig
