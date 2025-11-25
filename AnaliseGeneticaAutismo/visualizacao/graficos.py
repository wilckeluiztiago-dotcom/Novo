# visualizacao/graficos.py
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import networkx as nx
from configuracao import PALETA_CORES

class VisualizadorGenetico:
    """
    Gera visualizações avançadas e interativas para dados genéticos.
    """
    
    @staticmethod
    def criar_manhattan_plot(df_gwas: pd.DataFrame) -> go.Figure:
        """
        Cria um Manhattan Plot para visualizar SNPs significativos.
        Eixo X: SNPs (agrupados por cromossomo - simulado aqui como índice)
        Eixo Y: -log10(P-valor)
        """
        df_gwas['ind'] = range(len(df_gwas))
        
        fig = px.scatter(
            df_gwas, 
            x='ind', 
            y='Log10_P',
            color='Log10_P',
            hover_data=['SNP', 'P_Valor', 'Odds_Ratio'],
            title='Manhattan Plot: Associação Genética (GWAS Simulado)',
            labels={'ind': 'SNPs (Índice Genômico)', 'Log10_P': '-log10(P-valor)'},
            color_continuous_scale='Viridis'
        )
        
        # Linha de significância (p < 0.05 corrigido, ex: 5e-8 para GWAS real, aqui simplificado)
        fig.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="red", annotation_text="Significância (p=0.05)")
        
        fig.update_layout(
            template='plotly_dark',
            plot_bgcolor=PALETA_CORES['fundo'],
            paper_bgcolor=PALETA_CORES['fundo'],
            font_color=PALETA_CORES['texto']
        )
        return fig

    @staticmethod
    def criar_heatmap_expressao(df_expressao: pd.DataFrame, df_fenotipos: pd.DataFrame) -> go.Figure:
        """
        Cria um mapa de calor da expressão gênica, comparando Casos vs Controles.
        """
        # Calcular média de expressão por grupo
        df_expressao['Grupo'] = df_fenotipos['Grupo']
        media_expressao = df_expressao.groupby('Grupo').mean().T
        
        fig = px.imshow(
            media_expressao,
            labels=dict(x="Grupo", y="Gene", color="Expressão Média"),
            x=['Caso', 'Controle'],
            y=media_expressao.index,
            title='Expressão Gênica Diferencial: Casos vs Controles',
            color_continuous_scale='RdBu_r'
        )
        
        fig.update_layout(
            template='plotly_dark',
            plot_bgcolor=PALETA_CORES['fundo'],
            paper_bgcolor=PALETA_CORES['fundo'],
            font_color=PALETA_CORES['texto']
        )
        return fig

    @staticmethod
    def criar_rede_interacao(genes: list) -> go.Figure:
        """
        Cria um grafo de rede simulando interações proteína-proteína (PPI).
        """
        G = nx.random_geometric_graph(len(genes), 0.5)
        
        # Atribuir nomes aos nós
        mapping = {i: gene for i, gene in enumerate(genes)}
        G = nx.relabel_nodes(G, mapping)
        
        # Posições dos nós
        pos = nx.spring_layout(G)
        
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_x = []
        node_y = []
        node_text = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="top center",
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                reversescale=True,
                color=[],
                size=20,
                colorbar=dict(
                    thickness=15,
                    title=dict(text='Conectividade', side='right'),
                    xanchor='left'
                ),
                line_width=2))
        
        # Colorir por grau de conexão
        node_adjacencies = []
        for node, adjacencies in enumerate(G.adjacency()):
            node_adjacencies.append(len(adjacencies[1]))
        node_trace.marker.color = node_adjacencies

        fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title=dict(
                    text='Rede de Interação Gênica (PPI Simulado)',
                    font=dict(size=16)
                ),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                template='plotly_dark',
                plot_bgcolor=PALETA_CORES['fundo'],
                paper_bgcolor=PALETA_CORES['fundo'],
                font_color=PALETA_CORES['texto'],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
        return fig

    @staticmethod
    def criar_pca_3d(df_pca: pd.DataFrame, df_fenotipos: pd.DataFrame) -> go.Figure:
        """
        Cria um gráfico de dispersão 3D dos componentes principais (PCA).
        Permite visualizar a estratificação populacional e separação de casos/controles.
        """
        df_plot = df_pca.copy()
        df_plot['Grupo'] = df_fenotipos['Grupo']
        df_plot['PRS'] = df_fenotipos['PRS']
        
        fig = px.scatter_3d(
            df_plot, 
            x='PC1', 
            y='PC2', 
            z='PC3',
            color='Grupo',
            symbol='Grupo',
            size_max=10,
            opacity=0.7,
            color_discrete_map={'Caso': PALETA_CORES['caso'], 'Controle': PALETA_CORES['controle']},
            title='Estratificação Populacional 3D (PCA)',
            hover_data=['PRS']
        )
        
        fig.update_layout(
            template='plotly_dark',
            scene=dict(
                xaxis_title='PC1 (Variância Maior)',
                yaxis_title='PC2',
                zaxis_title='PC3',
                bgcolor=PALETA_CORES['fundo']
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            paper_bgcolor=PALETA_CORES['fundo'],
            font_color=PALETA_CORES['texto']
        )
        return fig
