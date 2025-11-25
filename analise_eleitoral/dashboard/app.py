"""
Dashboard de Análise Eleitoral - Deputados Federais e Estaduais
Sistema completo de análise estatística para eleições brasileiras
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Adicionar diretório pai ao path
sys.path.append(str(Path(__file__).parent.parent))

from utils.dados import (
    gerar_dados_eleitorais, gerar_dados_historicos,
    gerar_series_temporais_partido, gerar_matriz_transferencia_votos
)
from modelos.basicos import ModeloRegressao, ModeloARIMA, ModeloPCA
from modelos.avancados import ModeloRandomForest, ModeloGradientBoosting
from modelos.bayesianos import ModeloDirichlet
from modelos.eleitorais import QuocienteEleitoral, NumeroEfetivoPartidos, IndiceNacionalizacao
from analises.coligacoes import AnalisadorColigacoes
from analises.volatilidade import CalculadorVolatilidade
from analises.fragmentacao import AnalisadorFragmentacao
from analises.competitividade import IndiceCompetitividade

# ==================== CONFIGURAÇÃO DA PÁGINA ====================
st.set_page_config(
    page_title="Análise Eleitoral Brasil",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== ESTILO CSS MODERNO (FUNDO AZUL) ====================
st.markdown("""
<style>
    /* Fundo azul moderno */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
    }
    
    /* Cards e containers */
    .stMarkdown, .stDataFrame, .stPlotlyChart {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Métricas */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
        color: #1e3c72;
    }
    
    /* Títulos */
    h1, h2, h3 {
        color: white;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    /* Botões */
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
    }
    
    /* Selectbox e inputs */
    .stSelectbox, .stMultiSelect, .stSlider {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 5px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: white;
        font-weight: bold;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# ==================== FUNÇÕES AUXILIARES ====================

@st.cache_data
def carregar_dados():
    """Carrega dados eleitorais simulados."""
    return gerar_dados_eleitorais(n_candidatos=500, ano=2026, tipo='federal')

@st.cache_data
def carregar_dados_historicos():
    """Carrega dados históricos."""
    return gerar_dados_historicos(anos=[2010, 2014, 2018, 2022])

def criar_grafico_barras(df, x, y, title, color_seq=None):
    """Cria gráfico de barras estilizado."""
    if color_seq is None:
        color_seq = px.colors.sequential.Blues_r
    
    fig = px.bar(df, x=x, y=y, title=title, color=y,
                 color_continuous_scale=color_seq)
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#1e3c72', size=12),
        title_font=dict(size=18, color='#1e3c72', family='Arial Black')
    )
    return fig

def criar_grafico_linha(df, x, y, title, labels=None):
    """Cria gráfico de linha estilizado."""
    fig = px.line(df, x=x, y=y, title=title, labels=labels, markers=True)
    fig.update_traces(line=dict(width=3, color='#667eea'))
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#1e3c72', size=12),
        title_font=dict(size=18, color='#1e3c72', family='Arial Black')
    )
    return fig

# ==================== SIDEBAR ====================

with st.sidebar:
    st.image("https://via.placeholder.com/300x100/1e3c72/ffffff?text=Análise+Eleitoral", use_container_width=True)
    
    st.title("⚙️ Configurações")
    
    ano_eleicao = st.selectbox(
        "Ano da Eleição",
        [2026, 2022, 2018, 2014, 2010],
        index=0
    )
    
    tipo_eleicao = st.selectbox(
        "Tipo de Eleição",
        ["Deputado Federal", "Deputado Estadual"],
        index=0
    )
    
    estado_selecionado = st.selectbox(
        "Estado",
        ['Todos', 'SP', 'MG', 'RJ', 'BA', 'PR', 'RS', 'PE', 'CE', 'PA', 'SC'],
        index=0
    )
    
    st.markdown("---")
    
    st.subheader("📊 Modelos Disponíveis")
    st.markdown("""
    - ✅ Regressão Linear/Logística
    - ✅ ARIMA/SARIMA
    - ✅ Random Forest
    - ✅ Gradient Boosting
    - ✅ LSTM
    - ✅ Bayesiano Hierárquico
    - ✅ Dirichlet-Multinomial
    - ✅ Quociente Eleitoral
    - ✅ Análise de Coligações
    - ✅ Volatilidade (Pedersen)
    - ✅ Fragmentação (NEP)
    - ✅ Competitividade
    """)

# ==================== HEADER ====================

st.title("🗳️ Sistema Avançado de Análise Eleitoral")
st.markdown(f"""
### Eleições para {tipo_eleicao} - {ano_eleicao}
Sistema completo de análise estatística com modelos avançados de ciência de dados e análise eleitoral.
""")

# ==================== ABAS PRINCIPAIS ====================

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Visão Geral",
    "🤖 Modelos Preditivos",
    "🤝 Coligações",
    "📈 Volatilidade",
    "🔀 Fragmentação",
    "⚔️ Competitividade",
    "🎯 Simulador"
])

# ==================== TAB 1: VISÃO GERAL ====================

with tab1:
    st.header("📊 Visão Geral das Eleições")
    
    # Carregar dados
    dados = carregar_dados()
    
    if estado_selecionado != 'Todos':
        dados = dados[dados['estado'] == estado_selecionado]
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total de Candidatos", f"{len(dados):,}")
    
    with col2:
        st.metric("Total de Votos", f"{dados['votos'].sum():,}")
    
    with col3:
        n_eleitos = dados['eleito'].sum()
        st.metric("Candidatos Eleitos", f"{int(n_eleitos):,}")
    
    with col4:
        n_partidos = dados['partido'].nunique()
        st.metric("Partidos Concorrendo", n_partidos)
    
    st.markdown("---")
    
    # Gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Votos por Partido")
        votos_partido = dados.groupby('partido')['votos'].sum().sort_values(ascending=False).head(10)
        fig = criar_grafico_barras(
            votos_partido.reset_index(),
            'partido', 'votos',
            'Top 10 Partidos por Votos'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Distribuição de Cadeiras")
        cadeiras_partido = dados[dados['eleito'] == 1].groupby('partido').size().sort_values(ascending=False).head(10)
        fig = criar_grafico_barras(
            cadeiras_partido.reset_index(),
            'partido', 0,
            'Top 10 Partidos por Cadeiras'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Análise NEP
    st.subheader("📊 Número Efetivo de Partidos (NEP)")
    
    nep_calc = NumeroEfetivoPartidos()
    votos_por_partido = dados.groupby('partido')['votos'].sum()
    cadeiras_por_partido = dados[dados['eleito'] == 1].groupby('partido').size()
    
    indices = nep_calc.calcular_indices_fragmentacao(votos_por_partido, cadeiras_por_partido)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("NEP (Votos)", f"{indices['NEP_votos']:.2f}")
    
    with col2:
        st.metric("NEP (Cadeiras)", f"{indices['NEP_cadeiras']:.2f}")
    
    with col3:
        st.metric("Índice de Gallagher", f"{indices['Indice_Gallagher']:.2f}")

# ==================== TAB 2: MODELOS PREDITIVOS ====================

with tab2:
    st.header("🤖 Modelos Preditivos")
    
    modelo_selecionado = st.selectbox(
        "Selecione o Modelo",
        ["Regressão Linear", "Random Forest", "Gradient Boosting", "Bayesiano (Dirichlet)"]
    )
    
    if st.button("🚀 Executar Modelo"):
        with st.spinner("Treinando modelo..."):
            dados = carregar_dados()
            
            if modelo_selecionado == "Regressão Linear":
                # Preparar features
                X = dados[['gasto_campanha', 'tempo_tv_segundos', 'incumbente', 
                          'idade', 'escolaridade', 'em_coligacao']].values
                y = dados['votos'].values
                
                # Treinar modelo
                modelo = ModeloRegressao()
                modelo.treinar(X, y, features_nomes=['Gastos', 'Tempo TV', 'Incumbente', 
                                                      'Idade', 'Escolaridade', 'Coligação'])
                
                # Resultados
                st.success("✅ Modelo treinado com sucesso!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Coeficientes do Modelo")
                    coefs = modelo.obter_coeficientes()
                    st.dataframe(coefs, use_container_width=True)
                
                with col2:
                    st.subheader("Importância das Variáveis")
                    fig = px.bar(coefs, x='importancia_abs', y='feature', orientation='h',
                                title='Importância Absoluta dos Coeficientes')
                    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)
                
                r2 = modelo.obter_r2(X, y)
                st.metric("R² do Modelo", f"{r2:.4f}")
            
            elif modelo_selecionado in ["Random Forest", "Gradient Boosting"]:
                X = dados[['gasto_campanha', 'tempo_tv_segundos', 'incumbente', 
                          'idade', 'escolaridade', 'em_coligacao']].values
                y = dados['votos'].values
                
                if modelo_selecionado == "Random Forest":
                    modelo = ModeloRandomForest(n_arvores=100, tipo='regressao')
                else:
                    modelo = ModeloGradientBoosting(n_estimadores=100, tipo='regressao')
                
                modelo.treinar(X, y, features_nomes=['Gastos', 'Tempo TV', 'Incumbente', 
                                                      'Idade', 'Escolaridade', 'Coligação'])
                
                st.success("✅ Modelo treinado com sucesso!")
                
                st.subheader("Importância das Features")
                importancias = modelo.obter_importancia_features()
                fig = px.bar(importancias, x='importancia', y='feature', orientation='h',
                            title=f'Importância das Features - {modelo_selecionado}')
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
                
                score = modelo.obter_score(X, y)
                st.metric("Score do Modelo", f"{score:.4f}")
            
            elif modelo_selecionado == "Bayesiano (Dirichlet)":
                votos_por_partido = dados.groupby('partido')['votos'].sum()
                
                modelo = ModeloDirichlet()
                modelo.treinar(votos_por_partido)
                
                st.success("✅ Modelo Bayesiano treinado!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Proporções Esperadas")
                    props = modelo.obter_proporcoes_esperadas().sort_values(ascending=False).head(10)
                    st.dataframe((props * 100).round(2), use_container_width=True)
                
                with col2:
                    st.subheader("Intervalos de Credibilidade (95%)")
                    intervalos = modelo.obter_intervalos_credibilidade()
                    st.dataframe(intervalos.head(10), use_container_width=True)
                
                # Probabilidade de vitória
                st.subheader("Probabilidade de Maior Votação")
                probs = modelo.probabilidade_vitoria(n_simulacoes=5000)
                fig = px.bar(probs.head(10), title='Probabilidade de Maior Votação (Top 10)')
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 3: COLIGAÇÕES ====================

with tab3:
    st.header("🤝 Análise de Coligações")
    
    dados = carregar_dados()
    
    # Definir coligações exemplo
    coligacoes_exemplo = {
        'Coligação Esquerda': ['PT', 'PSB', 'PCdoB', 'PSOL'],
        'Coligação Centro': ['MDB', 'PSDB', 'CIDADANIA'],
        'Coligação Direita': ['PL', 'PP', 'REPUBLICANOS', 'UNIÃO']
    }
    
    votos_por_partido = dados.groupby('partido')['votos'].sum().to_dict()
    cadeiras_por_partido = dados[dados['eleito'] == 1].groupby('partido').size().to_dict()
    
    analisador = AnalisadorColigacoes()
    
    st.subheader("Eficiência das Coligações")
    
    resultados = analisador.analisar_todas_coligacoes(
        votos_por_partido, cadeiras_por_partido, coligacoes_exemplo
    )
    
    st.dataframe(resultados, use_container_width=True)
    
    # Gráfico de eficiência
    fig = px.bar(resultados, x='nome_coligacao', y='eficiencia',
                 title='Eficiência das Coligações',
                 color='eficiencia',
                 color_continuous_scale='Blues')
    fig.add_hline(y=1.0, line_dash="dash", line_color="red", 
                  annotation_text="Eficiência Neutra")
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)
    
    # Análise de sobras
    st.subheader("Análise de Sobras Eleitorais")
    
    n_cadeiras = dados['eleito'].sum()
    sobras = analisador.analisar_sobras_eleitorais(votos_por_partido, int(n_cadeiras), coligacoes_exemplo)
    
    st.dataframe(sobras.head(15), use_container_width=True)

# ==================== TAB 4: VOLATILIDADE ====================

with tab4:
    st.header("📈 Análise de Volatilidade Eleitoral")
    
    calc_vol = CalculadorVolatilidade()
    
    # Gerar dados de duas eleições
    dados_2022 = gerar_dados_eleitorais(n_candidatos=400, ano=2022)
    dados_2026 = gerar_dados_eleitorais(n_candidatos=500, ano=2026)
    
    votos_2022 = dados_2022.groupby('partido')['votos'].sum()
    votos_2026 = dados_2026.groupby('partido')['votos'].sum()
    
    # Índice de Pedersen
    vol_pedersen = calc_vol.calcular_pedersen(votos_2022, votos_2026)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Índice de Pedersen", f"{vol_pedersen:.2f}")
    
    with col2:
        classificacao = calc_vol.classificar_estabilidade(vol_pedersen)
        st.metric("Classificação", classificacao)
    
    with col3:
        st.metric("Período", "2022-2026")
    
    # Partidos voláteis
    st.subheader("Partidos com Maior Mudança")
    
    volateis = calc_vol.identificar_partidos_volateis(votos_2022, votos_2026, threshold=3)
    
    if len(volateis) > 0:
        fig = px.bar(volateis.head(10), x='partido', y='mudanca',
                     title='Mudança de Votos por Partido (pontos percentuais)',
                     color='tipo',
                     color_discrete_map={'crescimento': '#00cc00', 'declínio': '#cc0000'})
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(volateis, use_container_width=True)
    else:
        st.info("Nenhum partido com mudança significativa detectada.")
    
    # Série temporal
    st.subheader("Evolução da Volatilidade")
    
    serie_votos = {
        2010: gerar_dados_eleitorais(n_candidatos=300, ano=2010).groupby('partido')['votos'].sum().to_dict(),
        2014: gerar_dados_eleitorais(n_candidatos=350, ano=2014).groupby('partido')['votos'].sum().to_dict(),
        2018: gerar_dados_eleitorais(n_candidatos=400, ano=2018).groupby('partido')['votos'].sum().to_dict(),
        2022: votos_2022.to_dict(),
        2026: votos_2026.to_dict()
    }
    
    evolucao = calc_vol.analisar_serie_temporal(serie_votos)
    
    fig = criar_grafico_linha(evolucao, 'periodo', 'volatilidade',
                              'Evolução da Volatilidade Eleitoral')
    st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 5: FRAGMENTAÇÃO ====================

with tab5:
    st.header("🔀 Análise de Fragmentação Partidária")
    
    dados = carregar_dados()
    analisador_frag = AnalisadorFragmentacao()
    
    votos_por_partido = dados.groupby('partido')['votos'].sum()
    cadeiras_por_partido = dados[dados['eleito'] == 1].groupby('partido').size()
    
    # Análise completa
    analise = analisador_frag.analisar_fragmentacao_completa(votos_por_partido, cadeiras_por_partido)
    
    # Métricas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("NEP (Votos)", f"{analise['NEP_votos']:.2f}")
        st.caption(analisador_frag.classificar_fragmentacao(analise['NEP_votos']))
    
    with col2:
        st.metric("NEP (Cadeiras)", f"{analise['NEP_cadeiras']:.2f}")
    
    with col3:
        st.metric("HHI (Votos)", f"{analise['HHI_votos']:.0f}")
    
    with col4:
        st.metric("Partidos Relevantes", f"{analise['n_partidos_relevantes']}")
    
    # Concentração
    st.subheader("Concentração de Votos")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Top 2 Partidos", f"{analise['concentracao_top2']:.1f}%")
    
    with col2:
        st.metric("Top 3 Partidos", f"{analise['concentracao_top3']:.1f}%")
    
    with col3:
        st.metric("Top 5 Partidos", f"{analise['concentracao_top5']:.1f}%")
    
    # Distribuição de tamanhos
    st.subheader("Distribuição de Tamanhos dos Partidos")
    
    dist = analisador_frag.analisar_distribuicao_tamanhos(votos_por_partido)
    
    categorias = pd.DataFrame({
        'Categoria': ['Grandes (≥10%)', 'Médios (5-10%)', 'Pequenos (1-5%)', 'Nanicos (<1%)'],
        'Quantidade': [dist['n_partidos_grandes'], dist['n_partidos_medios'], 
                      dist['n_partidos_pequenos'], dist['n_partidos_nanicos']],
        'Votos (%)': [dist['votos_grandes'], dist['votos_medios'], 
                     dist['votos_pequenos'], dist['votos_nanicos']]
    })
    
    fig = px.bar(categorias, x='Categoria', y='Quantidade',
                 title='Distribuição dos Partidos por Tamanho',
                 color='Votos (%)',
                 color_continuous_scale='Blues')
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 6: COMPETITIVIDADE ====================

with tab6:
    st.header("⚔️ Análise de Competitividade")
    
    dados = carregar_dados()
    indice_comp = IndiceCompetitividade()
    
    # Análise por estado
    st.subheader("Competitividade por Estado")
    
    estados = dados['estado'].unique()
    resultados_estados = {}
    
    for estado in estados[:10]:  # Top 10 estados
        votos_estado = dados[dados['estado'] == estado].groupby('partido')['votos'].sum()
        resultados_estados[estado] = votos_estado.to_dict()
    
    distritos_comp = indice_comp.identificar_distritos_competitivos(resultados_estados, threshold_margem=15)
    
    st.dataframe(distritos_comp, use_container_width=True)
    
    # Gráfico de competitividade
    fig = px.bar(distritos_comp.head(10), x='distrito', y='margem_percentual',
                 title='Margem de Vitória por Estado (Top 10)',
                 color='competitiva',
                 color_discrete_map={True: '#00cc00', False: '#cc0000'})
    fig.add_hline(y=10, line_dash="dash", line_color="orange", 
                  annotation_text="Threshold Competitivo")
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)
    
    # Taxa de renovação
    st.subheader("Taxa de Renovação Parlamentar")
    
    # Simular eleitos anteriores
    eleitos_2022 = set(dados[dados['eleito'] == 1].sample(frac=0.7)['candidato_id'])
    eleitos_2026 = set(dados[dados['eleito'] == 1]['candidato_id'])
    
    renovacao = indice_comp.calcular_taxa_renovacao(eleitos_2022, eleitos_2026)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Taxa de Renovação", f"{renovacao['taxa_renovacao']:.1f}%")
    
    with col2:
        st.metric("Novos Eleitos", renovacao['n_novos'])
    
    with col3:
        st.metric("Reeleitos", renovacao['n_reeleitos'])

# ==================== TAB 7: SIMULADOR ====================

with tab7:
    st.header("🎯 Simulador de Cenários Eleitorais")
    
    st.markdown("""
    Simule diferentes cenários eleitorais ajustando os parâmetros abaixo.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_candidatos_sim = st.slider("Número de Candidatos", 100, 1000, 500)
        n_cadeiras_sim = st.slider("Número de Cadeiras", 10, 100, 50)
    
    with col2:
        usar_coligacoes = st.checkbox("Simular com Coligações", value=True)
        ano_sim = st.selectbox("Ano da Simulação", [2026, 2030, 2034])
    
    if st.button("🎲 Executar Simulação"):
        with st.spinner("Simulando cenário eleitoral..."):
            # Gerar dados
            dados_sim = gerar_dados_eleitorais(n_candidatos=n_candidatos_sim, ano=ano_sim)
            
            votos_sim = dados_sim.groupby('partido')['votos'].sum()
            
            # Calcular distribuição de cadeiras
            qe = QuocienteEleitoral()
            
            if usar_coligacoes:
                coligacoes_sim = {
                    'Esquerda': ['PT', 'PSB', 'PCdoB', 'PSOL'],
                    'Centro': ['MDB', 'PSDB', 'CIDADANIA'],
                    'Direita': ['PL', 'PP', 'REPUBLICANOS', 'UNIÃO']
                }
                resultado_sim = qe.calcular_distribuicao(votos_sim, n_cadeiras_sim, coligacoes_sim)
            else:
                resultado_sim = qe.calcular_distribuicao(votos_sim, n_cadeiras_sim, None)
            
            st.success("✅ Simulação concluída!")
            
            # Resultados
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Distribuição de Cadeiras")
                fig = px.pie(resultado_sim.head(10), values='cadeiras', names='partido',
                            title='Distribuição de Cadeiras (Top 10)')
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Votos vs Cadeiras")
                fig = px.scatter(resultado_sim.head(15), x='percentual_votos', y='percentual_cadeiras',
                                text='partido', title='Proporcionalidade Votos x Cadeiras',
                                size='votos')
                fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode='lines',
                                        name='Linha de Proporcionalidade Perfeita',
                                        line=dict(dash='dash', color='red')))
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            
            # Tabela detalhada
            st.subheader("Resultados Detalhados")
            st.dataframe(resultado_sim, use_container_width=True)
            
            # Métricas do sistema
            st.subheader("Métricas do Sistema Partidário")
            
            nep_calc = NumeroEfetivoPartidos()
            indices_sim = nep_calc.calcular_indices_fragmentacao(
                resultado_sim.set_index('partido')['votos'],
                resultado_sim.set_index('partido')['cadeiras']
            )
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("NEP (Votos)", f"{indices_sim['NEP_votos']:.2f}")
            
            with col2:
                st.metric("NEP (Cadeiras)", f"{indices_sim['NEP_cadeiras']:.2f}")
            
            with col3:
                st.metric("Índice de Gallagher", f"{indices_sim['Indice_Gallagher']:.3f}")
            
            with col4:
                st.metric("Quociente Eleitoral", f"{qe.obter_quociente_eleitoral():,.0f}")

# ==================== FOOTER ====================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: white; padding: 20px;'>
    <p><strong>Sistema de Análise Eleitoral Avançado</strong></p>
    <p>Desenvolvido com Python, Streamlit, Scikit-learn, TensorFlow, PyMC3 e Plotly</p>
    <p>© 2024 - Todos os direitos reservados</p>
</div>
""", unsafe_allow_html=True)
