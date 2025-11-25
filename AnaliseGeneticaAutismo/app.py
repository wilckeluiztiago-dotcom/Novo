# app.py
import streamlit as st
import pandas as pd
import numpy as np
from dados import GeradorDadosGeneticos
from analise import AnalisadorEstatistico, CalculadoraRisco, AnalisadorMultivariado
from modelos import PreditorGeneticoIA
from visualizacao import VisualizadorGenetico
from configuracao import GENES_ALVO

# Configuração da Página
st.set_page_config(
    page_title="NeuroGen: Análise Genética de Autismo",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo Customizado (CSS)
st.markdown("""
<style>
    .reportview-container {
        background: #0E1117;
    }
    .sidebar .sidebar-content {
        background: #262730;
    }
    h1 {
        color: #4B8BBE;
    }
    h2 {
        color: #306998;
    }
    .stButton>button {
        color: white;
        background-color: #4B8BBE;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Título e Descrição
st.title("🧬 NeuroGen: Sistema Avançado de Análise Genética (TEA)")
st.markdown("""
Este software realiza análises complexas de dados genômicos focados em Transtorno do Espectro Autista (TEA) e síndromes relacionadas.
Utiliza simulação estatística avançada, testes de associação (GWAS), cálculo de risco poligênico (PRS), Análise Multivariada (PCA) e Inteligência Artificial (Deep Learning).
""")

# Sidebar - Controles
st.sidebar.header("Configurações da Simulação")
n_amostras = st.sidebar.slider("Número de Amostras", 100, 5000, 1000)
seed = st.sidebar.number_input("Semente Aleatória (Seed)", value=42)

if st.sidebar.button("Gerar Novos Dados e Analisar"):
    with st.spinner('Gerando genoma sintético e calculando fenótipos...'):
        # 1. Geração de Dados
        gerador = GeradorDadosGeneticos(n_amostras=n_amostras, seed=seed)
        df_genotipos, df_expressao, df_fenotipos, metadados_snps = gerador.gerar_dataset_completo()
        
        # Armazenar em cache na sessão
        st.session_state['dados'] = {
            'genotipos': df_genotipos,
            'expressao': df_expressao,
            'fenotipos': df_fenotipos,
            'metadados': metadados_snps
        }
        st.success("Dados gerados com sucesso!")

# Verificar se há dados
if 'dados' in st.session_state:
    dados = st.session_state['dados']
    
    # Abas Principais
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Visão Geral & Estatística", 
        "🧬 Análise de Variantes (GWAS)", 
        "🧠 Inteligência Artificial (Deep Learning)", 
        "🕸️ Redes Gênicas",
        "🧊 Análise Multivariada (PCA 3D)"
    ])
    
    with tab1:
        st.header("Estatísticas da Coorte Simulada")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total de Amostras", len(dados['fenotipos']))
        col2.metric("Casos (TEA)", dados['fenotipos']['Status'].sum())
        col3.metric("Controles", len(dados['fenotipos']) - dados['fenotipos']['Status'].sum())
        
        st.subheader("Distribuição do Score de Risco Poligênico (PRS)")
        st.line_chart(dados['fenotipos']['PRS'])
        
        st.subheader("Dados Brutos (Amostra)")
        st.dataframe(dados['fenotipos'].head())

    with tab2:
        st.header("Genome-Wide Association Study (GWAS Simulado)")
        
        # Executar GWAS
        analisador = AnalisadorEstatistico()
        df_gwas = analisador.teste_associacao_gwas(dados['genotipos'], dados['fenotipos'])
        
        st.dataframe(df_gwas.style.highlight_min(subset=['P_Valor'], color='lightgreen'))
        
        # Manhattan Plot
        fig_manhattan = VisualizadorGenetico.criar_manhattan_plot(df_gwas)
        st.plotly_chart(fig_manhattan, use_container_width=True)
        
    with tab3:
        st.header("Predição de Risco com IA (Ensemble)")
        st.markdown("Comparação entre **Random Forest** e **Rede Neural (MLP)**.")
        
        # Preparar dados e treinar
        preditor = PreditorGeneticoIA()
        X = preditor.preparar_dados(dados['genotipos'], dados['expressao'])
        y = dados['fenotipos']['Status']
        
        # Split treino/teste
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        with st.spinner("Treinando modelos de IA..."):
            preditor.treinar(X_train, y_train)
            metricas = preditor.avaliar(X_test, y_test)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌲 Random Forest")
            st.metric("Acurácia", f"{metricas['RandomForest']['Relatorio']['accuracy']:.2%}")
            st.metric("AUC-ROC", f"{metricas['RandomForest']['AUC_ROC']:.3f}")
            st.text("Matriz de Confusão:")
            st.write(metricas['RandomForest']['Matriz_Confusao'])
            
        with col2:
            st.subheader("🧠 Rede Neural (MLP)")
            st.metric("Acurácia", f"{metricas['RedeNeural']['Relatorio']['accuracy']:.2%}")
            st.metric("AUC-ROC", f"{metricas['RedeNeural']['AUC_ROC']:.3f}")
            st.text("Matriz de Confusão:")
            st.write(metricas['RedeNeural']['Matriz_Confusao'])
        
        st.subheader("Importância das Features (Genes/SNPs - RF)")
        st.bar_chart(preditor.features_importantes.set_index('Feature').head(10))
        
    with tab4:
        st.header("Biologia de Sistemas e Redes")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Expressão Diferencial")
            fig_heatmap = VisualizadorGenetico.criar_heatmap_expressao(dados['expressao'], dados['fenotipos'])
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
        with col2:
            st.subheader("Rede de Interação Proteica")
            fig_rede = VisualizadorGenetico.criar_rede_interacao(GENES_ALVO)
            st.plotly_chart(fig_rede, use_container_width=True)

    with tab5:
        st.header("Análise de Componentes Principais (PCA)")
        st.markdown("""
        Visualização 3D da estrutura populacional. 
        Os eixos representam os principais componentes de variância genética.
        """)
        
        analisador_mv = AnalisadorMultivariado()
        df_pca = analisador_mv.executar_pca(dados['genotipos'])
        
        fig_pca = VisualizadorGenetico.criar_pca_3d(df_pca, dados['fenotipos'])
        st.plotly_chart(fig_pca, use_container_width=True)

else:
    st.info("👈 Clique no botão na barra lateral para iniciar a simulação e análise.")
