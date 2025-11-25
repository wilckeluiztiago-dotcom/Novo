import streamlit as st
import pandas as pd
import sys
import os
import numpy as np

# Adicionar diretório raiz ao path para importações
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dados.gerador import gerar_dados_simulados
from dados.processamento import calcular_linha_pobreza, classificar_pobreza
from indicadores.pobreza import IndicadoresFGT
from indicadores.desigualdade import Desigualdade
from modelos.multidimensional import PobrezaMultidimensional
from visualizacao.graficos import plotar_curva_lorenz, plotar_distribuicao_renda, plotar_pobreza_por_uf

st.set_page_config(page_title="Índice de Pobreza Brasil", layout="wide")

st.markdown("""
<style>
    /* Global Theme */
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #64b5f6 !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        color: #4fc3f7;
        font-weight: bold;
    }
    div[data-testid="stMetricLabel"] {
        color: #b0bec5;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0e1117;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1c2331;
        border-radius: 8px 8px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #b0bec5;
        border: 1px solid #2c3e50;
        border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2b3b4f;
        color: #64b5f6;
        border-top: 2px solid #64b5f6;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
    }
    
    /* Buttons */
    div.stButton > button {
        background-color: #1976d2;
        color: white;
        border-radius: 8px;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #1565c0;
        color: white;
    }
    
    /* Dataframes */
    div[data-testid="stDataFrame"] {
        background-color: #1c2331;
        border-radius: 8px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 Sistema de Análise de Pobreza e Desigualdade")
st.markdown("---")

# Sidebar - Configurações
st.sidebar.header("Configurações da Simulação")
n_domicilios = st.sidebar.slider("Número de Domicílios", 1000, 50000, 10000)
seed = st.sidebar.number_input("Semente Aleatória", value=42)

if st.sidebar.button("Gerar Novos Dados"):
    st.session_state.df = gerar_dados_simulados(n_domicilios, seed)
    st.sidebar.success("Dados gerados com sucesso!")

if 'df' not in st.session_state:
    st.session_state.df = gerar_dados_simulados(n_domicilios, seed)

df = st.session_state.df

# Abas
tab1, tab2, tab3 = st.tabs(["Indicadores Clássicos", "Pobreza Multidimensional", "Dados Brutos"])

with tab1:
    st.header("Indicadores Unidimensionais (Renda)")
    
    col1, col2 = st.columns(2)
    with col1:
        metodo_lp = st.selectbox("Método Linha de Pobreza", ['relativo', 'absoluto'])
    with col2:
        if metodo_lp == 'relativo':
            param_lp = st.slider("% da Mediana", 0.4, 0.8, 0.6)
            lp = calcular_linha_pobreza(df, metodo='relativo', percentual_mediana=param_lp)
        else:
            param_lp = st.number_input("Valor Absoluto (R$)", 100.0, 2000.0, 600.0)
            lp = calcular_linha_pobreza(df, metodo='absoluto', valor_absoluto=param_lp)
            
    df = classificar_pobreza(df, lp)
    
    # Cálculos
    fgt = IndicadoresFGT(df, lp)
    desig = Desigualdade(df)
    
    # Métricas
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Incidência (P0)", f"{fgt.incidencia():.2%}")
    c2.metric("Hiato (P1)", f"{fgt.hiato():.2%}")
    c3.metric("Severidade (P2)", f"{fgt.severidade():.4f}")
    c4.metric("Gini", f"{desig.gini():.3f}")
    
    # Gráficos
    c_graf1, c_graf2 = st.columns(2)
    
    with c_graf1:
        st.subheader("Curva de Lorenz")
        pop_acum, renda_acum = desig.curva_lorenz()
        fig_lorenz = plotar_curva_lorenz(pop_acum, renda_acum, desig.gini())
        st.pyplot(fig_lorenz)
        
    with c_graf2:
        st.subheader("Distribuição de Renda")
        fig_dist = plotar_distribuicao_renda(df, lp)
        st.pyplot(fig_dist)
        
    st.subheader("Pobreza por UF")
    df_uf = df.groupby('uf')['is_pobre'].mean().reset_index()
    fig_uf = plotar_pobreza_por_uf(df_uf)
    st.pyplot(fig_uf)

with tab2:
    st.header("Índice de Pobreza Multidimensional (MPI)")
    
    st.markdown("### Dimensões e Pesos")
    col_w1, col_w2, col_w3, col_w4 = st.columns(4)
    w_agua = col_w1.number_input("Peso Água", 0.0, 1.0, 0.25)
    w_saneamento = col_w2.number_input("Peso Saneamento", 0.0, 1.0, 0.25)
    w_energia = col_w3.number_input("Peso Energia", 0.0, 1.0, 0.25)
    w_internet = col_w4.number_input("Peso Internet", 0.0, 1.0, 0.25)
    
    if not np.isclose(w_agua + w_saneamento + w_energia + w_internet, 1.0):
        st.error("A soma dos pesos deve ser 1.0")
    else:
        # Inverter lógica para privação (1 = privado)
        # Nos dados: 1 = tem acesso. Então privação = 1 - acesso
        df_mpi = df.copy()
        df_mpi['priv_agua'] = 1 - df_mpi['acesso_agua_potavel']
        df_mpi['priv_saneamento'] = 1 - df_mpi['saneamento_basico']
        df_mpi['priv_energia'] = 1 - df_mpi['energia_eletrica']
        df_mpi['priv_internet'] = 1 - df_mpi['internet']
        
        dimensoes = ['priv_agua', 'priv_saneamento', 'priv_energia', 'priv_internet']
        pesos = {
            'priv_agua': w_agua,
            'priv_saneamento': w_saneamento,
            'priv_energia': w_energia,
            'priv_internet': w_internet
        }
        
        k = st.slider("Corte de Pobreza (k)", 0.1, 1.0, 0.33)
        
        mpi_calc = PobrezaMultidimensional(df_mpi, dimensoes, pesos, k)
        indices = mpi_calc.calcular_indices()
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Incidência Multidimensional (H)", f"{indices['H']:.2%}")
        m2.metric("Intensidade Média (A)", f"{indices['A']:.2%}")
        m3.metric("MPI (M0)", f"{indices['MPI']:.3f}")
        
        # FIX: Usar mpi_calc.dados que contém as colunas calculadas
        st.dataframe(mpi_calc.dados[['uf', 'escore_privacao', 'is_pobre_mpi']].head(10))

with tab3:
    st.dataframe(df)
