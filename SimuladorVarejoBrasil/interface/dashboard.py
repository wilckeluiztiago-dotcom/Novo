import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
import os

# Adicionar diretório raiz ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modelo.dinamica import SimulacaoVarejo
from modelo.regioes import RegioesBrasil

st.set_page_config(page_title="Simulador Varejo Brasil", layout="wide", page_icon="🛒")

# Estilo Moderno
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    h1, h2, h3 {
        color: #00e676 !important;
    }
    div[data-testid="stMetricValue"] {
        color: #69f0ae;
    }
    .stButton>button {
        background-color: #2e7d32;
        color: white;
        border-radius: 8px;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛒 Simulador Varejo Brasil")
st.markdown("---")

# Sidebar - Configurações
st.sidebar.header("⚙️ Parâmetros da Simulação")

regiao = st.sidebar.selectbox("📍 Região", list(RegioesBrasil.DADOS.keys()))
meses = st.sidebar.slider("📅 Período (Meses)", 12, 60, 24)

st.sidebar.subheader("💰 Estratégia")
preco = st.sidebar.number_input("Preço Médio (R$)", 50.0, 500.0, 120.0)
marketing = st.sidebar.number_input("Investimento Marketing (Mensal)", 1000.0, 100000.0, 5000.0)
meta_estoque = st.sidebar.number_input("Meta de Estoque (Unidades)", 1000, 50000, 10000)

# Executar Simulação
sim = SimulacaoVarejo(regiao, preco, marketing, 5000, meta_estoque) # Estoque inicial fixo em 5000
df = sim.solver(t_max=meses)

# Métricas Consolidadas
receita_total = df['receita'].sum()
lucro_total = df['lucro'].sum()
vendas_total = df['vendas'].sum()
ruptura_total = df['ruptura'].sum()

# Exibir Métricas
col1, col2, col3, col4 = st.columns(4)
col1.metric("Receita Total", f"R$ {receita_total/1e6:.2f} M")
col2.metric("Lucro Total", f"R$ {lucro_total/1e6:.2f} M", delta_color="normal")
col3.metric("Vendas Totais", f"{vendas_total:,.0f}")
col4.metric("Ruptura Acumulada", f"{ruptura_total:,.0f}", delta_color="inverse")

# Gráficos
tab1, tab2, tab3 = st.tabs(["📈 Vendas e Estoque", "💵 Financeiro", "📊 Dados Detalhados"])

with tab1:
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        fig_vendas = go.Figure()
        fig_vendas.add_trace(go.Scatter(x=df['tempo'], y=df['vendas'], mode='lines', name='Vendas', line=dict(color='#00e676', width=2)))
        fig_vendas.add_trace(go.Scatter(x=df['tempo'], y=df['demanda_base'], mode='lines', name='Demanda Potencial', line=dict(color='#b9f6ca', dash='dash')))
        fig_vendas.update_layout(title="Vendas vs Demanda", xaxis_title="Mês", yaxis_title="Unidades", template="plotly_dark")
        st.plotly_chart(fig_vendas, use_container_width=True)
        
    with col_g2:
        fig_est = go.Figure()
        fig_est.add_trace(go.Scatter(x=df['tempo'], y=df['estoque'], mode='lines', name='Estoque', line=dict(color='#2979ff', width=2)))
        fig_est.add_trace(go.Scatter(x=df['tempo'], y=df['compras'], mode='lines', name='Compras', line=dict(color='#82b1ff', width=1)))
        fig_est.update_layout(title="Dinâmica de Estoque", xaxis_title="Mês", yaxis_title="Unidades", template="plotly_dark")
        st.plotly_chart(fig_est, use_container_width=True)

with tab2:
    fig_fin = go.Figure()
    fig_fin.add_trace(go.Scatter(x=df['tempo'], y=df['receita'], mode='lines', name='Receita', line=dict(color='#00e676')))
    fig_fin.add_trace(go.Scatter(x=df['tempo'], y=df['lucro'], mode='lines', name='Lucro', line=dict(color='#ffea00')))
    fig_fin.update_layout(title="Performance Financeira", xaxis_title="Mês", yaxis_title="Valor (R$)", template="plotly_dark")
    st.plotly_chart(fig_fin, use_container_width=True)

with tab3:
    st.dataframe(df)
