"""
Dashboard Streamlit para BitcoinPreditor

Interface web interativa para visualização de previsões e riscos.

Autor: Luiz Tiago Wilcke
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import sys
import os

# Adiciona raiz ao path para imports funcionarem
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader import DataLoader
from models.bates import BatesModel
from engine.calibration import Calibrator
import config

st.set_page_config(
    page_title=config.UI_TITLE,
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo CSS customizado
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .metric-card {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #41424b;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("₿ BitcoinPreditor AI")
    st.markdown("### Sistema de Previsão Estocástica (Modelo de Bates)")
    
    # Sidebar de Configuração
    st.sidebar.header("⚙️ Configurações")
    ticker = st.sidebar.text_input("Ticker", config.DATA_TICKER)
    horizonte = st.sidebar.slider("Horizonte de Previsão (Dias)", 7, 365, config.SIM_HORIZONTE_DIAS)
    num_simulacoes = st.sidebar.slider("Número de Simulações", 100, 5000, config.SIM_NUM_TRAJETORIAS)
    
    if st.sidebar.button("🚀 Executar Previsão"):
        run_prediction(ticker, horizonte, num_simulacoes)
    else:
        st.info("Configure os parâmetros na barra lateral e clique em 'Executar Previsão'.")

def run_prediction(ticker, days, n_sims):
    # 1. Carregar Dados
    with st.spinner('Baixando dados históricos...'):
        loader = DataLoader(ticker)
        try:
            df = loader.get_data()
            current_price = df['Price'].iloc[-1]
        except Exception as e:
            st.error(f"Erro ao carregar dados: {e}")
            return

    # 2. Calibrar Modelo
    with st.spinner('Calibrando modelo matemático (Bates/Heston)...'):
        calibrator = Calibrator(df)
        params = calibrator.calibrate()
        
    # Mostra parâmetros calibrados
    with st.expander("Ver Parâmetros Calibrados"):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Preço Atual", f"${params['S0']:.2f}")
        col2.metric("Volatilidade (v0)", f"{np.sqrt(params['v0']):.2%}")
        col3.metric("Reversão (κ)", f"{params['kappa']:.2f}")
        col4.metric("Vol-of-Vol (ξ)", f"{params['xi']:.2f}")
        
        col5, col6, col7 = st.columns(3)
        col5.metric("Intensidade Saltos (λ)", f"{params['lambda_j']:.2f}/ano")
        col6.metric("Média Salto (μ)", f"{params['mu_j']:.2%}")
        col7.metric("Correlação (ρ)", f"{params['rho']:.2f}")

    # 3. Simulação Monte Carlo
    with st.spinner(f'Simulando {n_sims} cenários futuros...'):
        model = BatesModel(params)
        T = days / 365
        dt = 1/365
        times, paths = model.simulate(T, dt, n_sims)
        
        # Ajusta tempos para dias
        days_axis = times * 365
        
    # 4. Visualização e Análise
    st.markdown("---")
    
    # Métricas Finais
    final_prices = paths[-1, :]
    mean_price = np.mean(final_prices)
    median_price = np.median(final_prices)
    var_95 = np.percentile(final_prices, 5)
    upside_95 = np.percentile(final_prices, 95)
    prob_up = np.mean(final_prices > current_price)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Preço Esperado (Média)", f"${mean_price:.2f}", f"{(mean_price/current_price - 1):.2%}")
    c2.metric("VaR 95% (Pior Caso)", f"${var_95:.2f}", f"{(var_95/current_price - 1):.2%}")
    c3.metric("Probabilidade de Alta", f"{prob_up:.1%}")
    
    # Gráfico de Trajetórias (Cone de Incerteza)
    fig = go.Figure()
    
    # Plota amostra de trajetórias (max 100 para não pesar)
    subset = paths[:, :min(100, n_sims)]
    for i in range(subset.shape[1]):
        fig.add_trace(go.Scatter(
            x=days_axis, y=subset[:, i],
            mode='lines',
            line=dict(color='rgba(0, 255, 255, 0.1)', width=1),
            showlegend=False,
            hoverinfo='skip'
        ))
        
    # Plota Média e Intervalos
    fig.add_trace(go.Scatter(
        x=days_axis, y=np.mean(paths, axis=1),
        mode='lines',
        line=dict(color='white', width=3),
        name='Média Esperada'
    ))
    
    fig.add_trace(go.Scatter(
        x=days_axis, y=np.percentile(paths, 5, axis=1),
        mode='lines',
        line=dict(color='red', width=2, dash='dash'),
        name='Limite Inferior (5%)'
    ))
    
    fig.add_trace(go.Scatter(
        x=days_axis, y=np.percentile(paths, 95, axis=1),
        mode='lines',
        line=dict(color='green', width=2, dash='dash'),
        name='Limite Superior (95%)'
    ))
    
    fig.update_layout(
        title=f"Projeção Estocástica: {days} Dias",
        xaxis_title="Dias Futuros",
        yaxis_title="Preço (USD)",
        template="plotly_dark",
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Histograma Final
    fig_hist = px.histogram(
        final_prices, 
        nbins=50, 
        title="Distribuição de Preços no Final do Período",
        labels={'value': 'Preço (USD)'},
        color_discrete_sequence=['#00cc96']
    )
    fig_hist.add_vline(x=current_price, line_dash="dash", line_color="white", annotation_text="Hoje")
    fig_hist.update_layout(template="plotly_dark")
    st.plotly_chart(fig_hist, use_container_width=True)

if __name__ == "__main__":
    main()
