"""
Terminal B3 - Dashboard Financeiro

Interface profissional estilo terminal para análise de ativos brasileiros.

Autor: Luiz Tiago Wilcke
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys
import os

# Adiciona raiz ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dados.carregador import CarregadorB3
from modelos.merton import ModeloMerton
from motor.calibracao import CalibradorMerton
import config

st.set_page_config(
    page_title=config.TITULO_APP,
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed" # Estilo terminal, mais limpo
)

# CSS Estilo Terminal Financeiro
st.markdown("""
<style>
    .stApp {
        background-color: #000000;
        color: #00ff00;
        font-family: 'Courier New', Courier, monospace;
    }
    .stTextInput > div > div > input {
        background-color: #111;
        color: #00ff00;
        border: 1px solid #333;
    }
    .metric-container {
        border: 1px solid #333;
        padding: 10px;
        background-color: #0a0a0a;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("📟 TERMINAL B3 [QUANT_SYSTEM_V1.0]")
    
    # Input estilo comando
    col_input, col_btn = st.columns([3, 1])
    ticker_input = col_input.text_input("DIGITE O CÓDIGO DO ATIVO (EX: PETR4, VALE3, ^BVSP):", value="^BVSP").upper()
    
    if col_btn.button("EXECUTAR ANÁLISE"):
        analisar_ativo(ticker_input)
    else:
        st.markdown("---")
        st.markdown("Aguardando comando... Digite um ticker acima.")

def analisar_ativo(ticker):
    # 1. Dados
    try:
        with st.spinner(f"ACESSANDO DATAFEED B3 PARA {ticker}..."):
            carregador = CarregadorB3(ticker)
            df = carregador.obter_dados()
            
            # Candlestick Chart (Últimos 6 meses)
            df_recent = df.tail(126)
            
            fig_candle = go.Figure(data=[go.Candlestick(
                x=df_recent.index,
                open=df_recent['Open'],
                high=df_recent['High'],
                low=df_recent['Low'],
                close=df_recent['Close'],
                name=ticker
            )])
            fig_candle.update_layout(
                title=f"HISTÓRICO DE PREÇOS: {ticker}",
                template="plotly_dark",
                xaxis_rangeslider_visible=False,
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_candle, use_container_width=True)
            
    except Exception as e:
        st.error(f"ERRO DE CONEXÃO: {e}")
        return

    # 2. Calibração e Modelo
    with st.spinner("CALCULANDO PARÂMETROS ESTOCÁSTICOS (MJD)..."):
        calibrador = CalibradorMerton(df)
        params = calibrador.calibrar()
        
        # Exibe Parâmetros
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("PREÇO ATUAL", f"R$ {params['S0']:.2f}")
        c2.metric("VOLATILIDADE", f"{params['sigma']:.1%}")
        c3.metric("INTENSIDADE SALTOS", f"{params['lambda_j']:.2f}/ano")
        c4.metric("RISCO SALTO", f"{params['mu_j']:.2%}")

    # 3. Simulação
    with st.spinner("RODANDO SIMULAÇÃO MONTE CARLO (2000 CENÁRIOS)..."):
        modelo = ModeloMerton(params)
        T = 1.0 # 1 ano
        dt = 1/252
        tempos, trajetorias = modelo.simular(T, dt, 2000)
        
        dias = tempos * 252
        
    # 4. Projeção e Risco
    st.markdown("### 📊 PROJEÇÃO PROBABILÍSTICA (1 ANO)")
    
    final_prices = trajetorias[-1, :]
    p01 = np.percentile(final_prices, 1)
    p05 = np.percentile(final_prices, 5)
    p50 = np.percentile(final_prices, 50)
    p95 = np.percentile(final_prices, 95)
    p99 = np.percentile(final_prices, 99)
    
    # Tabela de Risco
    risk_data = {
        "CENÁRIO": ["Crash Extremo (1%)", "Pessimista (5%)", "Base (Mediana)", "Otimista (95%)", "Lua (99%)"],
        "PREÇO PROJETADO": [f"R$ {p01:.2f}", f"R$ {p05:.2f}", f"R$ {p50:.2f}", f"R$ {p95:.2f}", f"R$ {p99:.2f}"],
        "VARIAÇÃO": [f"{p01/params['S0']-1:.1%}", f"{p05/params['S0']-1:.1%}", f"{p50/params['S0']-1:.1%}", f"{p95/params['S0']-1:.1%}", f"{p99/params['S0']-1:.1%}"]
    }
    st.table(pd.DataFrame(risk_data))
    
    # Gráfico de Cone
    fig_cone = go.Figure()
    
    # Amostras
    subset = trajetorias[:, :50]
    for i in range(subset.shape[1]):
        fig_cone.add_trace(go.Scatter(
            x=dias, y=subset[:, i],
            mode='lines',
            line=dict(color='rgba(0, 255, 0, 0.1)', width=1),
            showlegend=False,
            hoverinfo='skip'
        ))
        
    # Quantis
    fig_cone.add_trace(go.Scatter(x=dias, y=np.percentile(trajetorias, 50, axis=1), 
                                mode='lines', line=dict(color='white', width=2), name='Mediana'))
    fig_cone.add_trace(go.Scatter(x=dias, y=np.percentile(trajetorias, 5, axis=1), 
                                mode='lines', line=dict(color='red', width=2, dash='dot'), name='Risco 5%'))
    fig_cone.add_trace(go.Scatter(x=dias, y=np.percentile(trajetorias, 95, axis=1), 
                                mode='lines', line=dict(color='cyan', width=2, dash='dot'), name='Alvo 95%'))
    
    fig_cone.update_layout(
        title="CONE DE PROBABILIDADE FUTURA",
        xaxis_title="DIAS ÚTEIS",
        yaxis_title="PREÇO (BRL)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=500
    )
    st.plotly_chart(fig_cone, use_container_width=True)

if __name__ == "__main__":
    main()
