"""
Painel Streamlit para PrevisorPetroleo

Interface web interativa para visualização de previsões de commodities.

Autor: Luiz Tiago Wilcke
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import sys
import os

# Adiciona raiz ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dados.carregador import CarregadorDados
from modelos.mrsvj import ModeloPetroleo
from motor.calibracao import Calibrador
import config

st.set_page_config(
    page_title=config.TITULO_APP,
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo CSS
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("🛢️ PrevisorPetroleo AI")
    st.markdown("### Sistema de Previsão Estocástica (Modelo MRSVJ)")
    st.markdown("Modelagem de Reversão à Média com Volatilidade Estocástica e Saltos.")
    
    # Sidebar
    st.sidebar.header("⚙️ Configurações")
    ticker = st.sidebar.selectbox("Commodity", ["BZ=F", "CL=F"], format_func=lambda x: "Brent (Global)" if x == "BZ=F" else "WTI (USA)")
    horizonte = st.sidebar.slider("Horizonte (Dias)", 30, 365, config.SIM_HORIZONTE_DIAS)
    num_simulacoes = st.sidebar.slider("Simulações", 100, 5000, config.SIM_NUM_TRAJETORIAS)
    
    # Cenários de Estresse
    st.sidebar.subheader("🌪️ Teste de Estresse (Choque)")
    choque_geo = st.sidebar.checkbox("Simular Crise Geopolítica")
    intensidade_choque = 0.0
    if choque_geo:
        intensidade_choque = st.sidebar.slider("Intensidade do Choque (%)", 5, 50, 20) / 100
    
    if st.sidebar.button("🚀 Executar Previsão"):
        executar_previsao(ticker, horizonte, num_simulacoes, choque_geo, intensidade_choque)
    else:
        st.info("Configure os parâmetros e clique em 'Executar Previsão'.")

def executar_previsao(ticker, dias, n_sims, tem_choque, tamanho_choque):
    # 1. Carregar Dados
    with st.spinner('Baixando dados históricos do petróleo...'):
        carregador = CarregadorDados(ticker)
        try:
            df = carregador.obter_dados()
            preco_atual = df['Preco'].iloc[-1]
        except Exception as e:
            st.error(f"Erro ao carregar dados: {e}")
            return

    # 2. Calibrar Modelo
    with st.spinner('Calibrando modelo matemático (Reversão à Média)...'):
        calibrador = Calibrador(df)
        params = calibrador.calibrar()
        
        # Aplica choque se solicitado (aumenta intensidade de saltos e média)
        if tem_choque:
            params['lambda_j'] = 12.0 # 1 salto por mês em média durante crise
            params['mu_j'] = tamanho_choque # Tamanho do salto positivo
            params['sigma_j'] = tamanho_choque / 2 # Incerteza
            st.warning(f"⚠️ Simulando cenário de crise com choques de +{tamanho_choque:.0%}!")
            
    # Mostra parâmetros
    with st.expander("Ver Parâmetros Calibrados"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Preço Atual", f"${params['S0']:.2f}")
        c2.metric("Preço Equilíbrio (Longo Prazo)", f"${np.exp(params['theta_S']):.2f}")
        c3.metric("Velocidade Reversão", f"{params['kappa_S']:.2f}")
        c4.metric("Volatilidade Atual", f"{np.sqrt(params['v0']):.1%}")

    # 3. Simulação
    with st.spinner(f'Simulando {n_sims} cenários futuros...'):
        modelo = ModeloPetroleo(params)
        T = dias / 252 # Dias úteis
        dt = 1/252
        tempos, trajetorias = modelo.simular(T, dt, n_sims)
        
        dias_eixo = tempos * 252
        
    # 4. Visualização
    st.markdown("---")
    
    precos_finais = trajetorias[-1, :]
    media_preco = np.mean(precos_finais)
    p05 = np.percentile(precos_finais, 5)
    p95 = np.percentile(precos_finais, 95)
    
    col1, col2, col3 = st.columns(3)
    var_pct = (media_preco / preco_atual) - 1
    col1.metric("Preço Esperado", f"${media_preco:.2f}", f"{var_pct:+.2%}")
    col2.metric("Suporte (5%)", f"${p05:.2f}")
    col3.metric("Resistência (95%)", f"${p95:.2f}")
    
    # Gráfico de Cone
    fig = go.Figure()
    
    # Amostras
    subset = trajetorias[:, :min(100, n_sims)]
    for i in range(subset.shape[1]):
        fig.add_trace(go.Scatter(
            x=dias_eixo, y=subset[:, i],
            mode='lines',
            line=dict(color='rgba(255, 165, 0, 0.1)', width=1),
            showlegend=False,
            hoverinfo='skip'
        ))
        
    # Média e Limites
    fig.add_trace(go.Scatter(x=dias_eixo, y=np.mean(trajetorias, axis=1), 
                           mode='lines', line=dict(color='white', width=3), name='Média Esperada'))
    
    fig.add_trace(go.Scatter(x=dias_eixo, y=np.percentile(trajetorias, 5, axis=1),
                           mode='lines', line=dict(color='red', width=2, dash='dash'), name='Limite Inferior'))
    
    fig.add_trace(go.Scatter(x=dias_eixo, y=np.percentile(trajetorias, 95, axis=1),
                           mode='lines', line=dict(color='green', width=2, dash='dash'), name='Limite Superior'))
    
    # Linha de Equilíbrio
    fig.add_hline(y=np.exp(params['theta_S']), line_dash="dot", line_color="cyan", annotation_text="Preço Justo (Equilíbrio)")
    
    fig.update_layout(
        title=f"Projeção de Preço do Petróleo ({dias} dias)",
        xaxis_title="Dias Úteis Futuros",
        yaxis_title="Preço (USD/Barril)",
        template="plotly_dark",
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Histograma
    fig_hist = px.histogram(precos_finais, nbins=50, title="Distribuição de Probabilidade Final",
                          labels={'value': 'Preço (USD)'}, color_discrete_sequence=['#FFA500'])
    fig_hist.add_vline(x=preco_atual, line_dash="dash", line_color="white", annotation_text="Hoje")
    fig_hist.update_layout(template="plotly_dark")
    st.plotly_chart(fig_hist, use_container_width=True)

if __name__ == "__main__":
    main()
