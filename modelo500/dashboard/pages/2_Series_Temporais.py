import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from gerador_dados import GeradorDados
from modelos.series_temporais import ModeloSARIMA, ModeloGARCH

st.set_page_config(page_title="Séries Temporais", page_icon="⏳", layout="wide")

st.title("Análise de Séries Temporais")

@st.cache_data
def carregar_dados():
    gerador = GeradorDados(seed=42)
    return gerador.gerar_dados_brasil_simulados(anos=15)

df = carregar_dados()
serie_desemprego = df.set_index('data')['desemprego']

col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("Configuração do Modelo")
    modelo_tipo = st.selectbox("Tipo de Modelo", ["SARIMA", "GARCH (Volatilidade)"])
    
    horizonte = st.slider("Horizonte de Previsão (meses)", 1, 24, 12)
    
    if modelo_tipo == "SARIMA":
        st.markdown("#### Parâmetros ARIMA")
        p = st.number_input("p (Auto-regressivo)", 0, 5, 1)
        d = st.number_input("d (Integração)", 0, 2, 1)
        q = st.number_input("q (Média Móvel)", 0, 5, 1)
        st.markdown("#### Sazonalidade")
        P = st.number_input("P (Sazonal AR)", 0, 2, 1)
        D = st.number_input("D (Sazonal Diff)", 0, 1, 1)
        Q = st.number_input("Q (Sazonal MA)", 0, 2, 1)
        s = st.number_input("s (Período)", 1, 24, 12)
    else:
        st.markdown("#### Parâmetros GARCH")
        p_garch = st.number_input("p (Lag Volatilidade)", 1, 5, 1)
        q_garch = st.number_input("q (Lag Erro)", 1, 5, 1)

with col2:
    if st.button("Executar Modelo", type="primary"):
        with st.spinner("Ajustando modelo..."):
            if modelo_tipo == "SARIMA":
                modelo = ModeloSARIMA(serie_desemprego, ordem=(p,d,q), ordem_sazonal=(P,D,Q,s))
                resumo = modelo.ajustar()
                
                if resumo:
                    previsao = modelo.prever(passos=horizonte)
                    
                    # Plot
                    fig = go.Figure()
                    # Histórico
                    fig.add_trace(go.Scatter(x=serie_desemprego.index, y=serie_desemprego, name="Histórico", line=dict(color='white')))
                    # Previsão
                    idx_futuro = pd.date_range(start=serie_desemprego.index[-1], periods=horizonte+1, freq='M')[1:]
                    fig.add_trace(go.Scatter(x=idx_futuro, y=previsao['previsao'], name="Previsão", line=dict(color='#00ff00')))
                    # Intervalo de Confiança
                    fig.add_trace(go.Scatter(
                        x=idx_futuro.tolist() + idx_futuro.tolist()[::-1],
                        y=previsao['limite_superior'].tolist() + previsao['limite_inferior'].tolist()[::-1],
                        fill='toself',
                        fillcolor='rgba(0, 255, 0, 0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='Intervalo de Confiança 95%'
                    ))
                    
                    fig.update_layout(title="Previsão SARIMA - Taxa de Desemprego", template="plotly_dark", yaxis_tickformat=".1%")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    with st.expander("Ver Resumo Estatístico"):
                        st.text(resumo)
                else:
                    st.error("Erro ao ajustar modelo SARIMA.")
                    
            else: # GARCH
                modelo = ModeloGARCH(serie_desemprego, p=p_garch, q=q_garch)
                resumo = modelo.ajustar()
                
                if resumo:
                    volatilidade = modelo.prever_volatilidade(horizonte=horizonte)
                    
                    fig = go.Figure()
                    # Volatilidade condicional histórica (precisaria extrair do modelo ajustado, simplificando aqui mostrando apenas previsão)
                    idx_futuro = pd.date_range(start=serie_desemprego.index[-1], periods=horizonte+1, freq='M')[1:]
                    
                    fig.add_trace(go.Bar(x=idx_futuro, y=volatilidade, name="Volatilidade Prevista", marker_color='#ff5555'))
                    
                    fig.update_layout(title="Previsão de Volatilidade (Incerteza) - GARCH", template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    with st.expander("Ver Resumo Estatístico"):
                        st.text(resumo)
                else:
                    st.error("Erro ao ajustar modelo GARCH (verifique se 'arch' está instalado).")
    else:
        st.info("Configure os parâmetros e clique em 'Executar Modelo'.")
