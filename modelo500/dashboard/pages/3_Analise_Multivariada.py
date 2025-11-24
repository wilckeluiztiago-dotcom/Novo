import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from gerador_dados import GeradorDados
from modelos.multivariada import AnaliseMultivariada

st.set_page_config(page_title="Análise Multivariada", page_icon="🕸️", layout="wide")

st.title("Análise Multivariada (VAR/VECM)")

@st.cache_data
def carregar_dados():
    gerador = GeradorDados(seed=42)
    return gerador.gerar_dados_brasil_simulados(anos=15)

df = carregar_dados()
# Seleciona colunas numéricas relevantes
df_model = df[['desemprego', 'inflacao', 'selic', 'pib_crescimento']].set_index(df['data'])

st.markdown("### Interdependência entre Variáveis Macroeconômicas")

col1, col2 = st.columns([1, 3])

with col1:
    lags = st.slider("Número de Lags (Defasagens)", 1, 12, 3)
    periodos_irf = st.slider("Períodos Impulso-Resposta", 6, 24, 12)
    
    st.markdown("---")
    st.markdown("**Teste de Causalidade**")
    causa = st.selectbox("Causa (X)", df_model.columns)
    efeito = st.selectbox("Efeito (Y)", df_model.columns, index=1)

with col2:
    analise = AnaliseMultivariada(df_model)
    
    tab1, tab2, tab3 = st.tabs(["Impulso-Resposta", "Decomposição de Variância", "Causalidade de Granger"])
    
    # Ajusta modelo uma vez
    analise.ajustar_var(lags=lags)
    
    with tab1:
        st.markdown(f"#### Resposta a um Choque em {causa}")
        try:
            irf = analise.impulso_resposta(periodos_irf)
            # Plotly manual do IRF é complexo pois statsmodels retorna objeto plot matplotlib
            # Vamos usar matplotlib e renderizar no streamlit
            fig = irf.plot(impulse=causa, response=efeito, figsize=(10, 5))
            st.pyplot(fig)
            st.caption(f"Como '{efeito}' reage a um choque inesperado em '{causa}' ao longo do tempo.")
        except Exception as e:
            st.error(f"Erro ao gerar IRF: {e}")

    with tab2:
        st.markdown("#### Decomposição da Variância do Erro de Previsão")
        try:
            fevd = analise.decomposicao_variancia(periodos_irf)
            fig = fevd.plot(figsize=(10, 8))
            st.pyplot(fig)
            st.caption("Explica qual porcentagem da variação futura de uma variável é devida a choques nela mesma vs. outras variáveis.")
        except Exception as e:
            st.error(f"Erro ao gerar FEVD: {e}")
            
    with tab3:
        st.markdown(f"#### Teste de Granger: {causa} -> {efeito}?")
        p_valores = analise.teste_causalidade_granger(causa, efeito, max_lags=lags)
        
        res_df = pd.DataFrame(list(p_valores.items()), columns=['Lag', 'P-Valor'])
        st.dataframe(res_df.style.applymap(lambda x: 'color: red' if x < 0.05 else 'color: green', subset=['P-Valor']))
        
        if any(v < 0.05 for v in p_valores.values()):
            st.success(f"Há evidências de que **{causa}** causa (no sentido de Granger) **{efeito}**.")
        else:
            st.warning(f"Não há evidências estatísticas suficientes de causalidade.")
