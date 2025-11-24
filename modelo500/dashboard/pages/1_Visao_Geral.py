import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Path setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from gerador_dados import GeradorDados

st.set_page_config(page_title="Visão Geral - Desemprego", page_icon="📊", layout="wide")

st.title("Visão Geral Macroeconômica")

# Cache data generation
@st.cache_data
def carregar_dados():
    gerador = GeradorDados(seed=42)
    return gerador.gerar_dados_brasil_simulados(anos=15)

df = carregar_dados()

# KPIs
col1, col2, col3, col4 = st.columns(4)

ultimo = df.iloc[-1]
penultimo = df.iloc[-2]

def metric_delta(atual, anterior, formato="{:.2%}"):
    delta = atual - anterior
    return formato.format(atual), f"{delta*100:.2f} p.p."

val_desemp, delta_desemp = metric_delta(ultimo['desemprego'], penultimo['desemprego'])
val_inf, delta_inf = metric_delta(ultimo['inflacao']*12, penultimo['inflacao']*12) # Anualizado
val_selic, delta_selic = metric_delta(ultimo['selic'], penultimo['selic'])
val_pib, delta_pib = metric_delta(ultimo['pib_crescimento']*12, penultimo['pib_crescimento']*12)

with col1:
    st.metric("Taxa de Desemprego", val_desemp, delta_desemp, delta_color="inverse")
with col2:
    st.metric("Inflação (IPCA 12m)", val_inf, delta_inf, delta_color="inverse")
with col3:
    st.metric("Taxa Selic", val_selic, delta_selic)
with col4:
    st.metric("Crescimento PIB (12m)", val_pib, delta_pib)

# Gráficos Principais
st.markdown("### 📈 Evolução Histórica")

tab1, tab2 = st.tabs(["Desemprego vs Inflação", "Matriz de Correlação"])

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['data'], y=df['desemprego'], name='Desemprego', line=dict(color='#60a5fa', width=3)))
    fig.add_trace(go.Scatter(x=df['data'], y=df['inflacao']*12, name='Inflação (12m)', line=dict(color='#f87171', width=2), yaxis='y2'))
    
    fig.update_layout(
        title="Curva de Phillips Dinâmica (Brasil Simulado)",
        yaxis=dict(title="Desemprego", tickformat=".1%"),
        yaxis2=dict(title="Inflação", tickformat=".1%", overlaying='y', side='right'),
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.1)
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    corr = df[['desemprego', 'inflacao', 'selic', 'pib_crescimento', 'rendimento_medio']].corr()
    fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', title="Correlação entre Variáveis")
    fig_corr.update_layout(template="plotly_dark")
    st.plotly_chart(fig_corr, use_container_width=True)

# Dados Recentes
st.markdown("### 📋 Dados Recentes")
st.dataframe(
    df.tail(12).sort_values('data', ascending=False).style.format({
        'desemprego': '{:.2%}',
        'inflacao': '{:.2%}',
        'selic': '{:.2%}',
        'pib_crescimento': '{:.2%}',
        'rendimento_medio': 'R$ {:.2f}'
    })
)
