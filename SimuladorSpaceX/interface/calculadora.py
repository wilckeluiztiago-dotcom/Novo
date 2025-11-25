import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
import os

# Adicionar diretório raiz ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modelo.fisica import Foguete, solver_rk4
from modelo.parametros import ParametrosFalcon9 as P

st.set_page_config(page_title="Calculadora de Missão SpaceX", layout="wide", page_icon="🚀")

# Estilo Moderno
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    h1, h2, h3 {
        color: #64b5f6 !important;
    }
    div[data-testid="stMetricValue"] {
        color: #4fc3f7;
    }
    .stButton>button {
        background-color: #1976d2;
        color: white;
        border-radius: 8px;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

st.title("🚀 Calculadora de Missão - Falcon 9")
st.markdown("---")

# Sidebar - Configuração da Missão
st.sidebar.header("⚙️ Configuração da Missão")

carga_util = st.sidebar.slider("📦 Carga Útil (kg)", 0, 22800, 5000, help="Massa do satélite ou carga.")
combustivel_pct = st.sidebar.slider("⛽ Combustível (%)", 10, 100, 100, help="Porcentagem de combustível no tanque.")
tempo_simulacao = st.sidebar.number_input("⏱️ Tempo de Simulação (s)", 100, 600, 300)

combustivel_kg = P.MASSA_COMBUSTIVEL_MAX * (combustivel_pct / 100.0)

# Criar Foguete e Simular
foguete = Foguete(carga_util=carga_util, combustivel_inicial=combustivel_kg)
resultados = solver_rk4(foguete, t_max=tempo_simulacao)

# Extrair métricas finais
alt_max = np.max(resultados['altitude'])
vel_max = np.max(resultados['velocidade'])
tempo_queima = np.sum(resultados['empuxo'] > 0) * (resultados['tempo'][1] - resultados['tempo'][0])

# Exibir Métricas Principais
col1, col2, col3, col4 = st.columns(4)
col1.metric("Apogeu Estimado", f"{alt_max/1000:.1f} km")
col2.metric("Velocidade Máxima", f"{vel_max:.0f} m/s ({vel_max*3.6:.0f} km/h)")
col3.metric("Tempo de Queima", f"{tempo_queima:.1f} s")
col4.metric("Massa Total Inicial", f"{foguete.massa_total/1000:.1f} ton")

# Gráficos
tab1, tab2, tab3 = st.tabs(["📈 Telemetria", "🌍 Trajetória", "📊 Dados Brutos"])

with tab1:
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        fig_alt = go.Figure()
        fig_alt.add_trace(go.Scatter(x=resultados['tempo'], y=resultados['altitude']/1000, mode='lines', name='Altitude', line=dict(color='#4fc3f7', width=2)))
        fig_alt.update_layout(title="Altitude vs Tempo", xaxis_title="Tempo (s)", yaxis_title="Altitude (km)", template="plotly_dark")
        st.plotly_chart(fig_alt, use_container_width=True)
        
    with col_g2:
        fig_vel = go.Figure()
        fig_vel.add_trace(go.Scatter(x=resultados['tempo'], y=resultados['velocidade'], mode='lines', name='Velocidade', line=dict(color='#ffb74d', width=2)))
        fig_vel.update_layout(title="Velocidade vs Tempo", xaxis_title="Tempo (s)", yaxis_title="Velocidade (m/s)", template="plotly_dark")
        st.plotly_chart(fig_vel, use_container_width=True)

    col_g3, col_g4 = st.columns(2)
    with col_g3:
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(x=resultados['tempo'], y=resultados['aceleracao']/9.81, mode='lines', name='Aceleração (G)', line=dict(color='#ef5350', width=2)))
        fig_acc.update_layout(title="Aceleração (G-Force)", xaxis_title="Tempo (s)", yaxis_title="Aceleração (G)", template="plotly_dark")
        st.plotly_chart(fig_acc, use_container_width=True)
        
    with col_g4:
        fig_mass = go.Figure()
        fig_mass.add_trace(go.Scatter(x=resultados['tempo'], y=resultados['massa']/1000, mode='lines', name='Massa', line=dict(color='#81c784', width=2)))
        fig_mass.update_layout(title="Massa Total vs Tempo", xaxis_title="Tempo (s)", yaxis_title="Massa (ton)", template="plotly_dark")
        st.plotly_chart(fig_mass, use_container_width=True)

with tab2:
    st.markdown("### Visualização da Trajetória Vertical")
    # Gráfico simples de altitude, mas poderia ser algo mais elaborado se tivéssemos componente horizontal
    fig_traj = go.Figure()
    fig_traj.add_trace(go.Scatter(x=np.zeros_like(resultados['altitude']), y=resultados['altitude']/1000, mode='lines', line=dict(color='white')))
    fig_traj.update_layout(title="Trajetória Vertical", xaxis_title="Desvio Lateral (km)", yaxis_title="Altitude (km)", template="plotly_dark")
    st.plotly_chart(fig_traj, use_container_width=True)

with tab3:
    df_res = pd.DataFrame(resultados)
    st.dataframe(df_res)
