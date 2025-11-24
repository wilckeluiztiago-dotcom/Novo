import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from modelos.stiglitz import ModeloShapiroStiglitz

st.set_page_config(page_title="Modelo de Stiglitz", page_icon="⚖️", layout="wide")

st.title("Teoria do Salário Eficiência (Shapiro-Stiglitz)")

st.markdown("""
Este modelo explica o **desemprego involuntário** como um mecanismo de disciplina. 
Se não houvesse desemprego, os trabalhadores não teriam medo de serem demitidos se fossem pegos "vadiando" (shirking).
Portanto, as empresas pagam um salário acima do mercado (salário eficiência) para incentivar o esforço.
""")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Parâmetros")
    
    b = st.slider("Benefício Desemprego (b)", 0.0, 2000.0, 1000.0, help="Utilidade de estar desempregado (seguro-desemprego + lazer)")
    e = st.slider("Custo do Esforço (e)", 0.0, 1000.0, 500.0, help="Desutilidade de trabalhar duro")
    q = st.slider("Probabilidade de Detecção (q)", 0.01, 1.0, 0.2, help="Eficiência do monitoramento da empresa")
    rho = st.slider("Taxa de Desconto (rho)", 0.01, 0.2, 0.05)
    alpha = st.slider("Produtividade (Alpha)", 5000.0, 20000.0, 10000.0)

with col2:
    # Instancia modelo
    params = {'b': b, 'e': e, 'q': q, 'rho': rho, 'alpha': alpha}
    modelo = ModeloShapiroStiglitz(params)
    
    # Calcula equilíbrio
    eq = modelo.calcular_equilibrio()
    
    # Métricas
    m1, m2, m3 = st.columns(3)
    m1.metric("Desemprego de Equilíbrio", f"{eq['desemprego_equilibrio']:.2%}")
    m2.metric("Salário de Equilíbrio", f"R$ {eq['salario_equilibrio']:.2f}")
    m3.metric("Emprego Total", f"{eq['emprego_equilibrio']:.1f}")
    
    # Gráfico
    df_curvas = modelo.simular_curvas()
    
    fig = go.Figure()
    
    # Curva NSC (No-Shirking Condition)
    fig.add_trace(go.Scatter(
        x=df_curvas['emprego'], 
        y=df_curvas['salario_nsc'], 
        name='NSC (Salário Eficiência)',
        line=dict(color='#ef4444', width=3)
    ))
    
    # Curva de Demanda
    fig.add_trace(go.Scatter(
        x=df_curvas['emprego'], 
        y=df_curvas['salario_demanda'], 
        name='Demanda de Trabalho',
        line=dict(color='#3b82f6', width=3)
    ))
    
    # Ponto de Equilíbrio
    fig.add_trace(go.Scatter(
        x=[eq['emprego_equilibrio']],
        y=[eq['salario_equilibrio']],
        mode='markers',
        marker=dict(size=15, color='white', symbol='star'),
        name='Equilíbrio'
    ))
    
    fig.update_layout(
        title="Equilíbrio de Shapiro-Stiglitz",
        xaxis_title="Nível de Emprego (L)",
        yaxis_title="Salário Real (w)",
        template="plotly_dark",
        yaxis=dict(range=[0, max(eq['salario_equilibrio']*2, 3000)])
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("""
    **Interpretação:**
    - A curva vermelha (NSC) mostra o salário mínimo necessário para evitar a vadiagem. Ela sobe conforme o emprego aumenta (desemprego cai), pois o medo da demissão diminui.
    - A curva azul é a demanda das empresas.
    - O desemprego de equilíbrio é a distância entre a Força de Trabalho Total (100) e o Emprego de Equilíbrio.
    """)
