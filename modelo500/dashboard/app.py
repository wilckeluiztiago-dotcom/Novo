import streamlit as st
import sys
import os

# Adiciona diretório raiz ao path para importar módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

st.set_page_config(
    page_title="Sistema Avançado de Análise de Desemprego",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo CSS Customizado (Premium Dark Theme)
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .main .block-container {
        padding-top: 2rem;
    }
    h1 {
        color: #f0f2f6;
        font-family: 'Helvetica Neue', sans-serif;
    }
    h2, h3 {
        color: #aeb5bc;
    }
    .stMetric {
        background-color: #1f2937;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #374151;
    }
    .stMetric label {
        color: #9ca3af !important;
    }
    .stMetric div[data-testid="stMetricValue"] {
        color: #60a5fa !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 Sistema de Análise e Previsão de Desemprego")
st.markdown("### Modelagem Avançada com Séries Temporais, Análise Multivariada e Teoria Econômica")

st.markdown("""
---
Bem-vindo ao **Sistema Avançado de Análise de Desemprego**. Este dashboard integra dados macroeconômicos simulados do Brasil com modelos matemáticos de ponta.

### 🚀 Módulos Disponíveis:

1.  **Visão Geral**: Dashboard executivo com os principais indicadores (KPIs) e tendências recentes.
2.  **Séries Temporais**: Previsões univariadas usando SARIMA e análise de volatilidade com GARCH.
3.  **Análise Multivariada**: Relações dinâmicas entre Desemprego, Inflação, Juros e PIB usando VAR/VECM.
4.  **Modelo de Stiglitz**: Simulador interativo da teoria de Salário Eficiência e desemprego involuntário.

---
**Tecnologias:** Python, Streamlit, Statsmodels, NumPy, Pandas.
**Autor:** Luiz Tiago Wilcke
""")

# Sidebar info
st.sidebar.info("Navegue pelas páginas acima para acessar as análises detalhadas.")
st.sidebar.markdown("---")
st.sidebar.caption("Versão 1.0.0 | 2025")
