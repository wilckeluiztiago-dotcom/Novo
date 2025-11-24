"""
Configurações Globais do BitcoinPreditor

Autor: Luiz Tiago Wilcke
"""

import os

# Configurações de Dados
DATA_TICKER = "BTC-USD"
DATA_START_DATE = "2018-01-01"
DATA_CACHE_DIR = "data/cache"

# Configurações de Simulação
SIM_NUM_TRAJETORIAS = 1000
SIM_HORIZONTE_DIAS = 30
SIM_DT = 1/365  # Passo de tempo diário (anualizado)
SIM_SEED = 42

# Parâmetros Iniciais (Chute inicial para calibração)
# Baseado em literatura para criptoativos
PARAMS_INICIAIS = {
    'S0': 100.0,    # Preço inicial (será sobrescrito)
    'v0': 0.04,     # Variância inicial
    'kappa': 2.0,   # Velocidade de reversão da média da vol
    'theta': 0.04,  # Variância de longo prazo
    'xi': 0.5,      # Volatilidade da volatilidade (Vol-of-Vol)
    'rho': -0.5,    # Correlação entre preço e volatilidade
    'r': 0.05,      # Taxa livre de risco
    'lambda_j': 0.1, # Intensidade dos saltos (saltos por ano)
    'mu_j': -0.05,  # Média do tamanho do salto
    'sigma_j': 0.1  # Desvio padrão do tamanho do salto
}

# Configurações de Interface
UI_TITLE = "BitcoinPreditor - Stochastic AI"
UI_THEME = "dark"
