"""
Configurações Globais do PrevisorPetroleo

Autor: Luiz Tiago Wilcke
"""

import os

# Configurações de Dados
TICKER_DADOS = "BZ=F"  # Brent Crude Oil Futures
DATA_INICIO_DADOS = "2010-01-01"
DIRETORIO_CACHE = "dados/cache"

# Configurações de Simulação
SIM_NUM_TRAJETORIAS = 1000
SIM_HORIZONTE_DIAS = 90  # 3 meses
SIM_DT = 1/252  # Passo de tempo diário (dias úteis)
SIM_SEED = 42

# Parâmetros Iniciais (Chute inicial para calibração)
# Modelo MRSVJ: Mean-Reverting Stochastic Volatility with Jumps
PARAMS_INICIAIS = {
    'S0': 80.0,     # Preço inicial
    'v0': 0.10,     # Variância inicial (10% vol anual ao quadrado)
    'theta_S': 4.4, # Log-preço de equilíbrio (exp(4.4) ~= 81 USD)
    'kappa_S': 1.5, # Velocidade de reversão do preço
    'theta_v': 0.10,# Variância de longo prazo
    'kappa_v': 2.0, # Velocidade de reversão da variância
    'xi': 0.3,      # Volatilidade da variância (Vol-of-Vol)
    'rho': -0.3,    # Correlação (Preço x Vol) - geralmente negativa para commodities
    'lambda_j': 2.0,# Intensidade dos saltos (2 por ano)
    'mu_j': 0.05,   # Média do tamanho do salto (5%)
    'sigma_j': 0.1  # Desvio padrão do salto
}

# Configurações de Interface
TITULO_APP = "PrevisorPetroleo AI - Brent/WTI"
TEMA_APP = "dark"
