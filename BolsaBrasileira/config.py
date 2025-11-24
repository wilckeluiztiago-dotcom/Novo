"""
Configurações Globais do BolsaBrasileira

Autor: Luiz Tiago Wilcke
"""

import os

# Configurações de Dados
TICKER_PADRAO = "^BVSP"  # Ibovespa
DATA_INICIO = "2015-01-01"
DIRETORIO_CACHE = "dados/cache"

# Configurações de Simulação
SIM_NUM_CENARIOS = 2000
SIM_HORIZONTE_DIAS = 252  # 1 ano útil
SIM_DT = 1/252
SIM_SEED = 42

# Parâmetros Iniciais (Modelo Merton Jump Diffusion)
# Estimativas típicas para mercado brasileiro (alta volatilidade + saltos)
PARAMS_INICIAIS = {
    'S0': 100.0,      # Preço inicial
    'mu': 0.12,       # Retorno esperado (12% a.a. - Selic + Prêmio)
    'sigma': 0.20,    # Volatilidade difusiva (20%)
    'lambda_j': 5.0,  # Intensidade de saltos (5 por ano)
    'mu_j': -0.02,    # Média do salto (-2% - viés de baixa em choques)
    'sigma_j': 0.05   # Desvio padrão do salto
}

# Configurações de Interface
TITULO_APP = "Terminal B3 - Análise Quantitativa"
TEMA_APP = "dark"
