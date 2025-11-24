"""
Configurações Globais do Sistema de Modelagem de Desemprego

Define parâmetros padrão para simulações e visualizações.

Autor: Luiz Tiago Wilcke
Data: 2025-11-24
"""

import numpy as np
from typing import Dict, Any

# ============================================================================
# CONFIGURAÇÕES DE SIMULAÇÃO
# ============================================================================

# Parâmetros de tempo
TEMPO_FINAL_PADRAO = 10.0  # Anos
NUMERO_PASSOS_PADRAO = 1000
DELTA_T_PADRAO = TEMPO_FINAL_PADRAO / NUMERO_PASSOS_PADRAO

# Simulações Monte Carlo
NUMERO_TRAJETORIAS_PADRAO = 100
SEED_PADRAO = 42

# Método numérico padrão
METODO_NUMERICO_PADRAO = 'euler'  # 'euler', 'milstein', ou 'srk'

# ============================================================================
# PARÂMETROS DOS MODELOS
# ============================================================================

# Modelo de Goodwin
# Calibração para equilíbrio realista:
# u* = delta/beta ~ 0.94 (94% emprego)
# v* = gamma/alpha ~ 0.66 (66% parcela salarial)
PARAMETROS_GOODWIN = {
    'gamma': 0.025,     # Crescimento da produtividade (2.5%)
    'alpha': 0.038,     # Sensibilidade salário-emprego (ajustado)
    'beta': 0.032,      # Sensibilidade lucro-salário (ajustado)
    'delta': 0.03,      # Taxa de depreciação (3%)
    'sigma_u': 0.01,    # Volatilidade do emprego (reduzida para estabilidade)
    'sigma_v': 0.01,    # Volatilidade dos salários
    'u0': 0.94,         # Emprego inicial (94%)
    'v0': 0.65          # Parcela salarial inicial (65%)
}

# Modelo de Phillips
PARAMETROS_PHILLIPS = {
    'pi_star': 0.04,    # Inflação alvo (4%)
    'u_natural': 0.06,  # NAIRU (6%)
    'theta': 0.5,       # Ajuste inflação
    'kappa': 2.0,       # Sensibilidade Phillips
    'alpha': 0.3,       # Persistência
    'beta': 0.4,        # Reversão à média
    'sigma_pi': 0.02,   # Volatilidade inflação
    'sigma_u': 0.01,    # Volatilidade desemprego
    'pi0': 0.035,       # Inflação inicial
    'u0': 0.055         # Desemprego inicial
}

# Modelo Populacional
PARAMETROS_POPULACIONAL = {
    'mu': 0.015,        # Crescimento populacional (1.5%)
    'lambda_': 0.08,    # Criação de empregos
    'K': 100.0,         # Capacidade em milhões
    'delta': 0.04,      # Destruição de empregos
    'sigma_L': 0.005,   # Volatilidade força trabalho
    'sigma_E': 0.03,    # Volatilidade emprego
    'L0': 80.0,         # Força trabalho inicial
    'E0': 75.0          # Empregados iniciais
}

# Modelo de Markov
PARAMETROS_MARKOV = {
    # Taxas de transição (por ano)
    'taxa_formal_informal': 0.05,
    'taxa_formal_desemprego': 0.03,
    'taxa_informal_formal': 0.08,
    'taxa_informal_desemprego': 0.06,
    'taxa_desemprego_formal': 0.15,
    'taxa_desemprego_informal': 0.20,
    # Volatilidades
    'sigma_formal': 0.02,
    'sigma_informal': 0.03,
    'sigma_desemprego': 0.04,
    # Estado inicial (frações)
    'x_formal_0': 0.55,
    'x_informal_0': 0.30,
    'x_desemprego_0': 0.15
}

# Mapeamento de modelos para parâmetros
PARAMETROS_MODELOS = {
    'goodwin': PARAMETROS_GOODWIN,
    'phillips': PARAMETROS_PHILLIPS,
    'populacional': PARAMETROS_POPULACIONAL,
    'markov': PARAMETROS_MARKOV
}

# ============================================================================
# CONFIGURAÇÕES DE VISUALIZAÇÃO
# ============================================================================

# Resolução de gráficos
DPI_PADRAO = 150
DPI_ALTA_QUALIDADE = 300

# Tamanhos de figura
FIGSIZE_PADRAO = (12, 8)
FIGSIZE_GRANDE = (16, 10)
FIGSIZE_QUADRADO = (10, 10)

# Diretório para salvar resultados
DIRETORIO_RESULTADOS = 'resultados'

# Formatos de exportação
FORMATOS_IMAGEM = ['png', 'pdf', 'svg']
FORMATO_PADRAO = 'png'

# Estilo de gráficos
ESTILO_GRAFICO_PADRAO = 'seaborn-v0_8-darkgrid'

# Cores para múltiplos modelos
CORES_MODELOS = {
    'goodwin': '#1f77b4',      # Azul
    'phillips': '#ff7f0e',     # Laranja
    'populacional': '#2ca02c', # Verde
    'markov': '#d62728'        # Vermelho
}

# Paleta de cores
PALETA_CORES = 'tab10'

# ============================================================================
# CONFIGURAÇÕES DE ANÁLISE
# ============================================================================

# Níveis de confiança
NIVEIS_CONFIANCA = [0.90, 0.95, 0.99]
NIVEL_CONFIANCA_PADRAO = 0.95

# Lags para autocorrelação
MAX_LAG_ACF = 40

# Testes estatísticos
ALPHA_TESTES = 0.05  # Nível de significância

# ============================================================================
# CONFIGURAÇÕES DE GERAÇÃO DE DADOS
# ============================================================================

# Série temporal sintética
TAXA_BASE_DESEMPREGO = 0.06
TENDENCIA_PADRAO = 0.0
AMPLITUDE_SAZONAL = 0.01
VOLATILIDADE_PADRAO = 0.005

# Choques econômicos padrão
CHOQUES_PADRAO = [
    {'tempo': 30, 'magnitude': 0.03, 'duracao': 8},   # Recessão moderada
    {'tempo': 70, 'magnitude': -0.02, 'duracao': 6}   # Boom econômico
]

# ============================================================================
# CONFIGURAÇÕES DE CONVERGÊNCIA
# ============================================================================

# Valores de N para análise de convergência
N_VALORES_CONVERGENCIA = [50, 100, 200, 400, 800, 1600]

# Número de trajetórias para análise de convergência
N_TRAJETORIAS_CONVERGENCIA = 50

# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def obter_config_modelo(nome_modelo: str) -> Dict[str, Any]:
    """
    Obtém configuração completa para um modelo específico.
    
    Args:
        nome_modelo: Nome do modelo ('goodwin', 'phillips', etc.)
        
    Returns:
        Dicionário com configurações
    """
    if nome_modelo not in PARAMETROS_MODELOS:
        raise ValueError(f"Modelo '{nome_modelo}' não reconhecido. "
                        f"Opções: {list(PARAMETROS_MODELOS.keys())}")
    
    config = {
        'parametros': PARAMETROS_MODELOS[nome_modelo].copy(),
        'tempo_final': TEMPO_FINAL_PADRAO,
        'numero_passos': NUMERO_PASSOS_PADRAO,
        'metodo': METODO_NUMERICO_PADRAO,
        'seed': SEED_PADRAO
    }
    
    return config


def obter_config_simulacao(
    modelo: str = 'goodwin',
    T: float = None,
    N: int = None,
    num_trajetorias: int = None,
    metodo: str = None,
    seed: int = None
) -> Dict[str, Any]:
    """
    Cria dicionário de configuração para simulação.
    
    Args:
        modelo: Nome do modelo
        T: Tempo final (usa padrão se None)
        N: Número de passos (usa padrão se None)
        num_trajetorias: Número de trajetórias (usa padrão se None)
        metodo: Método numérico (usa padrão se None)
        seed: Semente aleatória (usa padrão se None)
        
    Returns:
        Dicionário de configuração
    """
    config = obter_config_modelo(modelo)
    
    if T is not None:
        config['tempo_final'] = T
    if N is not None:
        config['numero_passos'] = N
    if num_trajetorias is not None:
        config['num_trajetorias'] = num_trajetorias
    else:
        config['num_trajetorias'] = NUMERO_TRAJETORIAS_PADRAO
    if metodo is not None:
        config['metodo'] = metodo
    if seed is not None:
        config['seed'] = seed
    
    return config


def imprimir_configuracao(config: Dict[str, Any]) -> None:
    """
    Imprime configuração de forma legível.
    
    Args:
        config: Dicionário de configuração
    """
    print("\n" + "="*60)
    print("CONFIGURAÇÃO DA SIMULAÇÃO")
    print("="*60)
    
    for chave, valor in config.items():
        if isinstance(valor, dict):
            print(f"\n{chave.upper()}:")
            for sub_chave, sub_valor in valor.items():
                print(f"  {sub_chave:.<30} {sub_valor}")
        else:
            print(f"{chave:.<40} {valor}")
    
    print("="*60 + "\n")


# Configuração padrão completa
CONFIG_PADRAO = {
    'simulacao': {
        'tempo_final': TEMPO_FINAL_PADRAO,
        'numero_passos': NUMERO_PASSOS_PADRAO,
        'num_trajetorias': NUMERO_TRAJETORIAS_PADRAO,
        'metodo': METODO_NUMERICO_PADRAO,
        'seed': SEED_PADRAO
    },
    'visualizacao': {
        'dpi': DPI_PADRAO,
        'figsize': FIGSIZE_PADRAO,
        'estilo': ESTILO_GRAFICO_PADRAO,
        'formato': FORMATO_PADRAO,
        'diretorio': DIRETORIO_RESULTADOS
    },
    'analise': {
        'nivel_confianca': NIVEL_CONFIANCA_PADRAO,
        'max_lag': MAX_LAG_ACF,
        'alpha': ALPHA_TESTES
    }
}
