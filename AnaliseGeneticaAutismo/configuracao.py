# configuracao.py
# Configurações Globais para o Software de Análise Genética Avançada

import os

# Caminhos do Projeto
DIRETORIO_BASE = os.path.dirname(os.path.abspath(__file__))
DIRETORIO_DADOS = os.path.join(DIRETORIO_BASE, 'dados_gerados')

# Garantir que o diretório de dados existe
os.makedirs(DIRETORIO_DADOS, exist_ok=True)

# Lista de Genes de Alto Risco para TEA (Transtorno do Espectro Autista) e Síndromes Relacionadas
GENES_ALVO = [
    "SHANK3",  # Síndrome de Phelan-McDermid
    "MECP2",   # Síndrome de Rett
    "CHD8",    # Subtipo de autismo com macrocefalia
    "PTEN",    # Síndrome de Cowden / TEA com macrocefalia
    "ADNP",    # Síndrome de Helsmoortel-Van der Aa
    "SYNGAP1", # Deficiência intelectual e epilepsia
    "FOXP1",   # Atraso na fala e TEA
    "SCN2A",   # Epilepsia e TEA
    "ARID1B",  # Síndrome de Coffin-Siris
    "FMR1"     # Síndrome do X Frágil
]

# Configurações da Simulação
NUMERO_AMOSTRAS_PADRAO = 1000
PROPORCAO_CASOS = 0.4  # 40% de casos, 60% de controles (enriquecido para análise)
NUMERO_SNPS_POR_GENE = 5

# Pesos de Risco (Simulados)
# Variantes de perda de função (LoF) têm peso maior
PESO_VARIANTE_LOF = 5.0
PESO_VARIANTE_MISSENSE = 1.5
PESO_VARIANTE_SINONIMA = 0.0

# Configurações de Visualização
PALETA_CORES = {
    'caso': '#FF4B4B',
    'controle': '#1E88E5',
    'fundo': '#0E1117',
    'texto': '#FAFAFA'
}
