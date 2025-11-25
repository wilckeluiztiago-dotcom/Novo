# Configurações do Projeto de Montagem de Genoma

# Parâmetros de Montagem
TAMANHO_KMER = 31  # Tamanho do k-mer para o grafo de Bruijn
COBERTURA_MINIMA = 5  # Cobertura mínima para considerar um k-mer válido

# Parâmetros de Qualidade
QUALIDADE_MINIMA_PHRED = 20  # Score Phred mínimo para manter uma base
TAMANHO_MINIMO_READ = 50  # Tamanho mínimo de um read após trimagem

# Caminhos (Exemplos)
DIRETORIO_DADOS = "dados_entrada"
DIRETORIO_SAIDA = "resultados"
