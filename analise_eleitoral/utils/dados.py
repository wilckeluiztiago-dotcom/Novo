"""
Gerador de dados eleitorais simulados para análise.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def gerar_dados_eleitorais(n_candidatos=500, n_estados=27, ano=2026, tipo='federal'):
    """
    Gera dados eleitorais simulados realistas.
    
    Parâmetros:
    -----------
    n_candidatos : int
        Número de candidatos a gerar
    n_estados : int
        Número de estados (padrão: 27 para Brasil)
    ano : int
        Ano da eleição
    tipo : str
        'federal' ou 'estadual'
    
    Retorna:
    --------
    pd.DataFrame
        DataFrame com dados eleitorais simulados
    """
    np.random.seed(42)
    
    estados = ['AC', 'AL', 'AP', 'AM', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA', 
               'MT', 'MS', 'MG', 'PA', 'PB', 'PR', 'PE', 'PI', 'RJ', 'RN', 
               'RS', 'RO', 'RR', 'SC', 'SP', 'SE', 'TO']
    
    partidos = ['PT', 'PL', 'PP', 'UNIÃO', 'PSD', 'REPUBLICANOS', 'MDB', 'PSDB',
                'PDT', 'PSB', 'PSOL', 'CIDADANIA', 'PODE', 'PCdoB', 'PV', 
                'AVANTE', 'SOLIDARIEDADE', 'NOVO', 'REDE', 'PMB']
    
    # Espectro ideológico (1=esquerda, 10=direita)
    ideologia_partido = {
        'PT': 2, 'PSOL': 1, 'PCdoB': 2, 'PSB': 3, 'PDT': 3, 'PV': 4, 'REDE': 3,
        'CIDADANIA': 5, 'MDB': 6, 'PSDB': 7, 'PSD': 6, 'PODE': 6,
        'PP': 8, 'PL': 9, 'REPUBLICANOS': 8, 'UNIÃO': 7, 'NOVO': 8,
        'AVANTE': 5, 'SOLIDARIEDADE': 5, 'PMB': 4
    }
    
    dados = []
    
    for i in range(n_candidatos):
        estado = np.random.choice(estados)
        partido = np.random.choice(partidos)
        
        # População do estado (simplificado)
        pop_estado = {
            'SP': 46_000_000, 'MG': 21_000_000, 'RJ': 17_000_000, 'BA': 15_000_000,
            'PR': 11_500_000, 'RS': 11_400_000, 'PE': 9_600_000, 'CE': 9_200_000,
            'PA': 8_700_000, 'SC': 7_300_000, 'MA': 7_100_000, 'GO': 7_100_000,
            'AM': 4_200_000, 'ES': 4_100_000, 'PB': 4_000_000, 'RN': 3_500_000,
            'MT': 3_500_000, 'AL': 3_300_000, 'PI': 3_300_000, 'DF': 3_100_000,
            'MS': 2_800_000, 'SE': 2_300_000, 'RO': 1_800_000, 'TO': 1_600_000,
            'AC': 900_000, 'AP': 800_000, 'RR': 600_000
        }
        
        populacao = pop_estado.get(estado, 3_000_000)
        eleitores = int(populacao * 0.75)  # ~75% são eleitores
        
        # Gastos de campanha (correlacionado com votos)
        gasto_campanha = np.random.lognormal(11, 1.5)  # Distribuição log-normal
        
        # Tempo de TV (em segundos)
        tempo_tv = np.random.exponential(30)
        
        # Incumbência (candidato já é deputado)
        incumbente = np.random.choice([0, 1], p=[0.7, 0.3])
        
        # Idade
        idade = int(np.random.normal(48, 12))
        idade = max(25, min(80, idade))  # Limitar entre 25 e 80
        
        # Gênero
        genero = np.random.choice(['M', 'F'], p=[0.7, 0.3])
        
        # Escolaridade (1=fundamental, 2=médio, 3=superior, 4=pós)
        escolaridade = np.random.choice([1, 2, 3, 4], p=[0.05, 0.15, 0.5, 0.3])
        
        # Coligação (1 se está em coligação)
        em_coligacao = np.random.choice([0, 1], p=[0.3, 0.7])
        
        # Número de candidatos na coligação
        tamanho_coligacao = np.random.randint(1, 8) if em_coligacao else 1
        
        # Votos (modelo simplificado com múltiplos fatores)
        base_votos = np.random.lognormal(8, 2)
        
        # Ajustes por fatores
        fator_gasto = (gasto_campanha / 100000) ** 0.3
        fator_tv = (tempo_tv / 30) ** 0.2
        fator_incumbente = 1.5 if incumbente else 1.0
        fator_coligacao = 1.2 if em_coligacao else 1.0
        fator_estado = (populacao / 10_000_000) ** 0.5
        
        votos = int(base_votos * fator_gasto * fator_tv * fator_incumbente * 
                   fator_coligacao * fator_estado)
        votos = max(100, votos)  # Mínimo de 100 votos
        
        # Percentual de votos no estado
        percentual_votos = (votos / eleitores) * 100
        
        dados.append({
            'candidato_id': f'CAND_{i+1:04d}',
            'nome': f'Candidato {i+1}',
            'partido': partido,
            'ideologia': ideologia_partido[partido],
            'estado': estado,
            'tipo_eleicao': tipo,
            'ano': ano,
            'populacao_estado': populacao,
            'eleitores_estado': eleitores,
            'votos': votos,
            'percentual_votos': percentual_votos,
            'gasto_campanha': gasto_campanha,
            'tempo_tv_segundos': tempo_tv,
            'incumbente': incumbente,
            'idade': idade,
            'genero': genero,
            'escolaridade': escolaridade,
            'em_coligacao': em_coligacao,
            'tamanho_coligacao': tamanho_coligacao,
            'eleito': 0  # Será calculado depois
        })
    
    df = pd.DataFrame(dados)
    
    # Calcular quem foi eleito baseado no quociente eleitoral
    for estado in estados:
        df_estado = df[df['estado'] == estado].copy()
        
        # Número de vagas (simplificado - proporcional à população)
        if tipo == 'federal':
            vagas = max(8, min(70, int(df_estado['populacao_estado'].iloc[0] / 500000)))
        else:
            vagas = max(24, min(94, int(df_estado['populacao_estado'].iloc[0] / 200000)))
        
        # Ordenar por votos e marcar os eleitos
        df_estado_sorted = df_estado.sort_values('votos', ascending=False)
        indices_eleitos = df_estado_sorted.head(vagas).index
        df.loc[indices_eleitos, 'eleito'] = 1
    
    return df


def gerar_dados_historicos(anos=[2010, 2014, 2018, 2022], n_candidatos_por_ano=400):
    """
    Gera série histórica de dados eleitorais.
    
    Parâmetros:
    -----------
    anos : list
        Lista de anos eleitorais
    n_candidatos_por_ano : int
        Número de candidatos por ano
    
    Retorna:
    --------
    pd.DataFrame
        DataFrame com dados históricos
    """
    dados_historicos = []
    
    for ano in anos:
        # Adicionar tendência temporal
        seed_ano = 42 + (ano - 2010)
        np.random.seed(seed_ano)
        
        df_ano = gerar_dados_eleitorais(
            n_candidatos=n_candidatos_por_ano,
            ano=ano,
            tipo='federal'
        )
        dados_historicos.append(df_ano)
    
    return pd.concat(dados_historicos, ignore_index=True)


def gerar_series_temporais_partido(partido, anos=range(2010, 2027, 4)):
    """
    Gera série temporal de desempenho de um partido.
    
    Parâmetros:
    -----------
    partido : str
        Sigla do partido
    anos : range ou list
        Anos para gerar dados
    
    Retorna:
    --------
    pd.DataFrame
        Série temporal com votos, cadeiras, etc.
    """
    np.random.seed(hash(partido) % 2**32)
    
    dados_serie = []
    
    # Tendência base do partido (crescimento ou declínio)
    tendencia = np.random.choice([-0.02, 0, 0.02, 0.05])
    base_votos = np.random.uniform(5, 15)  # Percentual base
    
    for i, ano in enumerate(anos):
        # Adicionar tendência e ruído
        percentual = base_votos * (1 + tendencia * i) + np.random.normal(0, 1)
        percentual = max(0.5, min(30, percentual))  # Limitar entre 0.5% e 30%
        
        votos_totais = int(percentual * 1_500_000)  # ~150M eleitores
        cadeiras = int(percentual * 513 / 100)  # 513 deputados federais
        
        dados_serie.append({
            'ano': ano,
            'partido': partido,
            'votos': votos_totais,
            'percentual': percentual,
            'cadeiras': cadeiras,
            'percentual_cadeiras': (cadeiras / 513) * 100
        })
    
    return pd.DataFrame(dados_serie)


def gerar_matriz_transferencia_votos(n_partidos=10):
    """
    Gera matriz de transferência de votos entre partidos (Cadeia de Markov).
    
    Parâmetros:
    -----------
    n_partidos : int
        Número de partidos
    
    Retorna:
    --------
    np.ndarray
        Matriz de transição (n_partidos x n_partidos)
    """
    partidos = ['PT', 'PL', 'PP', 'UNIÃO', 'PSD', 'REPUBLICANOS', 'MDB', 'PSDB', 'PDT', 'PSB'][:n_partidos]
    
    # Criar matriz com probabilidades
    matriz = np.random.dirichlet(np.ones(n_partidos), size=n_partidos)
    
    # Aumentar probabilidade de manter no mesmo partido (diagonal)
    for i in range(n_partidos):
        matriz[i, i] += 0.3
        matriz[i] = matriz[i] / matriz[i].sum()  # Renormalizar
    
    return pd.DataFrame(matriz, index=partidos, columns=partidos)


def gerar_dados_municipais(estado='SP', n_municipios=50):
    """
    Gera dados eleitorais em nível municipal.
    
    Parâmetros:
    -----------
    estado : str
        Sigla do estado
    n_municipios : int
        Número de municípios a simular
    
    Retorna:
    --------
    pd.DataFrame
        Dados municipais
    """
    np.random.seed(42)
    
    dados = []
    partidos = ['PT', 'PL', 'PP', 'UNIÃO', 'PSD', 'REPUBLICANOS', 'MDB', 'PSDB']
    
    for i in range(n_municipios):
        municipio = f'Município_{i+1}'
        populacao = int(np.random.lognormal(10, 1.5))
        
        for partido in partidos:
            votos = int(np.random.lognormal(7, 2))
            
            dados.append({
                'estado': estado,
                'municipio': municipio,
                'populacao': populacao,
                'partido': partido,
                'votos': votos,
                'percentual': (votos / populacao) * 100
            })
    
    return pd.DataFrame(dados)
