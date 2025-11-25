import pandas as pd
import numpy as np

def gerar_dados_simulados(n_domicilios=10000, seed=42):
    """
    Gera um DataFrame simulando dados da PNAD Contínua.
    
    Parâmetros:
        n_domicilios (int): Número de domicílios a simular.
        seed (int): Semente aleatória para reprodutibilidade.
        
    Retorna:
        pd.DataFrame: DataFrame com dados socioeconômicos.
    """
    np.random.seed(seed)
    
    # UFs e Regiões
    ufs = ['AC', 'AL', 'AP', 'AM', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA', 'MT', 'MS', 'MG', 'PA', 'PB', 'PR', 'PE', 'PI', 'RJ', 'RN', 'RS', 'RO', 'RR', 'SC', 'SP', 'SE', 'TO']
    pesos_uf = np.random.dirichlet(np.ones(len(ufs)), size=1)[0] # Distribuição aleatória de população por UF
    
    dados = {
        'id_domicilio': range(1, n_domicilios + 1),
        'uf': np.random.choice(ufs, n_domicilios, p=pesos_uf),
        'zona': np.random.choice(['Urbana', 'Rural'], n_domicilios, p=[0.85, 0.15]),
        'tamanho_familia': np.random.poisson(3, n_domicilios) + 1, # Mínimo 1 pessoa
    }
    
    df = pd.DataFrame(dados)
    
    # Renda Domiciliar (Log-Normal para simular desigualdade)
    # Média e desvio padrão variam por UF (simplificação)
    df['renda_domiciliar_total'] = np.random.lognormal(mean=7.5, sigma=1.2, size=n_domicilios)
    
    # Ajuste de renda por zona (Rural tende a ser menor)
    df.loc[df['zona'] == 'Rural', 'renda_domiciliar_total'] *= 0.7
    
    # Renda per capita
    df['renda_pc'] = df['renda_domiciliar_total'] / df['tamanho_familia']
    
    # Indicadores de Privação (0 = Não, 1 = Sim) - Probabilidades baseadas em renda
    # Quanto maior a renda, menor a chance de privação
    prob_privacao = 1 / (1 + np.exp((df['renda_pc'] - 500) / 200)) # Sigmoide invertida
    
    df['acesso_agua_potavel'] = np.random.binomial(1, 1 - prob_privacao * 0.8) # 1 = Tem acesso
    df['saneamento_basico'] = np.random.binomial(1, 1 - prob_privacao * 0.9)
    df['energia_eletrica'] = np.random.binomial(1, 1 - prob_privacao * 0.2) # Quase todos têm
    df['internet'] = np.random.binomial(1, 1 - prob_privacao * 0.7)
    
    # Escolaridade do Chefe (Anos de estudo)
    # Correlacionado com renda
    df['anos_estudo_chefe'] = (np.log(df['renda_pc']) * 2).astype(int).clip(0, 20)
    
    # Saúde (Autoavaliação: 1=Ruim, 2=Regular, 3=Bom)
    df['saude_autoavaliacao'] = np.random.choice([1, 2, 3], n_domicilios, p=[0.1, 0.3, 0.6])
    
    return df

if __name__ == "__main__":
    df = gerar_dados_simulados()
    print(df.head())
    print(df.describe())
