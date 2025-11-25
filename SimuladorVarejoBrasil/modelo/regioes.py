class RegioesBrasil:
    """
    Parâmetros socioeconômicos aproximados por região do Brasil.
    """
    
    # Dados base (População em milhões, Fator PIB relativo)
    DADOS = {
        'Norte': {
            'populacao': 18.9e6,
            'fator_pib': 0.8, # Menor poder de compra relativo
            'sazonalidade': {'verao': 1.0, 'inverno': 0.95, 'natal': 1.8, 'black_friday': 1.5},
            'custo_logistica': 1.5 # Mais caro
        },
        'Nordeste': {
            'populacao': 57.6e6,
            'fator_pib': 0.7,
            'sazonalidade': {'verao': 1.2, 'inverno': 0.9, 'natal': 1.9, 'black_friday': 1.4},
            'custo_logistica': 1.2
        },
        'Centro-Oeste': {
            'populacao': 16.7e6,
            'fator_pib': 1.1, # Agronegócio forte
            'sazonalidade': {'verao': 1.0, 'inverno': 1.0, 'natal': 1.7, 'black_friday': 1.5},
            'custo_logistica': 1.1
        },
        'Sudeste': {
            'populacao': 89.6e6,
            'fator_pib': 1.4, # Maior concentração de renda
            'sazonalidade': {'verao': 1.0, 'inverno': 1.1, 'natal': 2.2, 'black_friday': 1.8},
            'custo_logistica': 0.8 # Melhor infraestrutura
        },
        'Sul': {
            'populacao': 30.4e6,
            'fator_pib': 1.3,
            'sazonalidade': {'verao': 1.1, 'inverno': 1.3, 'natal': 2.0, 'black_friday': 1.6},
            'custo_logistica': 0.9
        }
    }

    @staticmethod
    def get_parametros(regiao):
        return RegioesBrasil.DADOS.get(regiao, RegioesBrasil.DADOS['Sudeste'])
