import pandas as pd
import numpy as np

def calcular_linha_pobreza(df, metodo='relativo', percentual_mediana=0.6, valor_absoluto=600):
    """
    Calcula o valor da linha de pobreza.
    
    Parâmetros:
        df (pd.DataFrame): DataFrame com dados.
        metodo (str): 'relativo' ou 'absoluto'.
        percentual_mediana (float): Percentual da mediana para linha relativa.
        valor_absoluto (float): Valor fixo para linha absoluta.
        
    Retorna:
        float: Valor da linha de pobreza.
    """
    if metodo == 'relativo':
        mediana = df['renda_pc'].median()
        return mediana * percentual_mediana
    elif metodo == 'absoluto':
        return valor_absoluto
    else:
        raise ValueError("Método deve ser 'relativo' ou 'absoluto'")

def classificar_pobreza(df, linha_pobreza):
    """
    Cria uma coluna binária indicando se o domicílio é pobre.
    
    Parâmetros:
        df (pd.DataFrame): DataFrame com dados.
        linha_pobreza (float): Valor de corte.
        
    Retorna:
        pd.DataFrame: DataFrame com coluna 'is_pobre'.
    """
    df['is_pobre'] = (df['renda_pc'] < linha_pobreza).astype(int)
    return df

def tratar_valores_nulos(df):
    """
    Trata valores nulos no DataFrame.
    
    Parâmetros:
        df (pd.DataFrame): DataFrame bruto.
        
    Retorna:
        pd.DataFrame: DataFrame limpo.
    """
    # Preenchimento simples para simulação
    df = df.fillna(df.median(numeric_only=True))
    return df
