import pandas as pd
import numpy as np
from dados.gerador import gerar_dados_simulados
from dados.processamento import calcular_linha_pobreza, classificar_pobreza
from indicadores.pobreza import IndicadoresFGT
from indicadores.desigualdade import Desigualdade
from modelos.multidimensional import PobrezaMultidimensional

def main():
    print("="*50)
    print("   INDICE POBREZA - SISTEMA DE ANÁLISE")
    print("   Autor: Luiz Tiago Wilcke")
    print("="*50)
    
    # 1. Gerar Dados
    print("\n[1] Gerando dados simulados (PNAD)...")
    df = gerar_dados_simulados(n_domicilios=20000)
    print(f"    Total de domicílios: {len(df)}")
    
    # 2. Processamento
    print("\n[2] Calculando Linha de Pobreza (60% da mediana)...")
    lp = calcular_linha_pobreza(df, metodo='relativo', percentual_mediana=0.6)
    df = classificar_pobreza(df, lp)
    print(f"    Linha de Pobreza: R$ {lp:.2f}")
    
    # 3. Indicadores Clássicos
    print("\n[3] Calculando Indicadores Clássicos...")
    fgt = IndicadoresFGT(df, lp)
    desig = Desigualdade(df)
    
    print(f"    Incidência (P0): {fgt.incidencia():.2%}")
    print(f"    Hiato (P1):      {fgt.hiato():.2%}")
    print(f"    Severidade (P2): {fgt.severidade():.4f}")
    print(f"    Índice de Gini:  {desig.gini():.3f}")
    print(f"    Theil T:         {desig.theil_t():.3f}")
    
    # 4. Pobreza Multidimensional
    print("\n[4] Calculando MPI (Alkire-Foster)...")
    # Definindo pesos iguais para simplificação
    pesos = {
        'acesso_agua_potavel': 0.25,
        'saneamento_basico': 0.25,
        'energia_eletrica': 0.25,
        'internet': 0.25
    }
    # Criar colunas de privação (inverso do acesso)
    # Aqui vamos usar as colunas originais mas interpretar 0 como privação no peso?
    # O módulo espera colunas onde 1 = privação.
    # Nossos dados: 1 = acesso.
    # Vamos criar colunas temporárias de privação.
    df_mpi = df.copy()
    colunas_privacao = []
    for col in pesos.keys():
        col_priv = f"priv_{col}"
        df_mpi[col_priv] = 1 - df_mpi[col]
        colunas_privacao.append(col_priv)
        
    pesos_priv = {k: v for k, v in zip(colunas_privacao, pesos.values())}
    
    mpi_calc = PobrezaMultidimensional(df_mpi, colunas_privacao, pesos_priv, corte_k=0.33)
    indices_mpi = mpi_calc.calcular_indices()
    
    print(f"    Incidência Multidimensional (H): {indices_mpi['H']:.2%}")
    print(f"    Intensidade Média (A):           {indices_mpi['A']:.2%}")
    print(f"    MPI (M0):                        {indices_mpi['MPI']:.3f}")
    
    print("\n" + "="*50)
    print("Análise concluída com sucesso!")
    print("Para visualizar o dashboard, execute: streamlit run visualizacao/dashboard.py")
    print("="*50)

if __name__ == "__main__":
    main()
