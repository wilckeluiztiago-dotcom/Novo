import sys
import os
import pandas as pd

# Adicionar diretório raiz ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modelo.dinamica import SimulacaoVarejo

def verificar():
    print("Verificando Modelo de Varejo...")
    # Simulação: Sudeste, Preço 120, Mkt 5000, Estoque Inicial 5000, Meta 10000
    sim = SimulacaoVarejo('Sudeste', 120.0, 5000.0, 5000, 10000)
    df = sim.solver(t_max=12) # 1 ano
    
    total_vendas = df['vendas'].sum()
    total_receita = df['receita'].sum()
    total_lucro = df['lucro'].sum()
    
    print(f"Região: Sudeste")
    print(f"Vendas Totais (12 meses): {total_vendas:.0f}")
    print(f"Receita Total: R$ {total_receita:.2f}")
    print(f"Lucro Total: R$ {total_lucro:.2f}")
    print("Simulação concluída com sucesso.")

if __name__ == "__main__":
    verificar()
