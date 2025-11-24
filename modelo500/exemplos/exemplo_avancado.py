"""
Exemplo Avançado de Uso do Modelo de Desemprego

Demonstra como criar um pipeline personalizado de simulação,
análise e visualização sem usar a interface de linha de comando.

Autor: Luiz Tiago Wilcke
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from modelos_sde import ModeloGoodwinEstocastico
from simulador import SimuladorSDE
from visualizador import Visualizador

def main():
    print("Iniciando simulação avançada...")
    
    # 1. Configuração personalizada do modelo
    params = {
        'gamma': 0.03,      # Produtividade maior
        'alpha': 0.04,
        'beta': 0.035,
        'delta': 0.025,     # Depreciação menor
        'sigma_u': 0.015,   # Mais volatilidade no emprego
        'sigma_v': 0.01,
        'u0': 0.90,         # Começa com 10% de desemprego
        'v0': 0.60
    }
    
    modelo = ModeloGoodwinEstocastico(params)
    
    # 2. Simulação com método de Milstein (mais preciso)
    sim = SimuladorSDE(modelo, seed=123)
    
    print("Simulando cenários...")
    # Cenário A: Curto prazo (5 anos)
    t_curto, traj_curto = sim.simular_multiplas_trajetorias(
        T=5.0, N=500, num_trajetorias=50, metodo='milstein'
    )
    
    # Cenário B: Longo prazo (20 anos)
    t_longo, traj_longo = sim.simular_multiplas_trajetorias(
        T=20.0, N=2000, num_trajetorias=50, metodo='milstein'
    )
    
    # 3. Análise personalizada
    desemprego_longo = sim.calcular_desemprego_trajetorias(traj_longo)
    prob_crise = np.mean(desemprego_longo > 0.15, axis=0)  # Probabilidade de desemprego > 15%
    
    print(f"Probabilidade máxima de crise (>15% desemprego): {np.max(prob_crise):.2%}")
    
    # 4. Visualização customizada
    vis = Visualizador()
    
    # Plotar probabilidade de crise ao longo do tempo
    plt.figure(figsize=(12, 6))
    plt.plot(t_longo, prob_crise * 100, 'r-', linewidth=2)
    plt.fill_between(t_longo, 0, prob_crise * 100, alpha=0.3, color='red')
    plt.title('Probabilidade de Crise de Desemprego (>15%) ao Longo do Tempo')
    plt.xlabel('Anos')
    plt.ylabel('Probabilidade (%)')
    plt.grid(True, alpha=0.3)
    
    caminho = 'resultados/analise_risco_avancada.png'
    if not os.path.exists('resultados'):
        os.makedirs('resultados')
    plt.savefig(caminho)
    print(f"Gráfico de risco salvo em: {caminho}")

if __name__ == "__main__":
    main()
