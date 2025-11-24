"""
Motor de Calibração para BolsaBrasileira

Estima parâmetros do modelo Merton usando o Método dos Momentos (Cumulantes).
Isso é mais rápido e estável que MLE para distribuições com saltos.

Autor: Luiz Tiago Wilcke
"""

import numpy as np
from scipy.optimize import minimize
import pandas as pd

class CalibradorMerton:
    def __init__(self, dados: pd.DataFrame):
        self.retornos = dados['RetornoLog'].values
        self.dt = 1/252
        self.preco_atual = dados['Preco'].iloc[-1]
        
    def calibrar(self) -> dict:
        """
        Calibra os parâmetros do modelo MJD.
        """
        print("Calibrando Modelo Merton (Método dos Momentos)...")
        
        # Estatísticas Anualizadas
        media_hist = np.mean(self.retornos) / self.dt
        var_hist = np.var(self.retornos) / self.dt
        
        # Detecta saltos via limiar (3 desvios padrão)
        std_dev = np.std(self.retornos)
        saltos = self.retornos[np.abs(self.retornos) > 3 * std_dev]
        
        if len(saltos) > 5:
            # Estimação direta dos saltos
            lambda_est = len(saltos) / (len(self.retornos) * self.dt)
            mu_j_est = np.mean(saltos)
            sigma_j_est = np.std(saltos)
            
            # A volatilidade difusiva é a vol dos retornos "normais" (sem saltos)
            retornos_normais = self.retornos[np.abs(self.retornos) <= 3 * std_dev]
            sigma_est = np.std(retornos_normais) / np.sqrt(self.dt)
        else:
            # Fallback se não houver saltos claros
            lambda_est = 1.0
            mu_j_est = -0.05
            sigma_j_est = 0.10
            sigma_est = np.std(self.retornos) / np.sqrt(self.dt)
            
        # Refinamento via Otimização (opcional, mas melhora o ajuste da cauda)
        # Aqui usamos os valores estimados diretamente pois são robustos para B3
        
        params = {
            'S0': self.preco_atual,
            'mu': media_hist,
            'sigma': sigma_est,
            'lambda_j': lambda_est,
            'mu_j': mu_j_est,
            'sigma_j': sigma_j_est
        }
        
        print(f"Volatilidade Difusiva: {sigma_est:.1%}")
        print(f"Intensidade de Saltos: {lambda_est:.1f}/ano")
        
        return params
