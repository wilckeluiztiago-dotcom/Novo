"""
Motor de Calibração para PrevisorPetroleo

Ajusta os parâmetros do modelo MRSVJ aos dados históricos.
Foca na reversão à média e volatilidade.

Autor: Luiz Tiago Wilcke
"""

import numpy as np
from scipy.optimize import minimize
import pandas as pd

class Calibrador:
    def __init__(self, dados: pd.DataFrame):
        self.dados = dados
        self.precos = dados['Preco'].values
        self.retornos = dados['RetornoLog'].values
        self.dt = 1/252  # Dados diários (dias úteis)
        
    def calibrar(self) -> dict:
        """
        Realiza a calibração dos parâmetros.
        """
        print("Iniciando calibração do modelo MRSVJ...")
        
        # 1. Calibração da Reversão à Média (Processo Ornstein-Uhlenbeck)
        # d(ln S) = kappa * (theta - ln S) dt + sigma dW
        # Discretizado: ln S_{t+1} = ln S_t + kappa*(theta - ln S_t)*dt + erro
        # Regressão linear: Y = alpha + beta * X + erro
        # Onde Y = ln S_{t+1} - ln S_t, X = ln S_t
        # beta = -kappa * dt
        # alpha = kappa * theta * dt
        
        log_precos = np.log(self.precos)
        X = log_precos[:-1]
        Y = log_precos[1:] - log_precos[:-1]
        
        # Regressão Linear Simples
        A = np.vstack([X, np.ones(len(X))]).T
        beta, alpha = np.linalg.lstsq(A, Y, rcond=None)[0]
        
        # Recupera parâmetros OU
        kappa_S = -beta / self.dt
        theta_S = alpha / (kappa_S * self.dt)
        
        # Limites de segurança
        kappa_S = max(0.1, min(kappa_S, 10.0))
        
        # 2. Estimativa de Volatilidade
        # Variância histórica média
        var_hist = np.var(self.retornos) / self.dt
        theta_v = var_hist
        v0 = np.var(self.retornos[-21:]) / self.dt # Volatilidade do último mês
        
        # 3. Detecção de Saltos (Outliers)
        std_dev = np.std(self.retornos)
        saltos = self.retornos[np.abs(self.retornos) > 3 * std_dev]
        
        if len(saltos) > 0:
            lambda_j = len(saltos) / (len(self.retornos) * self.dt)
            mu_j = np.mean(saltos)
            sigma_j = np.std(saltos)
        else:
            lambda_j = 0.5
            mu_j = 0.0
            sigma_j = 0.05
            
        # Parâmetros padrão para vol estocástica (difícil calibrar sem opções)
        kappa_v = 2.0
        xi = 0.3
        rho = -0.3 # Commodities geralmente têm correlação negativa (efeito inverso)
        
        parametros = {
            'S0': self.precos[-1],
            'v0': v0,
            'theta_S': theta_S,
            'kappa_S': kappa_S,
            'theta_v': theta_v,
            'kappa_v': kappa_v,
            'xi': xi,
            'rho': rho,
            'lambda_j': lambda_j,
            'mu_j': mu_j,
            'sigma_j': sigma_j
        }
        
        print("Calibração concluída.")
        print(f"Equilíbrio estimado: ${np.exp(theta_S):.2f}")
        print(f"Velocidade de reversão: {kappa_S:.2f}")
        
        return parametros
