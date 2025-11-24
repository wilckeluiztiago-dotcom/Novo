"""
Motor de Calibração para BitcoinPreditor

Ajusta os parâmetros do modelo de Bates aos dados históricos do Bitcoin.
Usa uma abordagem híbrida:
1. Estimação estatística direta para parâmetros de salto e correlação.
2. Otimização numérica para parâmetros de volatilidade.

Autor: Luiz Tiago Wilcke
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import skew, kurtosis
import pandas as pd

class Calibrator:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.returns = data['LogReturn'].values
        self.dt = 1/365  # Dados diários
        
    def calibrate(self) -> dict:
        """
        Realiza a calibração completa.
        Retorna dicionário com parâmetros otimizados.
        """
        print("Iniciando calibração do modelo de Bates...")
        
        # 1. Estimativa Inicial baseada em momentos estatísticos
        # Média e variância dos retornos
        mu_hist = np.mean(self.returns) / self.dt
        var_hist = np.var(self.returns) / self.dt
        
        # Theta (média da variância de longo prazo) é aprox a variância histórica
        theta_est = var_hist
        
        # V0 é a variância recente (últimos 30 dias)
        v0_est = np.var(self.returns[-30:]) / self.dt
        
        # 2. Detecção de Saltos (Heurística simples)
        # Retornos > 3 desvios padrão são considerados saltos
        std_dev = np.std(self.returns)
        jumps = self.returns[np.abs(self.returns) > 3 * std_dev]
        
        if len(jumps) > 0:
            lambda_est = len(jumps) / (len(self.returns) * self.dt)
            mu_j_est = np.mean(jumps)
            sigma_j_est = np.std(jumps)
        else:
            lambda_est = 0.1
            mu_j_est = 0.0
            sigma_j_est = 0.05
            
        # 3. Otimização para Kappa, Xi, Rho (Parâmetros de Heston)
        # Função objetivo: Matching de momentos (Variância da Variância e Skewness)
        # Esta é uma simplificação; calibração rigorosa usaria preços de opções,
        # mas para séries temporais puras, usamos GMM (Generalized Method of Moments)
        
        def objective(params):
            kappa, xi, rho = params
            
            # Penalidades para restrições
            if kappa <= 0 or xi <= 0 or abs(rho) >= 1:
                return 1e6
                
            # Momentos teóricos aproximados do Heston (sem saltos para simplificar otimização rápida)
            # Skewness teórica ~ f(rho, xi, kappa)
            # Kurtosis teórica ~ f(xi, kappa)
            
            # Na prática, para cripto, focamos na dinâmica da volatilidade
            # Calculamos a série de volatilidade histórica (GARCH proxy ou Rolling Std)
            vol_proxy = pd.Series(self.returns).rolling(30).std().dropna().values * np.sqrt(365)
            
            # Propriedades da vol proxy
            vol_mean = np.mean(vol_proxy**2) # Deve ser perto de theta
            vol_std = np.std(vol_proxy**2)   # Relacionado a xi
            
            # Erro quadrático
            err_theta = (theta_est - vol_mean)**2
            
            # A variância da variância no modelo Heston é xi^2 * v / kappa (aprox)
            # Tentamos ajustar xi e kappa para bater com a volatilidade da vol observada
            # Var[v] = theta * xi^2 / (2*kappa)
            theo_var_v = theta_est * xi**2 / (2*kappa)
            obs_var_v = np.var(vol_proxy**2)
            
            err_var = (theo_var_v - obs_var_v)**2
            
            return err_var + err_theta
            
        # Otimização
        res = minimize(objective, x0=[2.0, 0.5, -0.5], method='Nelder-Mead')
        kappa_opt, xi_opt, rho_opt = res.x
        
        # Refinamento de Rho baseado na correlação histórica Retorno vs Vol
        # Calcula correlação rolling
        vol_series = pd.Series(self.returns).rolling(30).std()
        ret_series = pd.Series(self.returns)
        rho_hist = ret_series.corr(vol_series)
        
        if not np.isnan(rho_hist):
            rho_opt = rho_hist
            
        params = {
            'S0': self.data['Price'].iloc[-1],
            'v0': v0_est,
            'theta': theta_est,
            'kappa': abs(kappa_opt),
            'xi': abs(xi_opt),
            'rho': np.clip(rho_opt, -0.99, 0.99),
            'r': 0.05, # Taxa fixa por enquanto
            'lambda_j': max(0.1, lambda_est),
            'mu_j': mu_j_est,
            'sigma_j': max(0.01, sigma_j_est)
        }
        
        print("Calibração concluída.")
        print(f"Parâmetros: {params}")
        return params
