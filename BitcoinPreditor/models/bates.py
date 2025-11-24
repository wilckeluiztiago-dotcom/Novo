"""
Modelo de Bates (Stochastic Volatility with Jumps)

Implementação matemática do modelo de Bates para simulação de preços de ativos.
Este modelo combina a volatilidade estocástica de Heston com saltos de Merton.

Autor: Luiz Tiago Wilcke
"""

import numpy as np
from numba import jit
from typing import Dict, Tuple

class BatesModel:
    def __init__(self, params: Dict[str, float]):
        """
        Inicializa o modelo de Bates.
        
        Args:
            params: Dicionário com parâmetros:
                S0: Preço inicial
                v0: Variância inicial
                kappa: Velocidade de reversão à média da variância
                theta: Variância de longo prazo
                xi: Volatilidade da variância (Vol-of-Vol)
                rho: Correlação Browniana (Preço x Vol)
                r: Taxa livre de risco
                lambda_j: Intensidade dos saltos (Poisson)
                mu_j: Média do log-tamanho do salto
                sigma_j: Desvio padrão do log-tamanho do salto
        """
        self.params = params
        self._validate_params()
        
    def _validate_params(self):
        # Condição de Feller para garantir variância positiva (idealmente 2*kappa*theta > xi^2)
        # Mas na prática simulamos com Full Truncation ou Reflection se violado
        pass

    def simulate(self, T: float, dt: float, num_paths: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simula trajetórias usando esquema de Euler-Maruyama com discretização de saltos.
        Retorna (tempos, caminhos_preço).
        """
        N = int(T / dt)
        return self._simulate_numba(
            self.params['S0'], self.params['v0'], self.params['kappa'], 
            self.params['theta'], self.params['xi'], self.params['rho'], 
            self.params['r'], self.params['lambda_j'], self.params['mu_j'], 
            self.params['sigma_j'], T, dt, N, num_paths
        )

    @staticmethod
    @jit(nopython=True, parallel=False, fastmath=True)
    def _simulate_numba(S0, v0, kappa, theta, xi, rho, r, lambda_j, mu_j, sigma_j, T, dt, N, num_paths):
        """
        Núcleo de simulação otimizado com Numba.
        """
        # Arrays de resultado
        prices = np.zeros((N + 1, num_paths))
        vols = np.zeros((N + 1, num_paths))
        
        prices[0, :] = S0
        vols[0, :] = v0
        
        sqrt_dt = np.sqrt(dt)
        
        # Correção de drift para os saltos (compensador de martingale)
        # k = E[e^J - 1] = exp(mu_j + 0.5*sigma_j^2) - 1
        k = np.exp(mu_j + 0.5 * sigma_j**2) - 1
        drift_correction = lambda_j * k
        
        for t in range(N):
            # Gera choques correlacionados para Heston
            z1 = np.random.standard_normal(num_paths)
            z2 = np.random.standard_normal(num_paths)
            
            # Correlaciona z2 com z1
            # dW2 = rho*dW1 + sqrt(1-rho^2)*dZ
            w1 = z1
            w2 = rho * z1 + np.sqrt(1 - rho**2) * z2
            
            # Variância atual
            vt = vols[t, :]
            # Garante não-negatividade (Full Truncation)
            vt_pos = np.maximum(vt, 0.0)
            sqrt_vt = np.sqrt(vt_pos)
            
            # Evolução da Variância (Heston)
            # dv = kappa(theta - v)dt + xi*sqrt(v)*dW2
            dv = kappa * (theta - vt_pos) * dt + xi * sqrt_vt * w2 * sqrt_dt
            vols[t + 1, :] = vt + dv
            
            # Componente de Difusão do Preço
            # dS_diff = (r - lambda*k)S dt + sqrt(v)S dW1
            drift = (r - drift_correction - 0.5 * vt_pos) * dt
            diffusion = sqrt_vt * w1 * sqrt_dt
            
            # Componente de Salto (Merton/Poisson)
            # Gera número de saltos neste passo para cada caminho
            # Aproximação: Probabilidade de 1 salto é lambda*dt
            # Para maior precisão, usamos Poisson
            n_jumps = np.random.poisson(lambda_j * dt, num_paths)
            
            jump_factor = np.zeros(num_paths)
            for i in range(num_paths):
                if n_jumps[i] > 0:
                    # Soma dos log-saltos: J = sum(N(mu_j, sigma_j))
                    total_jump_size = np.sum(np.random.normal(mu_j, sigma_j, n_jumps[i]))
                    jump_factor[i] = total_jump_size
            
            # Atualiza Preço (em log para estabilidade)
            # d(ln S) = (r - lambda*k - 0.5*v)dt + sqrt(v)dW1 + J
            log_ret = drift + diffusion + jump_factor
            prices[t + 1, :] = prices[t, :] * np.exp(log_ret)
            
        times = np.linspace(0, T, N + 1)
        return times, prices
