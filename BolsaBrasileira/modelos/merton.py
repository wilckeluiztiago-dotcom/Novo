"""
Modelo de Merton (Jump Diffusion)

Implementação matemática do modelo de difusão com saltos para ativos financeiros.
Captura "Cisnes Negros" e caudas gordas na distribuição de retornos.

Autor: Luiz Tiago Wilcke
"""

import numpy as np
from numba import jit
from typing import Dict, Tuple

class ModeloMerton:
    def __init__(self, parametros: Dict[str, float]):
        """
        Inicializa o modelo MJD.
        
        Args:
            parametros:
                S0: Preço inicial
                mu: Deriva (drift) anual
                sigma: Volatilidade difusiva
                lambda_j: Intensidade dos saltos (Poisson)
                mu_j: Média do log-salto
                sigma_j: Desvio padrão do log-salto
        """
        self.parametros = parametros
        
    def simular(self, T: float, dt: float, num_cenarios: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simula trajetórias de preço.
        """
        N = int(T / dt)
        return self._simular_numba(
            self.parametros['S0'], self.parametros['mu'], self.parametros['sigma'],
            self.parametros['lambda_j'], self.parametros['mu_j'], self.parametros['sigma_j'],
            T, dt, N, num_cenarios
        )

    @staticmethod
    @jit(nopython=True, parallel=False, fastmath=True)
    def _simular_numba(S0, mu, sigma, lambda_j, mu_j, sigma_j, T, dt, N, num_cenarios):
        """
        Núcleo de simulação MJD otimizado.
        
        SDE: dS/S = (mu - lambda*k)dt + sigma*dW + (e^J - 1)dN
        Solução Exata (Log-Preço):
        ln S_t = ln S_0 + (mu - 0.5*sigma^2 - lambda*k)*t + sigma*W_t + sum(J_i)
        """
        precos = np.zeros((N + 1, num_cenarios))
        precos[0, :] = S0
        
        sqrt_dt = np.sqrt(dt)
        
        # Compensador de martingale para o drift
        # k = E[e^J - 1]
        k = np.exp(mu_j + 0.5 * sigma_j**2) - 1
        drift_corrigido = (mu - 0.5 * sigma**2 - lambda_j * k) * dt
        
        log_precos = np.full(num_cenarios, np.log(S0))
        
        for t in range(N):
            # 1. Componente Difusivo (Browniano)
            Z = np.random.standard_normal(num_cenarios)
            difusao = sigma * Z * sqrt_dt
            
            # 2. Componente de Salto (Poisson Composto)
            # Número de saltos neste passo dt
            n_saltos = np.random.poisson(lambda_j * dt, num_cenarios)
            
            saltos_acumulados = np.zeros(num_cenarios)
            for i in range(num_cenarios):
                if n_saltos[i] > 0:
                    # Soma J ~ N(mu_j, sigma_j)
                    saltos = np.random.normal(mu_j, sigma_j, n_saltos[i])
                    saltos_acumulados[i] = np.sum(saltos)
            
            # Atualiza Log-Preço
            log_precos += drift_corrigido + difusao + saltos_acumulados
            
            precos[t + 1, :] = np.exp(log_precos)
            
        tempos = np.linspace(0, T, N + 1)
        return tempos, precos
