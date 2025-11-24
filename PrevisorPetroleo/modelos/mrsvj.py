"""
Modelo MRSVJ (Mean-Reverting Stochastic Volatility with Jumps)

Implementação matemática do modelo para commodities (Petróleo).
Combina reversão à média (Ornstein-Uhlenbeck) com volatilidade estocástica e saltos.

Autor: Luiz Tiago Wilcke
"""

import numpy as np
from numba import jit
from typing import Dict, Tuple

class ModeloPetroleo:
    def __init__(self, parametros: Dict[str, float]):
        """
        Inicializa o modelo MRSVJ.
        
        Args:
            parametros: Dicionário com parâmetros:
                S0: Preço inicial
                v0: Variância inicial
                theta_S: Log-preço de equilíbrio (Longo Prazo)
                kappa_S: Velocidade de reversão do preço
                theta_v: Variância de longo prazo
                kappa_v: Velocidade de reversão da variância
                xi: Volatilidade da variância (Vol-of-Vol)
                rho: Correlação (Preço x Vol)
                lambda_j: Intensidade dos saltos
                mu_j: Média do tamanho do salto
                sigma_j: Desvio padrão do salto
        """
        self.parametros = parametros
        
    def simular(self, T: float, dt: float, num_trajetorias: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simula trajetórias usando esquema de Euler-Maruyama.
        Retorna (tempos, matriz_precos).
        """
        N = int(T / dt)
        return self._simular_numba(
            self.parametros['S0'], self.parametros['v0'], 
            self.parametros['theta_S'], self.parametros['kappa_S'],
            self.parametros['theta_v'], self.parametros['kappa_v'],
            self.parametros['xi'], self.parametros['rho'],
            self.parametros['lambda_j'], self.parametros['mu_j'], 
            self.parametros['sigma_j'], T, dt, N, num_trajetorias
        )

    @staticmethod
    @jit(nopython=True, parallel=False, fastmath=True)
    def _simular_numba(S0, v0, theta_S, kappa_S, theta_v, kappa_v, xi, rho, 
                       lambda_j, mu_j, sigma_j, T, dt, N, num_trajetorias):
        """
        Núcleo de simulação otimizado com Numba.
        
        Modelo (Schwartz 97 estendido):
        d(ln S) = kappa_S * (theta_S - ln S) * dt + sqrt(v) * dW1 + J dN
        dv = kappa_v * (theta_v - v) * dt + xi * sqrt(v) * dW2
        """
        # Arrays de resultado
        precos = np.zeros((N + 1, num_trajetorias))
        vols = np.zeros((N + 1, num_trajetorias))
        
        # Inicialização (trabalhamos com log-preço X = ln S)
        X = np.log(S0)
        precos[0, :] = S0
        vols[0, :] = v0
        
        sqrt_dt = np.sqrt(dt)
        
        # Estado atual (vetorizado para todas as trajetórias)
        X_t = np.full(num_trajetorias, X)
        v_t = np.full(num_trajetorias, v0)
        
        for t in range(N):
            # Gera choques correlacionados
            z1 = np.random.standard_normal(num_trajetorias)
            z2 = np.random.standard_normal(num_trajetorias)
            
            # Correlaciona z2 com z1
            w1 = z1
            w2 = rho * z1 + np.sqrt(1 - rho**2) * z2
            
            # Garante variância positiva (Full Truncation)
            v_t_pos = np.maximum(v_t, 0.0)
            sqrt_v_t = np.sqrt(v_t_pos)
            
            # 1. Evolução da Volatilidade (Heston / CIR)
            # dv = kappa_v * (theta_v - v) * dt + xi * sqrt(v) * dW2
            dv = kappa_v * (theta_v - v_t_pos) * dt + xi * sqrt_v_t * w2 * sqrt_dt
            v_t_novo = v_t + dv
            
            # 2. Evolução do Log-Preço (Ornstein-Uhlenbeck com Vol Estocástica)
            # dX = kappa_S * (theta_S - X) * dt + sqrt(v) * dW1
            dX_drift = kappa_S * (theta_S - X_t) * dt
            dX_diff = sqrt_v_t * w1 * sqrt_dt
            
            # 3. Componente de Salto (Poisson)
            # Gera número de saltos
            n_saltos = np.random.poisson(lambda_j * dt, num_trajetorias)
            
            salto_total = np.zeros(num_trajetorias)
            for i in range(num_trajetorias):
                if n_saltos[i] > 0:
                    # Soma dos saltos normais
                    salto_total[i] = np.sum(np.random.normal(mu_j, sigma_j, n_saltos[i]))
            
            # Atualiza Log-Preço
            X_t_novo = X_t + dX_drift + dX_diff + salto_total
            
            # Armazena
            X_t = X_t_novo
            v_t = v_t_novo
            
            precos[t + 1, :] = np.exp(X_t)
            vols[t + 1, :] = v_t
            
        tempos = np.linspace(0, T, N + 1)
        return tempos, precos
