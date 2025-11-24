"""
MÓDULO DE SOLVERS NUMÉRICOS - Sistema de Modelagem Climática
=============================================================

Métodos numéricos para integração temporal e resolução de EDPs

Autor: Luiz Tiago Wilcke
Data: 2025-11-24
"""

import numpy as np
from typing import Callable, Tuple


class SolverEuler:
    """Solver de Euler explícito para EDOs"""
    
    @staticmethod
    def passo(
        y: np.ndarray,
        t: float,
        dt: float,
        dy_dt: Callable
    ) -> np.ndarray:
        """
        Um passo do método de Euler.
        
        y_{n+1} = y_n + dt × f(t_n, y_n)
        
        Args:
            y: Estado atual
            t: Tempo atual
            dt: Passo de tempo
            dy_dt: Função derivada dy/dt = f(t, y)
        
        Returns:
            Novo estado
        """
        return y + dt * dy_dt(t, y)


class SolverRungeKutta4:
    """Solver Runge-Kutta de 4ª ordem"""
    
    @staticmethod
    def passo(
        y: np.ndarray,
        t: float,
        dt: float,
        dy_dt: Callable
    ) -> np.ndarray:
        """
        Um passo do método RK4.
        
        Args:
            y: Estado atual
            t: Tempo atual
            dt: Passo de tempo
            dy_dt: Função derivada
        
        Returns:
            Novo estado
        """
        k1 = dy_dt(t, y)
        k2 = dy_dt(t + dt/2, y + dt/2 * k1)
        k3 = dy_dt(t + dt/2, y + dt/2 * k2)
        k4 = dy_dt(t + dt, y + dt * k3)
        
        return y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)


class SolverDifusao:
    """Solver para equação de difusão"""
    
    @staticmethod
    def passo_explicito(
        campo: np.ndarray,
        difusividade: float,
        dx: float,
        dt: float
    ) -> np.ndarray:
        """
        Difusão 1D explícita.
        
        ∂T/∂t = κ ∂²T/∂x²
        
        Args:
            campo: Campo a difundir
            difusividade: Coeficiente de difusão (m²/s)
            dx: Espaçamento espacial (m)
            dt: Passo tempo (s)
        
        Returns:
            Campo difundido
        """
        # Número de Fourier (critério de estabilidade: Fo ≤ 0.5)
        Fo = difusividade * dt / dx**2
        
        if Fo > 0.5:
            print(f"Aviso: Número de Fourier = {Fo:.3f} > 0.5. Instável!")
        
        # Diferenças finitas centrais
        campo_novo = campo.copy()
        campo_novo[1:-1] = campo[1:-1] + Fo * (
            campo[2:] - 2*campo[1:-1] + campo[:-2]
        )
        
        return campo_novo


class SolverAdveccao:
    """Solver para equação de advecção"""
    
    @staticmethod
    def passo_upwind(
        campo: np.ndarray,
        velocidade: np.ndarray,
        dx: float,
        dt: float
    ) -> np.ndarray:
        """
        Advecção 1D usando esquema upwind.
        
        ∂T/∂t + u ∂T/∂x = 0
        
        Args:
            campo: Campo a advectar
            velocidade: Campo de velocidade
            dx: Espaçamento espacial
            dt: Passo de tempo
        
        Returns:
            Campo advectado
        """
        # Número de Courant (CFL: Co < 1 para estabilidade)
        Co = np.max(np.abs(velocidade)) * dt / dx
        
        if Co > 1.0:
            print(f"Aviso: Número de Courant = {Co:.3f} > 1.0. Instável!")
        
        campo_novo = campo.copy()
        
        for i in range(1, len(campo)-1):
            if velocidade[i] > 0:
                # Upwind à esquerda
                campo_novo[i] = campo[i] - velocidade[i] * dt/dx * (campo[i] - campo[i-1])
            else:
                # Upwind à direita
                campo_novo[i] = campo[i] - velocidade[i] * dt/dx * (campo[i+1] - campo[i])
        
        return campo_novo


def integrar_temporal(
    estado_inicial: np.ndarray,
    t_inicial: float,
    t_final: float,
    dt: float,
    funcao_derivada: Callable,
    metodo: str = 'rk4'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Integra sistema de EDOs no tempo.
    
    Args:
        estado_inicial: Condição inicial
        t_inicial: Tempo inicial
        t_final: Tempo final
        dt: Passo de tempo
        funcao_derivada: Função f(t, y) que retorna dy/dt
        metodo: 'euler' ou 'rk4'
    
    Returns:
        Tuple (tempos, estados)
    """
    n_passos = int((t_final - t_inicial) / dt)
    tempos = np.linspace(t_inicial, t_final, n_passos + 1)
    
    # Array para armazenar estados
    if estado_inicial.ndim == 0:
        estados = np.zeros(n_passos + 1)
    else:
        shape = (n_passos + 1,) + estado_inicial.shape
        estados = np.zeros(shape)
    
    estados[0] = estado_inicial
    
    # Escolher solver
    if metodo == 'euler':
        solver = SolverEuler()
    elif metodo == 'rk4':
        solver = SolverRungeKutta4()
    else:
        raise ValueError(f"Método desconhecido: {metodo}")
    
    # Integrar
    for i in range(n_passos):
        estados[i+1] = solver.passo(estados[i], tempos[i], dt, funcao_derivada)
    
    return tempos, estados
