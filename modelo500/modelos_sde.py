"""
Módulo de Modelos de Equações Diferenciais Estocásticas para Previsão de Desemprego

Este módulo contém implementações de diversos modelos matemáticos baseados em SDEs
(Stochastic Differential Equations) para modelar e prever a dinâmica do desemprego.

Autor: Luiz Tiago Wilcke
Data: 2025-11-24
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any


class ModeloSDEBase(ABC):
    """
    Classe base abstrata para todos os modelos SDE.
    
    Define a interface comum que todos os modelos devem implementar.
    """
    
    def __init__(self, parametros: Dict[str, float]):
        """
        Inicializa o modelo com parâmetros específicos.
        
        Args:
            parametros: Dicionário com parâmetros do modelo
        """
        self.parametros = parametros
        self.validar_parametros()
    
    @abstractmethod
    def validar_parametros(self) -> None:
        """Valida se os parâmetros estão dentro de limites aceitáveis."""
        pass
    
    @abstractmethod
    def drift(self, t: float, estado: np.ndarray) -> np.ndarray:
        """
        Calcula o termo de drift (tendência determinística) da SDE.
        
        Args:
            t: Tempo atual
            estado: Estado atual do sistema
            
        Returns:
            Vetor de drift
        """
        pass
    
    @abstractmethod
    def difusao(self, t: float, estado: np.ndarray) -> np.ndarray:
        """
        Calcula o termo de difusão (volatilidade) da SDE.
        
        Args:
            t: Tempo atual
            estado: Estado atual do sistema
            
        Returns:
            Matriz de difusão
        """
        pass
    
    @abstractmethod
    def condicao_inicial(self) -> np.ndarray:
        """
        Retorna a condição inicial do sistema.
        
        Returns:
            Vetor de estado inicial
        """
        pass
    
    def dimensao(self) -> int:
        """Retorna a dimensão do sistema."""
        return len(self.condicao_inicial())


class ModeloGoodwinEstocastico(ModeloSDEBase):
    """
    Modelo de Goodwin Estocástico para ciclos econômicos.
    
    Modela a dinâmica entre taxa de emprego e parcela salarial com flutuações
    estocásticas. O modelo captura ciclos predador-presa na economia.
    
    Equações:
    du = [u(γ - α*v)]dt + σ_u*u*dW_1
    dv = [v(β*u - δ)]dt + σ_v*v*dW_2
    
    onde:
    u = taxa de emprego
    v = parcela salarial
    γ = taxa de crescimento da produtividade
    α = sensibilidade dos salários ao emprego
    β = sensibilidade do lucro aos salários
    δ = depreciação
    σ_u, σ_v = volatilidades
    """
    
    def __init__(self, parametros: Dict[str, float] = None):
        if parametros is None:
            # Parâmetros padrão calibrados
            parametros = {
                'gamma': 0.02,      # Crescimento da produtividade
                'alpha': 0.8,       # Sensibilidade salário-emprego
                'beta': 0.9,        # Sensibilidade lucro-salário
                'delta': 0.03,      # Taxa de depreciação
                'sigma_u': 0.05,    # Volatilidade do emprego
                'sigma_v': 0.04,    # Volatilidade dos salários
                'u0': 0.95,         # Emprego inicial (95%)
                'v0': 0.65          # Parcela salarial inicial (65%)
            }
        super().__init__(parametros)
    
    def validar_parametros(self) -> None:
        """Valida parâmetros do modelo de Goodwin."""
        assert self.parametros['gamma'] > 0, "Taxa de crescimento deve ser positiva"
        assert self.parametros['alpha'] > 0, "Alpha deve ser positivo"
        assert self.parametros['beta'] > 0, "Beta deve ser positivo"
        assert self.parametros['delta'] >= 0, "Delta não pode ser negativo"
        assert self.parametros['sigma_u'] >= 0, "Volatilidade não pode ser negativa"
        assert self.parametros['sigma_v'] >= 0, "Volatilidade não pode ser negativa"
        assert 0 < self.parametros['u0'] <= 1, "Emprego inicial deve estar entre 0 e 1"
        assert 0 < self.parametros['v0'] <= 1, "Parcela salarial deve estar entre 0 e 1"
    
    def drift(self, t: float, estado: np.ndarray) -> np.ndarray:
        """Calcula o drift do modelo de Goodwin."""
        u, v = estado
        
        # Previne valores negativos
        u = max(u, 1e-6)
        v = max(v, 1e-6)
        
        du_dt = u * (self.parametros['gamma'] - self.parametros['alpha'] * v)
        dv_dt = v * (self.parametros['beta'] * u - self.parametros['delta'])
        
        return np.array([du_dt, dv_dt])
    
    def difusao(self, t: float, estado: np.ndarray) -> np.ndarray:
        """Calcula a difusão do modelo de Goodwin."""
        u, v = estado
        
        # Previne valores negativos
        u = max(u, 1e-6)
        v = max(v, 1e-6)
        
        # Matriz de difusão diagonal
        return np.array([
            [self.parametros['sigma_u'] * u, 0],
            [0, self.parametros['sigma_v'] * v]
        ])
    
    def condicao_inicial(self) -> np.ndarray:
        """Retorna condição inicial do modelo."""
        return np.array([self.parametros['u0'], self.parametros['v0']])
    
    def calcular_desemprego(self, estado: np.ndarray) -> float:
        """
        Calcula a taxa de desemprego a partir do estado.
        
        Args:
            estado: Vetor [u, v] onde u é a taxa de emprego
            
        Returns:
            Taxa de desemprego (1 - u)
        """
        u = estado[0]
        return 1.0 - u


class ModeloPhillipsEstocastico(ModeloSDEBase):
    """
    Curva de Phillips Estocástica.
    
    Modela a relação estocástica entre desemprego e inflação.
    
    Equações:
    dπ = [θ(π* - π) - κ*u]dt + σ_π*dW_1
    du = [-α*u + β*(u_n - u)]dt + σ_u*dW_2
    
    onde:
    π = taxa de inflação
    u = taxa de desemprego
    π* = inflação esperada
    u_n = taxa natural de desemprego (NAIRU)
    θ = velocidade de ajuste da inflação
    κ = sensibilidade da inflação ao desemprego
    α = persistência do desemprego
    β = velocidade de reversão à média
    """
    
    def __init__(self, parametros: Dict[str, float] = None):
        if parametros is None:
            parametros = {
                'pi_star': 0.04,    # Inflação alvo (4%)
                'u_natural': 0.06,  # NAIRU (6%)
                'theta': 0.5,       # Ajuste inflação
                'kappa': 2.0,       # Sensibilidade Phillips
                'alpha': 0.3,       # Persistência
                'beta': 0.4,        # Reversão à média
                'sigma_pi': 0.02,   # Volatilidade inflação
                'sigma_u': 0.01,    # Volatilidade desemprego
                'pi0': 0.035,       # Inflação inicial
                'u0': 0.055         # Desemprego inicial
            }
        super().__init__(parametros)
    
    def validar_parametros(self) -> None:
        """Valida parâmetros do modelo de Phillips."""
        assert self.parametros['theta'] > 0, "Theta deve ser positivo"
        assert self.parametros['kappa'] > 0, "Kappa deve ser positivo"
        assert self.parametros['alpha'] >= 0, "Alpha deve ser não-negativo"
        assert self.parametros['beta'] >= 0, "Beta deve ser não-negativo"
        assert 0 <= self.parametros['u_natural'] <= 1, "NAIRU inválido"
        assert 0 <= self.parametros['u0'] <= 1, "Desemprego inicial inválido"
    
    def drift(self, t: float, estado: np.ndarray) -> np.ndarray:
        """Calcula o drift do modelo de Phillips."""
        pi, u = estado
        
        dpi_dt = (self.parametros['theta'] * (self.parametros['pi_star'] - pi) - 
                  self.parametros['kappa'] * u)
        
        du_dt = (-self.parametros['alpha'] * u + 
                 self.parametros['beta'] * (self.parametros['u_natural'] - u))
        
        return np.array([dpi_dt, du_dt])
    
    def difusao(self, t: float, estado: np.ndarray) -> np.ndarray:
        """Calcula a difusão do modelo de Phillips."""
        return np.array([
            [self.parametros['sigma_pi'], 0],
            [0, self.parametros['sigma_u']]
        ])
    
    def condicao_inicial(self) -> np.ndarray:
        """Retorna condição inicial."""
        return np.array([self.parametros['pi0'], self.parametros['u0']])
    
    def calcular_desemprego(self, estado: np.ndarray) -> float:
        """Extrai taxa de desemprego do estado."""
        return estado[1]


class ModeloCrescimentoPopulacional(ModeloSDEBase):
    """
    Modelo de Crescimento Populacional com Choques Estocásticos.
    
    Modela a dinâmica da força de trabalho e emprego considerando
    crescimento populacional e choques econômicos.
    
    Equações:
    dL = [μ*L]dt + σ_L*L*dW_1           (força de trabalho)
    dE = [λ*E*(1 - E/K) - δ*E]dt + σ_E*E*dW_2  (empregados)
    
    onde:
    L = força de trabalho
    E = população empregada
    μ = taxa de crescimento populacional
    λ = taxa de criação de empregos
    K = capacidade (emprego máximo)
    δ = taxa de destruição de empregos
    """
    
    def __init__(self, parametros: Dict[str, float] = None):
        if parametros is None:
            parametros = {
                'mu': 0.015,        # Crescimento populacional (1.5%)
                'lambda_': 0.08,    # Criação de empregos
                'K': 100.0,         # Capacidade em milhões
                'delta': 0.04,      # Destruição de empregos
                'sigma_L': 0.005,   # Volatilidade força trabalho
                'sigma_E': 0.03,    # Volatilidade emprego
                'L0': 80.0,         # Força trabalho inicial
                'E0': 75.0          # Empregados iniciais
            }
        super().__init__(parametros)
    
    def validar_parametros(self) -> None:
        """Valida parâmetros do modelo populacional."""
        assert self.parametros['mu'] >= 0, "Taxa de crescimento inválida"
        assert self.parametros['lambda_'] > 0, "Lambda deve ser positivo"
        assert self.parametros['K'] > 0, "Capacidade deve ser positiva"
        assert self.parametros['delta'] >= 0, "Delta inválido"
        assert self.parametros['E0'] <= self.parametros['L0'], "Empregados > força trabalho"
    
    def drift(self, t: float, estado: np.ndarray) -> np.ndarray:
        """Calcula o drift do modelo populacional."""
        L, E = estado
        
        # Previne divisão por zero e valores negativos
        L = max(L, 1e-3)
        E = max(E, 1e-3)
        
        dL_dt = self.parametros['mu'] * L
        dE_dt = (self.parametros['lambda_'] * E * (1 - E / self.parametros['K']) - 
                 self.parametros['delta'] * E)
        
        return np.array([dL_dt, dE_dt])
    
    def difusao(self, t: float, estado: np.ndarray) -> np.ndarray:
        """Calcula a difusão do modelo populacional."""
        L, E = estado
        
        L = max(L, 1e-3)
        E = max(E, 1e-3)
        
        return np.array([
            [self.parametros['sigma_L'] * L, 0],
            [0, self.parametros['sigma_E'] * E]
        ])
    
    def condicao_inicial(self) -> np.ndarray:
        """Retorna condição inicial."""
        return np.array([self.parametros['L0'], self.parametros['E0']])
    
    def calcular_desemprego(self, estado: np.ndarray) -> float:
        """Calcula taxa de desemprego."""
        L, E = estado
        if L <= 0:
            return 0.0
        return max(0.0, (L - E) / L)


class ModeloMarkovEstocastico(ModeloSDEBase):
    """
    Modelo de Transição de Estados de Emprego (Markov Estocástico).
    
    Modela a dinâmica de transições entre estados de emprego usando
    um processo de difusão que aproxima uma cadeia de Markov contínua.
    
    Estados:
    - Empregado formal
    - Empregado informal
    - Desempregado
    
    dX = A*X*dt + Σ*dW
    
    onde X é o vetor de probabilidades de estado.
    """
    
    def __init__(self, parametros: Dict[str, float] = None):
        if parametros is None:
            parametros = {
                # Taxas de transição (por ano)
                'taxa_formal_informal': 0.05,
                'taxa_formal_desemprego': 0.03,
                'taxa_informal_formal': 0.08,
                'taxa_informal_desemprego': 0.06,
                'taxa_desemprego_formal': 0.15,
                'taxa_desemprego_informal': 0.20,
                # Volatilidades
                'sigma_formal': 0.02,
                'sigma_informal': 0.03,
                'sigma_desemprego': 0.04,
                # Estado inicial (frações)
                'x_formal_0': 0.55,
                'x_informal_0': 0.30,
                'x_desemprego_0': 0.15
            }
        super().__init__(parametros)
    
    def validar_parametros(self) -> None:
        """Valida parâmetros do modelo de Markov."""
        # Verifica se estado inicial soma 1
        soma = (self.parametros['x_formal_0'] + 
                self.parametros['x_informal_0'] + 
                self.parametros['x_desemprego_0'])
        assert abs(soma - 1.0) < 1e-6, "Estado inicial deve somar 1"
        
        # Verifica positividade
        for key, val in self.parametros.items():
            if 'taxa' in key or 'sigma' in key or 'x_' in key:
                assert val >= 0, f"{key} deve ser não-negativo"
    
    def _construir_matriz_transicao(self) -> np.ndarray:
        """Constrói a matriz de taxas de transição."""
        p = self.parametros
        
        # Taxa de saída de cada estado
        saida_formal = p['taxa_formal_informal'] + p['taxa_formal_desemprego']
        saida_informal = p['taxa_informal_formal'] + p['taxa_informal_desemprego']
        saida_desemprego = p['taxa_desemprego_formal'] + p['taxa_desemprego_informal']
        
        # Matriz de transição (gerador infinitesimal)
        A = np.array([
            [-saida_formal, p['taxa_informal_formal'], p['taxa_desemprego_formal']],
            [p['taxa_formal_informal'], -saida_informal, p['taxa_desemprego_informal']],
            [p['taxa_formal_desemprego'], p['taxa_informal_desemprego'], -saida_desemprego]
        ])
        
        return A
    
    def drift(self, t: float, estado: np.ndarray) -> np.ndarray:
        """Calcula o drift do modelo de Markov."""
        A = self._construir_matriz_transicao()
        return A @ estado
    
    def difusao(self, t: float, estado: np.ndarray) -> np.ndarray:
        """Calcula a difusão do modelo de Markov."""
        return np.diag([
            self.parametros['sigma_formal'],
            self.parametros['sigma_informal'],
            self.parametros['sigma_desemprego']
        ])
    
    def condicao_inicial(self) -> np.ndarray:
        """Retorna condição inicial."""
        return np.array([
            self.parametros['x_formal_0'],
            self.parametros['x_informal_0'],
            self.parametros['x_desemprego_0']
        ])
    
    def normalizar_estado(self, estado: np.ndarray) -> np.ndarray:
        """
        Normaliza o estado para garantir que soma 1 e valores são não-negativos.
        """
        estado = np.maximum(estado, 0)  # Remove negativos
        soma = np.sum(estado)
        if soma > 0:
            return estado / soma
        else:
            # Retorna distribuição uniforme se algo der errado
            return np.ones(3) / 3
    
    def calcular_desemprego(self, estado: np.ndarray) -> float:
        """Retorna a fração de desempregados."""
        estado_normalizado = self.normalizar_estado(estado)
        return estado_normalizado[2]  # Terceiro componente é desemprego


# Dicionário para facilitar a criação de modelos
MODELOS_DISPONIVEIS = {
    'goodwin': ModeloGoodwinEstocastico,
    'phillips': ModeloPhillipsEstocastico,
    'populacional': ModeloCrescimentoPopulacional,
    'markov': ModeloMarkovEstocastico
}


def criar_modelo(nome: str, parametros: Dict[str, float] = None) -> ModeloSDEBase:
    """
    Factory function para criar modelos.
    
    Args:
        nome: Nome do modelo ('goodwin', 'phillips', 'populacional', 'markov')
        parametros: Parâmetros opcionais do modelo
        
    Returns:
        Instância do modelo solicitado
        
    Raises:
        ValueError: Se o nome do modelo não for reconhecido
    """
    if nome not in MODELOS_DISPONIVEIS:
        raise ValueError(f"Modelo '{nome}' não reconhecido. "
                        f"Modelos disponíveis: {list(MODELOS_DISPONIVEIS.keys())}")
    
    return MODELOS_DISPONIVEIS[nome](parametros)
