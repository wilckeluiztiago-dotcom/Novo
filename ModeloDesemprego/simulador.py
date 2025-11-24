"""
Simulador Numérico para Equações Diferenciais Estocásticas

Implementa múltiplos métodos numéricos para resolver SDEs com alta precisão.

Autor: Luiz Tiago Wilcke
Data: 2025-11-24
"""

import numpy as np
from typing import Tuple, List, Callable, Optional
from tqdm import tqdm
from modelos_sde import ModeloSDEBase


class SimuladorSDE:
    """
    Simulador de Equações Diferenciais Estocásticas.
    
    Implementa múltiplos métodos numéricos:
    - Euler-Maruyama (ordem forte 0.5, ordem fraca 1.0)
    - Milstein (ordem forte 1.0)
    - Runge-Kutta Estocástico (SRK)
    """
    
    def __init__(self, modelo: ModeloSDEBase, seed: Optional[int] = None):
        """
        Inicializa o simulador.
        
        Args:
            modelo: Instância de um modelo SDE
            seed: Semente para geração de números aleatórios
        """
        self.modelo = modelo
        self.rng = np.random.RandomState(seed)
        self.dimensao = modelo.dimensao()
    
    def euler_maruyama(
        self, 
        T: float, 
        N: int, 
        X0: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Método de Euler-Maruyama para resolver SDEs.
        
        Resolve: dX = a(t,X)dt + b(t,X)dW
        Esquema: X_{n+1} = X_n + a(t_n,X_n)*dt + b(t_n,X_n)*sqrt(dt)*Z_n
        onde Z_n ~ N(0,1)
        
        Args:
            T: Tempo final de simulação
            N: Número de passos de tempo
            X0: Condição inicial (usa a do modelo se None)
            
        Returns:
            (tempos, trajetoria): Arrays com tempos e estados simulados
        """
        dt = T / N
        sqrt_dt = np.sqrt(dt)
        
        # Condição inicial
        if X0 is None:
            X0 = self.modelo.condicao_inicial()
        
        # Arrays para armazenar resultados
        tempos = np.linspace(0, T, N + 1)
        trajetoria = np.zeros((N + 1, self.dimensao))
        trajetoria[0] = X0
        
        # Simulação
        X = X0.copy()
        for i in range(N):
            t = tempos[i]
            
            # Calcula drift e difusão
            a = self.modelo.drift(t, X)
            b = self.modelo.difusao(t, X)
            
            # Gera ruído Gaussiano
            dW = self.rng.randn(self.dimensao) * sqrt_dt
            
            # Atualização de Euler-Maruyama
            X = X + a * dt + b @ dW
            
            # Armazena
            trajetoria[i + 1] = X
        
        return tempos, trajetoria
    
    def milstein(
        self, 
        T: float, 
        N: int, 
        X0: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Método de Milstein para resolver SDEs (maior precisão que Euler-Maruyama).
        
        Inclui correção de segunda ordem usando derivadas da difusão.
        
        Args:
            T: Tempo final de simulação
            N: Número de passos de tempo
            X0: Condição inicial
            
        Returns:
            (tempos, trajetoria): Arrays com tempos e estados simulados
        """
        dt = T / N
        sqrt_dt = np.sqrt(dt)
        
        if X0 is None:
            X0 = self.modelo.condicao_inicial()
        
        tempos = np.linspace(0, T, N + 1)
        trajetoria = np.zeros((N + 1, self.dimensao))
        trajetoria[0] = X0
        
        X = X0.copy()
        epsilon = 1e-7  # Para diferenças finitas
        
        for i in range(N):
            t = tempos[i]
            
            # Calcula drift e difusão
            a = self.modelo.drift(t, X)
            b = self.modelo.difusao(t, X)
            
            # Ruído
            dW = self.rng.randn(self.dimensao) * sqrt_dt
            
            # Termo de correção de Milstein (derivada de b em relação a X)
            # Aproximação por diferenças finitas
            db_dX = np.zeros((self.dimensao, self.dimensao, self.dimensao))
            for j in range(self.dimensao):
                X_plus = X.copy()
                X_plus[j] += epsilon
                b_plus = self.modelo.difusao(t, X_plus)
                db_dX[:, :, j] = (b_plus - b) / epsilon
            
            # Termo de Milstein
            milstein_correcao = np.zeros(self.dimensao)
            for k in range(self.dimensao):
                for j in range(self.dimensao):
                    milstein_correcao += b[:, j] * db_dX[:, j, k] * (dW[k]**2 - dt) / 2
            
            # Atualização de Milstein
            X = X + a * dt + b @ dW + milstein_correcao
            
            trajetoria[i + 1] = X
        
        return tempos, trajetoria
    
    def runge_kutta_estocastico(
        self, 
        T: float, 
        N: int, 
        X0: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Método de Runge-Kutta Estocástico (ordem superior).
        
        Implementa versão simplificada do método SRK para melhor estabilidade.
        
        Args:
            T: Tempo final
            N: Número de passos
            X0: Condição inicial
            
        Returns:
            (tempos, trajetoria)
        """
        dt = T / N
        sqrt_dt = np.sqrt(dt)
        
        if X0 is None:
            X0 = self.modelo.condicao_inicial()
        
        tempos = np.linspace(0, T, N + 1)
        trajetoria = np.zeros((N + 1, self.dimensao))
        trajetoria[0] = X0
        
        X = X0.copy()
        
        for i in range(N):
            t = tempos[i]
            
            # Ruído
            dW = self.rng.randn(self.dimensao) * sqrt_dt
            
            # Estágio 1
            a1 = self.modelo.drift(t, X)
            b1 = self.modelo.difusao(t, X)
            
            # Estágio 2 (ponto médio)
            X_temp = X + a1 * dt / 2 + b1 @ (dW / np.sqrt(2))
            a2 = self.modelo.drift(t + dt / 2, X_temp)
            b2 = self.modelo.difusao(t + dt / 2, X_temp)
            
            # Combinação
            X = X + a2 * dt + b2 @ dW
            
            trajetoria[i + 1] = X
        
        return tempos, trajetoria
    
    def simular_multiplas_trajetorias(
        self,
        T: float,
        N: int,
        num_trajetorias: int,
        metodo: str = 'euler',
        mostrar_progresso: bool = True,
        X0: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simula múltiplas trajetórias (Monte Carlo).
        
        Args:
            T: Tempo final
            N: Passos de tempo
            num_trajetorias: Número de trajetórias a simular
            metodo: 'euler', 'milstein', ou 'srk'
            mostrar_progresso: Se True, mostra barra de progresso
            X0: Condição inicial
            
        Returns:
            (tempos, trajetorias): tempos e array de trajetórias (num_traj x N+1 x dim)
        """
        # Seleciona o método
        metodos = {
            'euler': self.euler_maruyama,
            'milstein': self.milstein,
            'srk': self.runge_kutta_estocastico
        }
        
        if metodo not in metodos:
            raise ValueError(f"Método '{metodo}' não reconhecido. "
                           f"Use: {list(metodos.keys())}")
        
        simular = metodos[metodo]
        
        # Array para armazenar todas as trajetórias
        trajetorias = np.zeros((num_trajetorias, N + 1, self.dimensao))
        
        # Simula cada trajetória
        iterador = range(num_trajetorias)
        if mostrar_progresso:
            iterador = tqdm(iterador, desc=f"Simulando {num_trajetorias} trajetórias")
        
        for i in iterador:
            # Nova semente para cada trajetória
            self.rng = np.random.RandomState(self.rng.randint(0, 2**31))
            tempos, traj = simular(T, N, X0)
            trajetorias[i] = traj
        
        return tempos, trajetorias
    
    def calcular_estatisticas_ensemble(
        self,
        trajetorias: np.ndarray
    ) -> dict:
        """
        Calcula estatísticas de um ensemble de trajetórias.
        
        Args:
            trajetorias: Array de forma (num_traj, N+1, dim)
            
        Returns:
            Dicionário com estatísticas (média, std, quantis, etc.)
        """
        estatisticas = {
            'media': np.mean(trajetorias, axis=0),
            'desvio_padrao': np.std(trajetorias, axis=0),
            'variancia': np.var(trajetorias, axis=0),
            'mediana': np.median(trajetorias, axis=0),
            'q05': np.percentile(trajetorias, 5, axis=0),
            'q25': np.percentile(trajetorias, 25, axis=0),
            'q75': np.percentile(trajetorias, 75, axis=0),
            'q95': np.percentile(trajetorias, 95, axis=0),
            'minimo': np.min(trajetorias, axis=0),
            'maximo': np.max(trajetorias, axis=0)
        }
        
        return estatisticas
    
    def convergencia_forte(
        self,
        T: float,
        N_valores: List[int],
        num_trajetorias: int = 100,
        metodo: str = 'euler',
        X0: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Analisa convergência forte do método numérico.
        
        Calcula o erro forte: E[|X_N - X_ref|] para diferentes valores de N.
        
        Args:
            T: Tempo final
            N_valores: Lista de números de passos para testar
            num_trajetorias: Número de trajetórias para média
            metodo: Método numérico
            X0: Condição inicial
            
        Returns:
            (N_valores, erros): Arrays com N e erro médio
        """
        # Usa a simulação mais fina como referência
        N_ref = max(N_valores) * 4
        print(f"Calculando trajetória de referência com N={N_ref}...")
        
        # Fixa a semente para comparação justa
        seed_original = 42
        
        erros = []
        
        for N in tqdm(N_valores, desc="Testando convergência"):
            erros_N = []
            
            for i in range(num_trajetorias):
                # Mesma semente para referência e teste
                seed = seed_original + i
                
                # Trajetória de referência
                self.rng = np.random.RandomState(seed)
                _, traj_ref = self.euler_maruyama(T, N_ref, X0)
                
                # Trajetória de teste
                self.rng = np.random.RandomState(seed)
                _, traj_test = self.euler_maruyama(T, N, X0)
                
                # Erro no tempo final
                erro = np.linalg.norm(traj_test[-1] - traj_ref[-1])
                erros_N.append(erro)
            
            erros.append(np.mean(erros_N))
        
        return np.array(N_valores), np.array(erros)
    
    def calcular_desemprego_trajetorias(
        self,
        trajetorias: np.ndarray
    ) -> np.ndarray:
        """
        Calcula a taxa de desemprego para cada ponto de cada trajetória.
        
        Args:
            trajetorias: Array (num_traj, N+1, dim)
            
        Returns:
            Array (num_traj, N+1) com taxa de desemprego
        """
        num_traj, N_plus_1, _ = trajetorias.shape
        desemprego = np.zeros((num_traj, N_plus_1))
        
        for i in range(num_traj):
            for j in range(N_plus_1):
                desemprego[i, j] = self.modelo.calcular_desemprego(trajetorias[i, j])
        
        return desemprego


class AnaliseConvergencia:
    """
    Classe auxiliar para análise de convergência de métodos numéricos.
    """
    
    @staticmethod
    def calcular_taxa_convergencia(
        N_valores: np.ndarray,
        erros: np.ndarray
    ) -> Tuple[float, float]:
        """
        Estima a taxa de convergência por regressão linear em log-log.
        
        E ~ C * dt^p onde dt = T/N
        log(E) ~ log(C) + p*log(dt)
        
        Args:
            N_valores: Valores de N testados
            erros: Erros correspondentes
            
        Returns:
            (taxa, intercepto): Taxa de convergência p e constante log(C)
        """
        # Remove zeros para evitar problemas com log
        mask = erros > 0
        N_valores = N_valores[mask]
        erros = erros[mask]
        
        # Regressão linear em log-log
        log_dt = -np.log(N_valores)  # dt = T/N, assumindo T=1
        log_erro = np.log(erros)
        
        coef = np.polyfit(log_dt, log_erro, 1)
        taxa = coef[0]
        intercepto = coef[1]
        
        return taxa, intercepto
