import numpy as np

class Desigualdade:
    """
    Classe para cálculo de indicadores de desigualdade.
    """
    
    def __init__(self, dados, coluna_renda='renda_pc'):
        self.y = dados[coluna_renda].values
        self.y = self.y[self.y > 0] # Remover rendas zero ou negativas para Theil/Log
        self.n = len(self.y)
        self.mu = np.mean(self.y)
        
    def gini(self):
        """
        Calcula o Coeficiente de Gini.
        """
        # Ordenar rendas
        y_sorted = np.sort(self.y)
        index = np.arange(1, self.n + 1)
        
        return ((2 * np.sum(index * y_sorted)) / (self.n * np.sum(y_sorted))) - ((self.n + 1) / self.n)

    def curva_lorenz(self):
        """
        Gera os pontos da Curva de Lorenz.
        
        Retorna:
            tuple: (populacao_acumulada, renda_acumulada)
        """
        y_sorted = np.sort(self.y)
        renda_acumulada = np.cumsum(y_sorted) / np.sum(y_sorted)
        renda_acumulada = np.insert(renda_acumulada, 0, 0) # Começa em (0,0)
        
        populacao_acumulada = np.linspace(0, 1, len(renda_acumulada))
        
        return populacao_acumulada, renda_acumulada

    def theil_t(self):
        """
        Calcula o Índice de Theil T (GE(1)).
        T = (1/N) * sum((y_i / mu) * ln(y_i / mu))
        """
        razao = self.y / self.mu
        return np.mean(razao * np.log(razao))

    def theil_l(self):
        """
        Calcula o Índice de Theil L (GE(0)) ou Desvio Logarítmico Médio.
        L = (1/N) * sum(ln(mu / y_i))
        """
        razao = self.mu / self.y
        return np.mean(np.log(razao))
