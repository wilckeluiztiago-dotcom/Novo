import numpy as np

class IndicadoresFGT:
    """
    Classe para cálculo dos índices Foster-Greer-Thorbecke (FGT).
    """
    
    def __init__(self, dados, linha_pobreza, coluna_renda='renda_pc'):
        """
        Inicializa a classe.
        
        Parâmetros:
            dados (pd.DataFrame): DataFrame com os dados.
            linha_pobreza (float): Valor da linha de pobreza (z).
            coluna_renda (str): Nome da coluna de renda (y).
        """
        self.dados = dados
        self.z = linha_pobreza
        self.y = dados[coluna_renda].values
        self.n = len(self.y)
        
        # Identificar pobres: y_i < z
        self.indices_pobres = self.y < self.z
        self.q = np.sum(self.indices_pobres) # Quantidade de pobres
        self.y_pobres = self.y[self.indices_pobres]
        
    def calcular_fgt(self, alpha):
        """
        Calcula o índice FGT para um dado alpha.
        Fórmula: P_alpha = (1/N) * sum(((z - y_i) / z)^alpha) para y_i < z
        
        Parâmetros:
            alpha (float): Parâmetro de aversão à pobreza (0, 1, 2).
            
        Retorna:
            float: Valor do índice.
        """
        if self.q == 0:
            return 0.0
            
        hiato_relativo = (self.z - self.y_pobres) / self.z
        soma_hiatos = np.sum(np.power(hiato_relativo, alpha))
        
        return soma_hiatos / self.n

    def incidencia(self):
        """Retorna P0 (Incidência)."""
        return self.calcular_fgt(0)
    
    def hiato(self):
        """Retorna P1 (Hiato de Pobreza)."""
        return self.calcular_fgt(1)
    
    def severidade(self):
        """Retorna P2 (Severidade da Pobreza)."""
        return self.calcular_fgt(2)

    def indice_sen(self, gini_pobres):
        """
        Calcula o Índice de Sen.
        S = P0 * (1 - (1 - P1) * (1 + G_p)) aproximadamente.
        Ou S = P0 * G_p + P1 * (1 - G_p)
        
        Aqui usaremos a formulação: S = P0 * (P1 + (1 - P1) * G_p)
        Onde G_p é o Gini dos pobres.
        """
        p0 = self.incidencia()
        p1 = self.hiato()
        return p0 * (p1 + (1 - p1) * gini_pobres)
