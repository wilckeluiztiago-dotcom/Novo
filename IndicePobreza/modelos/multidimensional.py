import pandas as pd
import numpy as np

class PobrezaMultidimensional:
    """
    Implementação do método Alkire-Foster para Pobreza Multidimensional (MPI).
    """
    
    def __init__(self, dados, dimensoes, pesos, corte_k=0.33):
        """
        Inicializa o cálculo do MPI.
        
        Parâmetros:
            dados (pd.DataFrame): DataFrame com os dados.
            dimensoes (list): Lista de colunas que representam as privações (0=Não privado, 1=Privado).
            pesos (dict): Dicionário mapeando dimensão -> peso. Soma deve ser 1.
            corte_k (float): Corte de pobreza multidimensional (k). Geralmente 33% (1/3).
        """
        self.dados = dados.copy()
        self.dimensoes = dimensoes
        self.pesos = pesos
        self.k = corte_k
        self.n = len(dados)
        
        if not np.isclose(sum(pesos.values()), 1.0):
            raise ValueError("A soma dos pesos deve ser igual a 1.")
            
    def calcular_escore_privacao(self):
        """
        Calcula o escore de privação (c_i) para cada domicílio.
        c_i = soma(w_j * g_ij)
        """
        self.dados['escore_privacao'] = 0.0
        for dim in self.dimensoes:
            # Assumindo que a coluna já é binária (1 = privado)
            # Se não for, precisaria de uma função de corte para cada indicador
            self.dados['escore_privacao'] += self.dados[dim] * self.pesos[dim]
            
        return self.dados['escore_privacao']
    
    def identificar_pobres(self):
        """
        Identifica quem é pobre multidimensionalmente (c_i >= k).
        """
        if 'escore_privacao' not in self.dados.columns:
            self.calcular_escore_privacao()
            
        self.dados['is_pobre_mpi'] = (self.dados['escore_privacao'] >= self.k).astype(int)
        return self.dados['is_pobre_mpi']
    
    def calcular_indices(self):
        """
        Calcula H (Incidência), A (Intensidade) e MPI (M0).
        
        Retorna:
            dict: {H, A, MPI}
        """
        if 'is_pobre_mpi' not in self.dados.columns:
            self.identificar_pobres()
            
        # H: Incidência (Headcount Ratio) = q / n
        q = self.dados['is_pobre_mpi'].sum()
        H = q / self.n
        
        if q == 0:
            return {'H': 0.0, 'A': 0.0, 'MPI': 0.0}
        
        # A: Intensidade Média da Pobreza entre os pobres
        # Soma dos escores dos pobres / q
        soma_escores_pobres = self.dados.loc[self.dados['is_pobre_mpi'] == 1, 'escore_privacao'].sum()
        A = soma_escores_pobres / q
        
        # MPI = H * A
        MPI = H * A
        
        return {'H': H, 'A': A, 'MPI': MPI}

    def decompor_por_grupo(self, coluna_grupo):
        """
        Decompõe o MPI por grupos (ex: UF, Zona).
        """
        resultados = []
        grupos = self.dados[coluna_grupo].unique()
        
        for grupo in grupos:
            sub_dados = self.dados[self.dados[coluna_grupo] == grupo]
            mpi_calc = PobrezaMultidimensional(sub_dados, self.dimensoes, self.pesos, self.k)
            indices = mpi_calc.calcular_indices()
            indices[coluna_grupo] = grupo
            resultados.append(indices)
            
        return pd.DataFrame(resultados)
