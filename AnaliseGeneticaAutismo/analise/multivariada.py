# analise/multivariada.py
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class AnalisadorMultivariado:
    """
    Realiza análises multivariadas complexas, como PCA (Principal Component Analysis),
    para identificar estruturas populacionais e padrões latentes nos dados genéticos.
    """
    
    def __init__(self):
        self.pca = PCA(n_components=3)
        self.scaler = StandardScaler()
        
    def executar_pca(self, df_genotipos: pd.DataFrame) -> pd.DataFrame:
        """
        Executa PCA nos dados de genótipos (0, 1, 2).
        Retorna um DataFrame com os 3 primeiros componentes principais (PC1, PC2, PC3).
        """
        # Padronizar os dados (média 0, variância 1) é crucial para PCA
        X_scaled = self.scaler.fit_transform(df_genotipos)
        
        # Calcular componentes principais
        componentes = self.pca.fit_transform(X_scaled)
        
        df_pca = pd.DataFrame(
            data=componentes, 
            columns=['PC1', 'PC2', 'PC3'],
            index=df_genotipos.index
        )
        
        # Calcular variância explicada
        variancia = self.pca.explained_variance_ratio_
        print(f"Variância explicada por PC: {variancia}")
        
        return df_pca
