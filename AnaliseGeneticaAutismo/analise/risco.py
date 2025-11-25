# analise/risco.py
import pandas as pd
import numpy as np

class CalculadoraRisco:
    """
    Módulo dedicado ao cálculo de riscos genéticos complexos.
    """
    
    def __init__(self, metadados_snps: pd.DataFrame):
        self.metadados = metadados_snps.set_index(['Gene', 'SNP_ID'])
        
    def calcular_prs(self, df_genotipos: pd.DataFrame) -> pd.Series:
        """
        Calcula o Polygenic Risk Score (PRS) para novas amostras.
        PRS = Somatório(Genótipo_i * Peso_i)
        """
        prs_scores = np.zeros(len(df_genotipos))
        
        for col in df_genotipos.columns:
            gene, snp_id = col.split('_')
            
            # Buscar peso do SNP
            try:
                # Tenta acessar pelo índice composto
                # Se o índice não estiver configurado corretamente no dataframe de entrada, pode falhar
                # Assumindo que metadados tem colunas Gene e SNP_ID se não for index
                if 'Peso_Risco' in self.metadados.columns:
                     peso = self.metadados.loc[(gene, snp_id), 'Peso_Risco']
                else:
                    peso = 0
            except KeyError:
                peso = 0
                
            prs_scores += df_genotipos[col].values * peso
            
        return pd.Series(prs_scores, index=df_genotipos.index, name='PRS_Calculado')

    def classificar_risco(self, prs_series: pd.Series) -> pd.Series:
        """
        Classifica o risco em Baixo, Médio e Alto baseado em quantis.
        """
        q33 = prs_series.quantile(0.33)
        q66 = prs_series.quantile(0.66)
        
        def classificar(valor):
            if valor <= q33:
                return 'Baixo'
            elif valor <= q66:
                return 'Médio'
            else:
                return 'Alto'
                
        return prs_series.apply(classificar)
