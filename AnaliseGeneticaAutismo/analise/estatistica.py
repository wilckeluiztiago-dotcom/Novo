# analise/estatistica.py
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

class AnalisadorEstatistico:
    """
    Realiza testes estatísticos em dados genéticos para identificar associações com o fenótipo.
    """
    
    @staticmethod
    def calcular_frequencias_alelicas(df_genotipos: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula a frequência do alelo de risco (MAF) para cada variante.
        """
        freqs = {}
        n_amostras = len(df_genotipos)
        
        for col in df_genotipos.columns:
            # Genótipos são 0, 1, 2. Soma total de alelos de risco é a soma da coluna.
            # Total de alelos é 2 * n_amostras
            contagem_alelos_risco = df_genotipos[col].sum()
            maf = contagem_alelos_risco / (2 * n_amostras)
            freqs[col] = maf
            
        return pd.DataFrame.from_dict(freqs, orient='index', columns=['MAF_Calculado'])

    @staticmethod
    def teste_associacao_gwas(df_genotipos: pd.DataFrame, df_fenotipos: pd.DataFrame) -> pd.DataFrame:
        """
        Realiza um teste Qui-quadrado para cada SNP para testar associação com o status (Caso/Controle).
        Simula um GWAS (Genome-Wide Association Study).
        """
        resultados = []
        
        casos = df_genotipos[df_fenotipos['Status'] == 1]
        controles = df_genotipos[df_fenotipos['Status'] == 0]
        
        for snp in df_genotipos.columns:
            # Contagem de alelos (Referência vs Risco) em Casos e Controles
            # Risco = soma dos valores (0, 1, 2)
            # Ref = (2 * n) - Risco
            
            risco_casos = casos[snp].sum()
            ref_casos = (2 * len(casos)) - risco_casos
            
            risco_controles = controles[snp].sum()
            ref_controles = (2 * len(controles)) - risco_controles
            
            tabela = [[risco_casos, ref_casos], [risco_controles, ref_controles]]
            
            # Teste Qui-quadrado
            try:
                chi2, p_valor, _, _ = chi2_contingency(tabela)
            except:
                p_valor = 1.0
                chi2 = 0.0
                
            # Odds Ratio
            try:
                or_val = (risco_casos * ref_controles) / (risco_controles * ref_casos)
            except ZeroDivisionError:
                or_val = np.nan
                
            resultados.append({
                'SNP': snp,
                'P_Valor': p_valor,
                'Chi2': chi2,
                'Odds_Ratio': or_val,
                'Log10_P': -np.log10(p_valor) if p_valor > 0 else 0
            })
            
        return pd.DataFrame(resultados).sort_values('P_Valor')
