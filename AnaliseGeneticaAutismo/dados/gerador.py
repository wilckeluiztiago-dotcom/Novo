# dados/gerador.py
import numpy as np
import pandas as pd
from typing import Tuple, Dict
import random
from configuracao import GENES_ALVO, NUMERO_SNPS_POR_GENE, PESO_VARIANTE_LOF, PESO_VARIANTE_MISSENSE

class GeradorDadosGeneticos:
    """
    Classe responsável por gerar dados genéticos sintéticos complexos para simulação de estudos de associação.
    Gera genótipos (SNPs), dados de expressão gênica e fenótipos clínicos.
    """

    def __init__(self, n_amostras: int = 1000, seed: int = 42):
        self.n_amostras = n_amostras
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        
        self.ids_amostras = [f"AMOSTRA_{i:04d}" for i in range(n_amostras)]
        self.snps_info = self._gerar_info_snps()

    def _gerar_info_snps(self) -> pd.DataFrame:
        """Gera metadados para os SNPs simulados."""
        snps = []
        tipos_variante = ['Missense', 'Synonymous', 'LoF', 'Intronic']
        pesos_tipo = [0.4, 0.4, 0.1, 0.1] # Probabilidade de cada tipo

        for gene in GENES_ALVO:
            for i in range(NUMERO_SNPS_POR_GENE):
                rsid = f"rs{random.randint(10000, 999999)}"
                tipo = np.random.choice(tipos_variante, p=pesos_tipo)
                
                # Definir peso de risco baseado no tipo
                peso_risco = 0.0
                if tipo == 'LoF':
                    peso_risco = PESO_VARIANTE_LOF
                elif tipo == 'Missense':
                    peso_risco = PESO_VARIANTE_MISSENSE
                
                # Frequência do alelo menor (MAF)
                maf = np.random.beta(2, 10) # Distribuição beta para simular MAFs realistas (maioria rara)
                
                snps.append({
                    'SNP_ID': rsid,
                    'Gene': gene,
                    'Cromossomo': np.random.randint(1, 23),
                    'Posicao': np.random.randint(100000, 100000000),
                    'Tipo': tipo,
                    'MAF': maf,
                    'Peso_Risco': peso_risco
                })
        
        return pd.DataFrame(snps)

    def gerar_genotipos(self) -> pd.DataFrame:
        """
        Gera uma matriz de genótipos (0, 1, 2 alelos de risco) para todas as amostras.
        """
        genotipos = {}
        
        for _, row in self.snps_info.iterrows():
            # Simular genótipos baseados em Hardy-Weinberg
            maf = row['MAF']
            p_aa = (1 - maf) ** 2
            p_Aa = 2 * maf * (1 - maf)
            p_AA = maf ** 2
            
            # Gerar genótipos (0=Homozigoto Ref, 1=Heterozigoto, 2=Homozigoto Alt)
            g = np.random.choice([0, 1, 2], size=self.n_amostras, p=[p_aa, p_Aa, p_AA])
            genotipos[f"{row['Gene']}_{row['SNP_ID']}"] = g
            
        df_genotipos = pd.DataFrame(genotipos, index=self.ids_amostras)
        return df_genotipos

    def gerar_expressao_genica(self, fenotipos: pd.Series) -> pd.DataFrame:
        """
        Gera dados de expressão gênica (RNA-seq normalizado) correlacionados com o fenótipo.
        """
        expressao = {}
        
        for gene in GENES_ALVO:
            # Base de expressão
            base = np.random.normal(10, 2, self.n_amostras)
            
            # Adicionar efeito da doença (alguns genes superexpressos, outros subexpressos)
            efeito = np.random.choice([-1, 0, 1]) * 2.0
            
            # Aplicar efeito apenas nos casos
            ajuste_caso = np.where(fenotipos == 1, efeito, 0)
            
            # Adicionar ruído
            ruido = np.random.normal(0, 1, self.n_amostras)
            
            expressao[gene] = base + ajuste_caso + ruido
            
        return pd.DataFrame(expressao, index=self.ids_amostras)

    def gerar_fenotipos(self, df_genotipos: pd.DataFrame) -> pd.DataFrame:
        """
        Gera o fenótipo (Caso/Controle) baseado no Score de Risco Poligênico (PRS) derivado dos genótipos.
        """
        prs = np.zeros(self.n_amostras)
        
        # Calcular PRS
        for col in df_genotipos.columns:
            gene, snp_id = col.split('_')
            snp_info = self.snps_info[self.snps_info['SNP_ID'] == snp_id].iloc[0]
            peso = snp_info['Peso_Risco']
            
            # Somar peso * número de alelos de risco
            prs += df_genotipos[col].values * peso
            
        # Normalizar PRS
        prs = (prs - prs.mean()) / prs.std()
        
        # Definir probabilidade de ser caso (Sigmoide)
        probabilidade = 1 / (1 + np.exp(-(prs - 1))) # Shift para ajustar a prevalência
        
        # Gerar status (1 = Caso/Autismo, 0 = Controle)
        status = np.random.binomial(1, probabilidade)
        
        df_fenotipos = pd.DataFrame({
            'PRS': prs,
            'Probabilidade_Risco': probabilidade,
            'Status': status,
            'Grupo': ['Caso' if s == 1 else 'Controle' for s in status]
        }, index=self.ids_amostras)
        
        return df_fenotipos

    def gerar_dataset_completo(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Executa todo o pipeline de geração e retorna os DataFrames.
        Retorna: (Genótipos, Expressão, Fenótipos, Metadados SNPs)
        """
        print("Gerando genótipos...")
        df_genotipos = self.gerar_genotipos()
        
        print("Calculando fenótipos baseados em risco genético...")
        df_fenotipos = self.gerar_fenotipos(df_genotipos)
        
        print("Simulando expressão gênica diferencial...")
        df_expressao = self.gerar_expressao_genica(df_fenotipos['Status'])
        
        return df_genotipos, df_expressao, df_fenotipos, self.snps_info
