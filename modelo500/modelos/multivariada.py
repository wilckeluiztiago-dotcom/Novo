"""
Módulo de Análise Multivariada para Desemprego.

Inclui implementações de:
- VAR (Vector AutoRegression)
- VECM (Vector Error Correction Model)
- Testes de Causalidade de Granger

Autor: Luiz Tiago Wilcke
Data: 2025-11-24
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import VECM, select_order
from statsmodels.tsa.stattools import grangercausalitytests

class AnaliseMultivariada:
    """
    Classe para análise de múltiplas séries temporais macroeconômicas.
    """
    def __init__(self, dados: pd.DataFrame):
        """
        Args:
            dados: DataFrame com colunas representando diferentes variáveis (ex: desemprego, inflação, pib)
        """
        self.dados = dados.dropna()
        self.modelo_ajustado = None
        self.tipo_modelo = None
        
    def selecionar_ordem_var(self, max_lags: int = 12) -> int:
        """Seleciona a melhor ordem (lags) baseada no AIC."""
        modelo = VAR(self.dados)
        resultado = modelo.select_order(maxlags=max_lags)
        return resultado.aic
        
    def ajustar_var(self, lags: Optional[int] = None):
        """Ajusta um modelo VAR."""
        modelo = VAR(self.dados)
        if lags is None:
            lags = self.selecionar_ordem_var()
            
        self.modelo_ajustado = modelo.fit(lags)
        self.tipo_modelo = 'VAR'
        return self.modelo_ajustado.summary()
        
    def ajustar_vecm(self, deterministic: str = 'ci'):
        """
        Ajusta um modelo VECM (para séries cointegradas).
        """
        # Seleção automática de rank de cointegração é complexa, 
        # aqui assumimos rank=1 para simplificação ou usamos select_order
        try:
            modelo = VECM(self.dados, k_ar_diff=2, coint_rank=1, deterministic=deterministic)
            self.modelo_ajustado = modelo.fit()
            self.tipo_modelo = 'VECM'
            return self.modelo_ajustado.summary()
        except Exception as e:
            print(f"Erro ao ajustar VECM: {e}")
            return None

    def impulso_resposta(self, periodos: int = 12):
        """Gera funções de impulso-resposta (IRF)."""
        if self.modelo_ajustado is None:
            raise ValueError("Modelo não ajustado.")
            
        irf = self.modelo_ajustado.irf(periodos)
        return irf
        
    def decomposicao_variancia(self, periodos: int = 12):
        """Calcula decomposição da variância do erro de previsão."""
        if self.modelo_ajustado is None:
            raise ValueError("Modelo não ajustado.")
            
        fevd = self.modelo_ajustado.fevd(periodos)
        return fevd
        
    def teste_causalidade_granger(self, variavel_causa: str, variavel_efeito: str, max_lags: int = 4):
        """
        Testa se 'variavel_causa' Granger-causa 'variavel_efeito'.
        
        Returns:
            Dicionário com p-valores para diferentes lags.
        """
        if variavel_causa not in self.dados.columns or variavel_efeito not in self.dados.columns:
            raise ValueError("Variáveis não encontradas no DataFrame.")
            
        dados_teste = self.dados[[variavel_efeito, variavel_causa]]
        try:
            resultado = grangercausalitytests(dados_teste, maxlag=max_lags, verbose=False)
            # Extrai p-valores do teste F (ssr_ftest)
            p_valores = {lag: res[0]['ssr_ftest'][1] for lag, res in resultado.items()}
            return p_valores
        except Exception as e:
            print(f"Erro no teste de Granger: {e}")
            return {}

    def prever(self, passos: int = 12) -> pd.DataFrame:
        """Realiza previsões."""
        if self.modelo_ajustado is None:
            raise ValueError("Modelo não ajustado.")
            
        if self.tipo_modelo == 'VAR':
            # VAR forecast requer os últimos lags dados
            lag_order = self.modelo_ajustado.k_ar
            input_data = self.dados.values[-lag_order:]
            previsao = self.modelo_ajustado.forecast(y=input_data, steps=passos)
        elif self.tipo_modelo == 'VECM':
            previsao = self.modelo_ajustado.predict(steps=passos)
        else:
            return None
            
        idx_futuro = pd.date_range(start=self.dados.index[-1], periods=passos+1, freq=self.dados.index.freq)[1:]
        if len(idx_futuro) != passos: # Fallback se freq não estiver definida
             idx_futuro = np.arange(len(self.dados), len(self.dados) + passos)
             
        return pd.DataFrame(previsao, index=idx_futuro, columns=self.dados.columns)
