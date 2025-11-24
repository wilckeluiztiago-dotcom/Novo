"""
Módulo de Séries Temporais para Análise de Desemprego.

Inclui implementações de:
- SARIMA (Seasonal AutoRegressive Integrated Moving Average)
- GARCH (Generalized AutoRegressive Conditional Heteroskedasticity)

Autor: Luiz Tiago Wilcke
Data: 2025-11-24
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, Union
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import warnings

# Tenta importar arch para GARCH, se não tiver, usa implementação simplificada ou avisa
try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False

class ModeloSeriesTemporais:
    """
    Classe base para modelos de séries temporais.
    """
    def __init__(self, dados: pd.Series):
        self.dados = dados
        self.modelo_ajustado = None
        
    def testar_estacionariedade(self) -> Dict[str, float]:
        """Realiza teste Augmented Dickey-Fuller."""
        resultado = adfuller(self.dados.dropna())
        return {
            'estatistica_teste': resultado[0],
            'p_valor': resultado[1],
            'lags': resultado[2],
            'n_obs': resultado[3]
        }

class ModeloSARIMA(ModeloSeriesTemporais):
    """
    Implementação do modelo SARIMA para previsão de desemprego.
    """
    def __init__(self, dados: pd.Series, ordem: Tuple[int,int,int] = (1,1,1), 
                 ordem_sazonal: Tuple[int,int,int,int] = (1,1,1,12)):
        super().__init__(dados)
        self.ordem = ordem
        self.ordem_sazonal = ordem_sazonal
        
    def ajustar(self):
        """Ajusta o modelo aos dados."""
        try:
            modelo = SARIMAX(
                self.dados,
                order=self.ordem,
                seasonal_order=self.ordem_sazonal,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            self.modelo_ajustado = modelo.fit(disp=False)
            return self.modelo_ajustado.summary()
        except Exception as e:
            print(f"Erro ao ajustar SARIMA: {e}")
            return None
            
    def prever(self, passos: int = 12) -> pd.DataFrame:
        """
        Realiza previsões fora da amostra.
        
        Returns:
            DataFrame com previsão, intervalo de confiança inferior e superior.
        """
        if self.modelo_ajustado is None:
            raise ValueError("Modelo não ajustado. Execute ajustar() primeiro.")
            
        previsao = self.modelo_ajustado.get_forecast(steps=passos)
        media = previsao.predicted_mean
        conf_int = previsao.conf_int()
        
        df_resultado = pd.DataFrame({
            'previsao': media,
            'limite_inferior': conf_int.iloc[:, 0],
            'limite_superior': conf_int.iloc[:, 1]
        })
        return df_resultado
        
    def diagnostico(self):
        """Retorna diagnósticos do modelo."""
        if self.modelo_ajustado is None:
            return None
        return self.modelo_ajustado.plot_diagnostics(figsize=(10, 8))

class ModeloGARCH(ModeloSeriesTemporais):
    """
    Implementação de GARCH para modelar volatilidade do desemprego.
    Útil para analisar incerteza econômica.
    """
    def __init__(self, dados: pd.Series, p: int = 1, q: int = 1):
        super().__init__(dados)
        self.p = p
        self.q = q
        
    def ajustar(self):
        """Ajusta modelo GARCH(p,q)."""
        if not HAS_ARCH:
            warnings.warn("Biblioteca 'arch' não encontrada. GARCH não disponível.")
            return None
            
        # GARCH geralmente é aplicado aos retornos ou diferenças, não ao nível
        # Aqui vamos aplicar à primeira diferença da taxa de desemprego
        diff_dados = self.dados.diff().dropna() * 100 # Escala para facilitar convergência
        
        try:
            modelo = arch_model(diff_dados, vol='Garch', p=self.p, q=self.q)
            self.modelo_ajustado = modelo.fit(disp='off')
            return self.modelo_ajustado.summary()
        except Exception as e:
            print(f"Erro ao ajustar GARCH: {e}")
            return None
            
    def prever_volatilidade(self, horizonte: int = 12) -> pd.Series:
        """Prevê a volatilidade futura."""
        if self.modelo_ajustado is None:
            return pd.Series(np.zeros(horizonte))
            
        previsao = self.modelo_ajustado.forecast(horizon=horizonte)
        return np.sqrt(previsao.variance.values[-1, :])
