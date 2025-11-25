"""
Modelos estatísticos básicos para análise eleitoral.

Inclui:
- Regressão Linear Múltipla
- Regressão Logística
- ARIMA/SARIMA para séries temporais
- PCA (Análise de Componentes Principais)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')


class ModeloRegressao:
    """
    Modelo de Regressão Linear Múltipla para análise eleitoral.
    
    Equação:
    Y = β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ + ε
    
    Onde:
    - Y: votos do candidato
    - X₁, X₂, ..., Xₙ: variáveis explicativas (gastos, tempo TV, etc.)
    - β₀, β₁, ..., βₙ: coeficientes a estimar
    - ε: erro aleatório
    """
    
    def __init__(self):
        self.modelo = LinearRegression()
        self.scaler = StandardScaler()
        self.features_nomes = None
        self.coeficientes = None
        
    def treinar(self, X, y, features_nomes=None):
        """
        Treina o modelo de regressão.
        
        Parâmetros:
        -----------
        X : array-like ou DataFrame
            Features (variáveis independentes)
        y : array-like
            Target (votos)
        features_nomes : list, opcional
            Nomes das features
        """
        if isinstance(X, pd.DataFrame):
            self.features_nomes = X.columns.tolist()
            X = X.values
        else:
            self.features_nomes = features_nomes or [f'X{i}' for i in range(X.shape[1])]
        
        # Normalizar features
        X_scaled = self.scaler.fit_transform(X)
        
        # Treinar modelo
        self.modelo.fit(X_scaled, y)
        
        # Armazenar coeficientes
        self.coeficientes = pd.DataFrame({
            'feature': self.features_nomes,
            'coeficiente': self.modelo.coef_,
            'importancia_abs': np.abs(self.modelo.coef_)
        }).sort_values('importancia_abs', ascending=False)
        
        return self
    
    def prever(self, X):
        """Faz previsões com o modelo treinado."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_scaled = self.scaler.transform(X)
        return self.modelo.predict(X_scaled)
    
    def obter_coeficientes(self):
        """Retorna os coeficientes do modelo."""
        return self.coeficientes
    
    def obter_r2(self, X, y):
        """Calcula R² do modelo."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_scaled = self.scaler.transform(X)
        return self.modelo.score(X_scaled, y)


class ModeloRegressaoLogistica:
    """
    Modelo de Regressão Logística para prever eleição de candidatos.
    
    Equação:
    P(Y=1|X) = 1 / (1 + e^(-(β₀ + β₁X₁ + ... + βₙXₙ)))
    
    Onde:
    - P(Y=1|X): probabilidade de ser eleito
    - X₁, ..., Xₙ: variáveis explicativas
    - β₀, ..., βₙ: coeficientes a estimar
    """
    
    def __init__(self, max_iter=1000):
        self.modelo = LogisticRegression(max_iter=max_iter, random_state=42)
        self.scaler = StandardScaler()
        self.features_nomes = None
        
    def treinar(self, X, y, features_nomes=None):
        """
        Treina o modelo de regressão logística.
        
        Parâmetros:
        -----------
        X : array-like ou DataFrame
            Features
        y : array-like
            Target binário (0=não eleito, 1=eleito)
        features_nomes : list, opcional
            Nomes das features
        """
        if isinstance(X, pd.DataFrame):
            self.features_nomes = X.columns.tolist()
            X = X.values
        else:
            self.features_nomes = features_nomes or [f'X{i}' for i in range(X.shape[1])]
        
        X_scaled = self.scaler.fit_transform(X)
        self.modelo.fit(X_scaled, y)
        
        return self
    
    def prever_probabilidade(self, X):
        """Retorna probabilidade de eleição."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_scaled = self.scaler.transform(X)
        return self.modelo.predict_proba(X_scaled)[:, 1]
    
    def prever(self, X):
        """Retorna classe prevista (0 ou 1)."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_scaled = self.scaler.transform(X)
        return self.modelo.predict(X_scaled)
    
    def obter_coeficientes(self):
        """Retorna os coeficientes do modelo."""
        return pd.DataFrame({
            'feature': self.features_nomes,
            'coeficiente': self.modelo.coef_[0],
            'odds_ratio': np.exp(self.modelo.coef_[0])
        }).sort_values('odds_ratio', ascending=False)


class ModeloARIMA:
    """
    Modelo ARIMA para previsão de séries temporais eleitorais.
    
    ARIMA(p, d, q):
    - p: ordem autoregressiva (AR)
    - d: grau de diferenciação
    - q: ordem de média móvel (MA)
    
    Equação geral:
    (1 - φ₁B - ... - φₚBᵖ)(1-B)ᵈyₜ = (1 + θ₁B + ... + θᵧBᵧ)εₜ
    
    Onde:
    - yₜ: valor no tempo t
    - B: operador de atraso (Byₜ = yₜ₋₁)
    - φᵢ: parâmetros AR
    - θⱼ: parâmetros MA
    - εₜ: erro no tempo t
    """
    
    def __init__(self, order=(1, 1, 1)):
        """
        Parâmetros:
        -----------
        order : tuple
            (p, d, q) para ARIMA
        """
        self.order = order
        self.modelo = None
        self.resultado = None
        
    def treinar(self, serie_temporal):
        """
        Treina o modelo ARIMA.
        
        Parâmetros:
        -----------
        serie_temporal : array-like ou Series
            Série temporal de dados eleitorais
        """
        self.modelo = ARIMA(serie_temporal, order=self.order)
        self.resultado = self.modelo.fit()
        return self
    
    def prever(self, n_periodos=1):
        """
        Faz previsão para períodos futuros.
        
        Parâmetros:
        -----------
        n_periodos : int
            Número de períodos à frente para prever
        
        Retorna:
        --------
        array
            Previsões
        """
        return self.resultado.forecast(steps=n_periodos)
    
    def obter_resumo(self):
        """Retorna resumo estatístico do modelo."""
        return self.resultado.summary()
    
    def obter_aic_bic(self):
        """Retorna critérios de informação AIC e BIC."""
        return {
            'AIC': self.resultado.aic,
            'BIC': self.resultado.bic
        }


class ModeloSARIMA:
    """
    Modelo SARIMA para séries temporais com sazonalidade.
    
    SARIMA(p,d,q)(P,D,Q,s):
    - (p,d,q): componentes não-sazonais
    - (P,D,Q,s): componentes sazonais
    - s: período sazonal (ex: 4 para eleições a cada 4 anos)
    
    Útil para modelar padrões eleitorais que se repetem a cada ciclo eleitoral.
    """
    
    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 4)):
        """
        Parâmetros:
        -----------
        order : tuple
            (p, d, q) componentes não-sazonais
        seasonal_order : tuple
            (P, D, Q, s) componentes sazonais
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.modelo = None
        self.resultado = None
        
    def treinar(self, serie_temporal):
        """Treina o modelo SARIMA."""
        self.modelo = SARIMAX(
            serie_temporal,
            order=self.order,
            seasonal_order=self.seasonal_order
        )
        self.resultado = self.modelo.fit(disp=False)
        return self
    
    def prever(self, n_periodos=1):
        """Faz previsão para períodos futuros."""
        return self.resultado.forecast(steps=n_periodos)
    
    def obter_resumo(self):
        """Retorna resumo estatístico."""
        return self.resultado.summary()


class ModeloPCA:
    """
    Análise de Componentes Principais para redução dimensional.
    
    Transforma variáveis correlacionadas em componentes principais não-correlacionados:
    
    Z = XW
    
    Onde:
    - X: matriz de dados originais (n × p)
    - W: matriz de autovetores (p × k)
    - Z: componentes principais (n × k)
    
    Útil para:
    - Reduzir dimensionalidade de muitas variáveis eleitorais
    - Identificar padrões latentes
    - Visualização de dados multidimensionais
    """
    
    def __init__(self, n_componentes=None, variancia_explicada=0.95):
        """
        Parâmetros:
        -----------
        n_componentes : int, opcional
            Número de componentes a manter
        variancia_explicada : float
            Manter componentes que explicam esta % de variância
        """
        self.n_componentes = n_componentes
        self.variancia_explicada = variancia_explicada
        self.pca = None
        self.scaler = StandardScaler()
        self.features_nomes = None
        
    def treinar(self, X, features_nomes=None):
        """
        Treina o PCA.
        
        Parâmetros:
        -----------
        X : array-like ou DataFrame
            Dados originais
        features_nomes : list, opcional
            Nomes das features
        """
        if isinstance(X, pd.DataFrame):
            self.features_nomes = X.columns.tolist()
            X = X.values
        else:
            self.features_nomes = features_nomes or [f'X{i}' for i in range(X.shape[1])]
        
        # Normalizar dados
        X_scaled = self.scaler.fit_transform(X)
        
        # Aplicar PCA
        if self.n_componentes is None:
            # Determinar número de componentes pela variância explicada
            pca_temp = PCA()
            pca_temp.fit(X_scaled)
            var_acumulada = np.cumsum(pca_temp.explained_variance_ratio_)
            self.n_componentes = np.argmax(var_acumulada >= self.variancia_explicada) + 1
        
        self.pca = PCA(n_components=self.n_componentes)
        self.pca.fit(X_scaled)
        
        return self
    
    def transformar(self, X):
        """Transforma dados para espaço de componentes principais."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_scaled = self.scaler.transform(X)
        return self.pca.transform(X_scaled)
    
    def obter_variancia_explicada(self):
        """Retorna variância explicada por cada componente."""
        return pd.DataFrame({
            'componente': [f'PC{i+1}' for i in range(self.n_componentes)],
            'variancia_explicada': self.pca.explained_variance_ratio_,
            'variancia_acumulada': np.cumsum(self.pca.explained_variance_ratio_)
        })
    
    def obter_loadings(self):
        """Retorna loadings (contribuição de cada variável original)."""
        loadings = pd.DataFrame(
            self.pca.components_.T,
            columns=[f'PC{i+1}' for i in range(self.n_componentes)],
            index=self.features_nomes
        )
        return loadings
    
    def reconstruir(self, X_transformado):
        """Reconstrói dados originais a partir dos componentes principais."""
        X_reconstruido = self.pca.inverse_transform(X_transformado)
        return self.scaler.inverse_transform(X_reconstruido)
