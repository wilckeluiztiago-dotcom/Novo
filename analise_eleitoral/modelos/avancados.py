"""
Modelos avançados de machine learning para análise eleitoral.

Inclui:
- Random Forest
- Gradient Boosting (XGBoost)
- LSTM (Long Short-Term Memory) para séries temporais
- Ensemble Methods
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class ModeloRandomForest:
    """
    Random Forest para análise eleitoral.
    
    Ensemble de árvores de decisão que combina múltiplas previsões:
    
    ŷ = (1/B) Σᵢ₌₁ᴮ fᵢ(x)
    
    Onde:
    - B: número de árvores
    - fᵢ(x): previsão da i-ésima árvore
    
    Vantagens:
    - Captura relações não-lineares
    - Robusto a outliers
    - Fornece importância de features
    """
    
    def __init__(self, n_arvores=100, max_profundidade=None, tipo='regressao'):
        """
        Parâmetros:
        -----------
        n_arvores : int
            Número de árvores na floresta
        max_profundidade : int, opcional
            Profundidade máxima das árvores
        tipo : str
            'regressao' ou 'classificacao'
        """
        self.n_arvores = n_arvores
        self.max_profundidade = max_profundidade
        self.tipo = tipo
        self.features_nomes = None
        
        if tipo == 'regressao':
            self.modelo = RandomForestRegressor(
                n_estimators=n_arvores,
                max_depth=max_profundidade,
                random_state=42,
                n_jobs=-1
            )
        else:
            self.modelo = RandomForestClassifier(
                n_estimators=n_arvores,
                max_depth=max_profundidade,
                random_state=42,
                n_jobs=-1
            )
    
    def treinar(self, X, y, features_nomes=None):
        """Treina o modelo Random Forest."""
        if isinstance(X, pd.DataFrame):
            self.features_nomes = X.columns.tolist()
            X = X.values
        else:
            self.features_nomes = features_nomes or [f'X{i}' for i in range(X.shape[1])]
        
        self.modelo.fit(X, y)
        return self
    
    def prever(self, X):
        """Faz previsões."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.modelo.predict(X)
    
    def prever_probabilidade(self, X):
        """Retorna probabilidades (apenas para classificação)."""
        if self.tipo != 'classificacao':
            raise ValueError("Probabilidades disponíveis apenas para classificação")
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.modelo.predict_proba(X)
    
    def obter_importancia_features(self):
        """Retorna importância de cada feature."""
        importancias = pd.DataFrame({
            'feature': self.features_nomes,
            'importancia': self.modelo.feature_importances_
        }).sort_values('importancia', ascending=False)
        
        return importancias
    
    def obter_score(self, X, y):
        """Retorna score do modelo (R² ou acurácia)."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.modelo.score(X, y)


class ModeloGradientBoosting:
    """
    Gradient Boosting para análise eleitoral.
    
    Constrói modelo aditivo sequencialmente:
    
    Fₘ(x) = Fₘ₋₁(x) + ν·hₘ(x)
    
    Onde:
    - Fₘ(x): modelo na iteração m
    - hₘ(x): nova árvore que corrige erros
    - ν: taxa de aprendizado
    
    Cada nova árvore é treinada no gradiente negativo da função de perda.
    """
    
    def __init__(self, n_estimadores=100, taxa_aprendizado=0.1, max_profundidade=3, tipo='regressao'):
        """
        Parâmetros:
        -----------
        n_estimadores : int
            Número de boosting stages
        taxa_aprendizado : float
            Taxa de aprendizado
        max_profundidade : int
            Profundidade máxima de cada árvore
        tipo : str
            'regressao' ou 'classificacao'
        """
        self.tipo = tipo
        self.features_nomes = None
        
        if tipo == 'regressao':
            self.modelo = GradientBoostingRegressor(
                n_estimators=n_estimadores,
                learning_rate=taxa_aprendizado,
                max_depth=max_profundidade,
                random_state=42
            )
        else:
            self.modelo = GradientBoostingClassifier(
                n_estimators=n_estimadores,
                learning_rate=taxa_aprendizado,
                max_depth=max_profundidade,
                random_state=42
            )
    
    def treinar(self, X, y, features_nomes=None):
        """Treina o modelo Gradient Boosting."""
        if isinstance(X, pd.DataFrame):
            self.features_nomes = X.columns.tolist()
            X = X.values
        else:
            self.features_nomes = features_nomes or [f'X{i}' for i in range(X.shape[1])]
        
        self.modelo.fit(X, y)
        return self
    
    def prever(self, X):
        """Faz previsões."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.modelo.predict(X)
    
    def obter_importancia_features(self):
        """Retorna importância de features."""
        return pd.DataFrame({
            'feature': self.features_nomes,
            'importancia': self.modelo.feature_importances_
        }).sort_values('importancia', ascending=False)


class ModeloXGBoost:
    """
    XGBoost - Extreme Gradient Boosting.
    
    Versão otimizada de Gradient Boosting com regularização:
    
    Obj = Σᵢ L(yᵢ, ŷᵢ) + Σₖ Ω(fₖ)
    
    Onde:
    - L: função de perda
    - Ω: termo de regularização
    - fₖ: k-ésima árvore
    
    Regularização: Ω(f) = γT + (λ/2)||w||²
    - T: número de folhas
    - w: pesos das folhas
    """
    
    def __init__(self, n_estimadores=100, taxa_aprendizado=0.1, max_profundidade=6, tipo='regressao'):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost não está instalado. Instale com: pip install xgboost")
        
        self.tipo = tipo
        self.features_nomes = None
        
        params = {
            'n_estimators': n_estimadores,
            'learning_rate': taxa_aprendizado,
            'max_depth': max_profundidade,
            'random_state': 42
        }
        
        if tipo == 'regressao':
            self.modelo = xgb.XGBRegressor(**params)
        else:
            self.modelo = xgb.XGBClassifier(**params)
    
    def treinar(self, X, y, features_nomes=None):
        """Treina o modelo XGBoost."""
        if isinstance(X, pd.DataFrame):
            self.features_nomes = X.columns.tolist()
            X = X.values
        else:
            self.features_nomes = features_nomes or [f'X{i}' for i in range(X.shape[1])]
        
        self.modelo.fit(X, y)
        return self
    
    def prever(self, X):
        """Faz previsões."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.modelo.predict(X)
    
    def obter_importancia_features(self):
        """Retorna importância de features."""
        return pd.DataFrame({
            'feature': self.features_nomes,
            'importancia': self.modelo.feature_importances_
        }).sort_values('importancia', ascending=False)


class ModeloLSTM:
    """
    LSTM (Long Short-Term Memory) para séries temporais eleitorais.
    
    Rede neural recorrente que mantém memória de longo prazo:
    
    Equações do LSTM:
    - fₜ = σ(Wf·[hₜ₋₁, xₜ] + bf)  # forget gate
    - iₜ = σ(Wi·[hₜ₋₁, xₜ] + bi)  # input gate
    - C̃ₜ = tanh(Wc·[hₜ₋₁, xₜ] + bc)  # candidate values
    - Cₜ = fₜ * Cₜ₋₁ + iₜ * C̃ₜ  # cell state
    - oₜ = σ(Wo·[hₜ₋₁, xₜ] + bo)  # output gate
    - hₜ = oₜ * tanh(Cₜ)  # hidden state
    
    Ideal para capturar padrões temporais complexos em dados eleitorais.
    """
    
    def __init__(self, n_passos=5, n_features=1, n_unidades=50, n_camadas=2):
        """
        Parâmetros:
        -----------
        n_passos : int
            Número de time steps para olhar para trás
        n_features : int
            Número de features por time step
        n_unidades : int
            Número de unidades LSTM por camada
        n_camadas : int
            Número de camadas LSTM
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow não está instalado. Instale com: pip install tensorflow")
        
        self.n_passos = n_passos
        self.n_features = n_features
        self.n_unidades = n_unidades
        self.n_camadas = n_camadas
        self.modelo = None
        self.scaler = StandardScaler()
        
    def _criar_modelo(self):
        """Cria arquitetura do modelo LSTM."""
        modelo = Sequential()
        
        # Primeira camada LSTM
        modelo.add(LSTM(
            self.n_unidades,
            activation='tanh',
            return_sequences=True if self.n_camadas > 1 else False,
            input_shape=(self.n_passos, self.n_features)
        ))
        modelo.add(Dropout(0.2))
        
        # Camadas LSTM adicionais
        for i in range(1, self.n_camadas):
            return_seq = i < self.n_camadas - 1
            modelo.add(LSTM(self.n_unidades, activation='tanh', return_sequences=return_seq))
            modelo.add(Dropout(0.2))
        
        # Camada de saída
        modelo.add(Dense(1))
        
        modelo.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return modelo
    
    def _preparar_dados(self, serie):
        """Prepara dados em formato de sequências para LSTM."""
        X, y = [], []
        
        for i in range(len(serie) - self.n_passos):
            X.append(serie[i:i + self.n_passos])
            y.append(serie[i + self.n_passos])
        
        return np.array(X), np.array(y)
    
    def treinar(self, serie_temporal, epochs=50, batch_size=32, validacao_split=0.2):
        """
        Treina o modelo LSTM.
        
        Parâmetros:
        -----------
        serie_temporal : array-like
            Série temporal de dados eleitorais
        epochs : int
            Número de épocas de treinamento
        batch_size : int
            Tamanho do batch
        validacao_split : float
            Proporção de dados para validação
        """
        # Normalizar dados
        serie_scaled = self.scaler.fit_transform(
            np.array(serie_temporal).reshape(-1, 1)
        ).flatten()
        
        # Preparar sequências
        X, y = self._preparar_dados(serie_scaled)
        X = X.reshape((X.shape[0], X.shape[1], self.n_features))
        
        # Criar e treinar modelo
        self.modelo = self._criar_modelo()
        
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        self.modelo.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validacao_split,
            callbacks=[early_stop],
            verbose=0
        )
        
        return self
    
    def prever(self, serie_recente, n_periodos=1):
        """
        Faz previsões para períodos futuros.
        
        Parâmetros:
        -----------
        serie_recente : array-like
            Últimos n_passos valores da série
        n_periodos : int
            Número de períodos à frente para prever
        
        Retorna:
        --------
        array
            Previsões
        """
        # Normalizar
        serie_scaled = self.scaler.transform(
            np.array(serie_recente).reshape(-1, 1)
        ).flatten()
        
        previsoes = []
        entrada = serie_scaled[-self.n_passos:].copy()
        
        for _ in range(n_periodos):
            # Preparar entrada
            X_pred = entrada.reshape(1, self.n_passos, self.n_features)
            
            # Prever próximo valor
            y_pred = self.modelo.predict(X_pred, verbose=0)[0, 0]
            previsoes.append(y_pred)
            
            # Atualizar entrada (rolling window)
            entrada = np.append(entrada[1:], y_pred)
        
        # Desnormalizar previsões
        previsoes = self.scaler.inverse_transform(
            np.array(previsoes).reshape(-1, 1)
        ).flatten()
        
        return previsoes


class ModeloEnsemble:
    """
    Ensemble de múltiplos modelos para previsão robusta.
    
    Combina previsões de diferentes modelos:
    
    ŷ_ensemble = Σᵢ wᵢ·ŷᵢ
    
    Onde:
    - wᵢ: peso do modelo i
    - ŷᵢ: previsão do modelo i
    
    Métodos de combinação:
    - Média simples: wᵢ = 1/n
    - Média ponderada: wᵢ baseado em performance
    - Stacking: meta-modelo aprende pesos ótimos
    """
    
    def __init__(self, modelos, pesos=None):
        """
        Parâmetros:
        -----------
        modelos : list
            Lista de modelos treinados
        pesos : list, opcional
            Pesos para cada modelo (se None, usa média simples)
        """
        self.modelos = modelos
        self.pesos = pesos or [1/len(modelos)] * len(modelos)
        
        if len(self.pesos) != len(self.modelos):
            raise ValueError("Número de pesos deve ser igual ao número de modelos")
        
        # Normalizar pesos
        soma_pesos = sum(self.pesos)
        self.pesos = [p/soma_pesos for p in self.pesos]
    
    def prever(self, X):
        """
        Faz previsão combinando todos os modelos.
        
        Parâmetros:
        -----------
        X : array-like
            Dados para previsão
        
        Retorna:
        --------
        array
            Previsões combinadas
        """
        previsoes = []
        
        for modelo in self.modelos:
            pred = modelo.prever(X)
            previsoes.append(pred)
        
        # Combinar previsões com pesos
        previsoes = np.array(previsoes)
        previsao_final = np.average(previsoes, axis=0, weights=self.pesos)
        
        return previsao_final
    
    def ajustar_pesos_por_performance(self, X_val, y_val):
        """
        Ajusta pesos baseado em performance em conjunto de validação.
        
        Parâmetros:
        -----------
        X_val : array-like
            Features de validação
        y_val : array-like
            Target de validação
        """
        from sklearn.metrics import mean_squared_error
        
        erros = []
        for modelo in self.modelos:
            pred = modelo.prever(X_val)
            mse = mean_squared_error(y_val, pred)
            erros.append(mse)
        
        # Inverter erros para obter pesos (menor erro = maior peso)
        erros_inv = [1/e if e > 0 else 1 for e in erros]
        soma = sum(erros_inv)
        self.pesos = [e/soma for e in erros_inv]
        
        return self.pesos
