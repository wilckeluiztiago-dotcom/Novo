"""
Métricas de avaliação para modelos de análise eleitoral.
"""

import numpy as np
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


def calcular_metricas_regressao(y_real, y_pred, nome_modelo='Modelo'):
    """
    Calcula métricas para modelos de regressão.
    
    Parâmetros:
    -----------
    y_real : array-like
        Valores reais
    y_pred : array-like
        Valores preditos
    nome_modelo : str
        Nome do modelo para exibição
    
    Retorna:
    --------
    dict
        Dicionário com métricas
    """
    mae = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    r2 = r2_score(y_real, y_pred)
    
    # MAPE (Mean Absolute Percentage Error)
    # Evitar divisão por zero
    mask = y_real != 0
    mape = np.mean(np.abs((y_real[mask] - y_pred[mask]) / y_real[mask])) * 100
    
    # Erro médio
    erro_medio = np.mean(y_real - y_pred)
    
    # Erro padrão
    erro_padrao = np.std(y_real - y_pred)
    
    metricas = {
        'modelo': nome_modelo,
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2,
        'MAPE (%)': mape,
        'Erro Médio': erro_medio,
        'Erro Padrão': erro_padrao,
        'n_amostras': len(y_real)
    }
    
    return metricas


def calcular_metricas_classificacao(y_real, y_pred, nome_modelo='Modelo', labels=None):
    """
    Calcula métricas para modelos de classificação.
    
    Parâmetros:
    -----------
    y_real : array-like
        Classes reais
    y_pred : array-like
        Classes preditas
    nome_modelo : str
        Nome do modelo
    labels : list, opcional
        Lista de labels das classes
    
    Retorna:
    --------
    dict
        Dicionário com métricas
    """
    acuracia = accuracy_score(y_real, y_pred)
    
    # Para classificação binária
    if len(np.unique(y_real)) == 2:
        precisao = precision_score(y_real, y_pred, zero_division=0)
        recall = recall_score(y_real, y_pred, zero_division=0)
        f1 = f1_score(y_real, y_pred, zero_division=0)
    else:
        # Para multiclasse, usar média ponderada
        precisao = precision_score(y_real, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_real, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_real, y_pred, average='weighted', zero_division=0)
    
    # Matriz de confusão
    conf_matrix = confusion_matrix(y_real, y_pred)
    
    metricas = {
        'modelo': nome_modelo,
        'Acurácia': acuracia,
        'Precisão': precisao,
        'Recall': recall,
        'F1-Score': f1,
        'Matriz de Confusão': conf_matrix,
        'n_amostras': len(y_real)
    }
    
    return metricas


def calcular_intervalo_confianca(valores, confianca=0.95):
    """
    Calcula intervalo de confiança para uma série de valores.
    
    Parâmetros:
    -----------
    valores : array-like
        Valores para calcular IC
    confianca : float
        Nível de confiança (padrão: 0.95 para 95%)
    
    Retorna:
    --------
    tuple
        (limite_inferior, media, limite_superior)
    """
    from scipy import stats
    
    valores = np.array(valores)
    media = np.mean(valores)
    erro_padrao = stats.sem(valores)
    
    # Intervalo de confiança
    intervalo = stats.t.interval(
        confianca,
        len(valores) - 1,
        loc=media,
        scale=erro_padrao
    )
    
    return intervalo[0], media, intervalo[1]


def calcular_log_likelihood(y_real, y_pred_proba):
    """
    Calcula log-likelihood para modelos probabilísticos.
    
    Parâmetros:
    -----------
    y_real : array-like
        Valores reais (0 ou 1 para binário)
    y_pred_proba : array-like
        Probabilidades preditas
    
    Retorna:
    --------
    float
        Log-likelihood
    """
    # Evitar log(0)
    epsilon = 1e-15
    y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)
    
    log_likelihood = np.sum(
        y_real * np.log(y_pred_proba) + 
        (1 - y_real) * np.log(1 - y_pred_proba)
    )
    
    return log_likelihood


def calcular_aic_bic(log_likelihood, n_params, n_samples):
    """
    Calcula critérios de informação AIC e BIC.
    
    Parâmetros:
    -----------
    log_likelihood : float
        Log-likelihood do modelo
    n_params : int
        Número de parâmetros do modelo
    n_samples : int
        Número de amostras
    
    Retorna:
    --------
    tuple
        (AIC, BIC)
    """
    # AIC = 2k - 2ln(L)
    aic = 2 * n_params - 2 * log_likelihood
    
    # BIC = k*ln(n) - 2ln(L)
    bic = n_params * np.log(n_samples) - 2 * log_likelihood
    
    return aic, bic


def validacao_cruzada_temporal(modelo, X, y, n_splits=5):
    """
    Validação cruzada respeitando ordem temporal dos dados.
    
    Parâmetros:
    -----------
    modelo : objeto
        Modelo com métodos fit e predict
    X : array-like
        Features
    y : array-like
        Target
    n_splits : int
        Número de splits
    
    Retorna:
    --------
    dict
        Métricas médias e desvio padrão
    """
    from sklearn.model_selection import TimeSeriesSplit
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    scores_mae = []
    scores_rmse = []
    scores_r2 = []
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        
        scores_mae.append(mean_absolute_error(y_test, y_pred))
        scores_rmse.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        scores_r2.append(r2_score(y_test, y_pred))
    
    return {
        'MAE_medio': np.mean(scores_mae),
        'MAE_std': np.std(scores_mae),
        'RMSE_medio': np.mean(scores_rmse),
        'RMSE_std': np.std(scores_rmse),
        'R2_medio': np.mean(scores_r2),
        'R2_std': np.std(scores_r2)
    }


def calcular_erro_previsao_cadeiras(cadeiras_reais, cadeiras_previstas):
    """
    Calcula erro específico para previsão de cadeiras.
    
    Parâmetros:
    -----------
    cadeiras_reais : dict ou pd.Series
        Número real de cadeiras por partido
    cadeiras_previstas : dict ou pd.Series
        Número previsto de cadeiras por partido
    
    Retorna:
    --------
    dict
        Métricas de erro
    """
    import pandas as pd
    
    if isinstance(cadeiras_reais, dict):
        cadeiras_reais = pd.Series(cadeiras_reais)
    if isinstance(cadeiras_previstas, dict):
        cadeiras_previstas = pd.Series(cadeiras_previstas)
    
    # Alinhar índices
    partidos = cadeiras_reais.index.union(cadeiras_previstas.index)
    reais = cadeiras_reais.reindex(partidos, fill_value=0)
    previstas = cadeiras_previstas.reindex(partidos, fill_value=0)
    
    # Erro absoluto médio por partido
    mae_partidos = np.mean(np.abs(reais - previstas))
    
    # Erro total de cadeiras
    erro_total = np.sum(np.abs(reais - previstas))
    
    # Percentual de acerto (cadeiras corretas / total)
    acerto_percentual = 100 * (1 - erro_total / (2 * np.sum(reais)))
    
    # Partidos com previsão exata
    partidos_exatos = np.sum(reais == previstas)
    
    return {
        'MAE_por_partido': mae_partidos,
        'Erro_total_cadeiras': erro_total,
        'Acerto_percentual': acerto_percentual,
        'Partidos_previsao_exata': partidos_exatos,
        'Total_partidos': len(partidos)
    }
