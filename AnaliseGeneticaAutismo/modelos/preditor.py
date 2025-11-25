# modelos/preditor.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
import os

class PreditorGeneticoIA:
    """
    Utiliza algoritmos de Machine Learning para prever o risco de autismo/síndromes
    com base em genótipos e expressão gênica.
    """
    
    def __init__(self):
        self.modelo_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        from sklearn.neural_network import MLPClassifier
        self.modelo_nn = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        self.features_importantes = None
        
    def preparar_dados(self, df_genotipos: pd.DataFrame, df_expressao: pd.DataFrame) -> pd.DataFrame:
        """
        Combina genótipos e expressão gênica em um único vetor de features.
        """
        # Concatenar DataFrames (assumindo mesmo índice)
        X = pd.concat([df_genotipos, df_expressao], axis=1)
        return X
        
    def treinar(self, X: pd.DataFrame, y: pd.Series):
        """
        Treina os modelos de IA (Random Forest e Rede Neural).
        """
        print("Treinando Random Forest...")
        self.modelo_rf.fit(X, y)
        
        print("Treinando Rede Neural (Deep Learning)...")
        self.modelo_nn.fit(X, y)
        
        # Salvar importância das features (apenas RF tem isso nativo fácil)
        importancias = self.modelo_rf.feature_importances_
        self.features_importantes = pd.DataFrame({
            'Feature': X.columns,
            'Importancia': importancias
        }).sort_values('Importancia', ascending=False)
        
    def avaliar(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Avalia os modelos e retorna métricas detalhadas comparativas.
        """
        # Random Forest
        y_pred_rf = self.modelo_rf.predict(X)
        y_prob_rf = self.modelo_rf.predict_proba(X)[:, 1]
        auc_rf = roc_auc_score(y, y_prob_rf)
        
        # Rede Neural
        y_pred_nn = self.modelo_nn.predict(X)
        y_prob_nn = self.modelo_nn.predict_proba(X)[:, 1]
        auc_nn = roc_auc_score(y, y_prob_nn)
        
        report_rf = classification_report(y, y_pred_rf, output_dict=True)
        report_nn = classification_report(y, y_pred_nn, output_dict=True)
        
        return {
            'RandomForest': {
                'Relatorio': report_rf,
                'AUC_ROC': auc_rf,
                'Matriz_Confusao': confusion_matrix(y, y_pred_rf)
            },
            'RedeNeural': {
                'Relatorio': report_nn,
                'AUC_ROC': auc_nn,
                'Matriz_Confusao': confusion_matrix(y, y_pred_nn)
            }
        }
        
    def prever_novos(self, X: pd.DataFrame) -> np.ndarray:
        """
        Realiza predições para novos dados (média dos modelos).
        """
        prob_rf = self.modelo_rf.predict_proba(X)[:, 1]
        prob_nn = self.modelo_nn.predict_proba(X)[:, 1]
        return (prob_rf + prob_nn) / 2
