"""
Sistema de Treinamento de Redes Neurais
Autor: Luiz Tiago Wilcke
Data: 2025-11-25

Pipeline completo de treinamento com:
- Validação cruzada
- Early stopping
- Checkpointing
- Métricas detalhadas
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


class TreinadorRedeNeural:
    """Classe para treinar qualquer modelo neural"""
    
    def __init__(self,
                 modelo: nn.Module,
                 device: str = None,
                 checkpoint_dir: str = "checkpoints"):
        """
        Args:
            modelo: Modelo PyTorch
            device: Dispositivo (cuda/cpu)
            checkpoint_dir: Diretório para salvar checkpoints
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.modelo = modelo.to(self.device)
        self.checkpoint_dir = checkpoint_dir
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.historico = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
    
    def treinar(self,
                train_loader: DataLoader,
                val_loader: DataLoader,
                criterion,
                optimizer,
                num_epochs: int = 100,
                early_stopping_patience: int = 10,
                nome_modelo: str = "modelo"):
        """
        Treina o modelo com early stopping
        
        Args:
            train_loader: DataLoader de treino
            val_loader: DataLoader de validação
            criterion: Função de loss
            optimizer: Otimizador
            num_epochs: Número máximo de épocas
            early_stopping_patience: Paciência para early stopping
            nome_modelo: Nome para salvar checkpoint
        """
        melhor_val_loss = float('inf')
        epocas_sem_melhora = 0
        
        print(f"Treinando em: {self.device}")
        print(f"Modelo: {nome_modelo}")
        print("=" * 80)
        
        for epoch in range(num_epochs):
            # Treino
            self.modelo.train()
            train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward
                optimizer.zero_grad()
                outputs = self.modelo(batch_x)
                
                # Loss
                if isinstance(outputs, dict):  # Multitask
                    loss = criterion(outputs, batch_y)
                else:
                    loss = criterion(outputs, batch_y)
                
                # Backward
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validação
            val_loss = self.avaliar(val_loader, criterion)
            
            # Salvar histórico
            self.historico['train_loss'].append(train_loss)
            self.historico['val_loss'].append(val_loss)
            
            # Print
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Época {epoch+1}/{num_epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f}")
            
            # Early stopping e checkpoint
            if val_loss < melhor_val_loss:
                melhor_val_loss = val_loss
                epocas_sem_melhora = 0
                self.salvar_checkpoint(nome_modelo)
            else:
                epocas_sem_melhora += 1
            
            if epocas_sem_melhora >= early_stopping_patience:
                print(f"\nEarly stopping após {epoch+1} épocas")
                break
        
        print(f"\nTreinamento concluído!")
        print(f"Melhor Val Loss: {melhor_val_loss:.4f}")
        
        # Carregar melhor modelo
        self.carregar_checkpoint(nome_modelo)
    
    def avaliar(self, data_loader: DataLoader, criterion) -> float:
        """Avalia o modelo"""
        self.modelo.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.modelo(batch_x)
                
                if isinstance(outputs, dict):
                    loss = criterion(outputs, batch_y)
                else:
                    loss = criterion(outputs, batch_y)
                
                total_loss += loss.item()
        
        return total_loss / len(data_loader)
    
    def salvar_checkpoint(self, nome: str):
        """Salva checkpoint do modelo"""
        caminho = os.path.join(self.checkpoint_dir, f"{nome}_best.pth")
        torch.save({
            'model_state_dict': self.modelo.state_dict(),
            'historico': self.historico
        }, caminho)
    
    def carregar_checkpoint(self, nome: str):
        """Carrega checkpoint do modelo"""
        caminho = os.path.join(self.checkpoint_dir, f"{nome}_best.pth")
        if os.path.exists(caminho):
            checkpoint = torch.load(caminho, map_location=self.device)
            self.modelo.load_state_dict(checkpoint['model_state_dict'])
            self.historico = checkpoint['historico']
            print(f"✓ Checkpoint carregado de: {caminho}")
        else:
            print(f"⚠ Checkpoint não encontrado: {caminho}")
    
    def plotar_historico(self, salvar: str = None):
        """Plota curvas de aprendizado"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.historico['train_loss'], label='Train Loss', linewidth=2)
        plt.plot(self.historico['val_loss'], label='Val Loss', linewidth=2)
        plt.xlabel('Época')
        plt.ylabel('Loss')
        plt.title('Curvas de Aprendizado')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if salvar:
            plt.savefig(salvar, dpi=150, bbox_inches='tight')
        
        plt.close()


def preparar_dados_lstm(caminho_features: str, 
                        caminho_series: str,
                        test_size: float = 0.2,
                        batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    """Prepara dados para treinamento LSTM"""
    
    # Carregar
    X = np.load(caminho_features)
    y = np.load(caminho_series)
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Converter para tensores
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val)
    
    # DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader


def preparar_dados_classificacao(df: pd.DataFrame,
                                  test_size: float = 0.2,
                                  batch_size: int = 32) -> Tuple[DataLoader, DataLoader, LabelEncoder]:
    """Prepara dados para classificação de resposta"""
    
    # Features
    features_cols = ['idade', 'peso', 'imc', 'funcao_hepatica', 'funcao_renal',
                     'gravidade_sintomas', 'dose', 'Cmax', 'AUC', 'ocupacao_d2_media',
                     'eficacia_media']
    
    # One-hot encoding para categóricas
    df_encoded = df.copy()
    df_encoded['sexo_M'] = (df['sexo'] == 'M').astype(int)
    df_encoded['cyp3a4_lento'] = (df['cyp3a4'] == 'lento').astype(int)
    df_encoded['cyp3a4_rapido'] = (df['cyp3a4'] == 'rapido').astype(int)
    df_encoded['tratamento_previo'] = df['tratamento_previo'].astype(int)
    
    features_cols += ['sexo_M', 'cyp3a4_lento', 'cyp3a4_rapido', 'tratamento_previo']
    
    X = df_encoded[features_cols].values
    
    # Normalizar
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Labels
    le = LabelEncoder()
    y = le.fit_transform(df['resposta_clinica'])
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Tensores
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.LongTensor(y_val)
    
    # DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, le


if __name__ == "__main__":
    print("=" * 80)
    print("SISTEMA DE TREINAMENTO DE REDES NEURAIS")
    print("=" * 80)
    print()
    
    # Verificar PyTorch
    print(f"PyTorch versão: {torch.__version__}")
    print(f"CUDA disponível: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print("\n✓ Sistema de treinamento pronto!")
    print("Execute 'python treinar_modelos.py' para iniciar o treinamento")
