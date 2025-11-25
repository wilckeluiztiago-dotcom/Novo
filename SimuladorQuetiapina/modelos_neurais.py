"""
Modelos de Redes Neurais Avançados com PyTorch
Autor: Luiz Tiago Wilcke
Data: 2025-11-25

Implementa arquiteturas avançadas de redes neurais para:
1. Predição de séries temporais farmacocinéticas (LSTM)
2. Otimização de dosagem (Rede Feedforward)
3. Classificação de resposta terapêutica (Rede Profunda)
4. Análise de padrões (Autoencoder)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class LSTMFarmacocinetica(nn.Module):
    """
    LSTM para predição de séries temporais de concentração plasmática
    
    Arquitetura:
    - Camadas LSTM empilhadas com dropout
    - Camadas totalmente conectadas para decodificação
    - Predição de toda a série temporal
    """
    
    def __init__(self, 
                 input_size: int = 9,
                 hidden_size: int = 128,
                 num_layers: int = 3,
                 output_seq_len: int = 100,
                 dropout: float = 0.3):
        """
        Args:
            input_size: Número de características de entrada (idade, peso, etc.)
            hidden_size: Tamanho das camadas LSTM
            num_layers: Número de camadas LSTM
            output_seq_len: Comprimento da sequência de saída
            dropout: Taxa de dropout
        """
        super(LSTMFarmacocinetica, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_seq_len = output_seq_len
        
        # Camada de embedding para features
        self.feature_embedding = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM empilhada
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Camadas de decodificação
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, output_seq_len),
            nn.Softplus()  # Garante saídas positivas (concentrações)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Tensor de características [batch_size, input_size]
        
        Returns:
            Tensor de séries temporais preditas [batch_size, seq_len]
        """
        batch_size = x.size(0)
        
        # Embedding
        embedded = self.feature_embedding(x)  # [batch, 128]
        
        # Expandir para sequência (repetir embedding)
        embedded = embedded.unsqueeze(1).repeat(1, 10, 1)  # [batch, 10, 128]
        
        # LSTM
        lstm_out, _ = self.lstm(embedded)  # [batch, 10, hidden_size]
        
        # Pegar última saída
        last_output = lstm_out[:, -1, :]  # [batch, hidden_size]
        
        # Decodificar para série temporal completa
        output = self.decoder(last_output)  # [batch, output_seq_len]
        
        return output


class RedeOtimizacaoDose(nn.Module):
    """
    Rede neural profunda para otimização de dosagem
    
    Input: Características do paciente + dose atual
    Output: Dose ótima recomendada
    """
    
    def __init__(self, 
                 input_features: int = 12,
                 hidden_dims: list = [256, 512, 256, 128],
                 dropout: float = 0.25):
        """
        Args:
            input_features: Número de características de entrada
            hidden_dims: Dimensões das camadas ocultas
            dropout: Taxa de dropout
        """
        super(RedeOtimizacaoDose, self).__init__()
        
        # Construir camadas dinamicamente
        layers = []
        prev_dim = input_features
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Camada de saída
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # Saída entre 0 e 1, será escalada
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Características do paciente [batch_size, input_features]
        
        Returns:
            Dose ótima normalizada [batch_size, 1]
        """
        return self.network(x)
    
    def predict_dose(self, x: torch.Tensor, 
                     dose_min: float = 25.0, 
                     dose_max: float = 800.0) -> torch.Tensor:
        """
        Prediz dose real (não normalizada)
        
        Args:
            x: Características
            dose_min: Dose mínima (mg)
            dose_max: Dose máxima (mg)
        
        Returns:
            Dose em mg
        """
        normalized = self.forward(x)
        return dose_min + normalized * (dose_max - dose_min)


class ClassificadorRespostaTerapeutica(nn.Module):
    """
    Classificador de resposta terapêutica (boa/moderada/pobre)
    
    Usa arquitetura residual profunda com atenção
    """
    
    def __init__(self,
                 input_features: int = 15,
                 hidden_dim: int = 256,
                 num_classes: int = 3,
                 dropout: float = 0.3):
        """
        Args:
            input_features: Número de características
            hidden_dim: Dimensão das camadas ocultas
            num_classes: Número de classes (3: boa/moderada/pobre)
            dropout: Taxa de dropout
        """
        super(ClassificadorRespostaTerapeutica, self).__init__()
        
        # Camada de entrada
        self.input_layer = nn.Sequential(
            nn.Linear(input_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Blocos residuais
        self.res_block1 = self._make_residual_block(hidden_dim, dropout)
        self.res_block2 = self._make_residual_block(hidden_dim, dropout)
        
        # Mecanismo de atenção
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Softmax(dim=1)
        )
        
        # Classificador
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes)
        )
    
    def _make_residual_block(self, dim: int, dropout: float) -> nn.Module:
        """Cria bloco residual"""
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Características [batch_size, input_features]
        
        Returns:
            Logits de classe [batch_size, num_classes]
        """
        # Input
        out = self.input_layer(x)
        identity = out
        
        # Residual block 1
        out = self.res_block1(out)
        out = F.relu(out + identity)
        
        # Residual block 2
        identity = out
        out = self.res_block2(out)
        out = F.relu(out + identity)
        
        # Atenção
        attention_weights = self.attention(out)
        out = out * attention_weights
        
        # Classificação
        logits = self.classifier(out)
        
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna probabilidades de classe"""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


class AutoencoderPacientes(nn.Module):
    """
    Autoencoder variacional para análise de padrões em pacientes
    
    Aprende representação latente dos pacientes para:
    - Clustering automático
    - Detecção de outliers
    - Geração de pacientes sintéticos
    """
    
    def __init__(self,
                 input_dim: int = 20,
                 latent_dim: int = 8,
                 hidden_dims: list = [128, 64, 32]):
        """
        Args:
            input_dim: Dimensão de entrada
            latent_dim: Dimensão do espaço latente
            hidden_dims: Dimensões das camadas ocultas
        """
        super(AutoencoderPacientes, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Camadas de média e log-variância (VAE)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        decoder_layers.append(nn.Sigmoid())  # Normalizado entre 0 e 1
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Codifica entrada em espaço latente
        
        Returns:
            Tupla (mu, logvar)
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick para VAE"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodifica do espaço latente"""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Returns:
            Tupla (reconstrução, mu, logvar)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar
    
    def loss_function(self, 
                      recon_x: torch.Tensor, 
                      x: torch.Tensor,
                      mu: torch.Tensor, 
                      logvar: torch.Tensor,
                      beta: float = 1.0) -> torch.Tensor:
        """
        Loss do VAE (reconstrução + KL divergence)
        
        Args:
            recon_x: Reconstrução
            x: Original
            mu: Média
            logvar: Log-variância
            beta: Peso da KL divergence
        
        Returns:
            Loss total
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # KL divergence
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + beta * kl_div


class RedeMultitarefa(nn.Module):
    """
    Rede neural multitarefa que prediz múltiplos outputs simultaneamente:
    - Eficácia terapêutica (regressão)
    - Ocupação de D2 (regressão)
    - Risco de efeitos colaterais (regressão múltipla)
    - Resposta clínica (classificação)
    """
    
    def __init__(self, input_features: int = 15):
        super(RedeMultitarefa, self).__init__()
        
        # Tronco compartilhado
        self.shared_layers = nn.Sequential(
            nn.Linear(input_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Cabeça 1: Eficácia terapêutica
        self.eficacia_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 0-1, será escalado para 0-100
        )
        
        # Cabeça 2: Ocupação D2
        self.ocupacao_d2_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 0-1, será escalado para 0-100%
        )
        
        # Cabeça 3: Efeitos colaterais (6 outputs)
        self.efeitos_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 6),
            nn.Sigmoid()  # 0-1 para cada efeito
        )
        
        # Cabeça 4: Classificação de resposta
        self.resposta_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # 3 classes
        )
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass
        
        Returns:
            Dicionário com todas as predições
        """
        shared = self.shared_layers(x)
        
        return {
            'eficacia': self.eficacia_head(shared) * 100,  # Escalar para 0-100
            'ocupacao_d2': self.ocupacao_d2_head(shared) * 100,  # 0-100%
            'efeitos_colaterais': self.efeitos_head(shared) * 100,  # 0-100 para cada
            'resposta_logits': self.resposta_head(shared)
        }


if __name__ == "__main__":
    print("=" * 80)
    print("MODELOS DE REDES NEURAIS - TESTE DE ARQUITETURAS")
    print("=" * 80)
    print()
    
    # Testar LSTM
    print("1. LSTM Farmacocinética")
    lstm = LSTMFarmacocinetica()
    x_test = torch.randn(8, 9)  # Batch de 8 pacientes
    output = lstm(x_test)
    print(f"   Input: {x_test.shape}")
    print(f"   Output: {output.shape}")
    print(f"   Parâmetros: {sum(p.numel() for p in lstm.parameters()):,}")
    
    # Testar otimizador
    print("\n2. Rede Otimização de Dose")
    otimizador = RedeOtimizacaoDose()
    x_test = torch.randn(8, 12)
    output = otimizador.predict_dose(x_test)
    print(f"   Input: {x_test.shape}")
    print(f"   Output (doses): {output.shape}")
    print(f"   Parâmetros: {sum(p.numel() for p in otimizador.parameters()):,}")
    
    # Testar classificador
    print("\n3. Classificador de Resposta")
    classificador = ClassificadorRespostaTerapeutica()
    x_test = torch.randn(8, 15)
    output = classificador.predict_proba(x_test)
    print(f"   Input: {x_test.shape}")
    print(f"   Output (probabilidades): {output.shape}")
    print(f"   Parâmetros: {sum(p.numel() for p in classificador.parameters()):,}")
    
    # Testar autoencoder
    print("\n4. Autoencoder Variacional")
    vae = AutoencoderPacientes()
    x_test = torch.randn(8, 20)
    recon, mu, logvar = vae(x_test)
    print(f"   Input: {x_test.shape}")
    print(f"   Reconstrução: {recon.shape}")
    print(f"   Espaço latente: {mu.shape}")
    print(f"   Parâmetros: {sum(p.numel() for p in vae.parameters()):,}")
    
    # Testar multitarefa
    print("\n5. Rede Multitarefa")
    multitask = RedeMultitarefa()
    x_test = torch.randn(8, 15)
    outputs = multitask(x_test)
    print(f"   Input: {x_test.shape}")
    for k, v in outputs.items():
        print(f"   {k}: {v.shape}")
    print(f"   Parâmetros: {sum(p.numel() for p in multitask.parameters()):,}")
    
    print("\n" + "=" * 80)
    print("TODAS AS ARQUITETURAS TESTADAS COM SUCESSO!")
    print("=" * 80)
