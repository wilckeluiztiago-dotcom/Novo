"""
Redes Neurais para Análise Genética
Autor: Luiz Tiago Wilcke
Data: 2025-11-25

Arquiteturas neurais sofisticadas para análise de perfis genéticos:
1. Graph Neural Network (GNN) para vias metabólicas
2. Transformer para sequências genéticas
3. Attention-based model para interações gene-droga
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import numpy as np


class GNNViasMetabolicas(nn.Module):
    """
    Graph Neural Network para modelar vias metabólicas
    
    Nós = Genes/Enzimas/Metabólitos
    Arestas = Interações bioquímicas
    """
    
    def __init__(self,
                 num_genes: int = 15,
                 embedding_dim: int = 64,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 output_dim: int = 32):
        """
        Args:
            num_genes: Número de genes farmacogenômicos
            embedding_dim: Dimensão de embedding dos nós
            hidden_dim: Dimensão das camadas ocultas
            num_layers: Número de camadas GNN
            output_dim: Dimensão de saída (representação da via)
        """
        super(GNNViasMetabolicas, self).__init__()
        
        self.num_genes = num_genes
        self.embedding_dim = embedding_dim
        
        # Embedding de genes
        self.gene_embedding = nn.Embedding(num_genes, embedding_dim)
        
        # Camadas de convolução de grafo (Graph Convolution)
        self.conv_layers = nn.ModuleList()
        
        # Primeira camada
        self.conv_layers.append(GraphConvLayer(embedding_dim, hidden_dim))
        
        # Camadas intermediárias
        for _ in range(num_layers - 2):
            self.conv_layers.append(GraphConvLayer(hidden_dim, hidden_dim))
        
        # Última camada
        self.conv_layers.append(GraphConvLayer(hidden_dim, output_dim))
        
        # Pooling global (agregação de nós)
        self.global_pool = nn.Sequential(
            nn.Linear(output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        
        # Camada de predição
        self.predictor = nn.Sequential(
            nn.Linear(output_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # 3 outputs: metabolismo, resposta, risco
        )
    
    def forward(self, 
                gene_ids: torch.Tensor,
                adj_matrix: torch.Tensor,
                node_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            gene_ids: IDs dos genes [batch_size, num_genes]
            adj_matrix: Matriz de adjacência [batch_size, num_genes, num_genes]
            node_features: Features adicionais dos nós [batch_size, num_genes, feature_dim]
        
        Returns:
            Predições [batch_size, 3]
        """
        batch_size = gene_ids.size(0)
        
        # Embedding
        x = self.gene_embedding(gene_ids)  # [batch, num_genes, emb_dim]
        
        # Concatenar features adicionais se fornecidas
        if node_features is not None:
            x = torch.cat([x, node_features], dim=-1)
        
        # Aplicar camadas de convolução de grafo
        for conv in self.conv_layers:
            x = conv(x, adj_matrix)
            x = F.relu(x)
        
        # Global pooling (mean pooling)
        x_pooled = torch.mean(x, dim=1)  # [batch, output_dim]
        
        # Predição
        out = self.predictor(x_pooled)
        
        return out


class GraphConvLayer(nn.Module):
    """Camada de convolução de grafo simples"""
    
    def __init__(self, in_features: int, out_features: int):
        super(GraphConvLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Features dos nós [batch, num_nodes, in_features]
            adj: Matriz de adjacência [batch, num_nodes, num_nodes]
        
        Returns:
            Features atualizadas [batch, num_nodes, out_features]
        """
        # Normalizar adjacência (adicionar self-loops)
        adj_norm = adj + torch.eye(adj.size(1), device=adj.device).unsqueeze(0)
        
        # Normalização por grau
        degree = adj_norm.sum(dim=-1, keepdim=True)
        adj_norm = adj_norm / (degree + 1e-6)
        
        # Agregação: A * X * W
        out = torch.bmm(adj_norm, x)  # Agregar vizinhos
        out = self.linear(out)  # Transformar
        
        return out


class TransformerSequenciaGenetica(nn.Module):
    """
    Transformer para processar sequências de variantes genéticas
    
    Input: Sequência de variantes (alelos, genótipos)
    Output: Representação contextual do perfil genético
    """
    
    def __init__(self,
                 vocab_size: int = 100,  # Número de alelos possíveis
                 d_model: int = 128,
                 nhead: int = 8,
                 num_layers: int = 4,
                 dim_feedforward: int = 512,
                 max_seq_len: int = 50,
                 output_dim: int = 64):
        """
        Args:
            vocab_size: Tamanho do vocabulário de alelos
            d_model: Dimensão do modelo
            nhead: Número de cabeças de atenção
            num_layers: Número de camadas transformer
            dim_feedforward: Dimensão da camada feedforward
            max_seq_len: Comprimento máximo da sequência
            output_dim: Dimensão de saída
        """
        super(TransformerSequenciaGenetica, self).__init__()
        
        self.d_model = d_model
        
        # Embedding de alelos
        self.alelo_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Pooling e projeção
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, 
                alelo_ids: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            alelo_ids: IDs dos alelos [batch_size, seq_len]
            mask: Máscara de padding [batch_size, seq_len]
        
        Returns:
            Representação genética [batch_size, output_dim]
        """
        # Embedding
        x = self.alelo_embedding(alelo_ids) * np.sqrt(self.d_model)
        
        # Positional encoding
        x = self.pos_encoding(x)
        
        # Criar máscara de atenção se necessário
        src_key_padding_mask = None
        if mask is not None:
            src_key_padding_mask = (mask == 0)
        
        # Transformer
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # Global pooling (pegar CLS token ou fazer mean)
        x_pooled = torch.mean(x, dim=1)  # [batch, d_model]
        
        # Projeção
        out = self.output_projection(x_pooled)
        
        return out


class PositionalEncoding(nn.Module):
    """Positional encoding para Transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        # Criar positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        """
        return x + self.pe[:, :x.size(1), :]


class MultiHeadGeneticAttention(nn.Module):
    """
    Atenção multi-cabeça especializada para interações genéticas
    
    Aprende quais genes interagem mais fortemente no contexto da Quetiapina
    """
    
    def __init__(self,
                 num_genes: int = 15,
                 gene_feature_dim: int = 32,
                 num_heads: int = 8,
                 hidden_dim: int = 256):
        """
        Args:
            num_genes: Número de genes
            gene_feature_dim: Dimensão de features por gene
            num_heads: Número de cabeças de atenção
            hidden_dim: Dimensão oculta
        """
        super(MultiHeadGeneticAttention, self).__init__()
        
        self.num_heads = num_heads
        self.gene_feature_dim = gene_feature_dim
        self.head_dim = hidden_dim // num_heads
        
        # Projeções Q, K, V
        self.query = nn.Linear(gene_feature_dim, hidden_dim)
        self.key = nn.Linear(gene_feature_dim, hidden_dim)
        self.value = nn.Linear(gene_feature_dim, hidden_dim)
        
        # Projeção de saída
        self.out_proj = nn.Linear(hidden_dim, gene_feature_dim)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(gene_feature_dim)
        
    def forward(self, gene_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            gene_features: [batch, num_genes, gene_feature_dim]
        
        Returns:
            Tupla (features_atendidas, attention_weights)
        """
        batch_size, num_genes, _ = gene_features.size()
        
        # Projetar Q, K, V
        Q = self.query(gene_features)  # [batch, num_genes, hidden_dim]
        K = self.key(gene_features)
        V = self.value(gene_features)
        
        # Reshape para multi-head
        Q = Q.view(batch_size, num_genes, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, num_genes, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, num_genes, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Aplicar atenção
        attended = torch.matmul(attention_weights, V)
        
        # Concatenar cabeças
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, num_genes, -1
        )
        
        # Projeção de saída
        out = self.out_proj(attended)
        
        # Residual + layer norm
        out = self.layer_norm(gene_features + out)
        
        # Pegar attention weights da primeira cabeça para visualização
        attention_viz = attention_weights[:, 0, :, :]  # [batch, num_genes, num_genes]
        
        return out, attention_viz


class PreditorFarmacogenomicoIntegrado(nn.Module):
    """
    Modelo integrado que combina GNN, Transformer e Attention
    para predição farmacogenômica completa
    """
    
    def __init__(self):
        super(PreditorFarmacogenomicoIntegrado, self).__init__()
        
        # Componentes
        self.gnn = GNNViasMetabolicas(
            num_genes=15,
            output_dim=32
        )
        
        self.transformer = TransformerSequenciaGenetica(
            vocab_size=100,
            output_dim=64
        )
        
        self.attention = MultiHeadGeneticAttention(
            num_genes=15,
            gene_feature_dim=32
        )
        
        # Fusion layer (3 from out + 64 from transformer + 32 from attention = 99)
        self.fusion = nn.Sequential(
            nn.Linear(3 + 64 + 32, 256),  # GNN outputs 3, not 32
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Cabeças de predição
        self.metabolismo_head = nn.Linear(128, 1)  # Score de metabolismo
        self.resposta_head = nn.Linear(128, 1)  # Score de resposta
        self.dose_head = nn.Linear(128, 1)  # Dose recomendada
        self.risco_head = nn.Linear(128, 5)  # 5 tipos de riscos
        
    def forward(self,
                gene_ids: torch.Tensor,
                adj_matrix: torch.Tensor,
                alelo_sequence: torch.Tensor,
                gene_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass completo
        
        Args:
            gene_ids: IDs dos genes [batch, num_genes]
            adj_matrix: Matriz de adjacência [batch, num_genes, num_genes]
            alelo_sequence: Sequência de alelos [batch, seq_len]
            gene_features: Features dos genes [batch, num_genes, feature_dim]
        
        Returns:
            Dicionário com predições
        """
        # GNN para vias metabólicas
        gnn_out = self.gnn(gene_ids, adj_matrix)  # [batch, 32]
        
        # Transformer para sequência
        transformer_out = self.transformer(alelo_sequence)  # [batch, 64]
        
        # Attention para interações
        gene_features_attended, attention_weights = self.attention(gene_features)
        attention_out = torch.mean(gene_features_attended, dim=1)  # [batch, 32]
        
        # Fusion
        combined = torch.cat([gnn_out, transformer_out, attention_out], dim=-1)
        fused = self.fusion(combined)  # [batch, 128]
        
        # Predições
        return {
            'metabolismo_score': torch.sigmoid(self.metabolismo_head(fused)) * 100,
            'resposta_score': torch.sigmoid(self.resposta_head(fused)) * 100,
            'dose_otima': torch.sigmoid(self.dose_head(fused)) * 800,
            'riscos': torch.sigmoid(self.risco_head(fused)) * 100,
            'attention_weights': attention_weights
        }


if __name__ == "__main__":
    print("=" * 80)
    print("REDES NEURAIS PARA ANÁLISE GENÉTICA - TESTE")
    print("=" * 80)
    print()
    
    batch_size = 4
    num_genes = 15
    seq_len = 30
    
    # 1. Testar GNN
    print("1. Graph Neural Network (Vias Metabólicas)")
    gnn = GNNViasMetabolicas()
    gene_ids = torch.randint(0, 15, (batch_size, num_genes))
    adj_matrix = torch.rand(batch_size, num_genes, num_genes)
    out_gnn = gnn(gene_ids, adj_matrix)
    print(f"   Input: genes={gene_ids.shape}, adj={adj_matrix.shape}")
    print(f"   Output: {out_gnn.shape}")
    print(f"   Parâmetros: {sum(p.numel() for p in gnn.parameters()):,}")
    
    # 2. Testar Transformer
    print("\n2. Transformer (Sequências Genéticas)")
    transformer = TransformerSequenciaGenetica()
    alelo_ids = torch.randint(0, 100, (batch_size, seq_len))
    out_transformer = transformer(alelo_ids)
    print(f"   Input: alelos={alelo_ids.shape}")
    print(f"   Output: {out_transformer.shape}")
    print(f"   Parâmetros: {sum(p.numel() for p in transformer.parameters()):,}")
    
    # 3. Testar Attention
    print("\n3. Multi-Head Genetic Attention")
    attention = MultiHeadGeneticAttention()
    gene_features = torch.randn(batch_size, num_genes, 32)
    out_attention, attn_weights = attention(gene_features)
    print(f"   Input: features={gene_features.shape}")
    print(f"   Output: features={out_attention.shape}, attention={attn_weights.shape}")
    print(f"   Parâmetros: {sum(p.numel() for p in attention.parameters()):,}")
    
    # 4. Testar Modelo Integrado
    print("\n4. Preditor Farmacogenômico Integrado")
    modelo = PreditorFarmacogenomicoIntegrado()
    results = modelo(gene_ids, adj_matrix, alelo_ids, gene_features)
    print(f"   Predições:")
    for k, v in results.items():
        if k != 'attention_weights':
            print(f"     {k}: {v.shape}")
    print(f"   Parâmetros totais: {sum(p.numel() for p in modelo.parameters()):,}")
    
    print("\n" + "=" * 80)
    print("✓ TODAS AS ARQUITETURAS GENÉTICAS TESTADAS COM SUCESSO!")
    print("=" * 80)
