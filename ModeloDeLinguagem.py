import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, Dict

class EmbeddingPosicionalRotativa(nn.Module):
    """Embedding posicional rotativa (RoPE) - implementação simplificada e funcional"""
    
    def __init__(self, dimensao_cabeca: int, base: int = 10000):
        super().__init__()
        self.dimensao_cabeca = dimensao_cabeca
        self.base = base
        
        # Garantir que a dimensão seja par
        assert dimensao_cabeca % 2 == 0, "Dimensão da cabeça deve ser par para RoPE"
        
        # Calcular theta
        theta = 1.0 / (base ** (torch.arange(0, dimensao_cabeca, 2).float() / dimensao_cabeca))
        self.registrar_buffer('theta', theta)
    
    def registrar_buffer(self, nome, tensor):
        """Wrapper para register_buffer"""
        self.register_buffer(nome, tensor, persistent=False)
    
    def forward(self, x: torch.Tensor, comprimento_sequencia: int) -> torch.Tensor:
        batch_size, num_cabecas, seq_len, dim_cabeca = x.shape
        
        # Criar posições [0, 1, 2, ..., seq_len-1]
        posicoes = torch.arange(seq_len, device=x.device).float().view(1, 1, seq_len, 1)
        
        # Calcular ângulos para cada posição e dimensão
        indices = torch.arange(0, dim_cabeca, 2, device=x.device).float()
        angles = posicoes * self.theta.view(1, 1, 1, -1)  # [1, 1, seq_len, dim_cabeca//2]
        
        # Calcular seno e cosseno
        cos_angles = torch.cos(angles)  # [1, 1, seq_len, dim_cabeca//2]
        sin_angles = torch.sin(angles)  # [1, 1, seq_len, dim_cabeca//2]
        
        # Separar dimensões pares e ímpares
        x_reshaped = x.view(batch_size, num_cabecas, seq_len, dim_cabeca // 2, 2)
        x_real = x_reshaped[..., 0]  # Dimensões pares
        x_imag = x_reshaped[..., 1]  # Dimensões ímpares
        
        # Aplicar rotação
        x_real_rotated = x_real * cos_angles - x_imag * sin_angles
        x_imag_rotated = x_real * sin_angles + x_imag * cos_angles
        
        # Combinar de volta
        x_rotated = torch.stack([x_real_rotated, x_imag_rotated], dim=-1)
        x_rotated = x_rotated.view(batch_size, num_cabecas, seq_len, dim_cabeca)
        
        return x_rotated


class AtencaoMultiCabeca(nn.Module):
    """Mecanismo de atenção multi-cabeça com RoPE"""
    
    def __init__(self, dimensao_modelo: int, numero_cabecas: int, taxa_dropout: float = 0.1):
        super().__init__()
        self.dimensao_modelo = dimensao_modelo
        self.numero_cabecas = numero_cabecas
        self.dimensao_cabeca = dimensao_modelo // numero_cabecas
        
        # Verificar divisibilidade
        assert dimensao_modelo % numero_cabecas == 0, "dimensao_modelo deve ser divisível por numero_cabecas"
        
        self.projecao_query = nn.Linear(dimensao_modelo, dimensao_modelo)
        self.projecao_chave = nn.Linear(dimensao_modelo, dimensao_modelo)
        self.projecao_valor = nn.Linear(dimensao_modelo, dimensao_modelo)
        self.projecao_saida = nn.Linear(dimensao_modelo, dimensao_modelo)
        self.dropout = nn.Dropout(taxa_dropout)
        
        # RoPE apenas para Q e K
        self.rope_query = EmbeddingPosicionalRotativa(self.dimensao_cabeca)
        self.rope_chave = EmbeddingPosicionalRotativa(self.dimensao_cabeca)
        
        # Fator de escala para atenção
        self.escala = math.sqrt(self.dimensao_cabeca)
        
    def forward(self, consulta: torch.Tensor, chave: torch.Tensor, valor: torch.Tensor, 
                mascara: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        batch_size, comprimento_sequencia, _ = consulta.shape
        
        # Projeções lineares e reshape para multi-cabeça
        Q = self.projecao_query(consulta)
        K = self.projecao_chave(chave)
        V = self.projecao_valor(valor)
        
        # Reshape para [batch_size, seq_len, num_heads, head_dim]
        Q = Q.view(batch_size, comprimento_sequencia, self.numero_cabecas, self.dimensao_cabeca).transpose(1, 2)
        K = K.view(batch_size, comprimento_sequencia, self.numero_cabecas, self.dimensao_cabeca).transpose(1, 2)
        V = V.view(batch_size, comprimento_sequencia, self.numero_cabecas, self.dimensao_cabeca).transpose(1, 2)
        
        # Aplicar RoPE apenas a Q e K
        Q = self.rope_query(Q, comprimento_sequencia)
        K = self.rope_chave(K, comprimento_sequencia)
        
        # Calcular scores de atenção (Q * K^T)
        scores_atencao = torch.matmul(Q, K.transpose(-2, -1)) / self.escala
        
        # Aplicar máscara se fornecida
        if mascara is not None:
            # Ajustar dimensões da máscara para [batch_size, 1, 1, seq_len]
            if mascara.dim() == 2:
                mascara = mascara.unsqueeze(1).unsqueeze(2)
            elif mascara.dim() == 3:
                mascara = mascara.unsqueeze(1)
                
            scores_atencao = scores_atencao.masked_fill(mascara == 0, -1e9)
        
        pesos_atencao = F.softmax(scores_atencao, dim=-1)
        pesos_atencao = self.dropout(pesos_atencao)
        
        # Aplicar atenção aos valores
        saida = torch.matmul(pesos_atencao, V)
        saida = saida.transpose(1, 2).contiguous().view(batch_size, comprimento_sequencia, self.dimensao_modelo)
        
        return self.projecao_saida(saida)


class RedeFeedForward(nn.Module):
    """Rede feed-forward com ativação GELU"""
    
    def __init__(self, dimensao_modelo: int, dimensao_interna: int, taxa_dropout: float = 0.1):
        super().__init__()
        self.camada_linear_1 = nn.Linear(dimensao_modelo, dimensao_interna)
        self.camada_linear_2 = nn.Linear(dimensao_interna, dimensao_modelo)
        self.dropout = nn.Dropout(taxa_dropout)
        self.ativacao = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.camada_linear_2(self.dropout(self.ativacao(self.camada_linear_1(x))))


class BlocoTransformer(nn.Module):
    """Bloco transformer único com normalização e conexões residuais"""
    
    def __init__(self, dimensao_modelo: int, numero_cabecas: int, dimensao_ff: int, taxa_dropout: float = 0.1):
        super().__init__()
        self.norma_1 = nn.LayerNorm(dimensao_modelo)
        self.norma_2 = nn.LayerNorm(dimensao_modelo)
        self.atencao = AtencaoMultiCabeca(dimensao_modelo, numero_cabecas, taxa_dropout)
        self.feed_forward = RedeFeedForward(dimensao_modelo, dimensao_ff, taxa_dropout)
        self.dropout = nn.Dropout(taxa_dropout)
        
    def forward(self, x: torch.Tensor, mascara: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Atenção com conexão residual
        residuo = x
        x = self.norma_1(x)
        x_atencao = self.atencao(x, x, x, mascara)
        x = residuo + self.dropout(x_atencao)
        
        # Feed-forward com conexão residual
        residuo = x
        x = self.norma_2(x)
        x_ff = self.feed_forward(x)
        x = residuo + self.dropout(x_ff)
        
        return x


class ModeloLinguagemTransformer(nn.Module):
    """Modelo de linguagem transformer completo e sofisticado"""
    
    def __init__(self, 
                 tamanho_vocabulario: int,
                 dimensao_modelo: int = 512,
                 numero_cabecas: int = 8,
                 numero_camadas: int = 6,
                 dimensao_ff: int = 2048,
                 comprimento_maximo_sequencia: int = 512,
                 taxa_dropout: float = 0.1):
        
        super().__init__()
        
        self.tamanho_vocabulario = tamanho_vocabulario
        self.dimensao_modelo = dimensao_modelo
        self.numero_cabecas = numero_cabecas
        self.numero_camadas = numero_camadas
        self.comprimento_maximo_sequencia = comprimento_maximo_sequencia
        
        # Verificar divisibilidade
        assert dimensao_modelo % numero_cabecas == 0, "dimensao_modelo deve ser divisível por numero_cabecas"
        
        # Embeddings
        self.embedding_token = nn.Embedding(tamanho_vocabulario, dimensao_modelo)
        self.embedding_posicional = nn.Embedding(comprimento_maximo_sequencia, dimensao_modelo)
        
        # Dropout para embeddings
        self.dropout_embedding = nn.Dropout(taxa_dropout)
        
        # Camadas transformer
        self.camadas = nn.ModuleList([
            BlocoTransformer(dimensao_modelo, numero_cabecas, dimensao_ff, taxa_dropout)
            for _ in range(numero_camadas)
        ])
        
        # Normalização final
        self.norma_final = nn.LayerNorm(dimensao_modelo)
        
        # Cabeça de linguagem
        self.cabeca_linguagem = nn.Linear(dimensao_modelo, tamanho_vocabulario)
        
        # Inicialização de pesos
        self.aplicar_inicializacao_pesos()
        
    def aplicar_inicializacao_pesos(self):
        """Inicialização avançada de pesos"""
        for modulo in self.modules():
            if isinstance(modulo, nn.Linear):
                torch.nn.init.normal_(modulo.weight, mean=0.0, std=0.02)
                if modulo.bias is not None:
                    torch.nn.init.zeros_(modulo.bias)
            elif isinstance(modulo, nn.Embedding):
                torch.nn.init.normal_(modulo.weight, mean=0.0, std=0.02)
            elif isinstance(modulo, nn.LayerNorm):
                torch.nn.init.zeros_(modulo.bias)
                torch.nn.init.ones_(modulo.weight)
    
    def criar_mascara_autoregressiva(self, tamanho: int, dispositivo: torch.device) -> torch.Tensor:
        """Cria máscara autoregressiva para decoder"""
        mascara = torch.tril(torch.ones(tamanho, tamanho, device=dispositivo))
        return mascara.unsqueeze(0).unsqueeze(0)  # [1, 1, tamanho, tamanho]
    
    def forward(self, tokens_entrada: torch.Tensor, 
                mascara: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        
        batch_size, comprimento_sequencia = tokens_entrada.shape
        dispositivo = tokens_entrada.device
        
        # Verificar comprimento da sequência
        if comprimento_sequencia > self.comprimento_maximo_sequencia:
            raise ValueError(f"Comprimento da sequência ({comprimento_sequencia}) excede o máximo ({self.comprimento_maximo_sequencia})")
        
        # Criar máscara se não for fornecida
        if mascara is None:
            mascara = self.criar_mascara_autoregressiva(comprimento_sequencia, dispositivo)
        
        # Embeddings de tokens e posição
        posicoes = torch.arange(comprimento_sequencia, device=dispositivo).unsqueeze(0).expand(batch_size, comprimento_sequencia)
        embeddings_tokens = self.embedding_token(tokens_entrada)
        embeddings_posicao = self.embedding_posicional(posicoes)
        
        # Combinar embeddings
        x = embeddings_tokens + embeddings_posicao
        x = self.dropout_embedding(x)
        
        # Passar pelas camadas transformer
        for camada in self.camadas:
            x = camada(x, mascara)
        
        # Normalização final
        x = self.norma_final(x)
        
        # Logits do vocabulário
        logits = self.cabeca_linguagem(x)
        
        # Métricas para monitoramento
        metricas = {
            'norma_ativacoes': x.norm().item(),
            'media_ativacoes': x.mean().item(),
            'std_ativacoes': x.std().item(),
        }
        
        return logits, metricas
    
    def gerar_texto(self, token_inicial: torch.Tensor, 
                   comprimento_maximo: int, 
                   temperatura: float = 1.0,
                   top_k: Optional[int] = None) -> torch.Tensor:
        """Geração autoregressiva de texto"""
        self.eval()
        with torch.no_grad():
            tokens_gerados = token_inicial
            
            for i in range(comprimento_maximo):
                # Usar apenas os últimos tokens que cabem no contexto máximo
                if tokens_gerados.size(1) > self.comprimento_maximo_sequencia:
                    tokens_entrada = tokens_gerados[:, -self.comprimento_maximo_sequencia:]
                else:
                    tokens_entrada = tokens_gerados
                
                logits, _ = self.forward(tokens_entrada)
                logits_proximos = logits[:, -1, :] / max(temperatura, 1e-8)
                
                # Amostragem top-k
                if top_k is not None:
                    valores, indices = torch.topk(logits_proximos, min(top_k, logits_proximos.size(-1)))
                    mascara_baixos = logits_proximos < valores[:, -1].unsqueeze(-1)
                    logits_proximos[mascara_baixos] = -float('Inf')
                
                probabilidades = F.softmax(logits_proximos, dim=-1)
                proximo_token = torch.multinomial(probabilidades, num_samples=1)
                tokens_gerados = torch.cat([tokens_gerados, proximo_token], dim=1)
                
                # Print de progresso
                if (i + 1) % 10 == 0:
                    print(f"Gerados {i + 1}/{comprimento_maximo} tokens")
            
            return tokens_gerados


class TreinadorModeloLinguagem:
    """Classe para treinar o modelo de linguagem"""
    
    def __init__(self, modelo: ModeloLinguagemTransformer, taxa_aprendizado: float = 1e-4):
        self.modelo = modelo
        self.otimizador = torch.optim.AdamW(modelo.parameters(), lr=taxa_aprendizado, weight_decay=0.01)
        self.agendador = torch.optim.lr_scheduler.CosineAnnealingLR(self.otimizador, T_max=1000)
        
    def passo_treinamento(self, batch_tokens: torch.Tensor) -> Dict[str, float]:
        """Executa um passo de treinamento"""
        self.modelo.train()
        self.otimizador.zero_grad()
        
        # Separar entrada e alvo (deslocado por 1)
        entrada = batch_tokens[:, :-1]
        alvo = batch_tokens[:, 1:]
        
        # Forward pass
        logits, metricas = self.modelo(entrada)
        
        # Calcular perda - CORREÇÃO: usar reshape em vez de view
        logits_reshaped = logits.reshape(-1, self.modelo.tamanho_vocabulario)
        alvo_reshaped = alvo.reshape(-1)
        
        perda = F.cross_entropy(logits_reshaped, alvo_reshaped)
        
        # Backward pass
        perda.backward()
        
        # Clip de gradientes
        torch.nn.utils.clip_grad_norm_(self.modelo.parameters(), max_norm=1.0)
        
        self.otimizador.step()
        self.agendador.step()
        
        return {
            'perda': perda.item(),
            **metricas,
            'taxa_aprendizado': self.agendador.get_last_lr()[0]
        }


# Teste simplificado para verificar funcionamento
def teste_modelo():
    """Função de teste para verificar se o modelo funciona"""
    print("Iniciando teste do modelo...")
    
    # Configurações menores para teste rápido
    config = {
        'tamanho_vocabulario': 1000,
        'dimensao_modelo': 128,
        'numero_cabecas': 4,
        'numero_camadas': 2,
        'dimensao_ff': 256,
        'comprimento_maximo_sequencia': 64,
        'taxa_dropout': 0.1
    }
    
    # Criar modelo
    modelo = ModeloLinguagemTransformer(**config)
    
    print(f"Modelo criado com {sum(p.numel() for p in modelo.parameters()):,} parâmetros")
    
    # Teste com dados pequenos
    batch_size = 2
    seq_len = 16
    tokens_exemplo = torch.randint(0, config['tamanho_vocabulario'], (batch_size, seq_len))
    
    print(f"Tokens de entrada: {tokens_exemplo.shape}")
    
    # Teste forward pass
    try:
        logits, metricas = modelo(tokens_exemplo)
        print(f"✅ Forward pass bem-sucedido!")
        print(f"Shape dos logits: {logits.shape}")
        print(f"Métricas: {metricas}")
    except Exception as e:
        print(f"❌ Erro no forward pass: {e}")
        return
    
    # Teste treinamento
    try:
        treinador = TreinadorModeloLinguagem(modelo)
        metricas_treinamento = treinador.passo_treinamento(tokens_exemplo)
        print(f"✅ Treinamento bem-sucedido!")
        print(f"Métricas de treinamento: {metricas_treinamento}")
    except Exception as e:
        print(f"❌ Erro no treinamento: {e}")
        return
    
    # Teste geração
    try:
        print("\nTestando geração...")
        token_inicial = torch.randint(0, config['tamanho_vocabulario'], (1, 1))
        texto_gerado = modelo.gerar_texto(token_inicial, comprimento_maximo=5, temperatura=0.8)
        print(f"✅ Geração bem-sucedida!")
        print(f"Texto gerado shape: {texto_gerado.shape}")
        print(f"Tokens gerados: {texto_gerado}")
    except Exception as e:
        print(f"❌ Erro na geração: {e}")
        return
    
    print("\n🎉 Todos os testes passaram! O modelo está funcionando corretamente.")


# Exemplo de treinamento completo
def exemplo_treinamento_completo():
    """Exemplo de como treinar o modelo por múltiplas épocas"""
    print("\n" + "="*50)
    print("EXEMPLO DE TREINAMENTO COMPLETO")
    print("="*50)
    
    # Configurações
    config = {
        'tamanho_vocabulario': 5000,
        'dimensao_modelo': 256,
        'numero_cabecas': 8,
        'numero_camadas': 4,
        'dimensao_ff': 512,
        'comprimento_maximo_sequencia': 128,
        'taxa_dropout': 0.1
    }
    
    # Criar modelo e treinador
    modelo = ModeloLinguagemTransformer(**config)
    treinador = TreinadorModeloLinguagem(modelo, taxa_aprendizado=1e-4)
    
    print(f"Modelo para treinamento completo: {sum(p.numel() for p in modelo.parameters()):,} parâmetros")
    
    # Simular dados de treinamento
    num_batches = 10
    batch_size = 4
    seq_len = 32
    
    print(f"\nTreinando por {num_batches} batches...")
    
    for epoch in range(3):  # 3 épocas
        perda_total = 0
        num_steps = 0
        
        for batch_idx in range(num_batches):
            # Gerar batch simulado
            batch_tokens = torch.randint(0, config['tamanho_vocabulario'], (batch_size, seq_len))
            
            # Passo de treinamento
            metricas = treinador.passo_treinamento(batch_tokens)
            
            perda_total += metricas['perda']
            num_steps += 1
            
            if (batch_idx + 1) % 5 == 0:
                print(f"Época {epoch + 1}, Batch {batch_idx + 1}/{num_batches}, Perda: {metricas['perda']:.4f}")
        
        perda_media = perda_total / num_steps
        print(f"Época {epoch + 1} concluída - Perda média: {perda_media:.4f}")
    
    print("\nTreinamento concluído!")


if __name__ == "__main__":
    # Executar teste básico
    teste_modelo()
    
    # Executar exemplo de treinamento completo
    exemplo_treinamento_completo()