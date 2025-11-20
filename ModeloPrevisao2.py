import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime, timedelta
import seaborn as sns
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Configuração de dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

@dataclass
class ParametrosPetroleo:
    """Parâmetros do modelo de petróleo"""
    preco_equilibrio: float = 75.0
    volatilidade: float = 0.25
    velocidade_reversao: float = 0.3
    custo_armazenamento: float = 0.02
    conveniencia_yield: float = 0.05
    taxa_juros: float = 0.05
    correlacao_dolar: float = -0.6

class EquacaoDiferencialPetroleo:
    """
    Modelo de equações diferenciais estocásticas para petróleo
    Baseado no modelo de Schwartz (1997) com fatores de conveniência
    """
    
    def __init__(self, parametros: ParametrosPetroleo):
        self.parametros = parametros
        
    def modelo_schwartz_1fator(self, preco_atual: float, dt: float) -> float:
        """
        Modelo de Schwartz 1 fator: dS = κ(μ - lnS)Sdt + σSdW
        """
        kappa = self.parametros.velocidade_reversao
        mu = np.log(self.parametros.preco_equilibrio)
        sigma = self.parametros.volatilidade
        
        log_preco = np.log(preco_atual)
        drift = kappa * (mu - log_preco) * preco_atual * dt
        difusao = sigma * preco_atual * np.sqrt(dt) * np.random.normal()
        
        return drift + difusao
    
    def modelo_schwartz_2fatores(self, preco_atual: float, conveniencia_atual: float, 
                               dt: float) -> Tuple[float, float]:
        """
        Modelo de Schwartz 2 fatores:
        dS = (r - δ)Sdt + σ₁SdW₁
        dδ = κ(α - δ)dt + σ₂dW₂
        """
        r = self.parametros.taxa_juros
        sigma1 = self.parametros.volatilidade
        kappa = self.parametros.velocidade_reversao
        alpha = self.parametros.conveniencia_yield
        sigma2 = 0.1  # Volatilidade do convenience yield
        
        # Brownianos correlacionados
        dW1 = np.random.normal()
        dW2 = self.parametros.correlacao_dolar * dW1 + \
               np.sqrt(1 - self.parametros.correlacao_dolar**2) * np.random.normal()
        
        # Equação do preço
        dS = (r - conveniencia_atual) * preco_atual * dt + \
              sigma1 * preco_atual * np.sqrt(dt) * dW1
        
        # Equação do convenience yield
        ddelta = kappa * (alpha - conveniencia_atual) * dt + \
                 sigma2 * np.sqrt(dt) * dW2
        
        return dS, ddelta

class RedeNeuralAvancada(nn.Module):
    """
    Rede neural avançada com arquitetura híbrida LSTM + Attention
    """
    
    def __init__(self, dimensao_entrada: int, sequencia_temporal: int = 30):
        super().__init__()
        
        self.sequencia_temporal = sequencia_temporal
        self.dimensao_entrada = dimensao_entrada
        
        # LSTM bidirecional
        self.lstm = nn.LSTM(
            input_size=dimensao_entrada,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # Mecanismo de atenção
        self.attention = nn.MultiheadAttention(
            embed_dim=256,  # 128 * 2 (bidirecional)
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # Camadas fully connected
        self.fc_layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LSTM espera input: (batch_size, seq_len, input_size)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Atenção
        attn_out, attn_weights = self.attention(
            lstm_out, lstm_out, lstm_out
        )
        
        # Pooling global
        pooled = torch.mean(attn_out, dim=1)
        
        # Camada fully connected
        output = self.fc_layers(pooled)
        
        return output

class DatasetPetroleo(Dataset):
    """Dataset personalizado para dados de petróleo"""
    
    def __init__(self, dados: np.ndarray, alvo: np.ndarray, 
                 sequencia_temporal: int = 30):
        self.dados = dados
        self.alvo = alvo
        self.sequencia_temporal = sequencia_temporal
        
    def __len__(self):
        return len(self.dados)
    
    def __getitem__(self, idx):
        x = self.dados[idx]
        y = self.alvo[idx]
        return torch.FloatTensor(x), torch.FloatTensor([y])

class SistemaHibridoPetroleo:
    """
    Sistema híbrido combinando EDEs e redes neurais para previsão de petróleo
    """
    
    def __init__(self, parametros: ParametrosPetroleo):
        self.parametros = parametros
        self.equacao_diferencial = EquacaoDiferencialPetroleo(parametros)
        self.scaler = StandardScaler()
        self.modelo_neural = None
        self.historico_treinamento = []
        
    def criar_dados_sinteticos_realistas(self) -> pd.DataFrame:
        """Cria dados sintéticos realistas para demonstração"""
        np.random.seed(42)
        n_dados = 1000  # Reduzido para treinamento mais rápido
        
        # Gerar datas
        start_date = datetime(2015, 1, 1)
        dates = [start_date + timedelta(days=x) for x in range(n_dados)]
        
        # Simular preço do petróleo com modelo de mean reversion + tendência + sazonalidade
        tempo = np.arange(n_dados)
        
        # Componentes do preço
        tendencia = 50 + 0.015 * tempo  # Tendência de alta gradual
        sazonalidade = 8 * np.sin(2 * np.pi * tempo / 252)  # Sazonalidade anual
        ciclo_economico = 5 * np.sin(2 * np.pi * tempo / 1000)  # Ciclos longos
        
        # Preço base com mean reversion
        preco_base = tendencia + sazonalidade + ciclo_economico
        
        # Adicionar volatilidade e saltos
        ruido_vol = np.random.normal(0, 2, n_dados)
        
        # Simular eventos de crise (saltos)
        saltos = np.zeros(n_dados)
        indices_saltos = np.random.choice(n_dados, size=10, replace=False)
        saltos[indices_saltos] = np.random.normal(0, 10, 10)
        
        preco_petroleo = preco_base + ruido_vol + saltos
        preco_petroleo = np.maximum(preco_petroleo, 10)  # Preço mínimo
        
        # Simular outras variáveis correlacionadas
        preco_dolar = 90 + 0.005 * tempo + 2 * np.sin(2 * np.pi * tempo / 365) + np.random.normal(0, 1, n_dados)
        sp500 = 2500 + 2 * tempo + 50 * np.sin(2 * np.pi * tempo / 500) + np.random.normal(0, 30, n_dados)
        preco_ouro = 1200 + 0.3 * tempo + 20 * np.sin(2 * np.pi * tempo / 400) + np.random.normal(0, 15, n_dados)
        
        # Calcular retornos e volatilidade
        retorno_petroleo = np.zeros(n_dados)
        retorno_petroleo[1:] = np.diff(preco_petroleo) / preco_petroleo[:-1]
        
        volatilidade_petroleo = np.zeros(n_dados)
        for i in range(20, n_dados):
            volatilidade_petroleo[i] = np.std(retorno_petroleo[i-19:i+1])
        volatilidade_petroleo[:20] = 0.02
        
        # Criar DataFrame
        dados = pd.DataFrame({
            'data': dates,
            'preco_petroleo': preco_petroleo,
            'retorno_petroleo': retorno_petroleo,
            'volatilidade_petroleo': volatilidade_petroleo,
            'preco_dolar': preco_dolar,
            'retorno_dolar': np.zeros(n_dados),
            'sp500': sp500,
            'retorno_sp500': np.zeros(n_dados),
            'preco_ouro': preco_ouro
        })
        
        # Calcular retornos para outras variáveis
        dados['retorno_dolar'] = dados['preco_dolar'].pct_change().fillna(0)
        dados['retorno_sp500'] = dados['sp500'].pct_change().fillna(0)
        
        # Adicionar features técnicas
        dados['media_movel_20'] = dados['preco_petroleo'].rolling(20).mean().fillna(method='bfill')
        dados['media_movel_50'] = dados['preco_petroleo'].rolling(50).mean().fillna(method='bfill')
        dados['rsi'] = self.calcular_rsi(dados['preco_petroleo'])
        dados['macd'] = self.calcular_macd(dados['preco_petroleo'])
        
        # Remover NaN
        dados = dados.fillna(method='bfill').fillna(method='ffill')
        dados = dados.dropna()
        
        print(f"Dados sintéticos criados: {len(dados)} observações")
        print(f"Preço atual simulado: ${dados['preco_petroleo'].iloc[-1]:.2f}")
        
        return dados.reset_index(drop=True)
    
    def calcular_rsi(self, preco: pd.Series, periodo: int = 14) -> pd.Series:
        """Calcula RSI"""
        delta = preco.diff()
        ganho = delta.where(delta > 0, 0).rolling(window=periodo).mean()
        perda = (-delta.where(delta < 0, 0)).rolling(window=periodo).mean()
        rs = ganho / perda
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Preencher com valor neutro
    
    def calcular_macd(self, preco: pd.Series) -> pd.Series:
        """Calcula MACD"""
        ema_12 = preco.ewm(span=12).mean()
        ema_26 = preco.ewm(span=26).mean()
        macd = ema_12 - ema_26
        return macd.fillna(0)
    
    def preparar_dados_treinamento(self, dados: pd.DataFrame, 
                                 sequencia_temporal: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """Prepara dados para treinamento"""
        # Selecionar features
        features = ['preco_petroleo', 'retorno_petroleo', 'volatilidade_petroleo',
                   'preco_dolar', 'retorno_dolar', 'sp500', 'retorno_sp500',
                   'preco_ouro', 'media_movel_20', 'media_movel_50', 'rsi', 'macd']
        
        dados_features = dados[features].values
        
        # Verificar se há dados suficientes
        if len(dados_features) < sequencia_temporal + 1:
            raise ValueError(f"Dados insuficientes. Necessário: {sequencia_temporal + 1}, Disponível: {len(dados_features)}")
        
        # Normalizar dados
        dados_normalizados = self.scaler.fit_transform(dados_features)
        
        # Criar sequências
        X, y = [], []
        for i in range(sequencia_temporal, len(dados_normalizados)):
            X.append(dados_normalizados[i-sequencia_temporal:i])
            y.append(dados_normalizados[i, 0])  # preço do petróleo
            
        return np.array(X), np.array(y)
    
    def treinar_modelo_hibrido(self, dados: pd.DataFrame, epochs: int = 50):
        """Treina o modelo híbrido"""
        print("Preparando dados para treinamento...")
        
        try:
            X, y = self.preparar_dados_treinamento(dados)
            print(f"Dados preparados: {X.shape[0]} sequências de treinamento")
            print(f"Dimensões X: {X.shape}")  # (n_sequencias, seq_len, n_features)
            print(f"Dimensões y: {y.shape}")
        except Exception as e:
            print(f"Erro ao preparar dados: {e}")
            return
        
        # Dividir treino/teste
        split_idx = int(0.8 * len(X))
        X_treino, X_teste = X[:split_idx], X[split_idx:]
        y_treino, y_teste = y[:split_idx], y[split_idx:]
        
        # Criar datasets e dataloaders
        dataset_treino = DatasetPetroleo(X_treino, y_treino)
        dataset_teste = DatasetPetroleo(X_teste, y_teste)
        
        dataloader_treino = DataLoader(dataset_treino, batch_size=32, shuffle=True)
        dataloader_teste = DataLoader(dataset_teste, batch_size=32, shuffle=False)
        
        # Inicializar modelo
        dimensao_entrada = X.shape[2]  # número de features
        self.modelo_neural = RedeNeuralAvancada(dimensao_entrada).to(device)
        
        print(f"Modelo inicializado com {dimensao_entrada} features de entrada")
        
        # Otimizador e loss
        optimizer = optim.AdamW(self.modelo_neural.parameters(), lr=0.001, 
                              weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        criterion = nn.HuberLoss()
        
        print("Iniciando treinamento...")
        
        for epoch in range(epochs):
            # Fase de treino
            self.modelo_neural.train()
            perda_treino = 0
            for batch_X, batch_y in dataloader_treino:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                # Verificar dimensões
                if batch_X.dim() != 3:
                    print(f"Dimensão inválida do batch: {batch_X.shape}")
                    continue
                    
                optimizer.zero_grad()
                pred = self.modelo_neural(batch_X)
                loss = criterion(pred, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.modelo_neural.parameters(), 1.0)
                optimizer.step()
                
                perda_treino += loss.item()
            
            # Fase de validação
            self.modelo_neural.eval()
            perda_teste = 0
            with torch.no_grad():
                for batch_X, batch_y in dataloader_teste:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    
                    if batch_X.dim() != 3:
                        continue
                        
                    pred = self.modelo_neural(batch_X)
                    loss = criterion(pred, batch_y)
                    perda_teste += loss.item()
            
            if len(dataloader_treino) > 0:
                perda_treino_media = perda_treino / len(dataloader_treino)
            else:
                perda_treino_media = 0
                
            if len(dataloader_teste) > 0:
                perda_teste_media = perda_teste / len(dataloader_teste)
            else:
                perda_teste_media = 0
            
            scheduler.step(perda_teste_media)
            
            self.historico_treinamento.append({
                'epoch': epoch,
                'perda_treino': perda_treino_media,
                'perda_teste': perda_teste_media
            })
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}: Treino = {perda_treino_media:.6f}, Teste = {perda_teste_media:.6f}')
        
        print("Treinamento concluído!")
    
    def prever_com_ede(self, preco_atual: float, horizonte: int = 30, 
                      n_simulacoes: int = 500) -> Dict:
        """Faz previsões usando equações diferenciais estocásticas"""
        dt = 1/252
        trajetorias = np.zeros((n_simulacoes, horizonte))
        trajetorias[:, 0] = preco_atual
        
        for i in range(n_simulacoes):
            for j in range(1, horizonte):
                dS = self.equacao_diferencial.modelo_schwartz_1fator(
                    trajetorias[i, j-1], dt
                )
                trajetorias[i, j] = trajetorias[i, j-1] + dS
                # Garantir que o preço não fique negativo
                trajetorias[i, j] = max(trajetorias[i, j], 1.0)
        
        return {
            'trajetorias': trajetorias,
            'media': np.mean(trajetorias, axis=0),
            'percentil_5': np.percentile(trajetorias, 5, axis=0),
            'percentil_95': np.percentile(trajetorias, 95, axis=0),
            'volatilidade': np.std(trajetorias[:, -1])
        }
    
    def prever_com_neural(self, dados_recentes: np.ndarray) -> float:
        """Faz previsão usando rede neural"""
        if self.modelo_neural is None:
            raise ValueError("Modelo não treinado")
        
        self.modelo_neural.eval()
        with torch.no_grad():
            # Garantir que os dados têm a dimensão correta: (1, seq_len, n_features)
            if dados_recentes.ndim == 2:
                dados_recentes = dados_recentes[np.newaxis, :, :]
            
            dados_tensor = torch.FloatTensor(dados_recentes).to(device)
            pred = self.modelo_neural(dados_tensor)
            return pred.cpu().item()
    
    def previsao_hibrida(self, dados: pd.DataFrame, horizonte: int = 30) -> Dict:
        """Combina previsões da EDE e rede neural"""
        try:
            # Previsão neural
            dados_recentes = dados.tail(30)[[
                'preco_petroleo', 'retorno_petroleo', 'volatilidade_petroleo',
                'preco_dolar', 'retorno_dolar', 'sp500', 'retorno_sp500',
                'preco_ouro', 'media_movel_20', 'media_movel_50', 'rsi', 'macd'
            ]].values
            
            # Normalizar os dados recentes
            dados_recentes_normalizados = self.scaler.transform(dados_recentes)
            
            previsao_neural = self.prever_com_neural(dados_recentes_normalizados)
            
            # Converter de volta para escala original
            preco_atual = dados['preco_petroleo'].iloc[-1]
            dummy = np.zeros((1, len(self.scaler.scale_)))
            dummy[0, 0] = previsao_neural
            previsao_neural_desnormalizada = self.scaler.inverse_transform(dummy)[0, 0]
            
            # Previsão EDE
            previsao_ede = self.prever_com_ede(preco_atual, horizonte)
            
            # Combinação ponderada
            peso_neural = 0.7
            peso_ede = 0.3
            
            previsao_combinada = (
                peso_neural * previsao_neural_desnormalizada +
                peso_ede * previsao_ede['media'][-1]
            )
            
            return {
                'preco_atual': preco_atual,
                'previsao_neural': previsao_neural_desnormalizada,
                'previsao_ede': previsao_ede['media'][-1],
                'previsao_hibrida': previsao_combinada,
                'intervalo_confianca_ede': (
                    previsao_ede['percentil_5'][-1],
                    previsao_ede['percentil_95'][-1]
                ),
                'trajetorias_ede': previsao_ede
            }
            
        except Exception as e:
            print(f"Erro na previsão híbrida: {e}")
            # Fallback para previsão simples
            preco_atual = dados['preco_petroleo'].iloc[-1]
            previsao_fallback = preco_atual * (1 + np.random.normal(0, 0.05))
            return {
                'preco_atual': preco_atual,
                'previsao_neural': previsao_fallback,
                'previsao_ede': previsao_fallback,
                'previsao_hibrida': previsao_fallback,
                'intervalo_confianca_ede': (preco_atual * 0.9, preco_atual * 1.1),
                'trajetorias_ede': {'trajetorias': np.array([[preco_atual]]), 'media': [preco_atual]}
            }

# Funções de análise e visualização
def analisar_resultados(dados: pd.DataFrame, previsoes: Dict):
    """Analisa e visualiza resultados"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Preço histórico e previsão
    axes[0, 0].plot(dados.index[-100:], dados['preco_petroleo'].tail(100), 
                   label='Histórico', linewidth=2)
    axes[0, 0].axhline(y=previsoes['previsao_hibrida'], color='red', 
                      linestyle='--', linewidth=2, label='Previsão Híbrida')
    axes[0, 0].set_title('Preço Histórico do Petróleo vs Previsão')
    axes[0, 0].set_ylabel('Preço (USD)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Trajetórias EDE
    trajetorias = previsoes['trajetorias_ede']['trajetorias']
    for i in range(min(50, len(trajetorias))):
        axes[0, 1].plot(trajetorias[i], alpha=0.1, color='blue')
    axes[0, 1].plot(previsoes['trajetorias_ede']['media'], 'r-', linewidth=2, label='Média')
    axes[0, 1].fill_between(range(len(trajetorias[0])),
                           previsoes['trajetorias_ede']['percentil_5'],
                           previsoes['trajetorias_ede']['percentil_95'],
                           alpha=0.3, label='90% IC')
    axes[0, 1].set_title('Trajetórias Simuladas (EDE)')
    axes[0, 1].set_ylabel('Preço (USD)')
    axes[0, 1].set_xlabel('Dias')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Comparação de métodos
    metodos = ['Neural', 'EDE', 'Híbrido']
    valores = [previsoes['previsao_neural'], 
               previsoes['previsao_ede'],
               previsoes['previsao_hibrida']]
    bars = axes[1, 0].bar(metodos, valores, alpha=0.7, edgecolor='black', 
                         color=['blue', 'orange', 'red'])
    axes[1, 0].axhline(y=previsoes['preco_atual'], color='green', 
                      linestyle='--', label='Preço Atual')
    axes[1, 0].set_title('Comparação de Métodos de Previsão')
    axes[1, 0].set_ylabel('Preço Previsto (USD)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Adicionar valores nas barras
    for bar, valor in zip(bars, valores):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'${valor:.2f}', ha='center', va='bottom')
    
    # Distribuição de previsões
    axes[1, 1].hist(trajetorias[:, -1], bins=30, density=True, alpha=0.7, 
                   edgecolor='black', color='lightblue')
    axes[1, 1].axvline(x=previsoes['previsao_hibrida'], color='red', 
                      linestyle='--', linewidth=2, label='Previsão Híbrida')
    axes[1, 1].axvline(x=previsoes['preco_atual'], color='green', 
                      linestyle='--', linewidth=2, label='Preço Atual')
    axes[1, 1].set_title('Distribuição das Previsões EDE')
    axes[1, 1].set_xlabel('Preço do Barril (USD)')
    axes[1, 1].set_ylabel('Densidade')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plotar_historico_treinamento(historico: List[Dict]):
    """Plota o histórico de treinamento"""
    if not historico:
        return
        
    epochs = [h['epoch'] for h in historico]
    loss_treino = [h['perda_treino'] for h in historico]
    loss_teste = [h['perda_teste'] for h in historico]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_treino, label='Perda Treino', linewidth=2)
    plt.plot(epochs, loss_teste, label='Perda Teste', linewidth=2)
    plt.title('Evolução da Perda durante o Treinamento')
    plt.xlabel('Época')
    plt.ylabel('Perda')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Exemplo de uso
def executar_sistema_completo():
    """Executa o sistema completo de previsão de petróleo"""
    print("=== SISTEMA HÍBRIDO DE PREVISÃO DE PETRÓLEO ===\n")
    
    # Parâmetros do modelo
    parametros = ParametrosPetroleo(
        preco_equilibrio=75.0,
        volatilidade=0.25,
        velocidade_reversao=0.3,
        conveniencia_yield=0.05,
        taxa_juros=0.05,
        correlacao_dolar=-0.6
    )
    
    # Criar sistema
    sistema = SistemaHibridoPetroleo(parametros)
    
    # Criar dados sintéticos realistas
    print("Criando dados sintéticos realistas...")
    dados = sistema.criar_dados_sinteticos_realistas()
    
    # Treinar modelo
    print("\nTreinando modelo híbrido...")
    sistema.treinar_modelo_hibrido(dados, epochs=30)  # Reduzido para teste mais rápido
    
    # Fazer previsões
    print("\nGerando previsões...")
    previsoes = sistema.previsao_hibrida(dados)
    
    # Resultados
    print(f"\n=== RESULTADOS DA PREVISÃO ===")
    print(f"Preço atual: ${previsoes['preco_atual']:.2f}")
    print(f"Previsão Neural: ${previsoes['previsao_neural']:.2f}")
    print(f"Previsão EDE: ${previsoes['previsao_ede']:.2f}")
    print(f"Previsão Híbrida: ${previsoes['previsao_hibrida']:.2f}")
    print(f"Intervalo de Confiança (90%): [${previsoes['intervalo_confianca_ede'][0]:.2f}, ${previsoes['intervalo_confianca_ede'][1]:.2f}]")
    
    # Análise visual
    analisar_resultados(dados, previsoes)
    if sistema.historico_treinamento:
        plotar_historico_treinamento(sistema.historico_treinamento)
    
    return sistema, dados, previsoes

if __name__ == "__main__":
    # Executar sistema completo
    sistema, dados, previsoes = executar_sistema_completo()
    
    print("\n=== SISTEMA EXECUTADO COM SUCESSO ===")