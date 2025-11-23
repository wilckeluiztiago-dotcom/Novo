import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configurar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Configurar estilo dos gráficos
plt.style.use('seaborn-v0_8')
np.random.seed(42)
torch.manual_seed(42)

class BayesianLSTM(nn.Module):
    """LSTM Bayesiana com Dropout como aproximação variacional"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.2):
        super(BayesianLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_mean = nn.Linear(hidden_size, output_size)
        self.fc_std = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, return_uncertainty=False):
        # x shape: (batch_size, seq_len, input_size)
        
        # Forward pass pela LSTM
        out, _ = self.lstm(x)  # out shape: (batch_size, seq_len, hidden_size)
        out = self.dropout(out[:, -1, :])  # Pegar última sequência
        
        # Previsões com incerteza
        mean = self.fc_mean(out)
        std = torch.nn.functional.softplus(self.fc_std(out)) + 1e-6
        
        if return_uncertainty:
            return mean, std
        return mean

class RedeNeuralBayesianaFinancas:
    def __init__(self):
        self.modelo_volatilidade = None
        self.modelo_retornos = None
        self.historico_treinamento = None
        
    def baixar_dados_financeiros(self, ticker, periodo="2y"):
        """Baixa dados financeiros do Yahoo Finance"""
        print(f"Baixando dados para {ticker}...")
        try:
            # Método alternativo para evitar problemas de impersonation
            acao = yf.Ticker(ticker)
            dados = acao.history(period=periodo)
            
            if dados.empty:
                raise ValueError("Dados vazios")
                
            # Calcular retornos e volatilidade
            precos = dados['Close']
            retornos = np.log(precos / precos.shift(1)).dropna()
            volatilidade = precos.rolling(window=21).std().dropna()
            
            # Alinhar índices
            idx_comum = retornos.index.intersection(volatilidade.index)
            retornos = retornos.loc[idx_comum]
            volatilidade = volatilidade.loc[idx_comum]
            precos = precos.loc[idx_comum]
            
            print(f"Dados baixados com sucesso: {len(precos)} registros")
            
            return {
                'precos': precos,
                'retornos': retornos,
                'volatilidade_realizada': volatilidade,
                'data': precos.index
            }
        except Exception as e:
            print(f"Erro ao baixar dados: {e}")
            print("Gerando dados simulados...")
            return self.gerar_dados_simulados()
    
    def gerar_dados_simulados(self, n_dias=500):
        """Gera dados financeiros simulados para teste"""
        np.random.seed(42)
        
        # Gerar preços simulados com GBM e volatilidade estocástica
        precos = [100]
        volatilidades = [0.02]
        
        for i in range(1, n_dias):
            # Modelo GBM com volatilidade estocástica
            vol = volatilidades[-1]
            retorno = np.random.normal(0.0005, vol)
            novo_preco = precos[-1] * np.exp(retorno)
            precos.append(novo_preco)
            
            # Evolução da volatilidade (processo mean-reverting)
            nova_vol = max(0.01, volatilidades[-1] + 0.1 * (0.02 - volatilidades[-1]) + 0.01 * np.random.normal())
            volatilidades.append(nova_vol)
        
        precos = pd.Series(precos)
        retornos = np.log(precos / precos.shift(1)).dropna()
        volatilidade = pd.Series(volatilidades).iloc[1:]  # Alinhar com retornos
        datas = pd.date_range(start='2022-01-01', periods=n_dias, freq='D')
        
        return {
            'precos': precos,
            'retornos': retornos,
            'volatilidade_realizada': volatilidade,
            'data': datas
        }
    
    def preparar_dados(self, dados, lookback=30, horizonte=5):
        """Prepara dados para treinamento da rede neural"""
        retornos = dados['retornos'].values.astype(np.float32)
        volatilidade = dados['volatilidade_realizada'].values.astype(np.float32)
        
        # Garantir que temos dados suficientes
        min_dados = lookback + horizonte + 1
        if len(retornos) < min_dados:
            raise ValueError(f"Dados insuficientes. Necessário: {min_dados}, Disponível: {len(retornos)}")
        
        X, y_retornos, y_vol = [], [], []
        
        for i in range(lookback, len(retornos) - horizonte):
            X.append(retornos[i-lookback:i])
            y_retornos.append(retornos[i:i+horizonte])
            y_vol.append(volatilidade[i:i+horizonte])
        
        if len(X) == 0:
            raise ValueError("Não foi possível criar sequências de treinamento")
        
        X = np.array(X)
        y_retornos = np.array(y_retornos)
        y_vol = np.array(y_vol)
        
        print(f"Shape dos dados: X={X.shape}, y_retornos={y_retornos.shape}, y_vol={y_vol.shape}")
        
        # Normalizar os dados
        self.retorno_medio = np.mean(X)
        self.retorno_std = np.std(X)
        
        X_normalizado = (X - self.retorno_medio) / (self.retorno_std + 1e-8)
        
        # Adicionar dimensão de features para LSTM
        X_normalizado = X_normalizado.reshape(X_normalizado.shape[0], X_normalizado.shape[1], 1)
        
        return X_normalizado, y_retornos, y_vol
    
    def construir_modelo_volatilidade(self, input_size=1, hidden_size=64, output_size=5):
        """Constrói rede neural bayesiana para previsão de volatilidade"""
        return BayesianLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            output_size=output_size,
            dropout_rate=0.2
        ).to(device)
    
    def construir_modelo_retornos(self, input_size=1, hidden_size=128, output_size=5):
        """Constrói rede neural bayesiana para previsão de retornos"""
        return BayesianLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=3,
            output_size=output_size,
            dropout_rate=0.3
        ).to(device)
    
    def loss_volatilidade(self, y_true, mean_pred, std_pred):
        """Loss function para volatilidade (LogNormal)"""
        # Log likelihood para distribuição LogNormal
        log_y_true = torch.log(y_true + 1e-8)
        loss = 0.5 * torch.log(2 * np.pi * std_pred**2) + 0.5 * ((log_y_true - mean_pred) / std_pred)**2
        return torch.mean(loss)
    
    def loss_retornos(self, y_true, mean_pred, std_pred):
        """Loss function para retornos (Normal)"""
        # Log likelihood para distribuição Normal
        loss = 0.5 * torch.log(2 * np.pi * std_pred**2) + 0.5 * ((y_true - mean_pred) / std_pred)**2
        return torch.mean(loss)
    
    def treinar_modelo(self, modelo, X, y, loss_function, epochs=100, lr=0.001, model_name="modelo"):
        """Treina um modelo individual"""
        optimizer = optim.Adam(modelo.parameters(), lr=lr)
        
        # Converter para tensores PyTorch
        X_tensor = torch.from_numpy(X).float().to(device)
        y_tensor = torch.from_numpy(y).float().to(device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        historico_loss = []
        modelo.train()
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                
                # Forward pass com incerteza
                # batch_X shape: (batch_size, seq_len, input_size)
                mean_pred, std_pred = modelo(batch_X, return_uncertainty=True)
                
                # Calcular loss
                loss = loss_function(batch_y, mean_pred, std_pred)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            historico_loss.append(avg_loss)
            
            if epoch % 20 == 0:
                print(f'Epoch {epoch}, {model_name} Loss: {avg_loss:.6f}')
        
        return historico_loss
    
    def treinar_modelos(self, ticker="AAPL", lookback=30, horizonte=5, epochs=100):
        """Treina os modelos de volatilidade e retornos"""
        try:
            # Baixar e preparar dados
            dados = self.baixar_dados_financeiros(ticker)
            X, y_retornos, y_vol = self.preparar_dados(dados, lookback, horizonte)
            
            print(f"Shape final X: {X.shape}")
            
            # Construir modelos
            self.modelo_volatilidade = self.construir_modelo_volatilidade(output_size=horizonte)
            self.modelo_retornos = self.construir_modelo_retornos(output_size=horizonte)
            
            print("Treinando modelo de volatilidade...")
            historico_vol = self.treinar_modelo(
                self.modelo_volatilidade, X, y_vol, 
                self.loss_volatilidade, epochs=epochs, model_name="Volatilidade"
            )
            
            print("Treinando modelo de retornos...")
            historico_ret = self.treinar_modelo(
                self.modelo_retornos, X, y_retornos,
                self.loss_retornos, epochs=epochs, model_name="Retornos"
            )
            
            self.historico_treinamento = {
                'volatilidade': historico_vol,
                'retornos': historico_ret
            }
            
            self.dados = dados
            self.X = X
            self.y_retornos = y_retornos
            self.y_vol = y_vol
            
            return historico_vol, historico_ret
            
        except Exception as e:
            print(f"Erro durante o treinamento: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def prever_distribuicoes(self, X_input, n_amostras=1000):
        """Faz previsões probabilísticas usando inferência variacional"""
        self.modelo_volatilidade.eval()
        self.modelo_retornos.eval()
        
        # Garantir que X_input tem a dimensão correta
        if len(X_input.shape) == 2:
            X_input = X_input.reshape(X_input.shape[0], X_input.shape[1], 1)
        
        X_tensor = torch.from_numpy(X_input).float().to(device)
        
        with torch.no_grad():
            # Gerar múltiplas amostras usando dropout
            previsoes_vol = []
            previsoes_ret = []
            
            for _ in range(n_amostras):
                # Ativar dropout para inferência variacional
                self.modelo_volatilidade.train()
                self.modelo_retornos.train()
                
                mean_vol, std_vol = self.modelo_volatilidade(X_tensor, return_uncertainty=True)
                mean_ret, std_ret = self.modelo_retornos(X_tensor, return_uncertainty=True)
                
                # Amostrar das distribuições
                # Para volatilidade (LogNormal): amostrar na escala log e depois exponenciar
                vol_amostra = torch.exp(torch.normal(mean_vol, std_vol)).cpu().numpy()
                ret_amostra = torch.normal(mean_ret, std_ret).cpu().numpy()
                
                previsoes_vol.append(vol_amostra)
                previsoes_ret.append(ret_amostra)
        
        previsoes_vol = np.array(previsoes_vol)
        previsoes_ret = np.array(previsoes_ret)
        
        return {
            'volatilidade': {
                'amostras': previsoes_vol,
                'media': np.mean(previsoes_vol, axis=0),
                'intervalo_confianca': np.percentile(previsoes_vol, [2.5, 97.5], axis=0)
            },
            'retornos': {
                'amostras': previsoes_ret,
                'media': np.mean(previsoes_ret, axis=0),
                'intervalo_confianca': np.percentile(previsoes_ret, [2.5, 97.5], axis=0)
            }
        }
    
    def calcular_var_esperado(self, previsoes, nivel_confianca=0.05):
        """Calcula Value at Risk e Expected Shortfall"""
        amostras_retornos = previsoes['retornos']['amostras']
        
        # VaR e ES para cada horizonte
        var_results = []
        es_results = []
        
        for horizonte in range(amostras_retornos.shape[2]):
            retornos_horizonte = amostras_retornos[:, :, horizonte].flatten()
            
            var = np.percentile(retornos_horizonte, nivel_confianca * 100)
            es = retornos_horizonte[retornos_horizonte <= var].mean()
            
            var_results.append(var)
            es_results.append(es)
        
        return {
            'var': np.array(var_results),
            'expected_shortfall': np.array(es_results),
            'nivel_confianca': nivel_confianca
        }
    
    def analisar_risco_carteira(self, previsoes, valor_carteira=1000000):
        """Analisa risco de carteira baseado nas previsões"""
        metrics = self.calcular_var_esperado(previsoes)
        
        var_monetario = valor_carteira * (1 - np.exp(metrics['var']))
        es_monetario = valor_carteira * (1 - np.exp(metrics['expected_shortfall']))
        
        return {
            'var_percentual': metrics['var'],
            'expected_shortfall_percentual': metrics['expected_shortfall'],
            'var_monetario': var_monetario,
            'expected_shortfall_monetario': es_monetario,
            'horizontes': range(1, len(metrics['var']) + 1)
        }
    
    def plotar_resultados(self, previsoes, analise_risco, valor_carteira=1000000):
        """Gera gráficos sofisticados dos resultados"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        # 1. Série temporal de preços
        axes[0].plot(self.dados['data'][-100:], self.dados['precos'][-100:])
        axes[0].set_title('Série Temporal de Preços (Últimos 100 dias)')
        axes[0].set_xlabel('Data')
        axes[0].set_ylabel('Preço (USD)')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3)
        
        # 2. Retornos diários
        axes[1].plot(self.dados['data'][-99:], self.dados['retornos'][-99:])
        axes[1].set_title('Retornos Diários (Últimos 99 dias)')
        axes[1].set_xlabel('Data')
        axes[1].set_ylabel('Retorno Logarítmico')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        # 3. Previsão de volatilidade com incerteza
        horizonte = previsoes['volatilidade']['media'].shape[1]
        dias_futuros = range(1, horizonte + 1)
        
        media_vol = previsoes['volatilidade']['media'][0]
        ic_vol = previsoes['volatilidade']['intervalo_confianca'][:, 0]
        
        axes[2].plot(dias_futuros, media_vol, 'b-', label='Volatilidade Esperada', linewidth=2)
        axes[2].fill_between(dias_futuros, ic_vol[0], ic_vol[1], alpha=0.3, 
                           label='Intervalo 95% Confiança')
        axes[2].set_title('Previsão de Volatilidade com Incerteza')
        axes[2].set_xlabel('Dias no Futuro')
        axes[2].set_ylabel('Volatilidade')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # 4. Previsão de retornos com incerteza
        media_ret = previsoes['retornos']['media'][0]
        ic_ret = previsoes['retornos']['intervalo_confianca'][:, 0]
        
        axes[3].plot(dias_futuros, media_ret, 'g-', label='Retorno Esperado', linewidth=2)
        axes[3].fill_between(dias_futuros, ic_ret[0], ic_ret[1], alpha=0.3,
                           label='Intervalo 95% Confiança')
        axes[3].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[3].set_title('Previsão de Retornos com Incerteza')
        axes[3].set_xlabel('Dias no Futuro')
        axes[3].set_ylabel('Retorno Logarítmico')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        # 5. VaR e Expected Shortfall
        axes[4].plot(analise_risco['horizontes'], analise_risco['var_percentual'] * 100, 
                   'ro-', label='VaR', linewidth=2)
        axes[4].plot(analise_risco['horizontes'], analise_risco['expected_shortfall_percentual'] * 100,
                   'bs-', label='Expected Shortfall', linewidth=2)
        axes[4].set_title('VaR e Expected Shortfall por Horizonte')
        axes[4].set_xlabel('Dias no Futuro')
        axes[4].set_ylabel('Perda Esperada (%)')
        axes[4].legend()
        axes[4].grid(True, alpha=0.3)
        
        # 6. Perda monetária esperada
        axes[5].plot(analise_risco['horizontes'], analise_risco['var_monetario'], 
                   'ro-', label='VaR Monetário', linewidth=2)
        axes[5].plot(analise_risco['horizontes'], analise_risco['expected_shortfall_monetario'],
                   'bs-', label='ES Monetário', linewidth=2)
        axes[5].set_title(f'Risco Monetário (Carteira: ${valor_carteira:,.0f})')
        axes[5].set_xlabel('Dias no Futuro')
        axes[5].set_ylabel('Perda Esperada (USD)')
        axes[5].legend()
        axes[5].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def gerar_relatorio_risco(self, previsoes, analise_risco, valor_carteira=1000000):
        """Gera relatório numérico detalhado de risco"""
        print("=" * 70)
        print("RELATÓRIO DE RISCO - REDE NEURAL BAYESIANA (PyTorch)")
        print("=" * 70)
        
        print(f"\n1. METADOS DO MODELO:")
        print(f"   - Dispositivo: {device}")
        print(f"   - Valor da carteira: ${valor_carteira:,.2f}")
        print(f"   - Nível de confiança VaR: {(1-analise_risco['nivel_confianca'])*100}%")
        
        print(f"\n2. PREVISÕES PARA PRÓXIMOS {len(previsoes['retornos']['media'][0])} DIAS:")
        print(f"   Dia | Retorno Esperado | Volatilidade Esperada")
        print(f"   ----|------------------|----------------------")
        
        for i in range(len(previsoes['retornos']['media'][0])):
            ret_esperado = previsoes['retornos']['media'][0][i] * 100
            vol_esperada = previsoes['volatilidade']['media'][0][i] * 100
            
            print(f"   {i+1:2d}  | {ret_esperado:7.3f}%        | {vol_esperada:7.3f}%")
        
        print(f"\n3. METRICAS DE RISCO - VaR 95%:")
        print(f"   Horizonte | VaR (%)  | ES (%)   | VaR (USD)   | ES (USD)")
        print(f"   ----------|----------|----------|-------------|----------")
        
        for i, horizonte in enumerate(analise_risco['horizontes']):
            var_pct = analise_risco['var_percentual'][i] * 100
            es_pct = analise_risco['expected_shortfall_percentual'][i] * 100
            var_usd = analise_risco['var_monetario'][i]
            es_usd = analise_risco['expected_shortfall_monetario'][i]
            
            print(f"   {horizonte:2d} dias    | {var_pct:6.3f}%  | {es_pct:6.3f}%  | ${var_usd:9,.0f} | ${es_usd:9,.0f}")
        
        # Calcular probabilidade de retorno negativo
        prob_negativo = np.mean(previsoes['retornos']['amostras'][:, 0, 0] < 0) * 100
        print(f"\n4. PROBABILIDADE DE RETORNO NEGATIVO (Dia 1): {prob_negativo:.2f}%")
        
        print(f"\n5. RECOMENDAÇÕES:")
        if analise_risco['var_monetario'][0] > valor_carteira * 0.02:
            print("   ⚠️  ALTO RISCO: Considerar reduzir exposição")
        elif analise_risco['var_monetario'][0] > valor_carteira * 0.01:
            print("   🔸 RISCO MODERADO: Manter com monitoramento constante")
        else:
            print("   ✅ RISCO BAIXO: Posição dentro dos limites aceitáveis")
        
        print("=" * 70)

# Exemplo de uso completo
if __name__ == "__main__":
    # Instanciar e treinar o modelo
    rn_bayesiana = RedeNeuralBayesianaFinancas()
    
    print("Iniciando treinamento da rede neural bayesiana com PyTorch...")
    historico_vol, historico_ret = rn_bayesiana.treinar_modelos(
        ticker="AAPL", 
        lookback=30, 
        horizonte=5, 
        epochs=50  # Reduzido para teste rápido
    )
    
    if historico_vol is not None and historico_ret is not None:
        # Fazer previsões
        ultimos_dados = rn_bayesiana.X[-1:]  # Já está no formato correto
        previsoes = rn_bayesiana.prever_distribuicoes(ultimos_dados, n_amostras=500)
        
        # Analisar risco
        valor_carteira = 1000000  # USD
        analise_risco = rn_bayesiana.analisar_risco_carteira(previsoes, valor_carteira)
        
        # Gerar gráficos
        fig = rn_bayesiana.plotar_resultados(previsoes, analise_risco, valor_carteira)
        
        # Gerar relatório numérico
        rn_bayesiana.gerar_relatorio_risco(previsoes, analise_risco, valor_carteira)
        
        # Plotar histórico de treinamento
        if rn_bayesiana.historico_treinamento:
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(rn_bayesiana.historico_treinamento['volatilidade'])
            plt.title('Loss do Modelo de Volatilidade')
            plt.xlabel('Época')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            plt.plot(rn_bayesiana.historico_treinamento['retornos'])
            plt.title('Loss do Modelo de Retornos')
            plt.xlabel('Época')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
    else:
        print("Falha no treinamento dos modelos.")