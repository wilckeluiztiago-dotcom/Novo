import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['font.family'] = 'DejaVu Sans'
np.random.seed(42)
tf.random.set_seed(42)

# =============================================
# GERADOR DE DADOS SINTÉTICOS COMPLEXOS OTIMIZADO
# =============================================

class GeradorDadosPetroleo:
    def __init__(self, n_amostras=1500):  # Reduzido para melhor performance
        self.n_amostras = n_amostras
        self.tempo = np.linspace(0, 8, n_amostras)  # Período menor
        
    def gerar_variaveis_macro(self):
        """Gera variáveis macroeconômicas sintéticas mais realistas"""
        # Tendência cíclica para preço de equilíbrio
        tendencia_ciclica = 65 + 15 * np.sin(0.3 * self.tempo) + 8 * np.sin(1.5 * self.tempo)
        
        # Variáveis explicativas com mais correlação
        producao_opep = 80 + 8 * np.sin(0.2 * self.tempo + 1)
        demanda_global = 95 + 12 * np.cos(0.35 * self.tempo + 2)
        
        # Tensão geopolítica com eventos específicos
        tensao = np.zeros(self.n_amostras)
        for i in range(1, self.n_amostras):
            tensao[i] = 0.95 * tensao[i-1] + np.random.normal(0, 0.08)
            # Adicionar alguns picos de tensão
            if i % 200 == 0:
                tensao[i] += np.random.exponential(1.5)
        tensao = np.clip(tensao, 0, 8)
        
        self.dados = {
            'tempo': self.tempo,
            'producao_opep': producao_opep,
            'demanda_global': demanda_global,
            'estoques_eua': 50 + 6 * np.sin(0.5 * self.tempo + 3),
            'tensao_geopolitica': tensao,
            'valor_dolar': 85 + 4 * np.sin(0.6 * self.tempo + 4),
            'preco_equilibrio': tendencia_ciclica
        }
        
        return pd.DataFrame(self.dados)
    
    def nova_equacao_diferencial_estocastica(self, df):
        """Resolve nova EDE personalizada mais estável"""
        dt = self.tempo[1] - self.tempo[0]
        precos = np.zeros(self.n_amostras)
        precos[0] = 70.0  # Preço inicial mais próximo da média
        
        # Parâmetros otimizados
        theta = 0.12  # Velocidade de mean reversion reduzida
        alpha = 0.05  # Efeito memória reduzido
        beta = 0.03   # Sensibilidade a choques reduzida
        gamma = 0.02  # Efeito não-linear reduzido
        
        for i in range(1, self.n_amostras):
            # Termo de mean reversion suavizado
            diff_equilibrio = df['preco_equilibrio'].iloc[i] - precos[i-1]
            mean_reversion = theta * np.tanh(diff_equilibrio / 10) * 10  # Suavizado
            
            # Termo de memória suave
            memoria = alpha * np.arctan((precos[i-1] - 70) / 5) * 5 if i > 3 else 0
            
            # Termo de choque estocástico com volatilidade controlada
            volatilidade = 0.015 + 0.008 * np.sin(0.3 * self.tempo[i])
            choque_estocastico = beta * volatilidade * np.random.normal()
            
            # Termo de tensão geopolítica controlado
            choque_geopolitico = gamma * df['tensao_geopolitica'].iloc[i] * np.random.normal(0, 0.3)
            
            # Nova EDE mais estável
            dS = mean_reversion + memoria + choque_estocastico + choque_geopolitico
            
            precos[i] = precos[i-1] + dS * dt + 0.05 * np.random.normal()  # Ruído reduzido
            
        return precos

# =============================================
# MODELO DE REDE NEURAL SIMPLIFICADO
# =============================================

class ModeloPetroleoAvancado(tf.keras.Model):
    def __init__(self, num_variaveis):
        super().__init__()
        # Arquitetura mais simples e estável
        self.camada_entrada = tf.keras.layers.Dense(64, activation='relu')
        self.camada_oculta1 = tf.keras.layers.Dense(128, activation='relu')
        self.camada_oculta2 = tf.keras.layers.Dense(64, activation='relu')
        self.camada_saida = tf.keras.layers.Dense(1, activation='linear')
        
        # Batch normalization para estabilidade
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        
    def call(self, inputs, training=False):
        x = self.camada_entrada(inputs)
        x = self.bn1(x, training=training)
        x = self.camada_oculta1(x)
        x = self.bn2(x, training=training)
        x = self.camada_oculta2(x)
        return self.camada_saida(x)

# =============================================
# FUNÇÃO DE PERDA SIMPLIFICADA
# =============================================

def perda_fisica_simplificada(modelo, entradas_numpy, precos_reais_numpy):
    """Função de perda simplificada e mais estável"""
    
    # Converter para tensores TensorFlow
    entradas = tf.constant(entradas_numpy, dtype=tf.float32)
    precos_reais = tf.constant(precos_reais_numpy, dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(entradas)
        precos_previstos = modelo(entradas, training=True)
    
    # Primeira derivada (dS/dt) - apenas esta para simplificar
    dS_dt = tape.gradient(precos_previstos, entradas)[:, 0:1]
    
    # Equação física simplificada - apenas mean reversion básica
    S = precos_previstos
    theta = 0.1
    mu = 70.0
    
    # Resíduo físico simplificado
    residuo_fisico = dS_dt - theta * (mu - S)
    
    # Perda de dados (peso maior)
    perda_dados = tf.reduce_mean(tf.square(precos_previstos - precos_reais))
    
    # Perda física (peso menor)
    perda_fisica = tf.reduce_mean(tf.square(residuo_fisico))
    
    return perda_dados + 0.1 * perda_fisica  # Peso físico reduzido

# =============================================
# TREINAMENTO OTIMIZADO
# =============================================

def executar_simulacao_completa():
    print("🔧 Gerando dados sintéticos complexos...")
    gerador = GeradorDadosPetroleo(n_amostras=1500)
    df = gerador.gerar_variaveis_macro()
    df['preco_petroleo'] = gerador.nova_equacao_diferencial_estocastica(df)
    
    # Preparar dados para o modelo
    variaveis = ['producao_opep', 'demanda_global', 'estoques_eua', 
                'tensao_geopolitica', 'valor_dolar', 'tempo']
    
    X = df[variaveis].values.astype(np.float32)
    y = df['preco_petroleo'].values.astype(np.float32).reshape(-1, 1)
    
    # Normalização robusta
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X_std = np.where(X_std == 0, 1.0, X_std)  # Evitar divisão por zero
    y_mean, y_std = y.mean(), y.std()
    
    X_normalizado = (X - X_mean) / X_std
    y_normalizado = (y - y_mean) / y_std
    
    print("🧠 Criando e treinando modelo avançado...")
    modelo = ModeloPetroleoAvancado(num_variaveis=len(variaveis))
    
    # Otimizador com learning rate decay
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01,
        decay_steps=200,
        decay_rate=0.9
    )
    otimizador = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    # Treinamento com early stopping implícito
    n_epocas = 500
    perdas = []
    melhor_perda = float('inf')
    paciencia = 50
    contador_paciencia = 0
    
    for epoca in range(n_epocas):
        with tf.GradientTape() as tape:
            perda = perda_fisica_simplificada(modelo, X_normalizado, y_normalizado)
        
        gradientes = tape.gradient(perda, modelo.trainable_variables)
        otimizador.apply_gradients(zip(gradientes, modelo.trainable_variables))
        perdas.append(perda.numpy())
        
        # Early stopping
        if perda.numpy() < melhor_perda:
            melhor_perda = perda.numpy()
            contador_paciencia = 0
        else:
            contador_paciencia += 1
            
        if contador_paciencia >= paciencia:
            print(f"🏁 Parada antecipada na época {epoca}")
            break
        
        if epoca % 50 == 0:
            print(f"Época {epoca}, Perda: {perda.numpy():.6f}, LR: {otimizador.learning_rate.numpy():.6f}")
    
    # Previsões finais
    print("📊 Gerando previsões e gráficos...")
    X_tensor = tf.constant(X_normalizado, dtype=tf.float32)
    previsoes_normalizado = modelo(X_tensor, training=False)
    previsoes = previsoes_normalizado.numpy() * y_std + y_mean
    
    return df, previsoes, perdas, modelo

# =============================================
# VISUALIZAÇÕES MELHORADAS
# =============================================

def criar_visualizacoes_avancadas(df, previsoes, perdas):
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Evolução temporal dos preços
    plt.subplot(3, 3, 1)
    plt.plot(df['tempo'], df['preco_petroleo'], label='Real', linewidth=2, alpha=0.8, color='blue')
    plt.plot(df['tempo'], previsoes, label='Previsto', linewidth=2, alpha=0.8, color='red', linestyle='--')
    plt.xlabel('Tempo')
    plt.ylabel('Preço do Petróleo (USD)')
    plt.title('EVOLUÇÃO TEMPORAL: Real vs Previsto')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Histograma dos resíduos
    plt.subplot(3, 3, 2)
    residuos = df['preco_petroleo'] - previsoes.flatten()
    plt.hist(residuos, bins=50, density=True, alpha=0.7, color='green', edgecolor='black')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, np.mean(residuos), np.std(residuos))
    plt.plot(x, p, 'k', linewidth=2, label='Distribuição Normal')
    plt.title('DISTRIBUIÇÃO DOS RESÍDUOS')
    plt.xlabel('Resíduos')
    plt.ylabel('Densidade')
    plt.legend()
    
    # 3. Correlação entre variáveis
    plt.subplot(3, 3, 3)
    correlacao = df[['preco_petroleo', 'producao_opep', 'demanda_global', 
                    'estoques_eua', 'tensao_geopolitica', 'valor_dolar']].corr()
    mask = np.triu(np.ones_like(correlacao, dtype=bool))
    sns.heatmap(correlacao, mask=mask, annot=True, cmap='RdYlBu', center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, fmt='.2f')
    plt.title('MATRIZ DE CORRELAÇÃO')
    
    # 4. Comportamento da perda durante treinamento
    plt.subplot(3, 3, 4)
    plt.plot(perdas, color='darkred', linewidth=2)
    plt.xlabel('Época')
    plt.ylabel('Perda')
    plt.title('EVOLUÇÃO DA FUNÇÃO DE PERDA')
    plt.grid(True, alpha=0.3)
    
    # 5. Preço vs Demanda Global
    plt.subplot(3, 3, 5)
    scatter = plt.scatter(df['demanda_global'], df['preco_petroleo'], 
                         c=df['tensao_geopolitica'], cmap='viridis', alpha=0.7, s=30)
    plt.colorbar(scatter, label='Tensão Geopolítica')
    plt.xlabel('Demanda Global')
    plt.ylabel('Preço do Petróleo')
    plt.title('PREÇO vs DEMANDA')
    
    # 6. Efeito da produção da OPEP
    plt.subplot(3, 3, 6)
    plt.scatter(df['producao_opep'], df['preco_petroleo'], 
               c=df['valor_dolar'], cmap='plasma', alpha=0.7, s=30)
    plt.colorbar(label='Valor do Dólar')
    plt.xlabel('Produção OPEP')
    plt.ylabel('Preço do Petróleo')
    plt.title('EFEITO DA PRODUÇÃO DA OPEP')
    
    # 7. Série temporal multivariada
    plt.subplot(3, 3, 7)
    # Normalizar para plotar juntas
    for coluna in ['preco_petroleo', 'preco_equilibrio']:
        dados_normalizados = (df[coluna] - df[coluna].min()) / (df[coluna].max() - df[coluna].min())
        plt.plot(df['tempo'], dados_normalizados, label=coluna, alpha=0.8, linewidth=2)
    plt.xlabel('Tempo')
    plt.ylabel('Valores Normalizados')
    plt.title('SÉRIE TEMPORAL MULTIVARIADA')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 8. Distribuição de probabilidade dos preços
    plt.subplot(3, 3, 8)
    sns.kdeplot(df['preco_petroleo'], label='Real', fill=True, alpha=0.5, color='blue')
    sns.kdeplot(previsoes.flatten(), label='Previsto', fill=True, alpha=0.5, color='red')
    plt.xlabel('Preço do Petróleo (USD)')
    plt.ylabel('Densidade')
    plt.title('DISTRIBUIÇÃO DE PROBABILIDADE')
    plt.legend()
    
    # 9. Análise de resíduos ao longo do tempo
    plt.subplot(3, 3, 9)
    plt.plot(df['tempo'], residuos, color='purple', alpha=0.7, linewidth=1)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=2)
    plt.fill_between(df['tempo'], residuos, alpha=0.3, color='purple')
    plt.xlabel('Tempo')
    plt.ylabel('Resíduos')
    plt.title('RESÍDUOS AO LONGO DO TEMPO')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Gráfico 3D adicional
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Amostrar pontos para melhor visualização
    indices = np.random.choice(len(df), size=400, replace=False)
    
    scatter = ax.scatter(df['demanda_global'].iloc[indices], 
                        df['producao_opep'].iloc[indices], 
                        df['preco_petroleo'].iloc[indices],
                        c=df['tensao_geopolitica'].iloc[indices], 
                        cmap='hot', alpha=0.8, s=40, depthshade=True)
    
    ax.set_xlabel('Demanda Global', fontsize=12, labelpad=10)
    ax.set_ylabel('Produção OPEP', fontsize=12, labelpad=10)
    ax.set_zlabel('Preço do Petróleo (USD)', fontsize=12, labelpad=10)
    ax.set_title('SUPERFÍCIE 3D: DEMANDA vs PRODUÇÃO vs PREÇO', fontsize=14, pad=20)
    
    # Adicionar barra de cores
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, aspect=20, pad=0.1)
    cbar.set_label('Tensão Geopolítica', rotation=270, labelpad=15)
    
    # Melhorar ângulo de visualização
    ax.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    plt.show()

# =============================================
# EXECUÇÃO PRINCIPAL
# =============================================

if __name__ == "__main__":
    print("🚀 INICIANDO SIMULAÇÃO AVANÇADA DE PREÇOS DO PETRÓLEO")
    print("=" * 60)
    
    try:
        df, previsoes, perdas, modelo = executar_simulacao_completa()
        
        print("\n📈 GERANDO VISUALIZAÇÕES AVANÇADAS...")
        criar_visualizacoes_avancadas(df, previsoes, perdas)
        
        # Métricas de desempenho
        from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
        
        mae = mean_absolute_error(df['preco_petroleo'], previsoes)
        mse = mean_squared_error(df['preco_petroleo'], previsoes)
        r2 = r2_score(df['preco_petroleo'], previsoes)
        rmse = np.sqrt(mse)
        
        print("\n📊 MÉTRICAS DE DESEMPENHO:")
        print(f"   MAE  (Erro Absoluto Médio): {mae:.4f}")
        print(f"   RMSE (Raiz do Erro Quadrático Médio): {rmse:.4f}")
        print(f"   R²   (Coeficiente de Determinação): {r2:.4f}")
        print(f"   Preço Médio Real: {df['preco_petroleo'].mean():.2f} USD")
        print(f"   Preço Médio Previsto: {previsoes.mean():.2f} USD")
        
        print("\n🔍 ANÁLISE DA NOVA EQUAÇÃO DIFERENCIAL ESTOCÁSTICA:")
        print("   ✓ Mean reversion não-linear suavizada")
        print("   ✓ Volatilidade dependente do tempo controlada")
        print("   ✓ Choques geopolíticos com distribuição normal")
        print("   ✓ Termos de memória com função arco-tangente")
        print("   ✓ Estabilidade numérica melhorada")
        
        print(f"\n📈 ESTATÍSTICAS DOS DADOS:")
        print(f"   Média do Preço: {df['preco_petroleo'].mean():.2f} USD")
        print(f"   Desvio Padrão: {df['preco_petroleo'].std():.2f} USD")
        print(f"   Mínimo: {df['preco_petroleo'].min():.2f} USD")
        print(f"   Máximo: {df['preco_petroleo'].max():.2f} USD")
        print(f"   Amostras: {len(df)} pontos temporais")
        
        print("\n✅ SIMULAÇÃO CONCLUÍDA COM SUCESSO!")
        
    except Exception as e:
        print(f"\n❌ ERRO DURANTE A EXECUÇÃO: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 Dica: Verifique se todas as bibliotecas estão instaladas:")
   