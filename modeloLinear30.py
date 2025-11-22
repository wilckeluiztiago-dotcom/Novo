import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Configurações de estilo
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# 1. GERAR DADOS SIMULADOS (em um caso real, estes seriam seus dados históricos)
np.random.seed(42)
n_amostras = 1000

dados = {
    'preco_produto': np.random.normal(50, 15, n_amostras),
    'investimento_marketing': np.random.normal(10000, 3000, n_amostras),
    'renda_consumidor': np.random.normal(3000, 800, n_amostras),
    'preco_concorrente': np.random.normal(48, 12, n_amostras),
    'temperatura': np.random.normal(25, 8, n_amostras),  # para produtos sensíveis ao clima
    'promocao_ativa': np.random.choice([0, 1], n_amostras, p=[0.7, 0.3]),
    'dia_da_semana': np.random.randint(1, 8, n_amostras),
    'mes': np.random.randint(1, 13, n_amostras)
}

# Criar demanda com relações não-lineares e interações
df = pd.DataFrame(dados)
df['demanda'] = (
    5000 
    - 20 * df['preco_produto'] 
    + 0.3 * df['investimento_marketing'] 
    + 0.5 * df['renda_consumidor']
    - 15 * df['preco_concorrente']
    + 10 * df['temperatura']
    + 800 * df['promocao_ativa']
    + 50 * np.sin(2 * np.pi * df['mes'] / 12)  # Sazonalidade
    - 0.001 * df['preco_produto'] * df['preco_concorrente']  # Interação
    + np.random.normal(0, 200, n_amostras)  # Ruído
)

# Garantir que demanda não seja negativa
df['demanda'] = df['demanda'].clip(lower=0)

print("📊 Primeiras linhas do dataset:")
print(df.head())
print(f"\n📈 Dimensões do dataset: {df.shape}")

# 2. ANÁLISE EXPLORATÓRIA DOS DADOS
print("\n" + "="*50)
print("ANÁLISE EXPLORATÓRIA DOS DADOS")
print("="*50)

# Estatísticas descritivas
print("\n📈 Estatísticas Descritivas:")
print(df.describe())

# Correlações
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, fmt='.2f')
plt.title('Mapa de Correlação das Variáveis')
plt.tight_layout()
plt.show()

# Distribuição da demanda
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(df['demanda'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Distribuição da Demanda')
plt.xlabel('Demanda')
plt.ylabel('Frequência')

plt.subplot(1, 3, 2)
plt.scatter(df['preco_produto'], df['demanda'], alpha=0.6)
plt.title('Demanda vs Preço do Produto')
plt.xlabel('Preço do Produto')
plt.ylabel('Demanda')

plt.subplot(1, 3, 3)
plt.scatter(df['investimento_marketing'], df['demanda'], alpha=0.6, color='green')
plt.title('Demanda vs Investimento em Marketing')
plt.xlabel('Investimento em Marketing')
plt.ylabel('Demanda')

plt.tight_layout()
plt.show()

# 3. FEATURE ENGINEERING
print("\n" + "="*50)
print("FEATURE ENGINEERING")
print("="*50)

# Criar features derivadas
df['preco_relativo'] = df['preco_produto'] / df['preco_concorrente']
df['orcamento_marketing_per_capita'] = df['investimento_marketing'] / df['renda_consumidor']
df['estacao'] = df['mes'] % 4  # Simular estações do ano
df['final_de_semana'] = df['dia_da_semana'].isin([6, 7]).astype(int)

# Transformações não-lineares
df['preco_quadrado'] = df['preco_produto'] ** 2
df['log_marketing'] = np.log1p(df['investimento_marketing'])

print("Novas features criadas:")
print(df[['preco_relativo', 'orcamento_marketing_per_capita', 
          'estacao', 'final_de_semana']].head())

# 4. PREPARAÇÃO DOS DADOS PARA MODELAGEM
print("\n" + "="*50)
print("PREPARAÇÃO DOS DADOS")
print("="*50)

# Definir variáveis preditoras e target
variaveis_numericas = [
    'preco_produto', 'investimento_marketing', 'renda_consumidor',
    'preco_concorrente', 'temperatura', 'preco_relativo',
    'orcamento_marketing_per_capita', 'preco_quadrado', 'log_marketing'
]

variaveis_categoricas = [
    'promocao_ativa', 'estacao', 'final_de_semana'
]

X = df[variaveis_numericas + variaveis_categoricas]
y = df['demanda']

# Dividir em treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Tamanho do conjunto de treino: {X_treino.shape}")
print(f"Tamanho do conjunto de teste: {X_teste.shape}")

# 5. MODELAGEM COM SCKIT-LEARN
print("\n" + "="*50)
print("TREINAMENTO DO MODELO LINEAR")
print("="*50)

# Pipeline com pré-processamento e modelo
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

# Treinar o modelo
pipeline.fit(X_treino, y_treino)

# Fazer previsões
y_pred_treino = pipeline.predict(X_treino)
y_pred_teste = pipeline.predict(X_teste)

# 6. AVALIAÇÃO DO MODELO
print("\n" + "="*50)
print("AVALIAÇÃO DO MODELO")
print("="*50)

# Métricas de avaliação
def avaliar_modelo(y_real, y_previsto, nome_dataset):
    mae = mean_absolute_error(y_real, y_previsto)
    mse = mean_squared_error(y_real, y_previsto)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_real, y_previsto)
    
    print(f"\n📊 Métricas para {nome_dataset}:")
    print(f"MAE (Mean Absolute Error): {mae:.2f}")
    print(f"MSE (Mean Squared Error): {mse:.2f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}

# Avaliar nos conjuntos de treino e teste
metricas_treino = avaliar_modelo(y_treino, y_pred_treino, "TREINO")
metricas_teste = avaliar_modelo(y_teste, y_pred_teste, "TESTE")

# Validação cruzada
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
print(f"\n🎯 Validação Cruzada (R²): {cv_scores}")
print(f"R² Médio na Validação Cruzada: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# 7. ANÁLISE DOS RESÍDUOS
print("\n" + "="*50)
print("ANÁLISE DOS RESÍDUOS")
print("="*50)

residuos_teste = y_teste - y_pred_teste

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(y_pred_teste, residuos_teste, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Resíduos vs Valores Preditos')
plt.xlabel('Valores Preditos')
plt.ylabel('Resíduos')

plt.subplot(1, 3, 2)
plt.hist(residuos_teste, bins=30, alpha=0.7, color='orange', edgecolor='black')
plt.title('Distribuição dos Resíduos')
plt.xlabel('Resíduos')
plt.ylabel('Frequência')

plt.subplot(1, 3, 3)
plt.scatter(y_pred_teste, y_teste, alpha=0.6, color='green')
plt.plot([y_teste.min(), y_teste.max()], [y_teste.min(), y_teste.max()], 'red', lw=2)
plt.title('Valores Reais vs Preditos')
plt.xlabel('Valores Preditos')
plt.ylabel('Valores Reais')

plt.tight_layout()
plt.show()

# 8. MODELO STATSMAPS PARA ANÁLISE DETALHADA
print("\n" + "="*50)
print("ANÁLISE ESTATÍSTICA DETALHADA (STATSMODELS)")
print("="*50)

# Adicionar constante para o intercept
X_com_constante = sm.add_constant(X_treino)
modelo_stats = sm.OLS(y_treino, X_com_constante).fit()

print(modelo_stats.summary())

# 9. INTERPRETAÇÃO DO MODELO
print("\n" + "="*50)
print("INTERPRETAÇÃO DO MODELO")
print("="*50)

# Coeficientes do modelo
coeficientes = pd.DataFrame({
    'Variavel': ['Intercepto'] + list(X.columns),
    'Coeficiente': [pipeline.named_steps['model'].intercept_] + 
                   list(pipeline.named_steps['model'].coef_)
})

coeficientes['Impacto_Absoluto'] = abs(coeficientes['Coeficiente'])
coeficientes = coeficientes.sort_values('Impacto_Absoluto', ascending=False)

print("\n📊 Coeficientes do Modelo (ordenados por impacto):")
print(coeficientes.to_string(index=False))

# 10. SIMULAÇÃO DE CENÁRIOS
print("\n" + "="*50)
print("SIMULAÇÃO DE CENÁRIOS")
print("="*50)

# Criar cenários de exemplo
cenarios = pd.DataFrame({
    'preco_produto': [45, 50, 55],
    'investimento_marketing': [12000, 10000, 8000],
    'renda_consumidor': [3200, 3000, 2800],
    'preco_concorrente': [50, 48, 46],
    'temperatura': [30, 25, 20],
    'preco_relativo': [45/50, 50/48, 55/46],
    'orcamento_marketing_per_capita': [12000/3200, 10000/3000, 8000/2800],
    'preco_quadrado': [45**2, 50**2, 55**2],
    'log_marketing': [np.log1p(12000), np.log1p(10000), np.log1p(8000)],
    'promocao_ativa': [1, 0, 1],
    'estacao': [1, 2, 3],
    'final_de_semana': [1, 0, 1]
})

# Fazer previsões para os cenários
previsoes_cenarios = pipeline.predict(cenarios)

print("\n🎯 Previsões para diferentes cenários:")
for i, (idx, cenario) in enumerate(cenarios.iterrows()):
    print(f"\nCenário {i+1}:")
    print(f"  Preço: R${cenario['preco_produto']:.2f}")
    print(f"  Marketing: R${cenario['investimento_marketing']:.2f}")
    print(f"  Promoção: {'Sim' if cenario['promocao_ativa'] == 1 else 'Não'}")
    print(f"  → Demanda Prevista: {previsoes_cenarios[i]:.0f} unidades")

# 11. RECOMENDAÇÕES BASEADAS NO MODELO
print("\n" + "="*50)
print("RECOMENDAÇÕES E INSIGHTS")
print("="*50)

print("""
💡 INSIGHTS DO MODELO:

1. 📊 VARIÁVEIS MAIS IMPORTANTES:
   - Identifique quais variáveis têm maior impacto na demanda
   - Foque nas com maiores coeficientes absolutos

2. 💰 EFEITO PREÇO:
   - Analise a sensibilidade da demanda ao preço
   - Considere estratégias de precificação dinâmica

3. 🎯 MARKETING:
   - Avalie o ROI do investimento em marketing
   - Otimize o mix de canais de marketing

4. 📈 SAZONALIDADE:
   - Planeje para períodos de alta e baixa demanda
   - Ajuste estoques e produção accordingly

5. 🔄 INTERAÇÕES:
   - Considere efeitos combinados entre variáveis
   - Ex: Efeito do preço combinado com promoções
""")

# 12. SALVAR O MODELO E RESULTADOS
print("\n" + "="*50)
print("EXPORTANDO RESULTADOS")
print("="*50)

# Criar dataframe com resultados
resultados = pd.DataFrame({
    'Real': y_teste,
    'Previsto': y_pred_teste,
    'Residuo': residuos_teste,
    'Erro_Percentual': np.abs(residuos_teste / y_teste) * 100
})

# Salvar resultados
resultados.to_csv('previsoes_demanda.csv', index=False)
coeficientes.to_csv('coeficientes_modelo.csv', index=False)

print("✅ Arquivos salvos:")
print("   - previsoes_demanda.csv: Previsões e resíduos")
print("   - coeficientes_modelo.csv: Coeficientes do modelo")

print("\n🎯 MODELO DE PREVISÃO DE DEMANDA FINALIZADO!")
print("   Pronto para uso em decisões de negócio!")