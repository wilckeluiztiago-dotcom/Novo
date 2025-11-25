import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plotar_curva_lorenz(pop_acumulada, renda_acumulada, gini_valor):
    """
    Plota a Curva de Lorenz.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(pop_acumulada, renda_acumulada, label='Curva de Lorenz', color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='red', label='Igualdade Perfeita')
    plt.fill_between(pop_acumulada, pop_acumulada, renda_acumulada, alpha=0.1, color='blue')
    
    plt.title(f'Curva de Lorenz (Gini = {gini_valor:.3f})')
    plt.xlabel('População Acumulada (%)')
    plt.ylabel('Renda Acumulada (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    return plt

def plotar_distribuicao_renda(dados, linha_pobreza):
    """
    Plota o histograma da distribuição de renda com a linha de pobreza.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(dados['renda_pc'], bins=50, kde=True, color='green')
    plt.axvline(linha_pobreza, color='red', linestyle='--', label=f'Linha de Pobreza (R$ {linha_pobreza:.2f})')
    
    plt.title('Distribuição da Renda Domiciliar per Capita')
    plt.xlabel('Renda (R$)')
    plt.ylabel('Frequência')
    plt.xlim(0, dados['renda_pc'].quantile(0.95)) # Limitar eixo X para melhor visualização
    plt.legend()
    return plt

def plotar_pobreza_por_uf(df_agrupado):
    """
    Plota um gráfico de barras da pobreza por UF.
    """
    plt.figure(figsize=(12, 6))
    sns.barplot(x='uf', y='is_pobre', data=df_agrupado.sort_values('is_pobre', ascending=False), hue='uf', palette='viridis', legend=False)
    
    plt.title('Taxa de Pobreza por UF')
    plt.xlabel('Unidade Federativa')
    plt.ylabel('Proporção de Pobres')
    plt.xticks(rotation=45)
    return plt
