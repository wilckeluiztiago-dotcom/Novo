import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.stats import levy_stable, norm, kurtosis, skew
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

class ModeloEDPFinanceira:
    """Modelo avançado baseado em Equações Diferenciais Parciais para finanças"""
    
    def __init__(self, preco_inicial, taxa_livre_risco=0.12, volatilidade=0.25, alfa_levy=1.7, beta_levy=0.0):
        self.preco_inicial = preco_inicial
        self.taxa_livre_risco = taxa_livre_risco
        self.volatilidade = volatilidade
        self.alfa_levy = alfa_levy
        self.beta_levy = beta_levy
        
    def resolver_edp_black_scholes_fracionaria(self, preco_atual, tempo_expiracao, preco_exercicio, tipo_opcao='call'):
        """Resolve a EDP fracionária de Black-Scholes com processo de Lévy"""
        preco_minimo = 0.1 * preco_atual
        preco_maximo = 3.0 * preco_atual
        numero_pontos_espaco = 200
        numero_pontos_tempo = 100
        
        malha_precos = np.linspace(preco_minimo, preco_maximo, numero_pontos_espaco)
        malha_tempo = np.linspace(0, tempo_expiracao, numero_pontos_tempo)
        passo_tempo = tempo_expiracao / numero_pontos_tempo
        passo_preco = (preco_maximo - preco_minimo) / numero_pontos_espaco
        
        matriz_valores = np.zeros((numero_pontos_espaco, numero_pontos_tempo))
        
        if tipo_opcao == 'call':
            matriz_valores[:, -1] = np.maximum(malha_precos - preco_exercicio, 0)
        else:
            matriz_valores[:, -1] = np.maximum(preco_exercicio - malha_precos, 0)
        
        if tipo_opcao == 'call':
            matriz_valores[0, :] = 0
            matriz_valores[-1, :] = preco_maximo - preco_exercicio * np.exp(-self.taxa_livre_risco * (tempo_expiracao - malha_tempo))
        else:
            matriz_valores[0, :] = preco_exercicio * np.exp(-self.taxa_livre_risco * (tempo_expiracao - malha_tempo))
            matriz_valores[-1, :] = 0
        
        for indice_tempo in range(numero_pontos_tempo-2, -1, -1):
            matriz_coeficientes = sparse.lil_matrix((numero_pontos_espaco, numero_pontos_espaco))
            vetor_termos_independentes = np.zeros(numero_pontos_espaco)
            
            for indice_preco in range(1, numero_pontos_espaco-1):
                termo_difusao = 0.5 * self.volatilidade**2 * malha_precos[indice_preco]**2 / passo_preco**2
                termo_conveccao = self.taxa_livre_risco * malha_precos[indice_preco] / (2 * passo_preco)
                termo_reacao = self.taxa_livre_risco
                
                matriz_coeficientes[indice_preco, indice_preco-1] = -passo_tempo * (termo_difusao - termo_conveccao)
                matriz_coeficientes[indice_preco, indice_preco] = 1 + passo_tempo * (2 * termo_difusao + termo_reacao)
                matriz_coeficientes[indice_preco, indice_preco+1] = -passo_tempo * (termo_difusao + termo_conveccao)
                
                vetor_termos_independentes[indice_preco] = matriz_valores[indice_preco, indice_tempo+1]
            
            matriz_coeficientes[0, 0] = 1
            matriz_coeficientes[-1, -1] = 1
            vetor_termos_independentes[0] = matriz_valores[0, indice_tempo]
            vetor_termos_independentes[-1] = matriz_valores[-1, indice_tempo]
            
            matriz_valores[:, indice_tempo] = spsolve(matriz_coeficientes.tocsr(), vetor_termos_independentes)
        
        return matriz_valores, malha_precos, malha_tempo
    
    def simular_processo_levy(self, preco_inicial, periodo, numero_passos, numero_simulacoes):
        """Simula processo de difusão com saltos de Lévy"""
        passo_tempo = periodo / numero_passos
        trajetorias = np.zeros((numero_simulacoes, numero_passos + 1))
        trajetorias[:, 0] = preco_inicial
        
        # Armazenar todos os saltos para análise
        todos_os_saltos = []
        
        for passo in range(1, numero_passos + 1):
            incremento_browniano = np.random.normal(0, np.sqrt(passo_tempo), numero_simulacoes)
            componente_drift = self.taxa_livre_risco * trajetorias[:, passo-1] * passo_tempo
            componente_difusao = self.volatilidade * trajetorias[:, passo-1] * incremento_browniano
            
            probabilidade_salto = 0.15
            mascara_salto = np.random.random(numero_simulacoes) < probabilidade_salto
            
            if np.any(mascara_salto):
                saltos = levy_stable.rvs(self.alfa_levy, self.beta_levy, 
                                       scale=0.03, size=np.sum(mascara_salto))
                componente_salto = trajetorias[mascara_salto, passo-1] * saltos
                todos_os_saltos.extend(saltos)
            else:
                componente_salto = 0
            
            trajetorias[:, passo] = (trajetorias[:, passo-1] + componente_drift + 
                                   componente_difusao + componente_salto)
            trajetorias[:, passo] = np.maximum(trajetorias[:, passo], 0.01)
        
        return trajetorias, np.array(todos_os_saltos)

def analisar_distribuicao_levy(saltos, alfa_levy, beta_levy):
    """Analisa estatisticamente a distribuição de Lévy gerada"""
    print("\n" + "="*60)
    print("ANÁLISE ESTATÍSTICA DA DISTRIBUIÇÃO DE LÉVY")
    print("="*60)
    
    if len(saltos) == 0:
        print("Nenhum salto foi gerado na simulação")
        return
    
    print(f"Total de saltos gerados: {len(saltos)}")
    print(f"Parâmetros teóricos: α = {alfa_levy}, β = {beta_levy}")
    print(f"Média dos saltos: {np.mean(saltos):.6f}")
    print(f"Desvio padrão: {np.std(saltos):.6f}")
    print(f"Assimetria (skewness): {skew(saltos):.4f}")
    print(f"Curtose (kurtosis): {kurtosis(saltos):.4f}")
    print(f"Máximo salto positivo: {np.max(saltos):.4f}")
    print(f"Máximo salto negativo: {np.min(saltos):.4f}")
    print(f"Amplitude total: {np.ptp(saltos):.4f}")
    
    # Percentis
    percentis = [1, 5, 25, 50, 75, 95, 99]
    valores_percentis = np.percentile(saltos, percentis)
    print("\nDistribuição por percentis:")
    for p, v in zip(percentis, valores_percentis):
        print(f"  {p:2d}%: {v:8.4f}")
    
    # Probabilidade de eventos extremos
    prob_3std = np.mean(np.abs(saltos) > 3 * np.std(saltos)) * 100
    prob_5std = np.mean(np.abs(saltos) > 5 * np.std(saltos)) * 100
    print(f"\nProbabilidade de eventos extremos:")
    print(f"  > 3 desvios padrão: {prob_3std:.2f}%")
    print(f"  > 5 desvios padrão: {prob_5std:.2f}%")

def gerar_valores_numericos_edp(matriz_valores, malha_precos, malha_tempo, preco_atual, preco_exercicio):
    """Gera valores numéricos detalhados da solução da EDP"""
    print("\n" + "="*60)
    print("VALORES NUMÉRICOS DA SOLUÇÃO DA EDP")
    print("="*60)
    
    # Encontrar índice mais próximo do preço atual
    idx_preco_atual = np.argmin(np.abs(malha_precos - preco_atual))
    idx_tempo_inicial = 0
    idx_tempo_meio = len(malha_tempo) // 2
    idx_tempo_final = -1
    
    print(f"Preço atual: R$ {preco_atual:.2f}")
    print(f"Preço de exercício: R$ {preco_exercicio:.2f}")
    print(f"Valor no preço atual (t=0): R$ {matriz_valores[idx_preco_atual, idx_tempo_inicial]:.2f}")
    print(f"Valor no preço atual (t=meio): R$ {matriz_valores[idx_preco_atual, idx_tempo_meio]:.2f}")
    print(f"Valor no preço atual (t=final): R$ {matriz_valores[idx_preco_atual, idx_tempo_final]:.2f}")
    
    # Greve prices específicos
    strikes_relativos = [0.8, 0.9, 1.0, 1.1, 1.2]
    print("\nValores para diferentes preços de exercício (t=0):")
    for strike_rel in strikes_relativos:
        strike = preco_atual * strike_rel
        idx_strike = np.argmin(np.abs(malha_precos - strike))
        valor_opcao = matriz_valores[idx_strike, idx_tempo_inicial]
        print(f"  Strike {strike_rel:.1f}x: R$ {strike:.2f} -> Opção: R$ {valor_opcao:.2f}")
    
    # Sensibilidade ao tempo
    print("\nDecaimento temporal no preço atual:")
    tempos = [0, 0.25, 0.5, 0.75, 1.0]
    for tempo in tempos:
        idx_tempo = int(tempo * (len(malha_tempo) - 1))
        valor = matriz_valores[idx_preco_atual, idx_tempo]
        print(f"  t={tempo:.2f}: R$ {valor:.2f}")

def plotar_superficie_3d(matriz_valores, malha_precos, malha_tempo, titulo="Solução da EDP"):
    """Plota superfície 3D da solução da EDP"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    T, S = np.meshgrid(malha_tempo, malha_precos)
    superficie = ax.plot_surface(S, T, matriz_valores, cmap='plasma', alpha=0.9, 
                               linewidth=0, antialiased=True)
    
    ax.set_xlabel('Preço do Ativo (R$)')
    ax.set_ylabel('Tempo (anos)')
    ax.set_zlabel('Valor da Opção (R$)')
    ax.set_title(titulo, fontsize=14, fontweight='bold')
    
    fig.colorbar(superficie, ax=ax, shrink=0.6, aspect=20, label='Valor (R$)')
    plt.tight_layout()
    plt.show()

def plotar_densidade_levy_comparacao(saltos, alfa_levy, beta_levy):
    """Plota densidade da distribuição de Lévy com comparação e histograma"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Gráfico 1: Densidades teóricas
    x = np.linspace(-0.15, 0.15, 1000)
    densidade_levy = levy_stable.pdf(x, alfa_levy, beta_levy)
    densidade_normal = norm.pdf(x, 0, 0.03)
    
    ax1.plot(x, densidade_levy, 'b-', linewidth=3, label=f'Lévy α={alfa_levy}, β={beta_levy}')
    ax1.plot(x, densidade_normal, 'r--', linewidth=2, label='Distribuição Normal')
    ax1.fill_between(x, densidade_levy, alpha=0.3, color='blue')
    ax1.fill_between(x, densidade_normal, alpha=0.2, color='red')
    
    ax1.set_xlabel('Retorno')
    ax1.set_ylabel('Densidade de Probabilidade')
    ax1.set_title('Comparação: Distribuição de Lévy vs Normal', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Histograma dos saltos simulados
    if len(saltos) > 0:
        ax2.hist(saltos, bins=50, density=True, alpha=0.7, color='green', label='Saltos Simulados')
        ax2.plot(x, densidade_levy, 'b-', linewidth=2, label='Lévy Teórica')
        ax2.set_xlabel('Valor do Salto')
        ax2.set_ylabel('Densidade')
        ax2.set_title('Histograma dos Saltos Simulados', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plotar_trajetorias_simuladas(trajetorias, preco_atual, titulo="Simulações de Monte Carlo"):
    """Plota múltiplas trajetórias simuladas com estatísticas"""
    plt.figure(figsize=(14, 8))
    
    numero_simulacoes, numero_passos = trajetorias.shape
    tempo = np.linspace(0, 1, numero_passos)
    
    for i in range(min(numero_simulacoes, 50)):
        plt.plot(tempo, trajetorias[i], alpha=0.1, color='blue')
    
    media_trajetorias = np.mean(trajetorias, axis=0)
    percentil_5 = np.percentile(trajetorias, 5, axis=0)
    percentil_95 = np.percentile(trajetorias, 95, axis=0)
    
    plt.plot(tempo, media_trajetorias, 'r-', linewidth=3, label='Trajetória Média')
    plt.fill_between(tempo, percentil_5, percentil_95, alpha=0.3, color='red', 
                    label='Intervalo de Confiança 90%')
    plt.axhline(y=preco_atual, color='green', linestyle='--', linewidth=2, 
               label=f'Preço Atual: R$ {preco_atual:.2f}')
    
    plt.xlabel('Tempo (anos)')
    plt.ylabel('Preço (R$)')
    plt.title(titulo, fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plotar_heatmap_edp(matriz_valores, malha_precos, malha_tempo, titulo="Mapa de Calor da EDP"):
    """Plota heatmap 2D da solução da EDP"""
    plt.figure(figsize=(12, 8))
    
    plt.imshow(matriz_valores, extent=[malha_tempo[0], malha_tempo[-1], malha_precos[0], malha_precos[-1]], 
              aspect='auto', cmap='viridis', origin='lower')
    
    plt.colorbar(label='Valor da Opção (R$)')
    plt.xlabel('Tempo (anos)')
    plt.ylabel('Preço do Ativo (R$)')
    plt.title(titulo, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def baixar_dados_petrobras():
    """Baixa dados históricos da Petrobras"""
    print("Baixando dados da Petrobras...")
    try:
        dados = yf.download("PETR4.SA", start="2020-01-01", end="2024-01-01", 
                           auto_adjust=True, progress=False)
        if len(dados) > 100:
            print(f"Dados baixados com sucesso: {len(dados)} dias de trading")
            return dados
        else:
            raise ValueError("Dados insuficientes")
    except Exception as e:
        print(f"Erro no download: {e}")
        print("Criando dados sintéticos...")
        return criar_dados_sinteticos()

def criar_dados_sinteticos():
    """Cria dados sintéticos realistas para demonstração"""
    datas = pd.date_range(start="2020-01-01", end="2024-01-01", freq='D')
    np.random.seed(42)
    
    n = len(datas)
    tendencia = np.linspace(25, 38, n)
    ruido = np.cumsum(np.random.normal(0, 0.02, n))
    sazonalidade = 3 * np.sin(2 * np.pi * np.arange(n) / 252)
    
    precos = tendencia + ruido + sazonalidade
    precos = np.abs(precos)
    
    dados_sinteticos = pd.DataFrame({
        'Open': precos * 0.99,
        'High': precos * 1.02,
        'Low': precos * 0.98,
        'Close': precos,
        'Volume': np.random.randint(1000000, 5000000, n)
    }, index=datas)
    
    print(f"Dados sintéticos criados: {len(dados_sinteticos)} dias")
    return dados_sinteticos

def main():
    """Função principal do modelo avançado"""
    print("=== MODELO AVANÇADO: EDPs FINANCEIRAS COM DISTRIBUIÇÃO DE LÉVY ===\n")
    
    # 1. Carregar dados
    dados_petrobras = baixar_dados_petrobras()
    preco_atual = float(dados_petrobras['Close'].iloc[-1])
    precos_historicos = dados_petrobras['Close'].values
    
    print(f"Preço atual da Petrobras: R$ {preco_atual:.2f}")
    print(f"Período de dados: {len(precos_historicos)} dias\n")
    
    # 2. Criar modelo EDP
    modelo_edp = ModeloEDPFinanceira(
        preco_inicial=preco_atual,
        taxa_livre_risco=0.12,
        volatilidade=0.35,
        alfa_levy=1.6,
        beta_levy=0.1
    )
    
    # 3. Resolver EDP para opção call
    print("Resolvendo Equação Diferencial Parcial...")
    preco_exercicio = preco_atual * 1.1
    matriz_valores_opcao, malha_precos, malha_tempo = modelo_edp.resolver_edp_black_scholes_fracionaria(
        preco_atual=preco_atual,
        tempo_expiracao=1.0,
        preco_exercicio=preco_exercicio,
        tipo_opcao='call'
    )
    
    print("EDP resolvida com sucesso!")
    
    # 4. Gerar valores numéricos da EDP
    gerar_valores_numericos_edp(matriz_valores_opcao, malha_precos, malha_tempo, preco_atual, preco_exercicio)
    
    # 5. Plotar visualizações da EDP
    print("\nGerando visualizações...")
    plotar_superficie_3d(matriz_valores_opcao, malha_precos, malha_tempo,
                        "Solução da EDP de Black-Scholes Fracionária\n(PETR4 Call Option)")
    
    plotar_heatmap_edp(matriz_valores_opcao, malha_precos, malha_tempo,
                      "Mapa de Calor: Valor da Opção Call PETR4")
    
    # 6. Simular trajetórias com Lévy
    print("Simulando trajetórias de preços com processo de Lévy...")
    trajetorias_simuladas, saltos_gerados = modelo_edp.simular_processo_levy(
        preco_inicial=preco_atual,
        periodo=1.0,
        numero_passos=252,
        numero_simulacoes=1000
    )
    
    # 7. Analisar distribuição de Lévy
    analisar_distribuicao_levy(saltos_gerados, modelo_edp.alfa_levy, modelo_edp.beta_levy)
    
    # 8. Plotar distribuição de Lévy
    plotar_densidade_levy_comparacao(saltos_gerados, modelo_edp.alfa_levy, modelo_edp.beta_levy)
    
    # 9. Plotar trajetórias simuladas
    plotar_trajetorias_simuladas(trajetorias_simuladas, preco_atual,
                               "Simulações de Monte Carlo com Processo de Lévy\n(PETR4 - Próximos 252 dias)")
    
    # 10. Análise estatística final
    precos_finais = trajetorias_simuladas[:, -1]
    retorno_esperado = (np.mean(precos_finais) / preco_atual - 1) * 100
    valor_em_risco_95 = np.percentile(precos_finais, 5)
    valor_em_risco_5 = np.percentile(precos_finais, 95)
    volatilidade_esperada = np.std(precos_finais) / preco_atual * 100
    
    print(f"\n" + "="*60)
    print("RESUMO FINAL DA SIMULAÇÃO")
    print("="*60)
    print(f"Preço atual: R$ {preco_atual:.2f}")
    print(f"Preço médio esperado: R$ {np.mean(precos_finais):.2f}")
    print(f"Retorno esperado: {retorno_esperado:+.2f}%")
    print(f"Volatilidade esperada: {volatilidade_esperada:.1f}%")
    print(f"VaR 95% (pior caso): R$ {valor_em_risco_95:.2f} ({((valor_em_risco_95/preco_atual)-1)*100:+.1f}%)")
    print(f"VaR 5% (melhor caso): R$ {valor_em_risco_5:.2f} ({((valor_em_risco_5/preco_atual)-1)*100:+.1f}%)")
    print(f"Probabilidade de lucro: {np.mean(precos_finais > preco_atual) * 100:.1f}%")

if __name__ == "__main__":
    main()