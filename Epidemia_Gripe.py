import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
from datetime import datetime, timedelta

class ModeloEpidemicoSofisticado:
    def __init__(self):
        # Parâmetros epidemiológicos sofisticados
        self.parametros = {
            'taxa_transmissao_basica': 0.3,  # β - taxa de transmissão
            'taxa_incubacao': 1/2,           # σ - taxa de incubação (2 dias)
            'taxa_recuperacao': 1/7,         # γ - taxa de recuperação (7 dias)
            'taxa_mortalidade': 0.02,        # μ - taxa de mortalidade
            'fator_sazonalidade': 0.1,       # amplitude da sazonalidade
            'taxa_isolamento': 0.1,          # taxa de isolamento voluntário
            'eficacia_isolamento': 0.7       # eficácia do isolamento
        }
        
        # Variáveis de estado
        self.variaveis = {
            'suscetiveis': 0.99,      # S
            'expostos': 0.01,         # E  
            'infectados': 0.0,        # I
            'recuperados': 0.0,       # R
            'obitos': 0.0,            # D
            'isolados': 0.0           # Q
        }
    
    def forcas_externas(self, t):
        """Funções de forças externas que afetam os parâmetros"""
        # Sazonalidade - variação sazonal na transmissão
        sazonalidade = 1 + self.parametros['fator_sazonalidade'] * np.sin(2 * np.pi * t / 365)
        
        # Intervenções não-farmacêuticas (máscaras, distanciamento)
        if t > 30 and t < 90:  # Período de intervenção
            fator_intervencao = 0.6
        else:
            fator_intervencao = 1.0
            
        # Comportamento populacional - redução voluntária de contatos
        comportamento = 1 - self.parametros['taxa_isolamento'] * np.tanh(t/50)
        
        return sazonalidade * fator_intervencao * comportamento
    
    def sistema_equacoes_diferenciais(self, t, y):
        """
        Sistema de equações diferenciais do modelo SEIRDQ avançado
        y = [S, E, I, R, D, Q]
        """
        S, E, I, R, D, Q = y
        
        # Parâmetros variáveis no tempo
        fator_temporal = self.forcas_externas(t)
        beta_efetivo = self.parametros['taxa_transmissao_basica'] * fator_temporal
        sigma = self.parametros['taxa_incubacao']
        gamma = self.parametros['taxa_recuperacao']
        mu = self.parametros['taxa_mortalidade']
        eta = self.parametros['taxa_isolamento']
        epsilon = self.parametros['eficacia_isolamento']
        
        # População total (normalizada)
        N = S + E + I + R + D + Q
        
        # Equações diferenciais principais
        dSdt = -beta_efetivo * S * I / N - eta * S
        dEdt = beta_efetivo * S * I / N - sigma * E
        dIdt = sigma * E - gamma * I - mu * I
        dRdt = gamma * I + epsilon * eta * S
        dDdt = mu * I
        dQdt = eta * S - epsilon * eta * S
        
        return [dSdt, dEdt, dIdt, dRdt, dDdt, dQdt]
    
    def simular_epidemia(self, dias=180, populacao_total=1e6):
        """Executa a simulação da epidemia"""
        # Condições iniciais
        y0 = [
            self.variaveis['suscetiveis'] * populacao_total,
            self.variaveis['expostos'] * populacao_total,
            self.variaveis['infectados'] * populacao_total,
            self.variaveis['recuperados'] * populacao_total,
            self.variaveis['obitos'] * populacao_total,
            self.variaveis['isolados'] * populacao_total
        ]
        
        # Tempo de simulação
        t_span = (0, dias)
        t_eval = np.linspace(0, dias, dias)
        
        # Resolver sistema de EDOs
        solucao = solve_ivp(
            self.sistema_equacoes_diferenciais,
            t_span,
            y0,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-6
        )
        
        return solucao.t, solucao.y
    
    def analise_sensibilidade(self, parametro, valores, dias=180):
        """Análise de sensibilidade para um parâmetro"""
        resultados = []
        
        for valor in valores:
            self.parametros[parametro] = valor
            t, y = self.simular_epidemia(dias)
            pico_infectados = np.max(y[2])
            total_obitos = y[4][-1]
            resultados.append((valor, pico_infectados, total_obitos))
        
        return resultados

def criar_visualizacoes(t, y, populacao_total):
    """Cria visualizações sofisticadas dos resultados"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Gráfico 1: Evolução temporal das compartimentos
    ax1.plot(t, y[0], label='Suscetíveis', linewidth=2)
    ax1.plot(t, y[1], label='Expostos', linewidth=2)
    ax1.plot(t, y[2], label='Infectados', linewidth=2, color='red')
    ax1.plot(t, y[3], label='Recuperados', linewidth=2)
    ax1.plot(t, y[4], label='Óbitos', linewidth=2, color='black')
    ax1.plot(t, y[5], label='Isolados', linewidth=2, linestyle='--')
    ax1.set_title('Evolução Temporal da Epidemia', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Dias')
    ax1.set_ylabel('Indivíduos')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Infectados ativos (escala logarítmica)
    ax2.semilogy(t, y[2], label='Infectados', color='red', linewidth=2)
    ax2.set_title('Infectados Ativos (Escala Logarítmica)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Dias')
    ax2.set_ylabel('Infectados (log)')
    ax2.grid(True, alpha=0.3)
    
    # Gráfico 3: Novos casos diários
    novos_casos = np.diff(y[2])  # Aproximação de novos casos
    ax3.plot(t[1:], novos_casos, label='Novos Casos Diários', color='orange', linewidth=2)
    ax3.set_title('Novos Casos Diários', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Dias')
    ax3.set_ylabel('Novos Casos')
    ax3.grid(True, alpha=0.3)
    
    # Gráfico 4: Proporções da população
    total_populacao = np.sum(y[:, -1])
    proporcoes_finais = [y[i][-1] / total_populacao for i in range(6)]
    categorias = ['Suscetíveis', 'Expostos', 'Infectados', 'Recuperados', 'Óbitos', 'Isolados']
    cores = ['blue', 'yellow', 'red', 'green', 'black', 'purple']
    
    ax4.bar(categorias, proporcoes_finais, color=cores, alpha=0.7)
    ax4.set_title('Distribuição Final da População', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Proporção da População')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

def calcular_metricas_epidemicas(y, populacao_total):
    """Calcula métricas epidemiológicas importantes"""
    infectados = y[2]
    obitos = y[4]
    
    metricas = {
        'pico_infectados': np.max(infectados),
        'dia_pico': np.argmax(infectados),
        'total_obitos': obitos[-1],
        'taxa_ataque': (populacao_total - y[0][-1]) / populacao_total * 100,
        'taxa_letalidade': (obitos[-1] / np.max(infectados)) * 100 if np.max(infectados) > 0 else 0
    }
    
    return metricas

# EXECUÇÃO PRINCIPAL
if __name__ == "__main__":
    # Criar e configurar o modelo
    modelo = ModeloEpidemicoSofisticado()
    
    # Parâmetros de simulação
    populacao_total = 1_000_000
    dias_simulacao = 200
    
    print("=== MODELO EPIDÊMICO SOFISTICADO DE INFLUENZA ===")
    print(f"População: {populacao_total:,} habitantes")
    print(f"Período de simulação: {dias_simulacao} dias")
    print("\nParâmetros iniciais:")
    for param, valor in modelo.parametros.items():
        print(f"  {param}: {valor}")
    
    # Executar simulação principal
    print("\nExecutando simulação...")
    tempo, resultados = modelo.simular_epidemia(dias_simulacao, populacao_total)
    
    # Calcular métricas
    metricas = calcular_metricas_epidemicas(resultados, populacao_total)
    
    print("\n=== RESULTADOS NUMÉRICOS ===")
    print(f"Pico de infectados: {metricas['pico_infectados']:,.0f} pessoas (dia {metricas['dia_pico']})")
    print(f"Total de óbitos: {metricas['total_obitos']:,.0f} pessoas")
    print(f"Taxa de ataque: {metricas['taxa_ataque']:.2f}%")
    print(f"Taxa de letalidade entre infectados: {metricas['taxa_letalidade']:.2f}%")
    
    # Análise de sensibilidade
    print("\n=== ANÁLISE DE SENSIBILIDADE ===")
    valores_beta = [0.2, 0.3, 0.4, 0.5]
    sensibilidade = modelo.analise_sensibilidade('taxa_transmissao_basica', valores_beta)
    
    print("Variação da taxa de transmissão (β):")
    for beta, pico, obitos in sensibilidade:
        print(f"β = {beta}: Pico = {pico:,.0f}, Óbitos = {obitos:,.0f}")
    
    # Criar visualizações
    print("\nGerando visualizações...")
    criar_visualizacoes(tempo, resultados, populacao_total)
    
    # Tabela resumo temporal
    print("\n=== RESUMO TEMPORAL (a cada 30 dias) ===")
    indices = range(0, len(tempo), 30)
    dados_tabela = []
    
    for i in indices:
        if i < len(tempo):
            dados_tabela.append({
                'Dia': int(tempo[i]),
                'Suscetíveis': f"{resultados[0][i]:,.0f}",
                'Infectados': f"{resultados[2][i]:,.0f}",
                'Recuperados': f"{resultados[3][i]:,.0f}",
                'Óbitos': f"{resultados[4][i]:,.0f}"
            })
    
    df_resumo = pd.DataFrame(dados_tabela)
    print(df_resumo.to_string(index=False))