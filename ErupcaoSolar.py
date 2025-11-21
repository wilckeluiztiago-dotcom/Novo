import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import pandas as pd
from datetime import datetime, timedelta

class ModeloErupcoesSolares:
    """
    Modelo matemático avançado para previsão de erupções solares perigosas
    Baseado em equações diferenciais que descrevem a física do plasma solar
    """
    
    def __init__(self):
        # Constantes físicas
        self.constante_boltzmann = 1.380649e-23
        self.permeabilidade_vacuo = 4 * np.pi * 1e-7
        self.massa_proton = 1.6726219e-27
        self.carga_eletron = 1.60217662e-19
        
        # Parâmetros do modelo
        self.parametros = self._inicializar_parametros()
        
    def _inicializar_parametros(self):
        """Inicializa os parâmetros físicos do modelo"""
        return {
            'temperatura_corona': 1e6,  # K
            'densidade_plasma': 1e15,   # partículas/m³
            'campo_magnetico_inicial': 0.1,  # Tesla
            'gradiente_campo_magnetico': 1e-4,  # T/m
            'comprimento_caracteristico': 1e7,  # metros
            'taxa_ressurgimento_magnetico': 1e-3,
            'coeficiente_difusao': 1e10,
            'limiar_erupcao': 2.5e-3
        }
    
    def equacoes_diferenciais_principais(self, t, variaveis, parametros):
        """
        Sistema de equações diferenciais que descreve a evolução do sistema magnético solar
        """
        # Variáveis de estado
        energia_magnetica = variaveis[0]  # Energia magnética armazenada
        helicidade_magnetica = variaveis[1]  # Helicidade magnética
        tensao_cisalhamento = variaveis[2]  # Tensão por cisalhamento magnético
        corrente_eletrica = variaveis[3]   # Corrente elétrica
        
        # Parâmetros
        B0 = parametros['campo_magnetico_inicial']
        L = parametros['comprimento_caracteristico']
        η = parametros['coeficiente_difusao']
        α = parametros['taxa_ressurgimento_magnetico']
        
        # Equações diferenciais
        dEm_dt = -η * energia_magnetica / L**2 + α * B0**2 * tensao_cisalhamento
        dH_dt = -2 * η * helicidade_magnetica / L**2 + 2 * α * B0 * energia_magnetica
        dτ_dt = -tensao_cisalhamento / (10 * 3600) + 0.1 * corrente_eletrica  # Relaxamento em ~10 horas
        dI_dt = -corrente_eletrica / (3600) + 0.01 * energia_magnetica  # Decaimento em ~1 hora
        
        return [dEm_dt, dH_dt, dτ_dt, dI_dt]
    
    def modelo_termo_magnetico(self, t, y, parametros):
        """
        Modelo acoplado termo-magnético para regiões ativas solares
        """
        temperatura = y[0]
        campo_magnetico = y[1]
        pressao_plasma = y[2]
        vorticiade = y[3]
        
        T0 = parametros['temperatura_corona']
        ρ = parametros['densidade_plasma']
        B0 = parametros['campo_magnetico_inicial']
        L = parametros['comprimento_caracteristico']
        
        # Constantes derivadas
        velocidade_alfven = B0 / np.sqrt(self.permeabilidade_vacuo * ρ * self.massa_proton)
        tempo_alfven = L / velocidade_alfven
        
        # Equações do modelo termo-magnético
        dT_dt = - (temperatura - T0) / (3600) + 0.5 * campo_magnetico**2 / B0**2
        dB_dt = - campo_magnetico / tempo_alfven + 0.1 * vorticiade
        dP_dt = - pressao_plasma / (2 * 3600) + 0.2 * temperatura / T0
        dω_dt = - vorticiade / (3600) + 0.05 * campo_magnetico * pressao_plasma
        
        return [dT_dt, dB_dt, dP_dt, dω_dt]
    
    def calcular_criterio_erupcao(self, solucao):
        """
        Calcula critérios para prever erupções solares
        Baseado em parâmetros observacionais e teóricos
        """
        energia_magnetica = solucao.y[0]
        helicidade = solucao.y[1]
        tensao_cisalhamento = solucao.y[2]
        
        # Critério 1: Excesso de energia magnética
        criterio_energia = energia_magnetica / np.max(energia_magnetica)
        
        # Critério 2: Acúmulo de helicidade
        criterio_helicidade = helicidade / np.max(np.abs(helicidade))
        
        # Critério 3: Cisalhamento magnético
        criterio_cisalhamento = np.abs(tensao_cisalhamento) / np.max(np.abs(tensao_cisalhamento))
        
        # Critério combinado
        criterio_combinado = (0.4 * criterio_energia + 
                            0.35 * criterio_helicidade + 
                            0.25 * criterio_cisalhamento)
        
        return {
            'criterio_energia': criterio_energia,
            'criterio_helicidade': criterio_helicidade,
            'criterio_cisalhamento': criterio_cisalhamento,
            'criterio_combinado': criterio_combinado,
            'probabilidade_erupcao': self._calcular_probabilidade(criterio_combinado)
        }
    
    def _calcular_probabilidade(self, criterio_combinado):
        """Calcula probabilidade de erupção baseada no critério combinado"""
        # Função sigmoide para transformar em probabilidade
        return 1 / (1 + np.exp(-10 * (criterio_combinado - 0.6)))
    
    def simular_regiao_ativa(self, tempo_simulacao, condicoes_iniciais=None):
        """
        Simula a evolução de uma região ativa solar
        """
        if condicoes_iniciais is None:
            condicoes_iniciais = [1e20, 1e15, 0.0, 1e10]  # Valores iniciais típicos
        
        # Intervalo de tempo (em segundos)
        t_span = (0, tempo_simulacao)
        t_eval = np.linspace(0, tempo_simulacao, 1000)
        
        # Resolver equações diferenciais
        solucao = solve_ivp(
            self.equacoes_diferenciais_principais,
            t_span,
            condicoes_iniciais,
            args=(self.parametros,),
            t_eval=t_eval,
            method='RK45'
        )
        
        return solucao
    
    def prever_erupcao(self, dados_observacionais):
        """
        Faz previsão de erupção solar baseada em dados observacionais
        """
        # Processar dados de entrada
        dados_processados = self._processar_dados_entrada(dados_observacionais)
        
        # Simular evolução
        solucao = self.simular_regiao_ativa(
            tempo_simulacao=24 * 3600,  # 24 horas
            condicoes_iniciais=dados_processados['condicoes_iniciais']
        )
        
        # Calcular critérios de erupção
        criterios = self.calcular_criterio_erupcao(solucao)
        
        # Gerar previsão
        previsao = self._gerar_previsao(criterios, solucao)
        
        return previsao
    
    def _processar_dados_entrada(self, dados):
        """Processa dados observacionais para uso no modelo"""
        # Aqui você implementaria a conversão de dados reais
        # Para este exemplo, usamos valores simulados
        return {
            'condicoes_iniciais': [
                dados.get('energia_magnetica', 5e19),
                dados.get('helicidade', 2e14),
                dados.get('cisalhamento', 0.1),
                dados.get('corrente', 5e9)
            ],
            'parametros_ajustados': self.parametros.copy()
        }
    
    def _gerar_previsao(self, criterios, solucao):
        """Gera a previsão final baseada nos critérios calculados"""
        prob_maxima = np.max(criterios['probabilidade_erupcao'])
        tempo_max_prob = solucao.t[np.argmax(criterios['probabilidade_erupcao'])]
        
        # Classificar risco
        if prob_maxima > 0.8:
            risco = "ALTO"
            alerta = "ERUPÇÃO PROVÁVEL"
        elif prob_maxima > 0.6:
            risco = "MODERADO"
            alerta = "RISCO ELEVADO"
        elif prob_maxima > 0.4:
            risco = "BAIXO"
            alerta = "MONITORAR"
        else:
            risco = "MÍNIMO"
            alerta = "CONDIÇÕES ESTÁVEIS"
        
        return {
            'probabilidade_maxima': prob_maxima,
            'tempo_previsto': tempo_max_prob,
            'nivel_risco': risco,
            'alerta': alerta,
            'criterios_detalhados': criterios,
            'solucao_completa': solucao
        }

# Exemplo de uso e visualização
def demonstrar_modelo():
    """Demonstra o uso do modelo de previsão de erupções solares"""
    
    # Criar instância do modelo
    modelo = ModeloErupcoesSolares()
    
    # Dados de exemplo para uma região ativa
    dados_regiao_ativa = {
        'energia_magnetica': 8e19,
        'helicidade': 3e14,
        'cisalhamento': 0.15,
        'corrente': 8e9
    }
    
    print("=== MODELO DE PREVISÃO DE ERUPÇÕES SOLARES ===")
    print("Simulando região ativa solar...")
    
    # Fazer previsão
    previsao = modelo.prever_erupcao(dados_regiao_ativa)
    
    # Exibir resultados
    print(f"\n--- RESULTADOS DA PREVISÃO ---")
    print(f"Probabilidade máxima de erupção: {previsao['probabilidade_maxima']:.2%}")
    print(f"Nível de risco: {previsao['nivel_risco']}")
    print(f"Alerta: {previsao['alerta']}")
    print(f"Tempo previsto para pico: {previsao['tempo_previsto']/3600:.1f} horas")
    
    # Visualizar resultados
    visualizar_resultados(previsao)

def visualizar_resultados(previsao):
    """Visualiza os resultados da simulação"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    t_horas = previsao['solucao_completa'].t / 3600
    
    # Gráfico 1: Energia magnética e critérios
    axes[0, 0].plot(t_horas, previsao['solucao_completa'].y[0], 'b-', label='Energia Magnética')
    axes[0, 0].set_ylabel('Energia (J)')
    axes[0, 0].set_xlabel('Tempo (horas)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    axes[0, 0].set_title('Evolução da Energia Magnética')
    
    # Gráfico 2: Probabilidade de erupção
    axes[0, 1].plot(t_horas, previsao['criterios_detalhados']['probabilidade_erupcao'], 'r-', linewidth=2)
    axes[0, 1].axhline(y=0.6, color='orange', linestyle='--', label='Limiar Moderado')
    axes[0, 1].axhline(y=0.8, color='red', linestyle='--', label='Limiar Alto')
    axes[0, 1].set_ylabel('Probabilidade de Erupção')
    axes[0, 1].set_xlabel('Tempo (horas)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    axes[0, 1].set_title('Probabilidade de Erupção Solar')
    
    # Gráfico 3: Critérios individuais
    axes[1, 0].plot(t_horas, previsao['criterios_detalhados']['criterio_energia'], label='Energia')
    axes[1, 0].plot(t_horas, previsao['criterios_detalhados']['criterio_helicidade'], label='Helicidade')
    axes[1, 0].plot(t_horas, previsao['criterios_detalhados']['criterio_cisalhamento'], label='Cisalhamento')
    axes[1, 0].set_ylabel('Critérios Normalizados')
    axes[1, 0].set_xlabel('Tempo (horas)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    axes[1, 0].set_title('Critérios de Erupção Individual')
    
    # Gráfico 4: Variáveis de estado
    axes[1, 1].plot(t_horas, previsao['solucao_completa'].y[1], label='Helicidade')
    axes[1, 1].plot(t_horas, previsao['solucao_completa'].y[2], label='Tensão Cisalhamento')
    axes[1, 1].set_ylabel('Valores')
    axes[1, 1].set_xlabel('Tempo (horas)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    axes[1, 1].set_title('Variáveis de Estado Adicionais')
    
    plt.tight_layout()
    plt.show()

# Classe para análise em tempo real (extensão)
class AnalisadorTempoReal:
    """Analisador para dados em tempo real de observatórios solares"""
    
    def __init__(self, modelo):
        self.modelo = modelo
        self.historico = []
    
    def atualizar_com_dados_reais(self, dados_tempo_real):
        """Atualiza o modelo com dados em tempo real"""
        # Implementação para dados reais de satélites
        previsao_atualizada = self.modelo.prever_erupcao(dados_tempo_real)
        self.historico.append({
            'timestamp': datetime.now(),
            'previsao': previsao_atualizada,
            'dados_entrada': dados_tempo_real
        })
        return previsao_atualizada

if __name__ == "__main__":
    # Executar demonstração
    demonstrar_modelo()