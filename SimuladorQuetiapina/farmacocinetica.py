"""
Modelo Farmacocinético da Quetiapina
Autor: Luiz Tiago Wilcke
Data: 2025-11-25

Este módulo implementa um modelo farmacocinético compartimental avançado
para simular a absorção, distribuição, metabolismo e excreção (ADME) da Quetiapina.
"""

import numpy as np
from scipy.integrate import odeint
from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class ParametrosFarmacocineticos:
    """Parâmetros farmacocinéticos da Quetiapina ajustados por peso corporal"""
    
    # Peso do paciente (kg)
    peso_corporal: float
    
    # Taxa de absorção (1/h) - primeira ordem
    k_absorcao: float = 1.2
    
    # Volume de distribuição (L/kg) - altamente lipofílico
    volume_distribuicao_por_kg: float = 10.0
    
    # Clearance total (L/h/kg)
    clearance_por_kg: float = 1.2
    
    # Constante de eliminação (1/h)
    k_eliminacao: float = 0.12
    
    # Fração biodisponível (oral)
    biodisponibilidade: float = 0.73
    
    # Constante de distribuição cérebro-plasma
    k_cerebro: float = 0.3
    
    # Constante de retorno cérebro-plasma
    k_retorno_cerebro: float = 0.15
    
    # Ligação às proteínas plasmáticas (%)
    ligacao_proteica: float = 0.83
    
    @property
    def volume_distribuicao(self) -> float:
        """Volume total de distribuição (L)"""
        return self.volume_distribuicao_por_kg * self.peso_corporal
    
    @property
    def clearance_total(self) -> float:
        """Clearance total (L/h)"""
        return self.clearance_por_kg * self.peso_corporal
    
    @property
    def concentracao_livre(self) -> float:
        """Fração livre (não ligada a proteínas)"""
        return 1 - self.ligacao_proteica


class ModeloFarmacocinetico:
    """
    Modelo farmacocinético compartimentado para Quetiapina
    
    Compartimentos:
    1. TGI (Trato Gastrointestinal) - Absorção
    2. Plasma - Distribuição central
    3. Periférico - Tecidos periféricos
    4. Cérebro - Sistema nervoso central (alvo terapêutico)
    """
    
    def __init__(self, parametros: ParametrosFarmacocineticos):
        self.params = parametros
        
    def equacoes_diferenciais(self, estado: np.ndarray, t: float, 
                              dose_continua: float = 0.0) -> List[float]:
        """
        Sistema de equações diferenciais ordinárias (EDOs) do modelo ADME
        
        Equações:
        dA_tgi/dt = -k_abs * A_tgi + dose_continua
        dA_plasma/dt = k_abs * F * A_tgi - (CL/V) * A_plasma 
                       - k_cerebro * A_plasma + k_ret * A_cerebro
                       - k_periferico * A_plasma + k_ret_periferico * A_periferico
        dA_cerebro/dt = k_cerebro * A_plasma - k_ret * A_cerebro
        dA_periferico/dt = k_periferico * A_plasma - k_ret_periferico * A_periferico
        
        Args:
            estado: [quantidade_tgi, quantidade_plasma, quantidade_cerebro, quantidade_periferico]
            t: tempo (horas)
            dose_continua: taxa de administração contínua (mg/h)
        
        Returns:
            Derivadas de cada compartimento
        """
        A_tgi, A_plasma, A_cerebro, A_periferico = estado
        
        # Parâmetros
        k_abs = self.params.k_absorcao
        F = self.params.biodisponibilidade
        V_d = self.params.volume_distribuicao
        CL = self.params.clearance_total
        k_cb = self.params.k_cerebro
        k_ret_cb = self.params.k_retorno_cerebro
        
        # Constantes para compartimento periférico
        k_periferico = 0.08
        k_ret_periferico = 0.05
        
        # EDOs
        dA_tgi_dt = -k_abs * A_tgi + dose_continua
        
        dA_plasma_dt = (k_abs * F * A_tgi - 
                       (CL / V_d) * A_plasma -
                       k_cb * A_plasma + 
                       k_ret_cb * A_cerebro -
                       k_periferico * A_plasma +
                       k_ret_periferico * A_periferico)
        
        dA_cerebro_dt = k_cb * A_plasma - k_ret_cb * A_cerebro
        
        dA_periferico_dt = k_periferico * A_plasma - k_ret_periferico * A_periferico
        
        return [dA_tgi_dt, dA_plasma_dt, dA_cerebro_dt, dA_periferico_dt]
    
    def simular(self, dose_mg: float, tempo_horas: float = 72.0, 
                num_pontos: int = 1000, via: str = "oral") -> Tuple[np.ndarray, np.ndarray]:
        """
        Simula a farmacocinética da Quetiapina
        
        Args:
            dose_mg: Dose administrada (mg)
            tempo_horas: Tempo total de simulação (horas)
            num_pontos: Número de pontos temporais
            via: Via de administração ("oral" ou "intravenosa")
        
        Returns:
            Tupla (tempo, concentrações) onde concentrações é uma matriz 4xN
            com [C_tgi, C_plasma, C_cerebro, C_periferico]
        """
        # Vetor de tempo
        tempo = np.linspace(0, tempo_horas, num_pontos)
        
        # Condições iniciais
        if via == "oral":
            # Dose toda no TGI inicialmente
            estado_inicial = [dose_mg, 0, 0, 0]
        elif via == "intravenosa":
            # Dose toda no plasma inicialmente
            estado_inicial = [0, dose_mg, 0, 0]
        else:
            raise ValueError("Via deve ser 'oral' ou 'intravenosa'")
        
        # Resolver EDOs
        solucao = odeint(self.equacoes_diferenciais, estado_inicial, tempo)
        
        # Converter quantidades para concentrações
        concentracoes = np.zeros_like(solucao)
        concentracoes[:, 0] = solucao[:, 0]  # TGI (quantidade)
        concentracoes[:, 1] = solucao[:, 1] / self.params.volume_distribuicao  # Plasma (ng/mL)
        concentracoes[:, 2] = solucao[:, 2] / (0.02 * self.params.peso_corporal)  # Cérebro (ng/g)
        concentracoes[:, 3] = solucao[:, 3] / (0.8 * self.params.volume_distribuicao)  # Periférico
        
        return tempo, concentracoes
    
    def calcular_parametros_pk(self, tempo: np.ndarray, 
                               concentracao_plasma: np.ndarray) -> dict:
        """
        Calcula parâmetros farmacocinéticos importantes
        
        Args:
            tempo: Vetor de tempo (horas)
            concentracao_plasma: Concentração plasmática (ng/mL)
        
        Returns:
            Dicionário com parâmetros PK
        """
        # Concentração máxima
        C_max = np.max(concentracao_plasma)
        
        # Tempo para concentração máxima
        idx_max = np.argmax(concentracao_plasma)
        T_max = tempo[idx_max]
        
        # AUC (Área sob a curva) usando regra trapezoidal
        AUC = np.trapz(concentracao_plasma, tempo)
        
        # Tempo de meia-vida (aproximado)
        # Encontrar quando C cai para C_max/2
        idx_meia_vida = np.where(concentracao_plasma[idx_max:] <= C_max/2)[0]
        if len(idx_meia_vida) > 0:
            T_meia_vida = tempo[idx_max + idx_meia_vida[0]] - T_max
        else:
            T_meia_vida = np.nan
        
        # Clearance aparente
        CL_aparente = self.params.clearance_total
        
        # Volume de distribuição
        V_d = self.params.volume_distribuicao
        
        return {
            'Cmax_ng_mL': C_max,
            'Tmax_horas': T_max,
            'AUC_ng_h_mL': AUC,
            'Tmeia_vida_horas': T_meia_vida,
            'Clearance_L_h': CL_aparente,
            'Volume_distribuicao_L': V_d,
            'Concentracao_livre_%': self.params.concentracao_livre * 100
        }


class RegimePosologico:
    """Simula regimes de dosagem múltipla"""
    
    def __init__(self, modelo: ModeloFarmacocinetico):
        self.modelo = modelo
    
    def simular_doses_multiplas(self, dose_mg: float, intervalo_horas: float,
                                num_doses: int, tempo_total_horas: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simula administração de doses múltiplas
        
        Args:
            dose_mg: Dose por administração (mg)
            intervalo_horas: Intervalo entre doses (horas)
            num_doses: Número de doses
            tempo_total_horas: Tempo total de simulação
        
        Returns:
            Tupla (tempo, concentrações)
        """
        if tempo_total_horas is None:
            tempo_total_horas = intervalo_horas * num_doses + 24
        
        num_pontos = int(tempo_total_horas * 20)  # 20 pontos por hora
        tempo = np.linspace(0, tempo_total_horas, num_pontos)
        
        # Inicializar estado
        estado = np.array([0.0, 0.0, 0.0, 0.0])
        concentracoes = np.zeros((num_pontos, 4))
        
        # Simular passo a passo
        for i in range(num_pontos):
            t_atual = tempo[i]
            
            # Verificar se é hora de administrar dose
            dose_continua = 0.0
            for dose_num in range(num_doses):
                tempo_dose = dose_num * intervalo_horas
                if abs(t_atual - tempo_dose) < (tempo[1] - tempo[0]):
                    # Adicionar dose ao compartimento TGI
                    estado[0] += dose_mg
            
            # Resolver um pequeno passo de tempo
            if i > 0:
                dt = tempo[i] - tempo[i-1]
                derivadas = self.modelo.equacoes_diferenciais(estado, t_atual)
                estado = estado + np.array(derivadas) * dt
            
            # Converter para concentrações
            concentracoes[i, 0] = estado[0]
            concentracoes[i, 1] = estado[1] / self.modelo.params.volume_distribuicao
            concentracoes[i, 2] = estado[2] / (0.02 * self.modelo.params.peso_corporal)
            concentracoes[i, 3] = estado[3] / (0.8 * self.modelo.params.volume_distribuicao)
        
        return tempo, concentracoes


if __name__ == "__main__":
    # Exemplo de uso
    print("=" * 60)
    print("SIMULADOR FARMACOCINÉTICO DE QUETIAPINA")
    print("=" * 60)
    
    # Configurar parâmetros para um paciente de 70 kg
    params = ParametrosFarmacocineticos(peso_corporal=70.0)
    
    # Criar modelo
    modelo = ModeloFarmacocinetico(params)
    
    # Simular dose única de 300 mg via oral
    print("\nSimulando dose única de 300 mg via oral...")
    tempo, conc = modelo.simular(dose_mg=300.0, tempo_horas=48.0, via="oral")
    
    # Calcular parâmetros PK
    params_pk = modelo.calcular_parametros_pk(tempo, conc[:, 1])
    
    print("\nParâmetros Farmacocinéticos:")
    print("-" * 60)
    for param, valor in params_pk.items():
        print(f"{param}: {valor:.2f}")
    
    print("\n" + "=" * 60)
