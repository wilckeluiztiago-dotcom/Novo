"""
MÓDULO DE ATMOSFERA - Sistema de Modelagem Climática
=====================================================

Modelagem completa da dinâmica atmosférica incluindo:
- Equações primitivas (Navier-Stokes em esfera rotativa)
- Termodinâmica atmosférica
- Transporte de umidade
- Balanço de energia

Autor: Luiz Tiago Wilcke
Data: 2025-11-24
"""

import numpy as np
from typing import Dict, Tuple, Optional
from config import (
    ConstantesFisicas,
    ParametrosAtmosfera,
    ConfiguracaoSimulacao
)
from grid import GradeEspacial
from radiation import ModeloRadiacao
from solver import SolverRungeKutta4, SolverDifusao, SolverAdveccao


class EstadoAtmosferico:
    """Armazena o estado completo da atmosfera"""
    
    def __init__(self, grade: GradeEspacial):
        """
        Inicializa estado atmosférico.
        
        Args:
            grade: Grade espacial do modelo
        """
        self.grade = grade
        
        # Dimensões
        nlat = grade.num_lat
        nlon = grade.num_lon
        nlev = grade.num_niveis_atm
        
        # Campos 3D [nivel, lat, lon]
        self.temperatura = np.zeros((nlev, nlat, nlon))  # K
        self.u = np.zeros((nlev, nlat, nlon))  # m/s - vento zonal
        self.v = np.zeros((nlev, nlat, nlon))  # m/s - vento meridional
        self.w = np.zeros((nlev, nlat, nlon))  # Pa/s - velocidade vertical
        self.umidade_especifica = np.zeros((nlev, nlat, nlon))  # kg/kg
        
        # Campos 2D [lat, lon] - superfície
        self.temperatura_superficie = np.zeros((nlat, nlon))  # K
        self.pressao_superficie = np.ones((nlat, nlon)) * ConstantesFisicas.PRESSAO_NIVEL_MAR  # Pa
        self.precipitacao = np.zeros((nlat, nlon))  # kg/m²/s
        self.cobertura_nuvens = np.zeros((nlat, nlon))  # fração [0-1]
    
    def inicializar_temperatura_realista(self):
        """Inicializa campo de temperatura com distribuição realista"""
        # Temperatura da superfície: gradiente equador-polo
        T_eq = 303  # K no equador (~30°C)
        T_polo = 243  # K nos polos (~-30°C)
        
        lat_rad = self.grade.lat_rad_2d
        self.temperatura_superficie = T_eq - (T_eq - T_polo) * np.sin(np.abs(lat_rad))**2
        
        # Temperatura vertical: usar gradiente adiabático
        for k in range(self.grade.num_niveis_atm):
            p = self.grade.niveis_pressao[k]
            p_surf = ConstantesFisicas.PRESSAO_NIVEL_MAR
            
            # Altura aproximada (usando fórmula barométrica)
            H = ParametrosAtmosfera.ALTURA_ESCALA
            z = -H * np.log(p / p_surf)
            
            # Temperatura = T_surf - Γ * z
            gamma = ParametrosAtmosfera.TAXA_AMBIENTAL_PADRAO / 1000  # K/m
            self.temperatura[k, :, :] = self.temperatura_superficie - gamma * z
    
    def inicializar_vento_zonal(self):
        """Inicializa vento zonal com padrão de jatos subtropicais"""
        # Jatos subtropicais em ~30° N/S
        lat_rad = self.grade.lat_rad_2d
        
        for k in range(self.grade.num_niveis_atm):
            p = self.grade.niveis_pressao[k]
            
            # Jatos mais fortes em níveis médios/altos (~250 hPa)
            if p < 30000:  # Acima de 300 hPa
                # Dois jatos subtropicais
                u_max = 40.0  # m/s
                lat_jato = np.deg2rad(30)
                
                jato_norte = u_max * np.exp(-((lat_rad - lat_jato)**2) / 0.3)
                jato_sul = u_max * np.exp(-((lat_rad + lat_jato)**2) / 0.3)
                
                self.u[k, :, :] = jato_norte + jato_sul
            else:
                # Ventos de superfície (alísios)
                u_alisio = 7.0 * np.cos(lat_rad)
                self.u[k, :, :] = u_alisio
    
    def inicializar_umidade(self):
        """Inicializa campo de umidade específica"""
        for k in range(self.grade.num_niveis_atm):
            T = self.temperatura[k, :, :]
            p = self.grade.niveis_pressao[k]
            
            # Umidade relativa típica: 70% nos trópicos, 50% em latitudes médias
            lat_rad = self.grade.lat_rad_2d
            ur_tropical = 0.70
            ur_media = 0.50
            
            ur = ur_tropical - (ur_tropical - ur_media) * np.abs(np.sin(lat_rad))
            
            # Calcular umidade específica a partir de UR
            # e_sat usando Clausius-Clapeyron
            from utils import calcular_pressao_vapor_saturado
            e_sat = calcular_pressao_vapor_saturado(T)
            e = ur * e_sat
            
            # q = ε * e / (p - e*(1-ε))
            epsilon = ParametrosAtmosfera.MASSA_MOLAR['H2O'] / ParametrosAtmosfera.MASSA_MOLAR['ar_seco']
            self.umidade_especifica[k, :, :] = epsilon * e / (p - e * (1 - epsilon))
    
    def estado_para_vetor(self) -> np.ndarray:
        """Converte estado para vetor plano (para integração temporal)"""
        return np.concatenate([
            self.temperatura.ravel(),
            self.u.ravel(),
            self.v.ravel(),
            self.umidade_especifica.ravel(),
            self.temperatura_superficie.ravel(),
        ])
    
    def vetor_para_estado(self, vetor: np.ndarray):
        """Atualiza estado a partir de vetor plano"""
        nlat = self.grade.num_lat
        nlon = self.grade.num_lon
        nlev = self.grade.num_niveis_atm
        
        idx = 0
        
        # Temperatura 3D
        n_temp = nlev * nlat * nlon
        self.temperatura = vetor[idx:idx+n_temp].reshape((nlev, nlat, nlon))
        idx += n_temp
        
        # Vento u
        self.u = vetor[idx:idx+n_temp].reshape((nlev, nlat, nlon))
        idx += n_temp
        
        # Vento v
        self.v = vetor[idx:idx+n_temp].reshape((nlev, nlat, nlon))
        idx += n_temp
        
        # Umidade
        self.umidade_especifica = vetor[idx:idx+n_temp].reshape((nlev, nlat, nlon))
        idx += n_temp
        
        # Temperatura superfície
        n_surf = nlat * nlon
        self.temperatura_superficie = vetor[idx:idx+n_surf].reshape((nlat, nlon))


class ModeloAtmosferico:
    """Modelo atmosférico completo com equações primitivas"""
    
    def __init__(self, grade: GradeEspacial):
        """
        Inicializa modelo atmosférico.
        
        Args:
            grade: Grade espacial
        """
        self.grade = grade
        self.estado = EstadoAtmosferico(grade)
        self.radiacao = ModeloRadiacao()
        
        # Parâmetros físicos
        self.omega = 7.2921e-5  # rad/s - velocidade angular da Terra
        self.R = ConstantesFisicas.RAIO_TERRA
        self.g = ConstantesFisicas.GRAVIDADE
        self.R_gas = ParametrosAtmosfera.R_AR_SECO
        self.cp = ParametrosAtmosfera.CP_AR
        
        # Difusividades para parametrizações
        self.difusividade_horizontal = 1.0e5  # m²/s
        self.difusividade_vertical = 1.0  # m²/s
    
    def calcular_parametro_coriolis(self) -> np.ndarray:
        """
        Calcula parâmetro de Coriolis.
        
        f = 2Ω sin(φ)
        
        Returns:
            Parâmetro de Coriolis (1/s)
        """
        return 2 * self.omega * np.sin(self.grade.lat_rad_2d)
    
    def calcular_forca_coriolis(
        self,
        u: np.ndarray,
        v: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula força de Coriolis.
        
        F_u = f × v
        F_v = -f × u
        
        Args:
            u: Vento zonal (m/s)
            v: Vento meridional (m/s)
        
        Returns:
            Tuple (F_u, F_v) em m/s²
        """
        f = self.calcular_parametro_coriolis()
        
        F_u = f * v
        F_v = -f * u
        
        return F_u, F_v
    
    def calcular_adveccao_temperatura(
        self,
        T: np.ndarray,
        u: np.ndarray,
        v: np.ndarray
    ) -> np.ndarray:
        """
        Calcula advecção de temperatura.
        
        -V·∇T = -u ∂T/∂x - v ∂T/∂y
        
        Args:
            T: Temperatura (K)
            u: Vento zonal (m/s)
            v: Vento meridional (m/s)
        
        Returns:
            Taxa de mudança por advecção (K/s)
        """
        grad_x = self.grade.calcular_gradiente_x(T)
        grad_y = self.grade.calcular_gradiente_y(T)
        
        adveccao = -(u * grad_x + v * grad_y)
        
        return adveccao
    
    def calcular_forca_gradiente_pressao(
        self,
        temperatura: np.ndarray,
        pressao_nivel: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula força do gradiente de pressão.
        
        F = -∇Φ = -R*T*∇(ln p)
        
        Args:
            temperatura: Campo de temperatura (K)
            pressao_nivel: Pressão no nível (Pa)
        
        Returns:
            Tuple (F_x, F_y) em m/s²
        """
        # Para modelo simplificado, usar gradiente de temperatura
        # como proxy para gradiente de pressão
        grad_T_x = self.grade.calcular_gradiente_x(temperatura)
        grad_T_y = self.grade.calcular_gradiente_y(temperatura)
        
        # F ∝ R*T*∇T/T ≈ R*∇T
        F_x = -self.R_gas * grad_T_x
        F_y = -self.R_gas * grad_T_y
        
        return F_x, F_y
    
    def calcular_aquecimento_radiativo(
        self,
        temperatura_superficie: np.ndarray,
        temperatura_atmosfera: np.ndarray,
        dia_ano: float
    ) -> np.ndarray:
        """
        Calcula aquecimento radiativo da atmosfera.
        
        Args:
            temperatura_superficie: Temperatura da superfície (K)
            temperatura_atmosfera: Temperatura atmosférica (K)
            dia_ano: Dia do ano [0, 365]
        
        Returns:
            Taxa de aquecimento (K/s)
        """
        # Radiação solar incidente
        Q_solar = self.radiacao.calcular_radiacao_solar_topo(
            self.grade.latitudes,
            dia_ano
        )
        
        # Albedo simplificado
        albedo = 0.30 * np.ones_like(temperatura_superficie)
        
        # Radiação absorvida
        SW_abs = self.radiacao.calcular_radiacao_onda_curta_absorvida(
            Q_solar[:, np.newaxis],
            albedo
        )
        
        # Radiação de onda longa
        LW_emit = self.radiacao.calcular_radiacao_onda_longa_emitida(
            temperatura_superficie
        )
        
        # Balanço radiativo
        balanco_rad = SW_abs - LW_emit
        
        # Converter para taxa de aquecimento
        # dT/dt = Q / (ρ * cp * Δz)
        # Simplificação: distribuir sobre coluna atmosférica
        rho = 1.2  # kg/m³ densidade média
        delta_z = 10000  # m - escala de altura
        
        taxa_aquecimento = balanco_rad / (rho * self.cp * delta_z)
        
        return taxa_aquecimento
    
    def calcular_tendencias(
        self,
        tempo: float,
        estado_vetor: np.ndarray
    ) -> np.ndarray:
        """
        Calcula tendências temporais (dX/dt) para todas as variáveis.
        
        Esta é a função principal que integra todas as equações governantes.
        
        Args:
            tempo: Tempo atual (s)
            estado_vetor: Estado atual como vetor
        
        Returns:
            Vetor de tendências (derivadas temporais)
        """
        # Reconstruir estado a partir do vetor
        estado_temp = EstadoAtmosferico(self.grade)
        estado_temp.vetor_para_estado(estado_vetor)
        
        # Dimensões
        nlev = self.grade.num_niveis_atm
        nlat = self.grade.num_lat
        nlon = self.grade.num_lon
        
        # Inicializar tendências
        dT_dt = np.zeros_like(estado_temp.temperatura)
        du_dt = np.zeros_like(estado_temp.u)
        dv_dt = np.zeros_like(estado_temp.v)
        dq_dt = np.zeros_like(estado_temp.umidade_especifica)
        dT_surf_dt = np.zeros_like(estado_temp.temperatura_superficie)
        
        # Dia do ano (assumindo 1 ano = 365 dias)
        dia_ano = (tempo / 86400) % 365
        
        # Calcular aquecimento radiativo na superfície
        T_atm_baixo = estado_temp.temperatura[-1, :, :]  # Nível mais baixo
        aquec_rad_surf = self.calcular_aquecimento_radiativo(
            estado_temp.temperatura_superficie,
            T_atm_baixo,
            dia_ano
        )
        
        # Loop sobre níveis verticais
        for k in range(nlev):
            T_k = estado_temp.temperatura[k, :, :]
            u_k = estado_temp.u[k, :, :]
            v_k = estado_temp.v[k, :, :]
            q_k = estado_temp.umidade_especifica[k, :, :]
            p_k = self.grade.niveis_pressao[k]
            
            # ===== EQUAÇÃO DO MOMENTO (vento) =====
            # du/dt = -u ∂u/∂x - v ∂u/∂y + fv - (1/ρ)∂p/∂x + D_u
            # dv/dt = -u ∂v/∂x - v ∂v/∂y - fu - (1/ρ)∂p/∂y + D_v
            
            # Advecção de momento
            adv_u = self.calcular_adveccao_temperatura(u_k, u_k, v_k)
            adv_v = self.calcular_adveccao_temperatura(v_k, u_k, v_k)
            
            # Força de Coriolis
            F_cor_u, F_cor_v = self.calcular_forca_coriolis(u_k, v_k)
            
            # Gradiente de pressão
            F_pgf_u, F_pgf_v = self.calcular_forca_gradiente_pressao(T_k, p_k)
            
            # Difusão horizontal (parametrização de turbulência)
            difusao_u = self.difusividade_horizontal * self.grade.calcular_laplaciano(u_k)
            difusao_v = self.difusividade_horizontal * self.grade.calcular_laplaciano(v_k)
            
            # Tendências do vento
            du_dt[k, :, :] = adv_u + F_cor_u + F_pgf_u + difusao_u
            dv_dt[k, :, :] = adv_v + F_cor_v + F_pgf_v + difusao_v
            
            # ===== EQUAÇÃO TERMODINÂMICA (temperatura) =====
            # dT/dt = -u ∂T/∂x - v ∂T/∂y + Q_rad + Q_latente + D_T
            
            # Advecção de temperatura
            adv_T = self.calcular_adveccao_temperatura(T_k, u_k, v_k)
            
            # Aquecimento radiativo (simplificado - propagar da superfície)
            peso_vertical = np.exp(-k / 5)  # Decai com altura
            Q_rad = aquec_rad_surf * peso_vertical
            
            # Difusão térmica
            difusao_T = self.difusividade_horizontal * self.grade.calcular_laplaciano(T_k)
            
            # Tendência da temperatura
            dT_dt[k, :, :] = adv_T + Q_rad + difusao_T
            
            # ===== EQUAÇÃO DA UMIDADE =====
            # dq/dt = -u ∂q/∂x - v ∂q/∂y + E - P
            
            # Advecção de umidade
            adv_q = self.calcular_adveccao_temperatura(q_k, u_k, v_k)
            
            # Fonte/sumidouro (evaporação - precipitação)
            E = 0.0  # Simplificado
            P = 0.0
            
            # Difusão de umidade
            difusao_q = self.difusividade_horizontal * self.grade.calcular_laplaciano(q_k)
            
            dq_dt[k, :, :] = adv_q + E - P + difusao_q
        
        # Tendência da temperatura da superfície
        # Acoplamento com atmosfera + balanço radiativo
        fluxo_sensivel = 0.01 * (T_atm_baixo - estado_temp.temperatura_superficie)
        dT_surf_dt = aquec_rad_surf + fluxo_sensivel
        
        # Limitar tendências para estabilidade numérica
        max_rate_T = 10.0 / 86400  # máx 10 K/dia
        max_rate_u = 50.0 / 86400  # máx 50 m/s/dia
        
        dT_dt = np.clip(dT_dt, -max_rate_T, max_rate_T)
        du_dt = np.clip(du_dt, -max_rate_u, max_rate_u)
        dv_dt = np.clip(dv_dt, -max_rate_u, max_rate_u)
        dT_surf_dt = np.clip(dT_surf_dt, -max_rate_T, max_rate_T)
        
        # Converter para vetor
        tendencias = np.concatenate([
            dT_dt.ravel(),
            du_dt.ravel(),
            dv_dt.ravel(),
            dq_dt.ravel(),
            dT_surf_dt.ravel(),
        ])
        
        return tendencias
    
    def integrar(
        self,
        dias_simulacao: float,
        dt_horas: float = 1.0
    ) -> Dict[str, np.ndarray]:
        """
        Integra modelo no tempo.
        
        Args:
            dias_simulacao: Número de dias para simular
            dt_horas: Passo de tempo em horas
        
        Returns:
            Dicionário com histórico das variáveis
        """
        dt = dt_horas * 3600  # Converter para segundos
        tempo_final = dias_simulacao * 86400
        n_passos = int(tempo_final / dt)
        
        # Inicializar estado
        self.estado.inicializar_temperatura_realista()
        self.estado.inicializar_vento_zonal()
        self.estado.inicializar_umidade()
        
        # Arrays para armazenar histórico
        historico = {
            'tempo': np.zeros(n_passos + 1),
            'temperatura_media': np.zeros(n_passos + 1),
            'temperatura_superficie_media': np.zeros(n_passos + 1),
            'velocidade_vento_max': np.zeros(n_passos + 1),
        }
        
        # Solver
        solver = SolverRungeKutta4()
        
        # Condição inicial
        estado_vetor = self.estado.estado_para_vetor()
        historico['tempo'][0] = 0
        historico['temperatura_media'][0] = np.mean(self.estado.temperatura)
        historico['temperatura_superficie_media'][0] = self.grade.media_global(self.estado.temperatura_superficie)
        historico['velocidade_vento_max'][0] = np.max(np.sqrt(self.estado.u**2 + self.estado.v**2))
        
        print(f"Iniciando simulação: {dias_simulacao} dias, dt = {dt_horas}h")
        print("=" * 70)
        
        # Integração temporal
        for i in range(n_passos):
            tempo_atual = i * dt
            
            # Um passo RK4
            estado_vetor = solver.passo(
                estado_vetor,
                tempo_atual,
                dt,
                self.calcular_tendencias
            )
            
            # Atualizar estado
            self.estado.vetor_para_estado(estado_vetor)
            
            # Salvar histórico
            historico['tempo'][i+1] = tempo_atual + dt
            historico['temperatura_media'][i+1] = np.mean(self.estado.temperatura)
            historico['temperatura_superficie_media'][i+1] = self.grade.media_global(
                self.estado.temperatura_superficie
            )
            velocidade = np.sqrt(self.estado.u**2 + self.estado.v**2)
            historico['velocidade_vento_max'][i+1] = np.max(velocidade)
            
            # Progresso
            if (i + 1) % 24 == 0:  # A cada dia simulado (se dt=1h)
                dia = (i + 1) * dt_horas / 24
                T_media = historico['temperatura_media'][i+1]
                T_surf = historico['temperatura_superficie_media'][i+1]
                print(f"Dia {dia:6.1f}: T_média={T_media:6.2f} K, T_surf={T_surf:6.2f} K")
        
        print("=" * 70)
        print("Simulação concluída!")
        
        return historico


if __name__ == "__main__":
    # Teste do modelo atmosférico
    print("=" * 70)
    print("MODELO ATMOSFÉRICO - Teste")
    print("=" * 70)
    
    # Criar grade de baixa resolução para teste
    grade = GradeEspacial(num_lat=45, num_lon=90, num_niveis_atm=10)
    print(f"\n{grade}")
    
    # Criar modelo
    modelo = ModeloAtmosferico(grade)
    
    # Simular 10 dias
    print("\nExecutando simulação de 10 dias...")
    historico = modelo.integrar(dias_simulacao=10, dt_horas=1.0)
    
    # Mostrar resultados
    print("\n" + "=" * 70)
    print("RESULTADOS DA SIMULAÇÃO")
    print("=" * 70)
    
    tempo_dias = historico['tempo'] / 86400
    
    print(f"\nTemperatura atmosférica média:")
    print(f"  Inicial: {historico['temperatura_media'][0]:.2f} K")
    print(f"  Final:   {historico['temperatura_media'][-1]:.2f} K")
    print(f"  Variação: {historico['temperatura_media'][-1] - historico['temperatura_media'][0]:+.2f} K")
    
    print(f"\nTemperatura de superfície média:")
    print(f"  Inicial: {historico['temperatura_superficie_media'][0]:.2f} K")
    print(f"  Final:   {historico['temperatura_superficie_media'][-1]:.2f} K")
    
    print(f"\nVelocidade do vento:")
    print(f"  Máxima inicial: {historico['velocidade_vento_max'][0]:.2f} m/s")
    print(f"  Máxima final:   {historico['velocidade_vento_max'][-1]:.2f} m/s")
    
    print("\n" + "=" * 70)
