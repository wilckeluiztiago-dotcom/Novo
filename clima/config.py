"""
MÓDULO DE CONFIGURAÇÃO - Sistema de Modelagem Climática
=========================================================

Constantes físicas, parâmetros atmosféricos/oceânicos e configurações globais.
Todas as unidades no Sistema Internacional (SI).

Autor: Luiz Tiago Wilcke
Data: 2025-11-24
"""

import numpy as np
from typing import Dict, Any

# =============================================================================
# CONSTANTES FÍSICAS FUNDAMENTAIS
# =============================================================================

class ConstantesFisicas:
    """Constantes físicas fundamentais"""
    
    # Gravitação
    GRAVIDADE = 9.80665  # m/s² - aceleração da gravidade padrão
    CONSTANTE_GRAVITACIONAL = 6.67430e-11  # m³/kg/s² - Lei de Newton
    
    # Termodinâmica
    CONSTANTE_BOLTZMANN = 1.380649e-23  # J/K
    CONSTANTE_AVOGADRO = 6.02214076e23  # mol⁻¹
    CONSTANTE_GASES_UNIVERSAL = 8.314462618  # J/(mol·K)
    
    # Radiação
    CONSTANTE_STEFAN_BOLTZMANN = 5.670374419e-8  # W/(m²·K⁴)
    CONSTANTE_PLANCK = 6.62607015e-34  # J·s
    VELOCIDADE_LUZ = 299792458  # m/s
    
    # Terra
    RAIO_TERRA = 6.371e6  # m - raio médio
    MASSA_TERRA = 5.972e24  # kg
    EXCENTRICIDADE_ORBITAL = 0.0167  # adimensional
    OBLIQUIDADE = 23.44  # graus - inclinação eixo
    
    # Astronomia
    CONSTANTE_SOLAR = 1361.0  # W/m² - irradiância solar no topo da atmosfera
    DISTANCIA_SOL_TERRA = 1.496e11  # m - 1 UA
    TEMPERATURA_SOL = 5778  # K - temperatura efetiva
    
    # Oceano
    DENSIDADE_AGUA_MAR = 1025.0  # kg/m³
    SALINIDADE_MEDIA = 35.0  # PSU (Practical Salinity Units)
    CALOR_ESPECIFICO_AGUA = 4186.0  # J/(kg·K)
    
    # Atmosfera ao nível do mar
    PRESSAO_NIVEL_MAR = 101325.0  # Pa
    TEMPERATURA_PADRAO = 288.15  # K (15°C)
    DENSIDADE_AR_NIVEL_MAR = 1.225  # kg/m³


# =============================================================================
# PARÂMETROS ATMOSFÉRICOS
# =============================================================================

class ParametrosAtmosfera:
    """Parâmetros da atmosfera e gases"""
    
    # Composição atmosférica (fração molar)
    COMPOSICAO = {
        'N2': 0.7808,   # Nitrogênio
        'O2': 0.2095,   # Oxigênio
        'Ar': 0.0093,   # Argônio
        'CO2': 0.000415,  # Dióxido de carbono (415 ppm - pré-industrial ~280 ppm)
        'Ne': 0.000018,  # Neônio
        'He': 0.000005,  # Hélio
        'CH4': 0.000001900,  # Metano (1900 ppb)
        'Kr': 0.000001,  # Criptônio
        'H2': 0.0000005,  # Hidrogênio
        'N2O': 0.00000033,  # Óxido nitroso (330 ppb)
        'O3': 0.00000004,  # Ozônio (variável)
    }
    
    # Massas molares (kg/mol)
    MASSA_MOLAR = {
        'N2': 0.028014,
        'O2': 0.031998,
        'Ar': 0.039948,
        'CO2': 0.04401,
        'CH4': 0.01604,
        'H2O': 0.018015,
        'ar_seco': 0.028965,  # massa molar média do ar seco
    }
    
    # Constante específica de gases (J/kg/K)
    R_AR_SECO = 287.05  # R = R_universal / M_ar
    R_VAPOR_AGUA = 461.5
    
    # Calores específicos do ar (J/kg/K)
    CP_AR = 1005.0  # a pressão constante
    CV_AR = 718.0   # a volume constante
    RAZAO_CALORES_ESPECIFICOS = CP_AR / CV_AR  # γ ≈ 1.4
    
    # Parâmetros termodinâmicos
    CALOR_LATENTE_VAPORIZACAO = 2.501e6  # J/kg a 0°C
    CALOR_LATENTE_FUSAO = 3.34e5  # J/kg
    CALOR_LATENTE_SUBLIMACAO = 2.834e6  # J/kg
    
    # Pressão de vapor saturado (equação de Clausius-Clapeyron)
    # e_s = e_0 * exp[(L/R_v)(1/T_0 - 1/T)]
    PRESSAO_VAPOR_REF = 611.2  # Pa a 273.15 K
    TEMPERATURA_REF = 273.15  # K (0°C)
    
    # Gradiente térmico vertical
    TAXA_ADIAB_SECA = 9.8  # K/km - gradiente adiabático seco
    TAXA_ADIAB_UMIDA = 6.5  # K/km - gradiente adiabático úmido (média)
    TAXA_AMBIENTAL_PADRAO = 6.5  # K/km - atmosfera padrão
    
    # Altura de escala atmosférica
    ALTURA_ESCALA = 8400  # m - H = RT/Mg
    
    # Número de Reynolds típico
    VISCOSIDADE_CINEMATICA = 1.5e-5  # m²/s


# =============================================================================
# PARÂMETROS OCEÂNICOS
# =============================================================================

class ParametrosOceano:
    """Parâmetros do oceano"""
    
    # Propriedades físicas
    DENSIDADE_REFERENCIA = 1025.0  # kg/m³
    SALINIDADE_REFERENCIA = 35.0  # PSU
    TEMPERATURA_REFERENCIA = 10.0  # °C
    
    # Coeficientes termohalinos
    EXPANSAO_TERMICA = 2.0e-4  # K⁻¹
    CONTRACAO_HALINA = 7.6e-4  # PSU⁻¹
    
    # Difusividades
    DIFUSIVIDADE_TERMICA_VERTICAL = 1.0e-5  # m²/s
    DIFUSIVIDADE_TERMICA_HORIZONTAL = 1000.0  # m²/s
    DIFUSIVIDADE_SALINA_VERTICAL = 1.0e-5  # m²/s
    DIFUSIVIDADE_SALINA_HORIZONTAL = 1000.0  # m²/s
    
    # Viscosidades
    VISCOSIDADE_VERTICAL = 1.0e-3  # m²/s
    VISCOSIDADE_HORIZONTAL = 1.0e5  # m²/s
    
    # Profundidades características
    PROFUNDIDADE_CAMADA_MISTA = 50.0  # m
    PROFUNDIDADE_TERMOCLINA = 500.0  # m
    PROFUNDIDADE_MEDIA = 3700.0  # m
    PROFUNDIDADE_MAXIMA = 11000.0  # m - Fossa das Marianas
    
    # Capacidade térmica oceânica
    CAPACIDADE_TERMICA_OCEANO = 4.0e9  # J/(m²·K) para 1000m de profundidade


# =============================================================================
# PARÂMETROS RADIATIVOS
# =============================================================================

class ParametrosRadiacao:
    """Parâmetros de radiação e efeito estufa"""
    
    # Radiação solar
    CONSTANTE_SOLAR = 1361.0  # W/m²
    FATOR_GEOMETRICO = 0.25  # S₀/4 para média global
    
    # Albedo (refletividade)
    ALBEDO_PLANETARIO = 0.30  # albedo médio global
    ALBEDO_NUVENS = 0.50
    ALBEDO_GELO_MAR = 0.60
    ALBEDO_GELO_CONTINENTAL = 0.80
    ALBEDO_NEVE_FRESCA = 0.85
    ALBEDO_OCEANO = 0.06
    ALBEDO_FLORESTA = 0.15
    ALBEDO_GRAMA = 0.25
    ALBEDO_DESERTO = 0.40
    
    # Emissividade (corpo negro = 1.0)
    EMISSIVIDADE_TERRA = 0.95  # superfície
    EMISSIVIDADE_ATMOSFERA = 0.78  # com gases de efeito estufa
    
    # Forçante radiativa (W/m²)
    # ΔF = α * ln(C/C₀)
    FORCANTE_CO2_DOBRO = 3.7  # W/m² para duplicação de CO2
    COEF_FORCANTE_CO2 = 5.35  # ΔF = 5.35 * ln(C/C₀)
    COEF_FORCANTE_CH4 = 0.036  # W/m² por ppb
    COEF_FORCANTE_N2O = 0.12  # W/m² por ppb
    
    # Concentrações de referência (pré-industrial)
    CO2_PREINDUSTRIAL = 280.0  # ppm
    CH4_PREINDUSTRIAL = 700.0  # ppb
    N2O_PREINDUSTRIAL = 270.0  # ppb
    
    # Opacidade atmosférica
    PROFUNDIDADE_OTICA_VAPOR = 2.0
    PROFUNDIDADE_OTICA_CO2 = 0.5
    PROFUNDIDADE_OTICA_OZONIO = 0.3


# =============================================================================
# PARÂMETROS DA CRIOSFERA
# =============================================================================

class ParametrosCriosfera:
    """Parâmetros de gelo, neve e permafrost"""
    
    # Propriedades físicas do gelo
    DENSIDADE_GELO = 917.0  # kg/m³
    DENSIDADE_NEVE = 300.0  # kg/m³ (variável: 100-500)
    CALOR_ESPECIFICO_GELO = 2090.0  # J/(kg·K)
    CONDUTIVIDADE_TERMICA_GELO = 2.2  # W/(m·K)
    
    # Temperaturas críticas
    TEMPERATURA_FUSAO_GELO = 273.15  # K (0°C)
    TEMPERATURA_FUSAO_AGUA_MAR = 271.25  # K (-1.9°C com salinidade)
    
    # Espessuras características
    ESPESSURA_GELO_MAR_ARTICO = 2.5  # m (média)
    ESPESSURA_GELO_MAR_ANTARTICO = 1.5  # m (média)
    ESPESSURA_MANTO_GROENLANDIA = 1500.0  # m (média)
    ESPESSURA_MANTO_ANTARTICA = 2000.0  # m (média)
    
    # Taxas de crescimento/derretimento
    TAXA_CRESCIMENTO_GELO = 0.01  # m/dia (condições favoráveis)
    TAXA_DERRETIMENTO_GELO = 0.05  # m/dia (verão)
    
    # Feedback albedo-gelo
    GANHO_FEEDBACK_ALBEDO = 0.4  # amplificação polar


# =============================================================================
# PARÂMETROS DE SUPERFÍCIE TERRESTRE
# =============================================================================

class ParametrosSuperficie:
    """Parâmetros de uso do solo e vegetação"""
    
    # Tipos de cobertura vegetal e albedos
    TIPOS_VEGETACAO = {
        'floresta_tropical': {'albedo': 0.13, 'rugosidade': 2.0, 'lai': 6.0},
        'floresta_temperada': {'albedo': 0.15, 'rugosidade': 1.5, 'lai': 5.0},
        'floresta_boreal': {'albedo': 0.12, 'rugosidade': 1.0, 'lai': 4.0},
        'savana': {'albedo': 0.20, 'rugosidade': 0.5, 'lai': 2.0},
        'grama': {'albedo': 0.25, 'rugosidade': 0.05, 'lai': 1.5},
        'tundra': {'albedo': 0.20, 'rugosidade': 0.03, 'lai': 0.5},
        'agricultura': {'albedo': 0.20, 'rugosidade': 0.1, 'lai': 3.0},
        'deserto': {'albedo': 0.40, 'rugosidade': 0.01, 'lai': 0.1},
    }
    
    # Capacidade de água no solo
    CAPACIDADE_CAMPO = 0.30  # fração volumétrica
    PONTO_MURCHA = 0.10  # fração volumétrica
    SATURACAO = 0.45  # porosidade típica
    
    # Evapotranspiração
    RESISTENCIA_ESTOMATICA_MIN = 100.0  # s/m
    RESISTENCIA_AERODINAMICA = 50.0  # s/m
    COEF_EVAPOTRANSPIRACAO = 0.5  # adimensional
    
    # Condutividade térmica do solo
    CONDUTIVIDADE_SOLO_SECO = 0.3  # W/(m·K)
    CONDUTIVIDADE_SOLO_UMIDO = 1.5  # W/(m·K)
    CAPACIDADE_TERMICA_SOLO = 2.0e6  # J/(m³·K)


# =============================================================================
# PARÂMETROS DO CICLO DE CARBONO
# =============================================================================

class ParametrosCarbono:
    """Parâmetros do ciclo biogeoquímico de carbono"""
    
    # Reservatórios de carbono (GtC - gigatoneladas de carbono)
    RESERVATORIO_ATMOSFERA = 870.0  # pré-industrial ~600
    RESERVATORIO_OCEANO_SUPERFICIAL = 900.0
    RESERVATORIO_OCEANO_PROFUNDO = 37100.0
    RESERVATORIO_BIOSFERA_TERRESTRE = 2300.0
    RESERVATORIO_SOLO = 1500.0
    RESERVATORIO_COMBUSTIVEIS_FOSSEIS = 5000.0
    
    # Fluxos (GtC/ano)
    PRODUCAO_PRIMARIA_BRUTA = 120.0  # fotossíntese terrestre
    RESPIRACAO_PLANTAS = 60.0
    RESPIRACAO_SOLO = 60.0
    EMISSOES_ANTROPOGENICAS_2020 = 10.0  # combustíveis fósseis + uso da terra
    ABSORCAO_OCEANICA = 2.5
    ABSORCAO_TERRESTRE = 3.0
    
    # Constantes de tempo (anos)
    TEMPO_RESIDENCIA_ATMOSFERA = 4.0
    TEMPO_RESIDENCIA_BIOSFERA = 20.0
    TEMPO_RESIDENCIA_OCEANO_SUPERFICIE = 10.0
    TEMPO_RESIDENCIA_OCEANO_PROFUNDO = 1000.0
    
    # Solubilidade do CO2 em água (dependente de T)
    # K_H = K_0 * exp[d(ln K_H)/d(1/T) * (1/T - 1/T_0)]
    HENRY_CO2_REF = 3.4e-2  # mol/(L·atm) a 25°C
    ENTALPIA_DISSOLUCAO_CO2 = -2400.0  # J/mol


# =============================================================================
# CONFIGURAÇÕES DE SIMULAÇÃO
# =============================================================================

class ConfiguracaoSimulacao:
    """Parâmetros de discretização espacial e temporal"""
    
    # Resolução espacial
    NUM_LATITUDE = 90  # número de pontos em latitude (2° resolução)
    NUM_LONGITUDE = 180  # número de pontos em longitude (2° resolução)
    NUM_NIVEIS_ATMOSFERA = 20  # níveis verticais na atmosfera
    NUM_NIVEIS_OCEANO = 30  # níveis verticais no oceano
    
    # Domínio espacial
    LAT_MIN = -90.0  # graus
    LAT_MAX = 90.0
    LON_MIN = 0.0
    LON_MAX = 360.0
    
    # Níveis de pressão atmosférica (Pa)
    NIVEIS_PRESSAO = np.array([
        100, 200, 300, 500, 700, 850, 925, 1000, 1013.25
    ]) * 100  # convertido para Pa
    
    # Profundidades oceânicas (m)
    NIVEIS_PROFUNDIDADE = np.array([
        0, 10, 20, 30, 50, 75, 100, 125, 150, 200,
        250, 300, 400, 500, 600, 700, 800, 900, 1000,
        1100, 1200, 1300, 1400, 1500, 1750, 2000, 2500,
        3000, 3500, 4000
    ])
    
    # Resolução temporal
    PASSO_TEMPO_ATMOSFERA = 600  # segundos (10 minutos)
    PASSO_TEMPO_OCEANO = 3600  # segundos (1 hora)
    PASSO_TEMPO_GELO = 86400  # segundos (1 dia)
    PASSO_TEMPO_CARBONO = 86400  # segundos (1 dia)
    
    # Duração da simulação
    TEMPO_INICIAL = 0.0  # anos desde início
    TEMPO_FINAL = 100.0  # anos
    
    # Condição de Courant-Friedrichs-Lewy (CFL) para estabilidade
    # CFL = u * Δt / Δx < 1
    CFL_MAX = 0.8
    
    # Frequência de salvamento
    FREQ_SALVAMENTO_DADOS = 30  # dias
    FREQ_SALVAMENTO_CHECKPOINT = 365  # dias


# =============================================================================
# FUNÇÕES UTILITÁRIAS DE CONFIGURAÇÃO
# =============================================================================

def obter_todos_parametros() -> Dict[str, Any]:
    """
    Retorna dicionário com todos os parâmetros do modelo.
    
    Returns:
        Dict: Dicionário contendo todas as configurações
    """
    return {
        'fisica': vars(ConstantesFisicas),
        'atmosfera': vars(ParametrosAtmosfera),
        'oceano': vars(ParametrosOceano),
        'radiacao': vars(ParametrosRadiacao),
        'criosfera': vars(ParametrosCriosfera),
        'superficie': vars(ParametrosSuperficie),
        'carbono': vars(ParametrosCarbono),
        'simulacao': vars(ConfiguracaoSimulacao),
    }


def calcular_temperatura_equilibrio(
    albedo: float = 0.30,
    emissividade: float = 0.95,
    constante_solar: float = 1361.0
) -> float:
    """
    Calcula temperatura de equilíbrio radiativo simples.
    
    T_eq = [(S₀/4)(1-α) / (σε)]^(1/4)
    
    Args:
        albedo: Albedo planetário (0-1)
        emissividade: Emissividade da superfície (0-1)
        constante_solar: Irradiância solar (W/m²)
    
    Returns:
        float: Temperatura de equilíbrio (K)
    """
    sigma = ConstantesFisicas.CONSTANTE_STEFAN_BOLTZMANN
    fluxo_absorvido = (constante_solar / 4) * (1 - albedo)
    temp_eq = (fluxo_absorvido / (sigma * emissividade)) ** 0.25
    return temp_eq


def calcular_sensibilidade_climatica(forcante_radiativa: float = 3.7) -> float:
    """
    Estima sensibilidade climática (aquecimento por duplicação de CO2).
    
    ΔT = λ * ΔF
    onde λ é o parâmetro de sensibilidade climática (~0.8 K/(W/m²))
    
    Args:
        forcante_radiativa: Forçante radiativa (W/m²)
    
    Returns:
        float: Mudança de temperatura (K)
    """
    # Sensibilidade climática típica: 0.5 a 1.2 K/(W/m²)
    # Valor médio do IPCC
    parametro_sensibilidade = 0.8  # K/(W/m²)
    delta_temperatura = parametro_sensibilidade * forcante_radiativa
    return delta_temperatura


if __name__ == "__main__":
    # Testes e demonstrações
    print("=" * 70)
    print("SISTEMA DE MODELAGEM CLIMÁTICA - Configurações")
    print("=" * 70)
    
    print("\n1. Constantes Físicas:")
    print(f"   Gravidade: {ConstantesFisicas.GRAVIDADE} m/s²")
    print(f"   Constante Solar: {ConstantesFisicas.CONSTANTE_SOLAR} W/m²")
    print(f"   σ (Stefan-Boltzmann): {ConstantesFisicas.CONSTANTE_STEFAN_BOLTZMANN} W/(m²·K⁴)")
    
    print("\n2. Atmosfera:")
    print(f"   R_ar: {ParametrosAtmosfera.R_AR_SECO} J/(kg·K)")
    print(f"   Cp_ar: {ParametrosAtmosfera.CP_AR} J/(kg·K)")
    print(f"   CO₂ atual: {ParametrosAtmosfera.COMPOSICAO['CO2'] * 1e6:.1f} ppm")
    
    print("\n3. Radiação:")
    print(f"   Albedo planetário: {ParametrosRadiacao.ALBEDO_PLANETARIO}")
    print(f"   Forçante CO₂ (2x): {ParametrosRadiacao.FORCANTE_CO2_DOBRO} W/m²")
    
    print("\n4. Temperatura de Equilíbrio:")
    T_eq = calcular_temperatura_equilibrio()
    print(f"   Sem efeito estufa: {T_eq:.2f} K ({T_eq-273.15:.2f} °C)")
    
    # Com efeito estufa (emissividade efetiva menor)
    T_eq_greenhouse = calcular_temperatura_equilibrio(emissividade=0.62)
    print(f"   Com efeito estufa: {T_eq_greenhouse:.2f} K ({T_eq_greenhouse-273.15:.2f} °C)")
    
    print("\n5. Sensibilidade Climática:")
    delta_T = calcular_sensibilidade_climatica()
    print(f"   ΔT (duplicação CO₂): {delta_T:.2f} K")
    print(f"   Considerando feedbacks (fator 3): ~{delta_T * 3:.1f} K")
    
    print("\n6. Resolução Espacial:")
    print(f"   Grade: {ConfiguracaoSimulacao.NUM_LATITUDE} lat × {ConfiguracaoSimulacao.NUM_LONGITUDE} lon")
    print(f"   Níveis atmosféricos: {ConfiguracaoSimulacao.NUM_NIVEIS_ATMOSFERA}")
    print(f"   Níveis oceânicos: {ConfiguracaoSimulacao.NUM_NIVEIS_OCEANO}")
    
    print("\n" + "=" * 70)
