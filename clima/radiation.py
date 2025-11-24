"""
MÓDULO DE RADIAÇÃO - Sistema de Modelagem Climática
====================================================

Balanço radiativo global, albedo, efeito estufa e forçante radiativa.

Autor: Luiz Tiago Wilcke
Data: 2025-11-24
"""

import numpy as np
from typing import Tuple, Dict
from config import (
    ConstantesFisicas,
    ParametrosRadiacao,
    ParametrosAtmosfera,
    ParametrosCriosfera
)


class ModeloRadiacao:
    """Modelo de balanço radiativo global"""
    
    def __init__(self):
        """Inicializa modelo de radiação"""
        self.sigma = ConstantesFisicas.CONSTANTE_STEFAN_BOLTZMANN
        self.S0 = ConstantesFisicas.CONSTANTE_SOLAR
        
        # Concentrações atuais de GEE (Gases de Efeito Estufa)
        self.co2_atual = 415.0  # ppm
        self.ch4_atual = 1900.0  # ppb
        self.n2o_atual = 333.0  # ppb
    
    def calcular_radiacao_solar_topo(
        self,
        latitude: np.ndarray,
        dia_ano: float
    ) -> np.ndarray:
        """
        Calcula radiação solar incidente no topo da atmosfera.
        
        Considera:
        - Variação sazonal (declinação solar)
        - Variação latitudinal (ângulo zenital)
        - Ciclo diurno (média diária)
        
        Args:
            latitude: Latitude em graus [-90, 90]
            dia_ano: Dia do ano [0, 365]
        
        Returns:
            Radiação solar no topo (W/m²)
        """
        lat_rad = np.deg2rad(latitude)
        
        # Declinação solar (inclinação do eixo da Terra)
        # δ = -23.44° × cos(360°/365 × (dia + 10))
        declinacao = -23.44 * np.cos(2 * np.pi * (dia_ano + 10) / 365)
        dec_rad = np.deg2rad(declinacao)
        
        # Ângulo horário do pôr do sol (H₀)
        cos_H0 = -np.tan(lat_rad) * np.tan(dec_rad)
        cos_H0 = np.clip(cos_H0, -1, 1)  # Limitar para evitar erros numéricos
        H0 = np.arccos(cos_H0)
        
        # Radiação média diária
        # Q = (S₀/π) × (H₀ sin φ sin δ + cos φ cos δ sin H₀)
        termo1 = H0 * np.sin(lat_rad) * np.sin(dec_rad)
        termo2 = np.cos(lat_rad) * np.cos(dec_rad) * np.sin(H0)
        
        Q = (self.S0 / np.pi) * (termo1 + termo2)
        
        # Noite polar: Q = 0
        Q[np.isnan(Q)] = 0
        Q = np.maximum(Q, 0)
        
        return Q
    
    def calcular_albedo_planetario(
        self,
        albedo_superficie: np.ndarray,
        cobertura_nuvens: np.ndarray,
        fracao_gelo: np.ndarray = None
    ) -> np.ndarray:
        """
        Calcula albedo planetário total.
        
        α_p = f_nuvem × α_nuvem + (1 - f_nuvem) × α_superficie
        
        Args:
            albedo_superficie: Albedo da superfície [0-1]
            cobertura_nuvens: Fração de cobertura de nuvens [0-1]
            fracao_gelo: Fração de cobertura de gelo [0-1]
        
        Returns:
            Albedo planetário efetivo
        """
        # Albedo de nuvens
        alpha_nuvem = ParametrosRadiacao.ALBEDO_NUVENS
        
        # Modificar albedo de superfície se houver gelo
        if fracao_gelo is not None:
            alpha_gelo = ParametrosRadiacao.ALBEDO_GELO_MAR
            albedo_superficie = (
                fracao_gelo * alpha_gelo +
                (1 - fracao_gelo) * albedo_superficie
            )
        
        # Albedo planetário combinado
        albedo_planet = (
            cobertura_nuvens * alpha_nuvem +
            (1 - cobertura_nuvens) * albedo_superficie
        )
        
        return albedo_planet
    
    def calcular_forcante_radiativa_co2(
        self,
        concentracao_co2: float,
        co2_referencia: float = None
    ) -> float:
        """
        Calcula forçante radiativa do CO₂.
        
        ΔF = 5.35 × ln(C/C₀)
        
        Args:
            concentracao_co2: Concentração atual (ppm)
            co2_referencia: Concentração de referência (ppm)
        
        Returns:
            Forçante radiativa (W/m²)
        """
        if co2_referencia is None:
            co2_referencia = ParametrosRadiacao.CO2_PREINDUSTRIAL
        
        forcante = ParametrosRadiacao.COEF_FORCANTE_CO2 * np.log(concentracao_co2 / co2_referencia)
        
        return forcante
    
    def calcular_forcante_total_gee(
        self,
        co2: float = None,
        ch4: float = None,
        n2o: float = None
    ) -> float:
        """
        Calcula forçante radiativa total de todos os GEE.
        
        Args:
            co2: Concentração de CO₂ (ppm)
            ch4: Concentração de CH₄ (ppb)
            n2o: Concentração de N₂O (ppb)
        
        Returns:
            Forçante total (W/m²)
        """
        # Usar valores atuais se não especificado
        co2 = co2 or self.co2_atual
        ch4 = ch4 or self.ch4_atual
        n2o = n2o or self.n2o_atual
        
        # Forçante CO₂
        F_co2 = self.calcular_forcante_radiativa_co2(co2)
        
        # Forçante CH₄
        ch4_ref = ParametrosRadiacao.CH4_PREINDUSTRIAL
        F_ch4 = ParametrosRadiacao.COEF_FORCANTE_CH4 * (ch4 - ch4_ref)
        
        # Forçante N₂O
        n2o_ref = ParametrosRadiacao.N2O_PREINDUSTRIAL
        F_n2o = ParametrosRadiacao.COEF_FORCANTE_N2O * (n2o - n2o_ref)
        
        forcante_total = F_co2 + F_ch4 + F_n2o
        
        return forcante_total
    
    def calcular_emissividade_atmosfera(
        self,
        vapor_agua: float,
        co2: float,
        temperatura: float
    ) -> float:
        """
        Calcula emissividade efetiva da atmosfera.
        
        ε_a ≈ 0.77 × (1 + 0.2 × ln(e/e₀))
        
        Args:
            vapor_agua: Pressão de vapor d'água (Pa)
            co2: Concentração de CO₂ (ppm)
            temperatura: Temperatura atmosférica (K)
        
        Returns:
            Emissividade atmosférica [0-1]
        """
        # Emissividade base (céu claro)
        e_base = 0.77
        
        # Aumento devido ao vapor d'água
        e_ref = 1000.0  # Pa
        fator_vapor = 0.2 * np.log(vapor_agua / e_ref) if vapor_agua > 0 else 0
        
        # Aumento devido ao CO₂
        co2_ref = ParametrosRadiacao.CO2_PREINDUSTRIAL
        fator_co2 = 0.05 * np.log(co2 / co2_ref)
        
        emissividade = e_base * (1 + fator_vapor + fator_co2)
        emissividade = np.clip(emissividade, 0.6, 0.95)
        
        return emissividade
    
    def calcular_radiacao_onda_curta_absorvida(
        self,
        radiacao_solar_topo: np.ndarray,
        albedo: np.ndarray
    ) -> np.ndarray:
        """
        Calcula radiação de onda curta absorvida.
        
        SW_abs = (1 - α) × S
        
        Args:
            radiacao_solar_topo: Radiação solar incidente (W/m²)
            albedo: Albedo planetário [0-1]
        
        Returns:
            Radiação absorvida (W/m²)
        """
        return (1 - albedo) * radiacao_solar_topo
    
    def calcular_radiacao_onda_longa_emitida(
        self,
        temperatura_superficie: np.ndarray,
        emissividade_atm: float = None
    ) -> np.ndarray:
        """
        Calcula radiação de onda longa emitida ao espaço.
        
        Com efeito estufa:
        LW_emit = ε_a × σT_s⁴ + (1-ε_a) × σT_s⁴
        
        Args:
            temperatura_superficie: Temperatura da superfície (K)
            emissividade_atm: Emissividade atmosférica [0-1]
        
        Returns:
            Radiação emitida (W/m²)
        """
        if emissividade_atm is None:
            emissividade_atm = ParametrosRadiacao.EMISSIVIDADE_ATMOSFERA
        
        # Radiação de corpo negro da superfície
        radiacao_superficie = self.sigma * temperatura_superficie**4
        
        # Parte da radiação que escapa ao espaço
        # (1 - ε_a) representa a janela atmosférica
        radiacao_emitida = (1 - emissividade_atm) * radiacao_superficie
        
        # Radiação da atmosfera para o espaço
        # Assumindo T_atm ≈ 0.8 × T_s
        temp_atm = 0.8 * temperatura_superficie
        radiacao_atm = emissividade_atm * self.sigma * temp_atm**4
        
        radiacao_total = radiacao_emitida + radiacao_atm
        
        return radiacao_total
    
    def calcular_balanco_radiativo(
        self,
        radiacao_solar: np.ndarray,
        albedo: np.ndarray,
        temperatura: np.ndarray,
        emiss_atm: float = 0.78
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcula balanço radiativo completo.
        
        ΔR = SW_abs - LW_emit
        
        Args:
            radiacao_solar: Radiação solar incidente (W/m²)
            albedo: Albedo planetário
            temperatura: Temperatura da superfície (K)
            emiss_atm: Emissividade atmosférica
        
        Returns:
            Tuple (SW absorvida, LW emitida, balanço líquido)
        """
        sw_abs = self.calcular_radiacao_onda_curta_absorvida(radiacao_solar, albedo)
        lw_emit = self.calcular_radiacao_onda_longa_emitida(temperatura, emiss_atm)
        
        balanco = sw_abs - lw_emit
        
        return sw_abs, lw_emit, balanco
    
    def calcular_temperatura_equilibrio(
        self,
        radiacao_solar_media: float,
        albedo: float = 0.30,
        emissividade_atm: float = 0.78
    ) -> float:
        """
        Calcula temperatura de equilíbrio radiativo.
        
        Em equilíbrio: SW_abs = LW_emit
        
        Args:
            radiacao_solar_media: Radiação solar média (W/m²)
            albedo: Albedo médio
            emissividade_atm: Emissividade atmosférica
        
        Returns:
            Temperatura de equilíbrio (K)
        """
        # SW absorvida
        sw_abs = (1 - albedo) * radiacao_solar_media
        
        # Para equilíbrio, considerar efeito estufa simplificado
        # ε_eff = (1 - ε_a) + ε_a × (T_a/T_s)⁴
        # Aproximação: ε_eff ≈  0.6 para atmosfera real
        emiss_efetiva = 1 - emissividade_atm * 0.5
        
        # T_eq = [SW_abs / (σ × ε_eff)]^(1/4)
        T_eq = (sw_abs / (self.sigma * emiss_efetiva)) ** 0.25
        
        return T_eq


def criar_campo_albedo_superficie(
    latitude: np.ndarray,
    mascara_terra: np.ndarray = None
) -> np.ndarray:
    """
    Cria campo de albedo de superfície realista.
    
    Args:
        latitude: Array 2D de latitudes
        mascara_terra: Máscara booleana (True = terra)
    
    Returns:
        Albedo de superfície 2D
    """
    albedo = np.zeros_like(latitude)
    
    if mascara_terra is None:
        # Tudo oceano
        albedo[:] = ParametrosRadiacao.ALBEDO_OCEANO
    else:
        # Terra: varia com latitude (floresta tropical -> deserto -> gelo)
        albedo_terra = np.where(
            np.abs(latitude) < 23,  # Trópicos
            ParametrosRadiacao.ALBEDO_FLORESTA,
            np.where(
                np.abs(latitude) < 60,  # Latitudes médias
                ParametrosRadiacao.ALBEDO_GRAMA,
                ParametrosRadiacao.ALBEDO_GELO_CONTINENTAL  # Polares
            )
        )
        
        # Oceano
        albedo_oceano = ParametrosRadiacao.ALBEDO_OCEANO
        
        albedo = np.where(mascara_terra, albedo_terra, albedo_oceano)
    
    return albedo


if __name__ == "__main__":
    # Testes
    print("=" * 70)
    print("MÓDULO DE RADIAÇÃO - Teste")
    print("=" * 70)
    
    modelo_rad = ModeloRadiacao()
    
    # Teste 1: Radiação solar
    print("\n1. Radiação Solar no Topo da Atmosfera:")
    lats = np.array([-90, -60, -30, 0, 30, 60, 90])
    
    print("\n   Equinócio (dia 80):")
    Q_eq = modelo_rad.calcular_radiacao_solar_topo(lats, 80)
    for lat, q in zip(lats, Q_eq):
        print(f"   Lat {lat:3.0f}°: {q:6.1f} W/m²")
    
    print("\n   Solstício de Verão NH (dia 172):")
    Q_sol = modelo_rad.calcular_radiacao_solar_topo(lats, 172)
    for lat, q in zip(lats, Q_sol):
        print(f"   Lat {lat:3.0f}°: {q:6.1f} W/m²")
    
    # Teste 2: Forçante radiativa
    print("\n2. Forçante Radiativa dos GEE:")
    F_preindustrial = modelo_rad.calcular_forcante_total_gee(280, 700, 270)
    F_atual = modelo_rad.calcular_forcante_total_gee(415, 1900, 333)
    print(f"   Pré-industrial: {F_preindustrial:.2f} W/m²")
    print(f"   Atual (2024): {F_atual:.2f} W/m²")
    print(f"   Aumento: {F_atual - F_preindustrial:.2f} W/m²")
    
    # Teste 3: Duplicação de CO₂
    print("\n3. Cenários de CO₂:")
    for co2_ppm in [280, 415, 560, 840]:
        F = modelo_rad.calcular_forcante_radiativa_co2(co2_ppm, 280)
        print(f"   CO₂ = {co2_ppm:4.0f} ppm: ΔF = {F:+5.2f} W/m²")
    
    # Teste 4: Temperatura de equilíbrio
    print("\n4. Temperatura de Equilíbrio:")
    S_avg = modelo_rad.S0 / 4  # Média global
    
    # Sem efeito estufa
    T_sem_gee = modelo_rad.calcular_temperatura_equilibrio(S_avg, 0.30, 0.0)
    print(f"   Sem efeito estufa: {T_sem_gee:.1f} K ({T_sem_gee-273.15:.1f}°C)")
    
    # Com efeito estufa
    T_com_gee = modelo_rad.calcular_temperatura_equilibrio(S_avg, 0.30, 0.78)
    print(f"   Com efeito estufa: {T_com_gee:.1f} K ({T_com_gee-273.15:.1f}°C)")
    
    # Teste 5: Balanço radiativo
    print("\n5. Balanço Radiativo Global:")
    lat_2d = np.linspace(-90, 90, 91)[:, np.newaxis]
    temp_2d = 288 - 40 * np.sin(np.deg2rad(lat_2d))
    Q_2d = modelo_rad.calcular_radiacao_solar_topo(lat_2d.squeeze(), 80)[:, np.newaxis]
    albedo_2d = criar_campo_albedo_superficie(lat_2d)
    
    sw, lw, bal = modelo_rad.calcular_balanco_radiativo(Q_2d, albedo_2d, temp_2d)
    
    print(f"   SW absorvida média: {np.mean(sw):.1f} W/m²")
    print(f"   LW emitida média: {np.mean(lw):.1f} W/m²")
    print(f"   Balanço médio: {np.mean(bal):.2f} W/m² (deve ser ~0 em equilíbrio)")
    
    print("\n" + "=" * 70)
