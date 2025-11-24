"""
MÓDULO DE GRADE ESPACIAL - Sistema de Modelagem Climática
==========================================================

Sistema de coordenadas lat-lon, malhas verticais, interpolação e
operadores diferenciais.

Autor: Luiz Tiago Wilcke
Data: 2025-11-24
"""

import numpy as np
from typing import Tuple, Optional
from config import ConfiguracaoSimulacao, ConstantesFisicas


class GradeEspacial:
    """Classe para gerenciar grade espacial lat-lon com coordenadas verticais"""
    
    def __init__(
        self,
        num_lat: int = None,
        num_lon: int = None,
        num_niveis_atm: int = None,
        num_niveis_ocean: int = None
    ):
        """
        Inicializa grade espacial.
        
        Args:
            num_lat: Número de pontos em latitude
            num_lon: Número de pontos em longitude
            num_niveis_atm: Número de níveis atmosféricos
            num_niveis_ocean: Número de níveis oceânicos
        """
        # Usar valores padrão se não especificado
        self.num_lat = num_lat or ConfiguracaoSimulacao.NUM_LATITUDE
        self.num_lon = num_lon or ConfiguracaoSimulacao.NUM_LONGITUDE
        self.num_niveis_atm = num_niveis_atm or ConfiguracaoSimulacao.NUM_NIVEIS_ATMOSFERA
        self.num_niveis_ocean = num_niveis_ocean or ConfiguracaoSimulacao.NUM_NIVEIS_OCEANO
        
        # Criar coordenadas horizontais
        self._criar_coordenadas_horizontais()
        
        # Criar coordenadas verticais
        self._criar_coordenadas_verticais()
        
        # Calcular métricas
        self._calcular_metricas()
    
    def _criar_coordenadas_horizontais(self):
        """Cria coordenadas de latitude e longitude"""
        # Latitudes de -90 a 90 graus
        self.latitudes = np.linspace(-90, 90, self.num_lat)
        self.latitudes_rad = np.deg2rad(self.latitudes)
        
        # Longitudes de 0 a 360 graus
        self.longitudes = np.linspace(0, 360, self.num_lon, endpoint=False)
        self.longitudes_rad = np.deg2rad(self.longitudes)
        
        # Criar malha 2D
        self.lon_2d, self.lat_2d = np.meshgrid(self.longitudes, self.latitudes)
        self.lon_rad_2d, self.lat_rad_2d = np.meshgrid(self.longitudes_rad, self.latitudes_rad)
        
        # Calcular espaçamentos
        self.dlat = np.abs(self.latitudes[1] - self.latitudes[0])
        self.dlon = np.abs(self.longitudes[1] - self.longitudes[0])
        self.dlat_rad = np.deg2rad(self.dlat)
        self.dlon_rad = np.deg2rad(self.dlon)
    
    def _criar_coordenadas_verticais(self):
        """Cria coordenadas verticais (pressão para atmosfera, profundidade para oceano)"""
        # Atmosfera: níveis de pressão (Pa)
        # Distribuição log-linear da superfície ao topo
        p_superficie = 101325.0  # Pa
        p_topo = 100.0  # Pa
        self.niveis_pressao = np.logspace(
            np.log10(p_topo),
            np.log10(p_superficie),
            self.num_niveis_atm
        )[::-1]  # Inverter para ter topo primeiro
        
        # Oceano: profundidades (m)
        # Distribuição logarítmica para maior resolução perto da superfície
        prof_max = 5000.0  # m
        niveis_norm = np.linspace(0, 1, self.num_niveis_ocean)
        self.niveis_profundidade = prof_max * (niveis_norm ** 1.5)
    
    def _calcular_metricas(self):
        """Calcula métricas da grade (fatores de escala, áreas, volumes)"""
        R = ConstantesFisicas.RAIO_TERRA
        
        # Fatores de escala em coordenadas esféricas
        # dx = R cos(φ) dλ
        # dy = R dφ
        self.dx = R * np.cos(self.lat_rad_2d) * self.dlon_rad
        self.dy = R * self.dlat_rad * np.ones_like(self.lat_rad_2d)
        
        # Área de cada célula
        self.area_celula = self.dx * self.dy
        
        # Área total da Terra (para verificação)
        self.area_total = np.sum(self.area_celula)
        area_terra_teorica = 4 * np.pi * R**2
        
        # Pesos para média global (cosseno de latitude)
        self.pesos_lat = np.cos(self.lat_rad_2d)
        self.pesos_lat /= np.sum(self.pesos_lat)  # normalizar
    
    def calcular_gradiente_x(self, campo: np.ndarray) -> np.ndarray:
        """
        Calcula gradiente em direção x (zonal/longitude).
        
        ∂f/∂x = (1/(R cos φ)) ∂f/∂λ
        
        Args:
            campo: Campo 2D [lat, lon]
        
        Returns:
            Gradiente em x
        """
        # Diferenças finitas centrais com condições periódicas em longitude
        grad_lambda = np.gradient(campo, self.dlon_rad, axis=1)
        R = ConstantesFisicas.RAIO_TERRA
        grad_x = grad_lambda / (R * np.cos(self.lat_rad_2d))
        
        return grad_x
    
    def calcular_gradiente_y(self, campo: np.ndarray) -> np.ndarray:
        """
        Calcula gradiente em direção y (meridional/latitude).
        
        ∂f/∂y = (1/R) ∂f/∂φ
        
        Args:
            campo: Campo 2D [lat, lon]
        
        Returns:
            Gradiente em y
        """
        R = ConstantesFisicas.RAIO_TERRA
        grad_phi = np.gradient(campo, self.dlat_rad, axis=0)
        grad_y = grad_phi / R
        
        return grad_y
    
    def calcular_divergencia(
        self,
        u: np.ndarray,
        v: np.ndarray
    ) -> np.ndarray:
        """
        Calcula divergência de campo vetorial horizontal.
        
        ∇·V = (1/(R cos φ)) ∂u/∂λ + (1/R) ∂v/∂φ - (v tan φ)/R
        
        Args:
            u: Componente zonal do vento (m/s)
            v: Componente meridional do vento (m/s)
        
        Returns:
            Divergência (1/s)
        """
        R = ConstantesFisicas.RAIO_TERRA
        
        # Termos da divergência
        du_dlambda = np.gradient(u, self.dlon_rad, axis=1)
        dv_dphi = np.gradient(v, self.dlat_rad, axis=0)
        
        termo1 = du_dlambda / (R * np.cos(self.lat_rad_2d))
        termo2 = dv_dphi / R
        termo3 = -(v * np.tan(self.lat_rad_2d)) / R
        
        divergencia = termo1 + termo2 + termo3
        
        return divergencia
    
    def calcular_vorticidade(
        self,
        u: np.ndarray,
        v: np.ndarray
    ) -> np.ndarray:
        """
        Calcula vorticidade relativa vertical.
        
        ζ = (1/(R cos φ)) ∂v/∂λ - (1/R) ∂u/∂φ + (u tan φ)/R
        
        Args:
            u: Componente zonal
            v: Componente meridional
        
        Returns:
            Vorticidade (1/s)
        """
        R = ConstantesFisicas.RAIO_TERRA
        
        dv_dlambda = np.gradient(v, self.dlon_rad, axis=1)
        du_dphi = np.gradient(u, self.dlat_rad, axis=0)
        
        termo1 = dv_dlambda / (R * np.cos(self.lat_rad_2d))
        termo2 = -du_dphi / R
        termo3 = (u * np.tan(self.lat_rad_2d)) / R
        
        vorticidade = termo1 + termo2 + termo3
        
        return vorticidade
    
    def calcular_laplaciano(self, campo: np.ndarray) -> np.ndarray:
        """
        Calcula Laplaciano em coordenadas esféricas (simplificado).
        
        Args:
            campo: Campo 2D
        
        Returns:
            Laplaciano
        """
        grad_x = self.calcular_gradiente_x(campo)
        grad_y = self.calcular_gradiente_y(campo)
        
        lap_x = self.calcular_gradiente_x(grad_x)
        lap_y = self.calcular_gradiente_y(grad_y)
        
        return lap_x + lap_y
    
    def integrar_global(self, campo: np.ndarray) -> float:
        """
        Integra campo sobre toda a superfície terrestre.
        
        ∫∫ f dA = ∫∫ f R² cos φ dφ dλ
        
        Args:
            campo: Campo 2D
        
        Returns:
            Integral global
        """
        return np.sum(campo * self.area_celula)
    
    def media_global(self, campo: np.ndarray) -> float:
        """
        Calcula média global ponderada por área.
        
        Args:
            campo: Campo 2D
        
        Returns:
            Média global
        """
        return np.average(campo, weights=self.pesos_lat)
    
    def media_zonal(self, campo: np.ndarray) -> np.ndarray:
        """
        Calcula média zonal (ao longo de longitudes).
        
        Args:
            campo: Campo 2D [lat, lon]
        
        Returns:
            Perfil zonal [lat]
        """
        return np.mean(campo, axis=1)
    
    def interpolar_para_pressao(
        self,
        campo_3d: np.ndarray,
        pressao_alvo: float
    ) -> np.ndarray:
        """
        Interpolação vertical para um nível de pressão específico.
        
        Args:
            campo_3d: Campo 3D [nivel, lat, lon]
            pressao_alvo: Pressão alvo (Pa)
        
        Returns:
            Campo 2D interpolado [lat, lon]
        """
        # Encontrar níveis adjacentes
        idx = np.searchsorted(self.niveis_pressao, pressao_alvo)
        
        if idx == 0:
            return campo_3d[0, :, :]
        elif idx >= len(self.niveis_pressao):
            return campo_3d[-1, :, :]
        
        # Interpolação linear em log(p)
        p1 = self.niveis_pressao[idx-1]
        p2 = self.niveis_pressao[idx]
        
        peso = (np.log(pressao_alvo) - np.log(p1)) / (np.log(p2) - np.log(p1))
        
        campo_interp = (1 - peso) * campo_3d[idx-1, :, :] + peso * campo_3d[idx, :, :]
        
        return campo_interp
    
    def aplicar_condicao_contorno_periodica(self, campo: np.ndarray) -> np.ndarray:
        """
        Aplica condição de contorno periódica em longitude.
        
        Args:
            campo: Campo 2D
        
        Returns:
            Campo com condição de contorno
        """
        # Longitude é periódica (0° = 360°)
        campo_expandido = np.zeros((self.num_lat, self.num_lon + 2))
        campo_expandido[:, 1:-1] = campo
        campo_expandido[:, 0] = campo[:, -1]  # Borda oeste = borda leste
        campo_expandido[:, -1] = campo[:, 0]  # Borda leste = borda oeste
        
        return campo_expandido
    
    def criar_mascara_terra_oceano(
        self,
        fracao_terra: float = 0.29
    ) -> np.ndarray:
        """
        Cria máscara simples de terra/oceano.
        
        Args:
            fracao_terra: Fração de área terrestre
        
        Returns:
            Máscara booleana (True = terra, False = oceano)
        """
        # Distribuição simplificada: mais terra nas latitudes médias do hemisfério norte
        mascara = np.zeros_like(self.lat_2d, dtype=bool)
        
        # Terra predominante entre 20°N e 70°N
        idx_nh = (self.lat_2d > 20) & (self.lat_2d < 70)
        
        # Oceano predominante no hemisfério sul
        idx_sh = self.lat_2d < -20
        
        # Padrão aleatório mas reproduzível
        np.random.seed(42)
        ruido = np.random.rand(*self.lat_2d.shape)
        
        mascara[idx_nh] = ruido[idx_nh] < 0.5  # ~50% terra no HN
        mascara[idx_sh] = ruido[idx_sh] < 0.2  # ~20% terra no HS
        mascara[(self.lat_2d >= -20) & (self.lat_2d <= 20)] = ruido[(self.lat_2d >= -20) & (self.lat_2d <= 20)] < 0.35
        
        return mascara
    
    def __repr__(self) -> str:
        """Representação em string da grade"""
        return (
            f"GradeEspacial(\n"
            f"  Lat: {self.num_lat} pontos ({self.latitudes[0]:.1f}° a {self.latitudes[-1]:.1f}°)\n"
            f"  Lon: {self.num_lon} pontos ({self.longitudes[0]:.1f}° a {self.longitudes[-1]:.1f}°)\n"
            f"  Resolução: Δlat={self.dlat:.2f}°, Δlon={self.dlon:.2f}°\n"
            f"  Níveis atmosféricos: {self.num_niveis_atm}\n"
            f"  Níveis oceânicos: {self.num_niveis_ocean}\n"
            f"  Área total: {self.area_total/1e14:.2f} × 10¹⁴ m²\n"
            f")"
        )


if __name__ == "__main__":
    # Teste da grade
    print("=" * 70)
    print("SISTEMA DE GRADE ESPACIAL - Teste")
    print("=" * 70)
    
    grade = GradeEspacial()
    print(f"\n{grade}")
    
    # Teste de operadores
    print("\n1. Teste de campo e operadores:")
    # Criar campo de teste (temperatura sintética)
    temp_eq = 288 - 40 * np.sin(grade.lat_rad_2d)  # Gradiente equador-polo
    
    grad_x = grade.calcular_gradiente_x(temp_eq)
    grad_y = grade.calcular_gradiente_y(temp_eq)
    
    print(f"   Temperatura média global: {grade.media_global(temp_eq):.2f} K")
    print(f"   Gradiente Y (máx): {np.max(np.abs(grad_y)):.2e} K/m")
    
    # Teste de vento e vorticidade
    print("\n2. Teste de vento:")
    u = 10 * np.cos(grade.lat_rad_2d)  # Vento zonal
    v = np.zeros_like(u)  # Sem vento meridional
    
    div = grade.calcular_divergencia(u, v)
    vort = grade.calcular_vorticidade(u, v)
    
    print(f"   Divergência média: {np.mean(div):.2e} s⁻¹")
    print(f"   Vorticidade máxima: {np.max(np.abs(vort)):.2e} s⁻¹")
    
    # Teste de máscara
    print("\n3. Teste de máscara terra/oceano:")
    mascara = grade.criar_mascara_terra_oceano()
    fracao_terra_calculada = np.sum(mascara) / mascara.size
    print(f"   Fração de terra: {fracao_terra_calculada:.2%}")
    
    # Verificar conservação de área
    print("\n4. Verificação de área:")
    area_teorica = 4 * np.pi * (6.371e6)**2
    erro_area = abs(grade.area_total - area_teorica) / area_teorica
    print(f"   Área teórica: {area_teorica/1e14:.4f} × 10¹⁴ m²")
    print(f"   Área calculada: {grade.area_total/1e14:.4f} × 10¹⁴ m²")
    print(f"   Erro relativo: {erro_area:.2%}")
    
    print("\n" + "=" * 70)
