"""
MÓDULO UTILITÁRIO - Sistema de Modelagem Climática
===================================================

Funções auxiliares para conversão de unidades, estatísticas,
I/O de dados e ferramentas gerais.

Autor: Luiz Tiago Wilcke
Data: 2025-11-24
"""

import numpy as np
from typing import Union, Tuple, List, Any
import pickle
import json
from pathlib import Path


# =============================================================================
# CONVERSÕES DE UNIDADES
# =============================================================================

def celsius_para_kelvin(temp_celsius: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Converte temperatura de Celsius para Kelvin"""
    return temp_celsius + 273.15


def kelvin_para_celsius(temp_kelvin: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Converte temperatura de Kelvin para Celsius"""
    return temp_kelvin - 273.15


def pressao_pa_para_hpa(pressao_pa: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Converte pressão de Pascal para hectoPascal (milibar)"""
    return pressao_pa / 100.0


def pressao_hpa_para_pa(pressao_hpa: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Converte pressão de hectoPascal para Pascal"""
    return pressao_hpa * 100.0


def umidade_especifica_para_relativa(
    q: Union[float, np.ndarray],
    temperatura: Union[float, np.ndarray],
    pressao: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Converte umidade específica para umidade relativa.
    
    Args:
        q: Umidade específica (kg/kg)
        temperatura: Temperatura (K)
        pressao: Pressão (Pa)
    
    Returns:
        Umidade relativa (0-1)
    """
    from config import ParametrosAtmosfera
    
    # Pressão de vapor saturado (Clausius-Clapeyron)
    e_sat = calcular_pressao_vapor_saturado(temperatura)
    
    # Pressão de vapor atual
    epsilon = ParametrosAtmosfera.MASSA_MOLAR['H2O'] / ParametrosAtmosfera.MASSA_MOLAR['ar_seco']
    e = (q * pressao) / (epsilon + q)
    
    # Umidade relativa
    ur = e / e_sat
    return np.clip(ur, 0, 1)


def calcular_pressao_vapor_saturado(temperatura: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calcula pressão de vapor saturado usando equação de Clausius-Clapeyron.
    
    e_s = e_0 * exp[(L/R_v)(1/T_0 - 1/T)]
    
    Args:
        temperatura: Temperatura (K)
    
    Returns:
        Pressão de vapor saturado (Pa)
    """
    from config import ParametrosAtmosfera
    
    L = ParametrosAtmosfera.CALOR_LATENTE_VAPORIZACAO
    R_v = ParametrosAtmosfera.R_VAPOR_AGUA
    e_0 = ParametrosAtmosfera.PRESSAO_VAPOR_REF
    T_0 = ParametrosAtmosfera.TEMPERATURA_REF
    
    e_sat = e_0 * np.exp((L / R_v) * (1 / T_0 - 1 / temperatura))
    return e_sat


# =============================================================================
# FUNÇÕES ESTATÍSTICAS
# =============================================================================

def media_ponderada(
    dados: np.ndarray,
    pesos: np.ndarray,
    eixo: int = None
) -> Union[float, np.ndarray]:
    """
    Calcula média ponderada.
    
    Args:
        dados: Array de dados
        pesos: Array de pesos
        eixo: Eixo para calcular média
    
    Returns:
        Média ponderada
    """
    return np.average(dados, weights=pesos, axis=eixo)


def media_zonal(campo: np.ndarray) -> np.ndarray:
    """
    Calcula média zonal (média sobre longitudes).
    
    Args:
        campo: Campo 2D [lat, lon]
    
    Returns:
        Perfil zonal [lat]
    """
    return np.mean(campo, axis=-1)


def media_meridional(campo: np.ndarray) -> np.ndarray:
    """
    Calcula média meridional (média sobre latitudes).
    
    Args:
        campo: Campo 2D [lat, lon]
    
    Returns:
        Perfil meridional [lon]
    """
    return np.mean(campo, axis=0)


def media_global_ponderada(
    campo: np.ndarray,
    latitudes: np.ndarray
) -> float:
    """
    Calcula média global ponderada por área (cosseno de latitude).
    
    Args:
        campo: Campo 2D [lat, lon]
        latitudes: Array de latitudes em graus
    
    Returns:
        Média global
    """
    lat_rad = np.deg2rad(latitudes)
    pesos = np.cos(lat_rad)
    pesos_2d = pesos[:, np.newaxis] * np.ones((len(latitudes), campo.shape[1]))
    
    return np.average(campo, weights=pesos_2d)


def anomalia(
    dados: np.ndarray,
    referencia: np.ndarray = None,
    eixo: int = 0
) -> np.ndarray:
    """
    Calcula anomalia em relação à média de referência.
    
    Args:
        dados: Dados para calcular anomalia
        referencia: Período de referência (se None, usa média total)
        eixo: Eixo temporal
    
    Returns:
        Anomalias
    """
    if referencia is None:
        media_ref = np.mean(dados, axis=eixo, keepdims=True)
    else:
        media_ref = np.mean(referencia, axis=eixo, keepdims=True)
    
    return dados - media_ref


def tendencia_linear(tempo: np.ndarray, dados: np.ndarray) -> Tuple[float, float]:
    """
    Calcula tendência linear por mínimos quadrados.
    
    Args:
        tempo: Array de tempo
        dados: Série temporal
    
    Returns:
        Tuple (inclinação, intercept)
    """
    coef = np.polyfit(tempo, dados, deg=1)
    return coef[0], coef[1]


# =============================================================================
# OPERADORES ESPACIAIS
# =============================================================================

def gradiente_latitude(
    campo: np.ndarray,
    latitudes: np.ndarray
) -> np.ndarray:
    """
    Calcula gradiente em direção à latitude.
    
    ∂f/∂φ ≈ [f(φ+Δφ) - f(φ-Δφ)] / (2Δφ)
    
    Args:
        campo: Campo 2D [lat, lon]
        latitudes: Array de latitudes em graus
    
    Returns:
        Gradiente latitudinal
    """
    from config import ConstantesFisicas
    
    lat_rad = np.deg2rad(latitudes)
    dlat = np.gradient(lat_rad)
    a = ConstantesFisicas.RAIO_TERRA
    
    # ∂f/∂y = (1/a) ∂f/∂φ
    grad_lat = np.gradient(campo, axis=0) / (a * dlat[:, np.newaxis])
    
    return grad_lat


def gradiente_longitude(
    campo: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray
) -> np.ndarray:
    """
    Calcula gradiente em direção à longitude.
    
    ∂f/∂λ ≈ [f(λ+Δλ) - f(λ-Δλ)] / (2Δλ)
    
    Args:
        campo: Campo 2D [lat, lon]
        latitudes: Array de latitudes em graus
        longitudes: Array de longitudes em graus
    
    Returns:
        Gradiente longitudinal
    """
    from config import ConstantesFisicas
    
    lat_rad = np.deg2rad(latitudes)
    lon_rad = np.deg2rad(longitudes)
    dlon = np.gradient(lon_rad)
    a = ConstantesFisicas.RAIO_TERRA
    
    # ∂f/∂x = (1/(a cos φ)) ∂f/∂λ
    grad_lon = np.gradient(campo, axis=1) / (a * np.cos(lat_rad)[:, np.newaxis] * dlon)
    
    return grad_lon


def laplaciano(
    campo: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray
) -> np.ndarray:
    """
    Calcula Laplaciano em coordenadas esféricas (simplificado).
    
    ∇²f ≈ ∂²f/∂φ² + ∂²f/∂λ²
    
    Args:
        campo: Campo 2D [lat, lon]
        latitudes: Array de latitudes
        longitudes: Array de longitudes
    
    Returns:
        Laplaciano do campo
    """
    grad_lat = gradiente_latitude(campo, latitudes)
    grad2_lat = gradiente_latitude(grad_lat, latitudes)
    
    grad_lon = gradiente_longitude(campo, latitudes, longitudes)
    grad2_lon = gradiente_longitude(grad_lon, latitudes, longitudes)
    
    return grad2_lat + grad2_lon


# =============================================================================
# INTEGRAÇÃO E INTERPOLAÇÃO
# =============================================================================

def integrar_vertical(
    campo: np.ndarray,
    niveis: np.ndarray,
    eixo: int = 0
) -> np.ndarray:
    """
    Integra campo verticalmente (regra do trapézio).
    
    Args:
        campo: Campo 3D com dimensão vertical
        niveis: Níveis verticais (pressão ou profundidade)
        eixo: Eixo vertical
    
    Returns:
        Campo integrado verticalmente
    """
    return np.trapz(campo, niveis, axis=eixo)


def interpolar_linear(
    x: np.ndarray,
    y: np.ndarray,
    x_novo: np.ndarray
) -> np.ndarray:
    """
    Interpolação linear 1D.
    
    Args:
        x: Coordenadas originais
        y: Valores originais
        x_novo: Novas coordenadas
    
    Returns:
        Valores interpolados
    """
    return np.interp(x_novo, x, y)


# =============================================================================
# I/O DE DADOS
# =============================================================================

def salvar_array_numpy(
    arquivo: str,
    dados: np.ndarray,
    compress: bool = True
) -> None:
    """
    Salva array NumPy em arquivo.
    
    Args:
        arquivo: Caminho do arquivo
        dados: Array para salvar
        compress: Se True, usa compressão
    """
    arquivo_path = Path(arquivo)
    arquivo_path.parent.mkdir(parents=True, exist_ok=True)
    
    if compress:
        np.savez_compressed(arquivo, data=dados)
    else:
        np.save(arquivo, dados)


def carregar_array_numpy(arquivo: str) -> np.ndarray:
    """
    Carrega array NumPy de arquivo.
    
    Args:
        arquivo: Caminho do arquivo
    
    Returns:
        Array carregado
    """
    if arquivo.endswith('.npz'):
        return np.load(arquivo)['data']
    else:
        return np.load(arquivo)


def salvar_dicionario(
    arquivo: str,
    dados: dict,
    formato: str = 'pickle'
) -> None:
    """
    Salva dicionário em arquivo.
    
    Args:
        arquivo: Caminho do arquivo
        dados: Dicionário para salvar
        formato: 'pickle' ou 'json'
    """
    arquivo_path = Path(arquivo)
    arquivo_path.parent.mkdir(parents=True, exist_ok=True)
    
    if formato == 'pickle':
        with open(arquivo, 'wb') as f:
            pickle.dump(dados, f)
    elif formato == 'json':
        with open(arquivo, 'w') as f:
            json.dump(dados, f, indent=2)
    else:
        raise ValueError(f"Formato desconhecido: {formato}")


def carregar_dicionario(arquivo: str, formato: str = 'pickle') -> dict:
    """
    Carrega dicionário de arquivo.
    
    Args:
        arquivo: Caminho do arquivo
        formato: 'pickle' ou 'json'
    
    Returns:
        Dicionário carregado
    """
    if formato == 'pickle':
        with open(arquivo, 'rb') as f:
            return pickle.load(f)
    elif formato == 'json':
        with open(arquivo, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"Formato desconhecido: {formato}")


# =============================================================================
# FERRAMENTAS DE LOGGING
# =============================================================================

def criar_barra_progresso(
    iteracao: int,
    total: int,
    largura: int = 50,
    prefixo: str = "Progresso"
) -> str:
    """
    Cria barra de progresso em modo texto.
    
    Args:
        iteracao: Iteração atual
        total: Total de iterações
        largura: Largura da barra
        prefixo: Texto antes da barra
    
    Returns:
        String formatada da barra
    """
    percentual = 100 * (iteracao / float(total))
    preenchido = int(largura * iteracao // total)
    barra = '█' * preenchido + '-' * (largura - preenchido)
    
    return f'\r{prefixo} |{barra}| {percentual:.1f}% Completo'


def formatar_tempo_segundos(segundos: float) -> str:
    """
    Formata tempo em segundos para string legível.
    
    Args:
        segundos: Tempo em segundos
    
    Returns:
        String formatada (ex: "2h 15m 30s")
    """
    horas = int(segundos // 3600)
    minutos = int((segundos % 3600) // 60)
    segs = int(segundos % 60)
    
    partes = []
    if horas > 0:
        partes.append(f"{horas}h")
    if minutos > 0:
        partes.append(f"{minutos}m")
    partes.append(f"{segs}s")
    
    return " ".join(partes)


# =============================================================================
# FERRAMENTAS MATEMÁTICAS
# =============================================================================

def normalizar(dados: np.ndarray, minimo: float = 0.0, maximo: float = 1.0) -> np.ndarray:
    """
    Normaliza dados para intervalo [minimo, maximo].
    
    Args:
        dados: Dados para normalizar
        minimo: Valor mínimo do intervalo
        maximo: Valor máximo do intervalo
    
    Returns:
        Dados normalizados
    """
    d_min = np.min(dados)
    d_max = np.max(dados)
    
    if d_max == d_min:
        return np.full_like(dados, (minimo + maximo) / 2)
    
    normalizado = (dados - d_min) / (d_max - d_min)
    return normalizado * (maximo - minimo) + minimo


def suavizar_media_movel(
    dados: np.ndarray,
    janela: int,
    eixo: int = 0
) -> np.ndarray:
    """
    Aplica média móvel para suavização.
    
    Args:
        dados: Dados para suavizar
        janela: Tamanho da janela
        eixo: Eixo para aplicar suavização
    
    Returns:
        Dados suavizados
    """
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(dados, size=janela, axis=eixo, mode='nearest')


def calcular_fft(
    serie_temporal: np.ndarray,
    intervalo_tempo: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula FFT (Transformada de Fourier) de série temporal.
    
    Args:
        serie_temporal: Série temporal
        intervalo_tempo: Intervalo entre pontos (s)
    
    Returns:
        Tuple (frequências, espectro de potência)
    """
    n = len(serie_temporal)
    fft_valores = np.fft.fft(serie_temporal)
    frequencias = np.fft.fftfreq(n, d=intervalo_tempo)
    
    # Apenas frequências positivas
    idx_positivo = frequencias > 0
    freq_pos = frequencias[idx_positivo]
    potencia = np.abs(fft_valores[idx_positivo])**2
    
    return freq_pos, potencia


if __name__ == "__main__":
    # Testes das funções
    print("=" * 70)
    print("MÓDULO UTILITÁRIO - Testes")
    print("=" * 70)
    
    # Teste conversões
    print("\n1. Conversões de Temperatura:")
    temp_c = 25.0
    temp_k = celsius_para_kelvin(temp_c)
    print(f"   {temp_c}°C = {temp_k} K")
    print(f"   {temp_k} K = {kelvin_para_celsius(temp_k)}°C")
    
    # Teste pressão de vapor
    print("\n2. Pressão de Vapor Saturado:")
    T_test = np.array([273.15, 283.15, 293.15])  #  0°C, 10°C, 20°C
    e_sat = calcular_pressao_vapor_saturado(T_test)
    print(f"   T = {kelvin_para_celsius(T_test)} °C")
    print(f"   e_sat = {e_sat} Pa = {pressao_pa_para_hpa(e_sat)} hPa")
    
    # Teste médias
    print("\n3. Estatísticas:")
    latitudes = np.linspace(-90, 90, 91)
    longitudes = np.linspace(0, 360, 181)
    campo_teste = np.random.rand(len(latitudes), len(longitudes))
    
    media_glob = media_global_ponderada(campo_teste, latitudes)
    print(f"   Média global ponderada: {media_glob:.4f}")
    
    # Teste gradientes
    print("\n4. Operadores Espaciais:")
    grad_lat = gradiente_latitude(campo_teste, latitudes)
    grad_lon = gradiente_longitude(campo_teste, latitudes, longitudes)
    print(f"   Shape gradiente lat: {grad_lat.shape}")
    print(f"   Shape gradiente lon: {grad_lon.shape}")
    
    print("\n" + "=" * 70)
