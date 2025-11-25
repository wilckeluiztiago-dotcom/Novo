import numpy as np
from modules.constants import k_B, q, m_0

class Material:
    def __init__(self, name, Eg_0, alpha, beta, mn, mp, Nc_300, Nv_300):
        self.name = name
        self.Eg_0 = Eg_0      # Band gap a 0K (eV)
        self.alpha = alpha    # Coeficiente de temperatura (eV/K)
        self.beta = beta      # Coeficiente de temperatura (K)
        self.mn = mn          # Massa efetiva elétron (relativa)
        self.mp = mp          # Massa efetiva buraco (relativa)
        self.Nc_300 = Nc_300  # Densidade efetiva de estados na banda de condução a 300K (cm^-3)
        self.Nv_300 = Nv_300  # Densidade efetiva de estados na banda de valência a 300K (cm^-3)

# Materiais comuns
SILICON = Material("Silício (Si)", 1.17, 4.73e-4, 636, 1.08, 0.56, 2.8e19, 1.04e19)
GAAS = Material("Arsenieto de Gálio (GaAs)", 1.519, 5.405e-4, 204, 0.067, 0.45, 4.7e17, 7.0e18)
PEROVSKITE = Material("Perovskita (MAPI)", 1.6, 4e-4, 300, 0.2, 0.2, 1e19, 1e19) # Valores aproximados

def calculate_band_gap(material, T):
    """
    Calcula o Band Gap em função da temperatura usando a equação de Varshni:
    Eg(T) = Eg(0) - (alpha * T^2) / (T + beta)
    """
    Eg = material.Eg_0 - (material.alpha * T**2) / (T + material.beta)
    return Eg

def calculate_intrinsic_carrier_concentration(material, T, Eg):
    """
    Calcula a concentração intrínseca de portadores (ni).
    ni = sqrt(Nc * Nv) * exp(-Eg / 2kT)
    """
    # Ajustar Nc e Nv com a temperatura (proporcional a T^(3/2))
    Nc = material.Nc_300 * (T / 300.0)**1.5
    Nv = material.Nv_300 * (T / 300.0)**1.5
    
    # Energia térmica
    kT = k_B * T / q # em eV
    
    ni = np.sqrt(Nc * Nv) * np.exp(-Eg / (2 * kT))
    return ni, Nc, Nv
