import numpy as np
from math import pi
from modules.constants import h, c, k_B, q, epsilon_0

# Parâmetros do Sol–Terra para corpo negro (modelo simplificado)
RAIO_SOL = 6.9634e8          # m
DISTANCIA_SOL_TERRA = 1.496e11  # m
FATOR_GEOMETRICO_SOL_TERRA = (RAIO_SOL / DISTANCIA_SOL_TERRA) ** 2

def fluxo_fotons_corpo_negro(energia_J: np.ndarray, temperatura: float) -> np.ndarray:
    """
    Fluxo espectral de fótons (por unidade de energia) de um corpo negro
    ideal (por unidade de área da superfície do emissor),
    integrando sobre ângulo sólido (2π estérico).

    Φ(E) = 2π / (h^3 c^2) * E^2 / (exp(E / (k_B T)) - 1)
    [fótons / (m^2·s·J)]

    Parâmetros:
        energia_J : array de energias [J]
        temperatura : [K]
    
    Retorna:
        fluxo : array de fluxo de fótons
    """
    expoente = np.exp(energia_J / (k_B * temperatura)) - 1.0
    # Evitar overflow numérico
    expoente[expoente == 0] = np.inf

    fluxo = (2.0 * pi / (h ** 3 * c ** 2)) * (energia_J ** 2) / expoente
    return fluxo


def calcular_corrente_fotogerada_limite(energia_gap_eV: float,
                                        temperatura_sol: float = 5778.0,
                                        num_pontos_energia: int = 4000) -> float:
    """
    Calcula a corrente fotogerada J_ph (limite de Shockley–Queisser)
    para uma célula ideal com gap Eg, usando um Sol como corpo negro.

    J_ph = q * ∫_{Eg}^{E_max} Φ_inc(E) dE
    onde Φ_inc(E) = Φ_superfície(E) * fator_geométrico_sol_terra

    Parâmetros:
        energia_gap_eV : Energia de gap [eV]
        temperatura_sol : Temperatura do Sol [K]
        num_pontos_energia : Número de pontos para integração numérica
    
    Retorna:
        J_ph : Corrente fotogerada [A/m^2]
    """
    Eg_J = energia_gap_eV * q
    # Limite superior de energia (4 eV ~ ultravioleta)
    E_max_J = 4.0 * q

    # Malha de energia
    energia_J = np.linspace(Eg_J, E_max_J, num_pontos_energia)

    # Fluxo espectral na superfície do Sol
    fluxo_sol = fluxo_fotons_corpo_negro(energia_J, temperatura_sol)

    # Fluxo espectral na Terra (reduzido pelo fator geométrico)
    fluxo_terra = fluxo_sol * FATOR_GEOMETRICO_SOL_TERRA  # [fótons / (m^2·s·J)]

    # Integração numérica
    fluxo_total_fotons = np.trapz(fluxo_terra, energia_J)  # [fótons / (m^2·s)]

    # Corrente fotogerada
    J_ph = q * fluxo_total_fotons  # [A/m^2]
    return J_ph
