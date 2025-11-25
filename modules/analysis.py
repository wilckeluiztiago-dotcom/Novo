import numpy as np
from modules.constants import k_B, q

def extrair_parametros(tensoes_V, correntes_J, J_ph, J0, temperatura_celula, fator_idealidade):
    """
    Extrai J_sc, V_oc, P_max, FF e Eficiência a partir das curvas J-V.
    
    Parâmetros:
        tensoes_V : Array de tensões [V]
        correntes_J : Array de densidades de corrente [A/m^2]
        J_ph : Corrente fotogerada [A/m^2]
        J0 : Corrente de saturação [A/m^2]
        temperatura_celula : Temperatura da célula [K]
        fator_idealidade : Fator de idealidade do diodo
    
    Retorna:
        dicionário com parâmetros extraídos:
            - J_sc: Corrente de curto-circuito
            - V_oc_ideal: Tensão de circuito aberto (ideal)
            - V_oc_numerico: Tensão de circuito aberto (numérica)
            - P_max: Potência máxima
            - FF: Fator de preenchimento
            - Eficiencia: Eficiência de conversão
            - V_mp: Tensão no ponto de máxima potência
            - J_mp: Corrente no ponto de máxima potência
            - Potencias: Array de potências
    """
    # Corrente de curto-circuito (aproximação no ponto V=0)
    J_sc = correntes_J[0]

    # Tensão de circuito aberto (aproximação via diodo ideal)
    T = temperatura_celula
    n = fator_idealidade

    V_oc_ideal = (n * k_B * T / q) * np.log(J_ph / J0 + 1.0)
    # Aproximação numérica usando o ponto onde J≈0
    indice_voc = np.argmin(np.abs(correntes_J))
    V_oc_numerico = tensoes_V[indice_voc]

    # Potência P(V) = V * J(V)
    potencias = tensoes_V * correntes_J  # [W/m^2]
    indice_pmax = np.argmax(potencias)
    V_mp = tensoes_V[indice_pmax]
    J_mp = correntes_J[indice_pmax]
    P_max = potencias[indice_pmax]       # [W/m^2]

    # Fator de preenchimento (Fill Factor)
    FF = (V_mp * J_mp) / (V_oc_numerico * J_sc + 1e-30)

    # Eficiência em relação à irradiância padrão (1000 W/m^2)
    IRRADIANCIA_PADRAO = 1000.0
    eficiencia = P_max / IRRADIANCIA_PADRAO

    return {
        "J_sc": J_sc,
        "V_oc_ideal": V_oc_ideal,
        "V_oc_numerico": V_oc_numerico,
        "P_max": P_max,
        "FF": FF,
        "Eficiencia": eficiencia,
        "V_mp": V_mp,
        "J_mp": J_mp,
        "Potencias": potencias
    }
