import numpy as np
from modules.constants import k_B, q
from modules.solar import fluxo_fotons_corpo_negro

def calcular_corrente_saturacao_radiativa(energia_gap_eV: float,
                                          temperatura_celula: float = 300.0,
                                          num_pontos_energia: int = 4000) -> float:
    """
    Calcula J0 radiativa aproximada usando um modelo de corpo negro
    para a célula à temperatura T_célula (sem viés, V = 0).

    J0 ≈ q * ∫_{Eg}^{∞} Φ_emit(E, T_célula) dE

    Aqui, Φ_emit é o fluxo de fótons emitidos por unidade de área da célula
    ideal (modelo de emissor de corpo negro).
    
    Parâmetros:
        energia_gap_eV : Energia de gap [eV]
        temperatura_celula : Temperatura da célula [K]
        num_pontos_energia : Número de pontos para integração numérica
    
    Retorna:
        J0 : Corrente de saturação radiativa [A/m^2]
    """
    Eg_J = energia_gap_eV * q
    # Limite superior de energia
    E_max_J = 4.0 * q

    energia_J = np.linspace(Eg_J, E_max_J, num_pontos_energia)

    # Fluxo espectral emitido pela célula (corpo negro)
    fluxo_emitido = fluxo_fotons_corpo_negro(energia_J, temperatura_celula)

    # Integração e conversão em corrente
    fluxo_total_emitido = np.trapz(fluxo_emitido, energia_J)  # [fótons / (m^2·s)]
    J0 = q * fluxo_total_emitido      # [A/m^2]
    return J0


def curva_JV_diodo(J_ph: float,
                    J0: float,
                    temperatura_celula: float = 300.0,
                    fator_idealidade: float = 1.0,
                    resistencia_serie: float = 0.0,
                    resistencia_shunt: float = np.inf,
                    tensao_min: float = 0.0,
                    tensao_max: float = 1.2,
                    num_pontos_tensao: int = 400) -> tuple:
    """
    Gera a curva J(V) para o diodo fotovoltaico:

      J(V) = J_ph
             - J0 * [ exp(q (V + J Rs) / (n k_B T)) - 1 ]
             - (V + J Rs) / Rsh

    Usa método de Newton para resolver J em função de V quando Rs e/ou Rsh
    são finitos.

    Parâmetros:
        J_ph : Corrente fotogerada [A/m^2]
        J0 : Corrente de saturação [A/m^2]
        temperatura_celula : Temperatura da célula [K]
        fator_idealidade : Fator de idealidade do diodo
        resistencia_serie : Resistência série [Ω·m^2]
        resistencia_shunt : Resistência shunt [Ω·m^2]
        tensao_min : Tensão mínima [V]
        tensao_max : Tensão máxima [V]
        num_pontos_tensao : Número de pontos de tensão

    Retorna:
        tensoes_V : array de tensões [V]
        correntes_J : array de densidades de corrente [A/m^2]
    """
    T = temperatura_celula
    n = fator_idealidade
    Rs = resistencia_serie
    Rsh = resistencia_shunt

    tensoes_V = np.linspace(tensao_min, tensao_max, num_pontos_tensao)
    correntes_J = np.zeros_like(tensoes_V)

    # Palpite inicial para o método de Newton (começa em J_ph)
    J_inicial = J_ph

    for i, V in enumerate(tensoes_V):
        J = J_inicial  # palpite

        for _ in range(50):
            # Função f(J) = 0
            expoente = q * (V + J * Rs) / (n * k_B * T)
            # Limitar expoente para evitar overflow numérico
            expoente = np.clip(expoente, -100, 100)

            termo_exp = np.exp(expoente)
            if np.isinf(Rsh):
                termo_shunt = 0.0
                derivada_termo_shunt_dJ = 0.0
            else:
                termo_shunt = (V + J * Rs) / Rsh
                derivada_termo_shunt_dJ = Rs / Rsh

            f_J = (J_ph
                   - J0 * (termo_exp - 1.0)
                   - termo_shunt
                   - J)

            # Derivada df/dJ
            dfdJ = (-J0 * termo_exp * (q * Rs / (n * k_B * T))
                    - derivada_termo_shunt_dJ
                    - 1.0)

            # Atualização de Newton
            if abs(dfdJ) < 1e-20:
                break

            J_novo = J - f_J / dfdJ

            # Critério de convergência
            if abs(J_novo - J) < 1e-10:
                J = J_novo
                break

            J = J_novo

        correntes_J[i] = J
        # Usar o valor atual como palpite para o próximo V
        J_inicial = J

    return tensoes_V, correntes_J
