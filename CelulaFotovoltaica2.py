# ============================================================
# MODELO MATEMÁTICO COMPLETO (DIDÁTICO) — CÉLULA FOTOVOLTAICA DE SILÍCIO
# Autor: Luiz Tiago Wilcke (LT)
#
# - Nível quântico do material:
#     • Estrutura de bandas (via massa efetiva)
#     • Densidade de estados Nc, Nv
#     • Concentração intrínseca ni
#
# - Interação luz–matéria:
#     • Espectro solar aproximado por corpo negro (T_sol ~ 5778 K)
#     • Fator geométrico Sol–Terra
#     • Corrente fotogerada J_ph (limite de Shockley–Queisser)
#
# - Nível de dispositivo / circuito:
#     • Corrente de saturação J0 (recombinação radiativa)
#     • Equação de diodo com fator de idealidade n
#     • Resistência série Rs e shunt Rsh
#     • Curvas J–V, P–V, J_sc, V_oc, FF, eficiência
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from math import pi, sqrt, exp

# ============================================================
# 1. Constantes físicas fundamentais
# ============================================================

# Constantes SI
CARGA_ELEMENTAR = 1.602176634e-19      # Coulomb
CONSTANTE_BOLTZMANN = 1.380649e-23     # J/K
CONSTANTE_PLANCK = 6.62607015e-34      # J·s
VELOCIDADE_LUZ = 2.99792458e8          # m/s
PERMISSIVIDADE_VACUO = 8.8541878128e-12  # F/m
MASSA_ELETRON_LIVRE = 9.10938356e-31   # kg

# Parâmetros do Sol–Terra para corpo negro (modelo simplificado)
RAIO_SOL = 6.9634e8          # m
DISTANCIA_SOL_TERRA = 1.496e11  # m
FATOR_GEOMETRICO_SOL_TER = (RAIO_SOL / DISTANCIA_SOL_TERRA) ** 2

# Irradiância padrão aproximada (AM1.5) para cálculo da eficiência
IRRADIANCIA_PADRAO = 1000.0  # W/m^2

# ============================================================
# 2. Parâmetros do silício e funções quânticas
# ============================================================

class ParametrosSilicio:
    """
    Parâmetros físicos e quânticos do silício cristalino.
    """

    def __init__(self,
                 temperatura_celula: float = 300.0,     # K
                 energia_gap_eV: float = 1.12,          # eV
                 massa_efetiva_eletron_rel: float = 1.08,  # m*_n / m0 (aprox. efetiva)
                 massa_efetiva_lacuna_rel: float = 0.6):   # m*_p / m0 (aprox.)

        self.temperatura_celula = temperatura_celula
        self.energia_gap_eV = energia_gap_eV
        self.energia_gap_J = energia_gap_eV * CARGA_ELEMENTAR

        # Massas efetivas absolutas
        self.massa_efetiva_eletron = massa_efetiva_eletron_rel * MASSA_ELETRON_LIVRE
        self.massa_efetiva_lacuna = massa_efetiva_lacuna_rel * MASSA_ELETRON_LIVRE

        # Densidades de estados efetivas (Nc, Nv)
        self.Nc = self._calcular_densidade_de_estados_conducao()
        self.Nv = self._calcular_densidade_de_estados_valencia()

        # Concentração intrínseca ni
        self.ni = self._calcular_concentracao_intrinseca()

    def _calcular_densidade_de_estados_conducao(self) -> float:
        """
        Nc = 2 * (2π m*_n k_B T / h^2)^(3/2)
        """
        T = self.temperatura_celula
        m_star = self.massa_efetiva_eletron
        h = CONSTANTE_PLANCK
        kB = CONSTANTE_BOLTZMANN

        Nc = 2.0 * ((2.0 * pi * m_star * kB * T) / (h ** 2)) ** 1.5
        return Nc  # [m^-3]

    def _calcular_densidade_de_estados_valencia(self) -> float:
        """
        Nv = 2 * (2π m*_p k_B T / h^2)^(3/2)
        """
        T = self.temperatura_celula
        m_star = self.massa_efetiva_lacuna
        h = CONSTANTE_PLANCK
        kB = CONSTANTE_BOLTZMANN

        Nv = 2.0 * ((2.0 * pi * m_star * kB * T) / (h ** 2)) ** 1.5
        return Nv  # [m^-3]

    def _calcular_concentracao_intrinseca(self) -> float:
        """
        ni^2 = Nc * Nv * exp(-Eg / (k_B T))
        ni = sqrt(Nc * Nv) * exp(-Eg / (2 k_B T))
        """
        T = self.temperatura_celula
        Eg = self.energia_gap_J
        kB = CONSTANTE_BOLTZMANN

        ni = sqrt(self.Nc * self.Nv) * np.exp(-Eg / (2.0 * kB * T))
        return ni  # [m^-3]


# ============================================================
# 3. Espectro solar (corpo negro) e corrente fotogerada J_ph
# ============================================================

def fluxo_fotons_corpo_negro(E_J: np.ndarray,
                             temperatura: float) -> np.ndarray:
    """
    Fluxo espectral de fótons (por unidade de energia) de um corpo negro
    ideal (por unidade de área da superfície do emissor),
    integrando sobre ângulo sólido (2π estérico).

    Φ(E) = 2π / (h^3 c^2) * E^2 / (exp(E / (k_B T)) - 1)
    [fotons / (m^2·s·J)]

    E_J : array de energias [J]
    temperatura : [K]
    """
    h = CONSTANTE_PLANCK
    c = VELOCIDADE_LUZ
    kB = CONSTANTE_BOLTZMANN

    expoente = np.exp(E_J / (kB * temperatura)) - 1.0
    # Evitar overflow numérico
    expoente[expoente == 0] = np.inf

    fluxo = (2.0 * pi / (h ** 3 * c ** 2)) * (E_J ** 2) / expoente
    return fluxo


def calcular_corrente_fotogerada_limite(energia_gap_eV: float,
                                        temperatura_sol: float = 5778.0,
                                        num_pontos_energia: int = 4000) -> float:
    """
    Calcula a corrente fotogerada J_ph (limite de Shockley–Queisser)
    para uma célula ideal com gap Eg, usando um Sol como corpo negro.

    J_ph = q * ∫_{Eg}^{E_max} Φ_inc(E) dE
    onde Φ_inc(E) = Φ_superficie(E) * fator_geométrico_sol_terra

    Retorna J_ph em [A/m^2].
    """
    Eg_J = energia_gap_eV * CARGA_ELEMENTAR
    # Limite superior de energia (e.g. 4 eV ~ ultravioleta)
    E_max_J = 4.0 * CARGA_ELEMENTAR

    # Malha de energia
    E_J = np.linspace(Eg_J, E_max_J, num_pontos_energia)

    # Fluxo espectral na superfície do Sol
    fluxo_sol = fluxo_fotons_corpo_negro(E_J, temperatura_sol)

    # Fluxo espectral na Terra (reduzido pelo fator geométrico)
    fluxo_terra = fluxo_sol * FATOR_GEOMETRICO_SOL_TER  # [fotons / (m^2·s·J)]

    # Integração numérica
    fluxo_total_fotons = np.trapz(fluxo_terra, E_J)  # [fotons / (m^2·s)]

    # Corrente fotogerada
    J_ph = CARGA_ELEMENTAR * fluxo_total_fotons  # [A/m^2]
    return J_ph


# ============================================================
# 4. Corrente de saturação radiativa J0 (recombinação)
# ============================================================

def calcular_corrente_saturacao_radiativa(energia_gap_eV: float,
                                          temperatura_celula: float = 300.0,
                                          num_pontos_energia: int = 4000) -> float:
    """
    Calcula J0 radiativa aproximada usando um modelo de corpo negro
    para a célula à temperatura T_celula (sem viés, V = 0).

    J0 ≈ q * ∫_{Eg}^{∞} Φ_emit(E, T_celula) dE

    Aqui, Φ_emit é o fluxo de fótons emitidos por unidade de área da célula
    ideal (modelo de emissor de corpo negro).
    """
    Eg_J = energia_gap_eV * CARGA_ELEMENTAR
    # Limite superior de energia
    E_max_J = 4.0 * CARGA_ELEMENTAR

    E_J = np.linspace(Eg_J, E_max_J, num_pontos_energia)

    # Fluxo espectral emitido pela célula (corpo negro)
    fluxo_emit = fluxo_fotons_corpo_negro(E_J, temperatura_celula)

    # Integração e conversão em corrente
    fluxo_total_emit = np.trapz(fluxo_emit, E_J)  # [fotons / (m^2·s)]
    J0 = CARGA_ELEMENTAR * fluxo_total_emit      # [A/m^2]
    return J0


# ============================================================
# 5. Modelo de diodo equivalente (com Rs e Rsh)
# ============================================================

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

    Retorna:
        tensoes_V : array [V]
        correntes_J : array [A/m^2]
    """
    q = CARGA_ELEMENTAR
    kB = CONSTANTE_BOLTZMANN
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
            expoente = q * (V + J * Rs) / (n * kB * T)
            # Limitar expoente para evitar overflow numérico
            expoente = np.clip(expoente, -100, 100)

            termo_exp = np.exp(expoente)
            if np.isinf(Rsh):
                termo_shunt = 0.0
                d_termo_shunt_dJ = 0.0
            else:
                termo_shunt = (V + J * Rs) / Rsh
                d_termo_shunt_dJ = Rs / Rsh

            f_J = (J_ph
                   - J0 * (termo_exp - 1.0)
                   - termo_shunt
                   - J)

            # Derivada df/dJ
            dfdJ = (-J0 * termo_exp * (q * Rs / (n * kB * T))
                    - d_termo_shunt_dJ
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


# ============================================================
# 6. Função principal: montar o modelo completo e exibir resultados
# ============================================================

def executar_modelo_celula_silicio():
    # --------------------------------------------------------
    # 6.1. Parâmetros do material (nível quântico)
    # --------------------------------------------------------
    parametros = ParametrosSilicio(
        temperatura_celula=300.0,
        energia_gap_eV=1.12,
        massa_efetiva_eletron_rel=1.08,
        massa_efetiva_lacuna_rel=0.6
    )

    print("==============================================")
    print(" PARÂMETROS QUÂNTICOS DO SILÍCIO (DIDÁTICO)  ")
    print("==============================================")
    print(f"Temperatura da célula        : {parametros.temperatura_celula:.1f} K")
    print(f"Gap de energia (Eg)          : {parametros.energia_gap_eV:.3f} eV")
    print(f"Nc (densidade de estados)    : {parametros.Nc:.3e} m^-3")
    print(f"Nv (densidade de estados)    : {parametros.Nv:.3e} m^-3")
    print(f"Concentração intrínseca (ni) : {parametros.ni:.3e} m^-3")
    print()

    # --------------------------------------------------------
    # 6.2. Corrente fotogerada (limite Shockley–Queisser)
    # --------------------------------------------------------
    J_ph = calcular_corrente_fotogerada_limite(
        energia_gap_eV=parametros.energia_gap_eV,
        temperatura_sol=5778.0,
        num_pontos_energia=4000
    )

    # Converter para mA/cm^2 para comparação usual
    J_ph_mA_cm2 = J_ph * 0.1  # 1 A/m^2 = 0.1 mA/cm^2

    print("==============================================")
    print(" CORRENTE FOTOGERADA (LIMITe QUÂNTICO)        ")
    print("==============================================")
    print(f"J_ph ~ {J_ph:.3e} A/m^2  (~ {J_ph_mA_cm2:.2f} mA/cm^2)")
    print()

    # --------------------------------------------------------
    # 6.3. Corrente de saturação radiativa J0
    # --------------------------------------------------------
    J0 = calcular_corrente_saturacao_radiativa(
        energia_gap_eV=parametros.energia_gap_eV,
        temperatura_celula=parametros.temperatura_celula,
        num_pontos_energia=4000
    )
    J0_mA_cm2 = J0 * 0.1

    print("==============================================")
    print(" CORRENTE DE SATURAÇÃO RADIATIVA (J0)         ")
    print("==============================================")
    print(f"J0 ~ {J0:.3e} A/m^2  (~ {J0_mA_cm2:.4e} mA/cm^2)")
    print()

    # --------------------------------------------------------
    # 6.4. Parâmetros do diodo equivalente
    # --------------------------------------------------------
    fator_idealidade = 1.0
    resistencia_serie = 0.5   # ohm·m^2 (valor didático)
    resistencia_shunt = 1e4   # ohm·m^2 (quase ideal)

    tensoes_V, correntes_J = curva_JV_diodo(
        J_ph=J_ph,
        J0=J0,
        temperatura_celula=parametros.temperatura_celula,
        fator_idealidade=fator_idealidade,
        resistencia_serie=resistencia_serie,
        resistencia_shunt=resistencia_shunt,
        tensao_min=0.0,
        tensao_max=1.2,
        num_pontos_tensao=400
    )

    # --------------------------------------------------------
    # 6.5. Extração de J_sc, V_oc, P_max, FF, eficiência
    # --------------------------------------------------------
    # Corrente de curto-circuito (aprox. ponto V=0)
    J_sc = correntes_J[0]
    J_sc_mA_cm2 = J_sc * 0.1

    # Tensão de circuito aberto (aprox. via diodo ideal)
    q = CARGA_ELEMENTAR
    kB = CONSTANTE_BOLTZMANN
    T = parametros.temperatura_celula
    n = fator_idealidade

    V_oc_ideal = (n * kB * T / q) * np.log(J_ph / J0 + 1.0)
    # Aproximação numérica usando o ponto onde J≈0
    indice_voc = np.argmin(np.abs(correntes_J))
    V_oc_numerico = tensoes_V[indice_voc]

    # Potência P(V) = V * J(V)
    potencias = tensoes_V * correntes_J  # [W/m^2]
    indice_pmax = np.argmax(potencias)
    V_mp = tensoes_V[indice_pmax]
    J_mp = correntes_J[indice_pmax]
    P_max = potencias[indice_pmax]       # [W/m^2]

    # Fator de preenchimento (FF)
    FF = (V_mp * J_mp) / (V_oc_numerico * J_sc + 1e-30)

    # Eficiência em relação à irradiância padrão
    eficiencia = P_max / IRRADIANCIA_PADRAO  # fração -> ex: 0.28 = 28%

    print("==============================================")
    print(" PARÂMETROS ELÉTRICOS DA CÉLULA (DIDÁTICO)    ")
    print("==============================================")
    print(f"J_sc (curto-circuito) : {J_sc:.3e} A/m^2 (~ {J_sc_mA_cm2:.2f} mA/cm^2)")
    print(f"V_oc (ideal)          : {V_oc_ideal:.3f} V")
    print(f"V_oc (numérico)       : {V_oc_numerico:.3f} V")
    print(f"P_max                 : {P_max:.1f} W/m^2")
    print(f"Fator de preenchimento: {FF*100:.1f} %")
    print(f"Eficiência            : {eficiencia*100:.1f} % (referência 1000 W/m^2)")
    print()

    # --------------------------------------------------------
    # 6.6. Gráficos J–V e P–V
    # --------------------------------------------------------
    plt.figure()
    plt.plot(tensoes_V, correntes_J * 0.1)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Tensão [V]")
    plt.ylabel("Densidade de Corrente [mA/cm²]")
    plt.title("Curva J–V — Célula FV de Silício (modelo didático)")
    plt.grid(True)

    plt.figure()
    plt.plot(tensoes_V, potencias)
    plt.xlabel("Tensão [V]")
    plt.ylabel("Potência [W/m²]")
    plt.title("Curva P–V — Célula FV de Silício (modelo didático)")
    plt.grid(True)

    plt.show()


# ============================================================
# Execução direta
# ============================================================

if __name__ == "__main__":
    executar_modelo_celula_silicio()
