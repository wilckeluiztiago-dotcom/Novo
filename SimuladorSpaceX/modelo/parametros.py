class ParametrosFalcon9:
    """
    Parâmetros aproximados do Falcon 9 v1.2 Full Thrust.
    """
    # Constantes Físicas
    G = 6.67430e-11  # Constante gravitacional (m^3 kg^-1 s^-2)
    M_TERRA = 5.972e24  # Massa da Terra (kg)
    R_TERRA = 6371000  # Raio da Terra (m)
    G0 = 9.80665  # Gravidade padrão (m/s^2)
    RHO_AR_0 = 1.225  # Densidade do ar ao nível do mar (kg/m^3)
    H_ESCALA = 8500  # Altura de escala da atmosfera (m)

    # Parâmetros do Foguete (1º Estágio)
    MASSA_SECA = 25600  # Massa estrutural (kg)
    MASSA_COMBUSTIVEL_MAX = 395700  # Massa de propelente (kg)
    EMPUXO_NIVEL_MAR = 7607000  # Empuxo total (9 Merlin 1D) (N)
    EMPUXO_VACUO = 8227000  # Empuxo no vácuo (N)
    ISP_NIVEL_MAR = 282  # Impulso específico ao nível do mar (s)
    ISP_VACUO = 311  # Impulso específico no vácuo (s)
    AREA_FRONTAL = 10.8  # Área de seção transversal (m^2) - Diâmetro ~3.7m
    COEF_ARRASTO = 0.2  # Coeficiente de arrasto aproximado (adimensional)
    TAXA_QUEIMA = MASSA_COMBUSTIVEL_MAX / 162 # Queima total em ~162s
