# ============================================================
# Modelo Super Simplificado de Aquecimento Global (EBM 2-Caixas)
#   - Equações diferenciais ordinárias (EDOs)
#   - Constantes físicas com alta precisão
#   - Forçamento radiativo por CO₂ (ΔF = 5.35 ln(C/C0))
#   - Realimentação gelo–albedo
#   - Caixa de superfície + oceano profundo
#
# Autor: Luiz Tiago Wilcke 
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Literal
from scipy.integrate import solve_ivp

# ------------------------------------------------------------
# 1) Parâmetros físicos e de cenário
# ------------------------------------------------------------

@dataclass
class ParametrosModelo:
    # Tempo da simulação
    tempo_simulacao_anos: float = 200.0
    passo_saida_anos: float = 0.1

    # Constantes físicas
    constante_solar: float = 1361.0                    # W/m²
    sigma_SB: float = 5.670374419e-8                  # W/(m² K⁴) — Stefan-Boltzmann
    emissividade_efetiva: float = 0.6105405115455411  # Ajustada p/ equilíbrio em ~288 K

    # Geometria temporal
    segundos_por_ano: float = 365.25 * 24.0 * 3600.0

    # Calor específico efetivo (J/(m² K)) das caixas
    # Mistura de atmosfera + camada superior do oceano (~70 m)
    capacidade_calor_superficie_J: float = 4.2e8
    # Oceano profundo: muito maior
    capacidade_calor_profundidade_J: float = 3.0e9

    # Troca de calor entre superfície e oceano profundo
    taxa_troca_superficie_oceano: float = 0.8         # W/(m² K)

    # Albedo (refletividade)
    albedo_referencia: float = 0.30                   # valor típico atual
    usar_realimentacao_gelo: bool = True
    albedo_min: float = 0.28                          # planeta mais quente (menos gelo)
    albedo_max: float = 0.60                          # planeta frio (muito gelo)
    temperatura_char_albedo: float = 260.0            # K — transição de gelo
    delta_T_albedo: float = 10.0                      # "largura" da transição

    # CO₂ e forçamento de gases de efeito estufa
    co2_inicial_ppm: float = 280.0                    # ppm (pré-industrial)
    co2_referencia_forc_ppm: float = 280.0            # ppm p/ ΔF = 0
    tipo_cenario_co2: Literal["exponencial", "linear", "constante"] = "exponencial"
    taxa_crescimento_co2: float = 0.01                # 1% ao ano (duplica ~70 anos)
    incremento_anual_co2_ppm: float = 2.5             # usado se cenário linear

    # Forçamentos extras (aerossóis, solar, etc) — opcional
    forcamento_extra_constante: float = 0.0           # W/m²

    # Condições iniciais (em Kelvin)
    temperatura_superficie_inicial_K: float = 288.0   # ~15°C
    temperatura_profundidade_inicial_K: float = 286.0 # ligeiramente mais fria


# ------------------------------------------------------------
# 2) Funções auxiliares de física do clima
# ------------------------------------------------------------

def capacidade_calor_anos(param: ParametrosModelo):
    """
    Converte capacidades de calor de J/(m² K) para (W·ano)/(m² K),
    pois a EDO será escrita em termos de t (anos).

    dT/dt_anos = Fluxo (W/m²) / C_ano
    onde C_ano = C_J / segundos_por_ano
    """
    C_sup_ano = param.capacidade_calor_superficie_J / param.segundos_por_ano
    C_prof_ano = param.capacidade_calor_profundidade_J / param.segundos_por_ano
    return C_sup_ano, C_prof_ano


def albedo_dependente_de_T(temperatura_superficie_K: float,
                            param: ParametrosModelo) -> float:
    """
    Albedo com realimentação gelo–albedo:
      - Próximo a albedo_max em planetas frios (muito gelo)
      - Próximo a albedo_min em planetas quentes (pouco gelo)
    Usamos uma função logística suave em função da temperatura.
    """
    if not param.usar_realimentacao_gelo:
        return param.albedo_referencia

    T = temperatura_superficie_K
    a_min = param.albedo_min
    a_max = param.albedo_max
    Tc = param.temperatura_char_albedo
    dT = param.delta_T_albedo

    return a_min + (a_max - a_min) / (1.0 + np.exp((T - Tc) / dT))


def concentracao_co2_ppm(t_anos: float, param: ParametrosModelo) -> float:
    """
    Cenários simples de CO₂:
      - exponencial: C(t) = C0 * exp(r t)
      - linear: C(t) = C0 + incremento_anual * t
      - constante: C(t) = C0
    """
    C0 = param.co2_inicial_ppm

    if param.tipo_cenario_co2 == "exponencial":
        return C0 * np.exp(param.taxa_crescimento_co2 * t_anos)
    elif param.tipo_cenario_co2 == "linear":
        return C0 + param.incremento_anual_co2_ppm * t_anos
    else:
        return C0


def forcamento_gei_Wm2(t_anos: float, param: ParametrosModelo) -> float:
    """
    Forçamento radiativo de gases de efeito estufa (CO₂) em W/m²:
        ΔF_CO₂ = 5.35 * ln(C / C_ref)
    Fórmula de Myhre et al. (1998) usada em muitos modelos simples.
    """
    C = concentracao_co2_ppm(t_anos, param)
    C_ref = param.co2_referencia_forc_ppm
    return 5.35 * np.log(C / C_ref)


# ------------------------------------------------------------
# 3) Equações diferenciais do modelo (2 caixas)
# ------------------------------------------------------------

def sistema_edo(t_anos: float, estado: np.ndarray, param: ParametrosModelo) -> np.ndarray:
    """
    Sistema de EDOs:
      - Caixa de superfície (atmosfera + oceano raso)
      - Caixa de oceano profundo

    Variáveis de estado:
      estado[0] = T_superficie (K)
      estado[1] = T_profundidade (K)

    Equações (em termos de t em anos):
      C_sup_ano dT_sup/dt = (1 - α(T_sup)) * S0 / 4
                            + F_GEI(t) + F_extra
                            - ε σ T_sup^4
                            - κ (T_sup - T_prof)

      C_prof_ano dT_prof/dt = κ (T_sup - T_prof)
    """
    T_sup = estado[0]
    T_prof = estado[1]

    C_sup_ano, C_prof_ano = capacidade_calor_anos(param)

    # Albedo dependente de T (gelo–albedo)
    albedo = albedo_dependente_de_T(T_sup, param)

    # Entrada solar média na superfície (esfera: 1/4)
    fluxo_solar_entrada = (1.0 - albedo) * param.constante_solar / 4.0  # W/m²

    # Forçamento por GEI (CO₂) + extra (aerossóis, etc.)
    F_gei = forcamento_gei_Wm2(t_anos, param)
    F_extra = param.forcamento_extra_constante

    # Emissão infravermelha (Stefan-Boltzmann "ajustado")
    fluxo_OLR = param.emissividade_efetiva * param.sigma_SB * (T_sup ** 4)

    # Troca de calor superfície ↔ oceano profundo
    fluxo_troca = param.taxa_troca_superficie_oceano * (T_sup - T_prof)  # W/m²

    # Equações diferenciais em anos
    dT_sup_dt = (fluxo_solar_entrada + F_gei + F_extra - fluxo_OLR - fluxo_troca) / C_sup_ano
    dT_prof_dt = (fluxo_troca) / C_prof_ano

    return np.array([dT_sup_dt, dT_prof_dt])


# ------------------------------------------------------------
# 4) Simulação numérica
# ------------------------------------------------------------

def simular_aquecimento_global(param: ParametrosModelo):
    """
    Resolve o sistema de EDOs para o período especificado.
    Retorna tempos (anos) e matrizes de estado (T_superficie, T_profundidade).
    """
    t0 = 0.0
    tf = param.tempo_simulacao_anos
    t_avaliacao = np.linspace(t0, tf, int(tf / param.passo_saida_anos) + 1)

    # Estado inicial (K)
    estado_inicial = np.array([
        param.temperatura_superficie_inicial_K,
        param.temperatura_profundidade_inicial_K
    ])

    solucao = solve_ivp(
        fun=lambda t, y: sistema_edo(t, y, param),
        t_span=(t0, tf),
        y0=estado_inicial,
        t_eval=t_avaliacao,
        method="RK45",
        rtol=1e-6,
        atol=1e-8,
    )

    if not solucao.success:
        raise RuntimeError(f"Falha na integração: {solucao.message}")

    return solucao.t, solucao.y


# ------------------------------------------------------------
# 5) Geração de gráficos e diagnósticos
# ------------------------------------------------------------

def gerar_graficos(t_anos: np.ndarray, estado: np.ndarray, param: ParametrosModelo):
    """
    Gera vários gráficos:
      1) Temperaturas (superfície e oceano profundo) em °C
      2) Concentração de CO₂ (ppm)
      3) Forçamento radiativo total (GEI + extra)
      4) Albedo em função da temperatura da superfície
      5) Diagrama de fase T_superfície vs T_profundidade
    """
    T_sup_K = estado[0, :]
    T_prof_K = estado[1, :]

    T_sup_C = T_sup_K - 273.15
    T_prof_C = T_prof_K - 273.15

    # Séries auxiliares
    co2_series = np.array([concentracao_co2_ppm(t, param) for t in t_anos])
    forc_gei_series = np.array([forcamento_gei_Wm2(t, param) for t in t_anos])
    forc_total_series = forc_gei_series + param.forcamento_extra_constante
    albedo_series = np.array([albedo_dependente_de_T(T, param) for T in T_sup_K])

    # ------------------- Gráfico 1: Temperaturas -------------------
    plt.figure()
    plt.plot(t_anos, T_sup_C, label="Superfície (°C)")
    plt.plot(t_anos, T_prof_C, label="Oceano profundo (°C)")
    plt.xlabel("Tempo (anos)")
    plt.ylabel("Temperatura (°C)")
    plt.title("Evolução da temperatura média global")
    plt.grid(True)
    plt.legend()

    # ------------------- Gráfico 2: CO₂ -------------------
    plt.figure()
    plt.plot(t_anos, co2_series)
    plt.xlabel("Tempo (anos)")
    plt.ylabel("Concentração de CO₂ (ppm)")
    plt.title("Cenário de concentração de CO₂")
    plt.grid(True)

    # ------------------- Gráfico 3: Forçamento radiativo -------------------
    plt.figure()
    plt.plot(t_anos, forc_gei_series, label="Forçamento GEI (W/m²)")
    if param.forcamento_extra_constante != 0.0:
        plt.plot(t_anos, forc_total_series, "--", label="Forçamento total (W/m²)")
    plt.xlabel("Tempo (anos)")
    plt.ylabel("Forçamento (W/m²)")
    plt.title("Forçamento radiativo por gases de efeito estufa")
    plt.grid(True)
    plt.legend()

    # ------------------- Gráfico 4: Albedo vs Temperatura -------------------
    plt.figure()
    plt.plot(T_sup_C, albedo_series)
    plt.xlabel("Temperatura da superfície (°C)")
    plt.ylabel("Albedo planetário")
    plt.title("Realimentação gelo–albedo")
    plt.grid(True)

    # ------------------- Gráfico 5: Diagrama de fase -------------------
    plt.figure()
    plt.plot(T_sup_C, T_prof_C)
    plt.xlabel("Superfície (°C)")
    plt.ylabel("Oceano profundo (°C)")
    plt.title("Diagrama de fase T_superfície vs T_profundidade")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Imprime aquecimento final como diagnóstico rápido
    aquecimento_final_C = T_sup_C[-1] - T_sup_C[0]
    print(f"Aquecimento da superfície em {param.tempo_simulacao_anos:.1f} anos: "
          f"{aquecimento_final_C:.2f} °C")


# ------------------------------------------------------------
# 6) Função principal
# ------------------------------------------------------------

def main():
    # Você pode ajustar o cenário aqui facilmente
    param = ParametrosModelo(
        tempo_simulacao_anos=200.0,
        passo_saida_anos=0.2,
        tipo_cenario_co2="exponencial",     # "exponencial", "linear", "constante"
        taxa_crescimento_co2=0.01,         # 1% ao ano
        incremento_anual_co2_ppm=2.5,      # usado se "linear"
        usar_realimentacao_gelo=True,
        forcamento_extra_constante=0.0     # pode colocar, por ex, -1.0 p/ aerossóis
    )

    t_anos, estado = simular_aquecimento_global(param)
    gerar_graficos(t_anos, estado, param)


if __name__ == "__main__":
    main()
