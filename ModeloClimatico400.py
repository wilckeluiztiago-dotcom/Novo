# ============================================================
# Modelo Climático com Controle Ótimo via HJB
# Autor: Luiz Tiago Wilcke (LT)
# ============================================================

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 1. Parâmetros do modelo climático-óptimo
# ------------------------------------------------------------

@dataclass
class ParametrosClimaHJB:
    # Grade de temperatura (°C acima do pré-industrial)
    temperatura_min: float = 0.0
    temperatura_max: float = 6.0
    passos_temperatura: int = 301  # número de pontos na grade

    # Horizonte temporal (anos)
    horizonte_anos: int = 100
    dt: float = 1.0  # passo de tempo (anos)

    # Dinâmica climática simplificada
    aquecimento_base: float = 0.03          # tendência média de aquecimento (°C/ano)
    resfriamento_natural: float = 0.02      # força de retorno ao equilíbrio
    impacto_mitigacao: float = 0.05         # quanto a mitigação reduz o aquecimento (°C/ano por unidade de esforço)
    temperatura_natural: float = 0.0        # temperatura "natural" (referência)

    # Função de custo
    temperatura_referencia: float = 1.5     # meta de aquecimento (ex: 1.5 °C)
    alpha_dano: float = 2.0                 # peso do dano climático
    beta_custo: float = 0.5                 # peso do custo de mitigação

    # Conjunto de controles (esforços de mitigação)
    numero_controles: int = 51              # discretização de u em [0,1]


# ------------------------------------------------------------
# 2. Funções do modelo: dinâmica e custos
# ------------------------------------------------------------

def dinamica_clima(temperatura: float,
                   controle: float,
                   p: ParametrosClimaHJB) -> float:
    """
    dT/dt = aquecimento_base - resfriamento_natural*(T - T_natural) - impacto_mitigacao*u
    """
    termo_aquecimento = p.aquecimento_base
    termo_resfriamento = p.resfriamento_natural * (temperatura - p.temperatura_natural)
    termo_mitigacao = p.impacto_mitigacao * controle
    return termo_aquecimento - termo_resfriamento - termo_mitigacao


def custo_dano_climatico(temperatura: float,
                         p: ParametrosClimaHJB) -> float:
    """
    Dano climático quadrático acima da temperatura de referência.
    """
    excesso = max(temperatura - p.temperatura_referencia, 0.0)
    return p.alpha_dano * excesso**2


def custo_mitigacao(controle: float,
                    p: ParametrosClimaHJB) -> float:
    """
    Custo quadrático do esforço de mitigação.
    """
    return p.beta_custo * controle**2


def custo_instantaneo(temperatura: float,
                      controle: float,
                      p: ParametrosClimaHJB) -> float:
    """
    Custo total instantâneo = dano climático + custo de mitigação.
    """
    return custo_dano_climatico(temperatura, p) + custo_mitigacao(controle, p)


# ------------------------------------------------------------
# 3. Resolução numérica da HJB (diferenças finitas upwind)
# ------------------------------------------------------------

def resolver_hjb(p: ParametrosClimaHJB) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Resolve numericamente a HJB em uma grade de temperatura x tempo.
    Retorna:
      - grade_temperatura: vetor 1D de temperaturas
      - grade_tempo: vetor 1D de tempos
      - valor_otimo: matriz [tempo, temperatura] com V(t_k, T_i)
      - politica_otima: matriz [tempo, temperatura] com u*(t_k, T_i)
    """

    # Grade de temperatura
    grade_temperatura = np.linspace(
        p.temperatura_min,
        p.temperatura_max,
        p.passos_temperatura
    )
    dT = grade_temperatura[1] - grade_temperatura[0]

    # Grade de tempo (0, dt, ..., horizonte_anos)
    grade_tempo = np.arange(0.0, p.horizonte_anos + p.dt, p.dt)
    passos_tempo = len(grade_tempo)

    # Conjunto de controles (discretização de u em [0,1])
    grade_controles = np.linspace(0.0, 1.0, p.numero_controles)

    # Matrizes para valor ótimo e política ótima
    valor_otimo = np.zeros((passos_tempo, p.passos_temperatura))
    politica_otima = np.zeros((passos_tempo, p.passos_temperatura))

    # Condição terminal: custo terminal depende apenas da temperatura
    # V(T, T_final) = dano_climático_terminal(T)
    for i, T in enumerate(grade_temperatura):
        valor_otimo[-1, i] = custo_dano_climatico(T, p)

    # Backward in time (programação dinâmica)
    for k in range(passos_tempo - 2, -1, -1):
        V_proximo = valor_otimo[k + 1, :]

        for i, T in enumerate(grade_temperatura):
            melhor_valor = np.inf
            melhor_controle = 0.0

            for u in grade_controles:
                # Dinâmica local
                drift = dinamica_clima(T, u, p)

                # Aproximação de derivada dV/dT (upwind)
                if drift >= 0:
                    # deriva para temperaturas maiores: usar diferença para frente
                    if i < p.passos_temperatura - 1:
                        dV_dT = (V_proximo[i + 1] - V_proximo[i]) / dT
                    else:
                        # borda superior: usar diferença para trás
                        dV_dT = (V_proximo[i] - V_proximo[i - 1]) / dT
                else:
                    # deriva para temperaturas menores: usar diferença para trás
                    if i > 0:
                        dV_dT = (V_proximo[i] - V_proximo[i - 1]) / dT
                    else:
                        # borda inferior: usar diferença para frente
                        dV_dT = (V_proximo[i + 1] - V_proximo[i]) / dT

                # Custo instantâneo no estado (T,u)
                custo = custo_instantaneo(T, u, p)

                # Esquema de Euler implícito simples para HJB:
                # V_k(T_i) ≈ min_u { V_{k+1}(T_i) + dt * [custo + drift * dV_dT] }
                valor_candidato = V_proximo[i] + p.dt * (custo + drift * dV_dT)

                if valor_candidato < melhor_valor:
                    melhor_valor = valor_candidato
                    melhor_controle = u

            valor_otimo[k, i] = melhor_valor
            politica_otima[k, i] = melhor_controle

    return grade_temperatura, grade_tempo, valor_otimo, politica_otima


# ------------------------------------------------------------
# 4. Simulação de trajetória ótima (após resolver HJB)
# ------------------------------------------------------------

def simular_politica_otima(temperatura_inicial: float,
                           p: ParametrosClimaHJB,
                           grade_temperatura: np.ndarray,
                           grade_tempo: np.ndarray,
                           politica_otima: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simula a trajetória da temperatura e do controle ótimo
    usando a política encontrada pela HJB.
    """

    passos_tempo = len(grade_tempo)
    temperatura = float(temperatura_inicial)

    trajetoria_temperatura = np.zeros(passos_tempo)
    trajetoria_controle = np.zeros(passos_tempo - 1)

    trajetoria_temperatura[0] = temperatura

    for k in range(passos_tempo - 1):
        # encontra índice de temperatura mais próximo na grade
        idx_T = int(np.argmin(np.abs(grade_temperatura - temperatura)))
        controle_otimo = politica_otima[k, idx_T]

        # salva controle
        trajetoria_controle[k] = controle_otimo

        # integra a dinâmica (Euler explícito simples)
        dT_dt = dinamica_clima(temperatura, controle_otimo, p)
        temperatura = temperatura + p.dt * dT_dt

        # mantém dentro da grade
        temperatura = float(np.clip(temperatura,
                                    p.temperatura_min,
                                    p.temperatura_max))

        trajetoria_temperatura[k + 1] = temperatura

    return trajetoria_temperatura, trajetoria_controle


# ------------------------------------------------------------
# 5. Execução principal: resolve HJB e mostra resultados
# ------------------------------------------------------------

def executar_demo_modelo_climatico():
    # 1) Parâmetros
    p = ParametrosClimaHJB()

    # 2) Resolver HJB
    print("Resolvendo a HJB em uma grade (isso pode levar alguns segundos)...")
    grade_T, grade_t, V, politica = resolver_hjb(p)
    print("HJB resolvida!")

    # 3) Simular trajetória ótima a partir de T0
    temperatura_inicial = 1.0  # por exemplo, 1°C acima do pré-industrial
    trajetoria_T, trajetoria_u = simular_politica_otima(
        temperatura_inicial, p, grade_T, grade_t, politica
    )

    # 4) Plots
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    # (a) Valor inicial como função da temperatura
    axs[0].plot(grade_T, V[0, :])
    axs[0].set_title("Valor ótimo inicial V(0, T)")
    axs[0].set_xlabel("Temperatura (°C acima do pré-industrial)")
    axs[0].set_ylabel("Valor de custo esperado")

    # (b) Trajetória ótima de temperatura
    axs[1].plot(grade_t, trajetoria_T)
    axs[1].axhline(p.temperatura_referencia, linestyle="--", label="Meta climática")
    axs[1].set_title("Trajetória ótima da temperatura global")
    axs[1].set_xlabel("Tempo (anos)")
    axs[1].set_ylabel("Temperatura (°C acima do pré-industrial)")
    axs[1].legend()

    # (c) Trajetória do esforço de mitigação
    axs[2].plot(grade_t[:-1], trajetoria_u)
    axs[2].set_title("Esforço de mitigação ótimo u*(t)")
    axs[2].set_xlabel("Tempo (anos)")
    axs[2].set_ylabel("Controle (0 = sem política, 1 = máximo)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    executar_demo_modelo_climatico()
