# ============================================================
# Modelo de Gotejamento de Petróleo via Navier–Stokes (filme fino)
# Autor: Luiz Tiago Wilcke 
# ============================================================
"""
Ideia geral
-----------
Partimos das equações de Navier–Stokes incompressíveis para um fluido
newtoniano (petróleo):

(1)  ρ (∂u/∂t + u·∇u) = -∇p + μ ∇²u + ρ g
(2)  ∇·u = 0

Para um jato / filme fino vertical (gotejamento em torno da saída de
um tubo), usamos a aproximação de lubrificação:
- escoamento predominantemente na direção z (vertical),
- variações lentas em z comparadas ao raio do jato/filme,
- número de Reynolds moderado/baixo (escoamento laminar).

Escrevendo h(z,t) como a espessura local do filme de petróleo (ou o
raio efetivo de uma coluna axisimétrica) e integrando NS na direção
transversal, obtemos a equação de filme fino 1D com gravidade + tensão
superficial:

(3)  ∂h/∂t + ∂/∂z [ (ρ g h³)/(3 μ) ] =
         (σ / (3 μ)) ∂/∂z [ h³ ∂³h/∂z³ ]

O termo à esquerda representa o escoamento por gravidade que alimenta o
gotejamento; o termo à direita vem da curvatura (tensão superficial) que
controla a formação e estrangulamento (pinch-off) das gotas.

Generalização adimensional
--------------------------
Escolhendo escalas típicas:
  L  ~ comprimento característico (m)
  H0 ~ espessura típica (m)
  T  ~ escala de tempo

Definimos variáveis adimensionais:
  Z = z / L,     H = h / H0,     τ = t / T

Surgem números adimensionais clássicos:
  Bo (Bond)      = ρ g L² / σ
  Oh (Ohnesorge) = μ / √(ρ σ L)

Uma forma genérica da equação de filme fino adimensionalizada é

(4)  ∂H/∂τ + ∂/∂Z [ Bo H³ ] =
         ∂/∂Z [ (1/Oh²) H³ ∂³H/∂Z³ ]

Com isso, o mesmo código pode simular não só petróleo,
mas qualquer fluido newtoniano, bastando ajustar ρ, μ, σ e L.

Abaixo implementamos:
- cálculo dos números adimensionais;
- simulação numérica 1D da Eq. (3) (forma dimensional) com diferenças finitas;
- um cenário de gotejamento de petróleo a partir de uma "protuberância"
  inicial que evolui em direção à formação de gotas.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 1. Parâmetros físicos e de simulação
# ------------------------------------------------------------

@dataclass
class ParametrosFluido:
    densidade: float          # kg/m³
    viscosidade: float        # Pa.s
    tensao_superficial: float # N/m
    gravidade: float = 9.81   # m/s² (pode ser alterado p/ outros planetas)


@dataclass
class ParametrosSimulacao:
    comprimento: float = 0.02      # m (20 mm ao longo da coluna/filme)
    n_pontos: int = 200            # discretização espacial
    dt: float = 2.0e-6             # passo de tempo (s) - pequeno p/ estabilidade
    t_final: float = 0.02          # horizonte de simulação (s)
    salvar_cada: int = 500         # guardar perfis a cada N passos
    espessura_base: float = 1.0e-4 # espessura "de fundo" do filme (m)
    amplitude_gota: float = 6.0e-4 # amplitude da protuberância inicial (m)
    largura_gota: float = 0.003    # largura característica da protuberância (m)


# ------------------------------------------------------------
# 2. Números adimensionais — generalização do modelo
# ------------------------------------------------------------

def calcular_numeros_adimensionais(
    fluido: ParametrosFluido,
    comprimento_escala: float,
    espessura_escala: float
) -> Dict[str, float]:
    """
    Calcula números adimensionais (Bond e Ohnesorge) que generalizam
    a equação de filme fino derivada de Navier–Stokes.
    """
    rho = fluido.densidade
    mu = fluido.viscosidade
    sigma = fluido.tensao_superficial
    g = fluido.gravidade
    L = comprimento_escala

    # Número de Bond (gravidade vs tensão superficial)
    Bo = rho * g * L**2 / sigma

    # Número de Ohnesorge (viscosidade vs inércia + tensão superficial)
    Oh = mu / np.sqrt(rho * sigma * L)

    # Escala de tempo gravitacional (só para referência)
    T_g = np.sqrt(L / g)

    return {
        "Bond_Bo": Bo,
        "Ohnesorge_Oh": Oh,
        "tempo_escala_gravitacional": T_g,
        "comprimento_escala_L": L,
        "espessura_escala_H0": espessura_escala,
    }


# ------------------------------------------------------------
# 3. Operadores de derivada 1D (diferenças finitas centrais)
# ------------------------------------------------------------

def derivada_central(v: np.ndarray, dx: float) -> np.ndarray:
    """
    Primeira derivada com fronteiras de Neumann (gradiente zero nas bordas).
    """
    dv = np.zeros_like(v)
    dv[1:-1] = (v[2:] - v[:-2]) / (2.0 * dx)
    # fronteiras: gradiente ~ 0
    dv[0] = (v[1] - v[0]) / dx
    dv[-1] = (v[-1] - v[-2]) / dx
    return dv


def terceira_derivada_central(v: np.ndarray, dx: float) -> np.ndarray:
    """
    Terceira derivada aproximada aplicando derivadas centrais sucessivas.
    """
    d1 = derivada_central(v, dx)
    d2 = derivada_central(d1, dx)
    d3 = derivada_central(d2, dx)
    return d3


# ------------------------------------------------------------
# 4. Passo de tempo da equação de filme fino
# ------------------------------------------------------------

def passo_tempo_filme_fino(
    h: np.ndarray,
    fluido: ParametrosFluido,
    params: ParametrosSimulacao,
    dx: float
) -> np.ndarray:
    """
    Passo explícito de tempo para a equação:

    ∂h/∂t + ∂/∂z [ (ρ g h³)/(3 μ) ] =
          (σ / (3 μ)) ∂/∂z [ h³ ∂³h/∂z³ ]
    """
    rho = fluido.densidade
    mu = fluido.viscosidade
    sigma = fluido.tensao_superficial
    g = fluido.gravidade

    # Fluxo por gravidade: q_g = (ρ g h³)/(3 μ)
    q_g = (rho * g * h**3) / (3.0 * mu)
    dqg_dz = derivada_central(q_g, dx)

    # Termo de tensão superficial (curvatura): q_σ = - (σ h³ / 3 μ) ∂³h/∂z³
    d3h_dz3 = terceira_derivada_central(h, dx)
    q_sigma = - (sigma * h**3 * d3h_dz3) / (3.0 * mu)
    dq_sigma_dz = derivada_central(q_sigma, dx)

    dh_dt = - (dqg_dz + dq_sigma_dz)

    return h + params.dt * dh_dt


# ------------------------------------------------------------
# 5. Condição inicial — protuberância/gota pendente no topo
# ------------------------------------------------------------

def condicao_inicial_gotejamento(params: ParametrosSimulacao) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cria uma espessura inicial h(z,0) com:
      h(z,0) = h_base + A * exp( - (z / Lg)² )
    localizada no topo (z=0), representando uma gota pendente sob o orifício.
    """
    z = np.linspace(0.0, params.comprimento, params.n_pontos)
    h_base = params.espessura_base
    A = params.amplitude_gota
    Lg = params.largura_gota

    h0 = h_base + A * np.exp(-(z / Lg)**2)
    return z, h0


# ------------------------------------------------------------
# 6. Função principal de simulação
# ------------------------------------------------------------

def simular_gotejamento_petroleo(
    fluido: ParametrosFluido,
    params: ParametrosSimulacao
):
    """
    Simula a evolução da espessura h(z,t) de um filme de petróleo em queda
    (modelo 1D de gotejamento) e retorna:
      - coordenadas z
      - lista de perfis de espessura ao longo do tempo
      - tempos associados a cada perfil salvo
    """
    z, h = condicao_inicial_gotejamento(params)
    dx = z[1] - z[0]

    n_passos = int(params.t_final / params.dt)
    perfis = [h.copy()]
    tempos = [0.0]

    for passo in range(1, n_passos + 1):
        h = passo_tempo_filme_fino(h, fluido, params, dx)

        # Evita espessuras negativas por erro numérico
        h = np.maximum(h, 1e-8)

        if passo % params.salvar_cada == 0:
            perfis.append(h.copy())
            tempos.append(passo * params.dt)

    return z, np.array(perfis), np.array(tempos)


# ------------------------------------------------------------
# 7. Visualização — evolução da gota
# ------------------------------------------------------------

def plotar_resultados(z: np.ndarray, perfis: np.ndarray, tempos: np.ndarray):
    """
    Plota alguns perfis h(z,t) para visualizar a formação/alongamento da gota.
    """
    plt.figure(figsize=(8, 5))
    for i, (h, t) in enumerate(zip(perfis, tempos)):
        label = f"t = {t*1000:.2f} ms"
        plt.plot(z * 1000, h * 1000, label=label)
    plt.xlabel("z (mm)")
    plt.ylabel("espessura h (mm)")
    plt.title("Modelo 1D de gotejamento de petróleo (filme fino)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Propriedades típicas de um petróleo pesado (valores aproximados)
    fluido_petroleo = ParametrosFluido(
        densidade=900.0,          # kg/m³
        viscosidade=0.5,          # Pa.s (bem mais viscoso que água)
        tensao_superficial=0.03   # N/m (ordem de grandeza óleo/água)
    )

    params = ParametrosSimulacao(
        comprimento=0.02,
        n_pontos=200,
        dt=2.0e-6,
        t_final=0.02,
        salvar_cada=800,
        espessura_base=1.0e-4,
        amplitude_gota=7.0e-4,
        largura_gota=0.003
    )

    # Mostra generalização em termos de números adimensionais
    nums = calcular_numeros_adimensionais(
        fluido_petroleo,
        comprimento_escala=params.comprimento,
        espessura_escala=params.espessura_base
    )
    print("=== Números adimensionais (generalização do modelo) ===")
    for k, v in nums.items():
        print(f"{k:35s}: {v: .4e}")

    # Simulação
    z, perfis, tempos = simular_gotejamento_petroleo(fluido_petroleo, params)

    # Visualização
    plotar_resultados(z, perfis, tempos)
