# ============================================================
# COSMOLOGIA — ONDAS GRAVITACIONAIS PRIMORDIAIS (INFLAÇÃO)
# Autor: Luiz Tiago Wilcke (LT)
# ============================================================
# - Universo de Sitter (inflação): a(η) = -1/(H η)
# - Perturbações tensoriais h_k(η) (tensor modes)
# - Equação: h_k'' + 2ℋ h_k' + k^2 h_k = 0, ℋ = a'/a = -1/η
#   → h_k'' - (2/η) h_k' + k^2 h_k = 0
# - Forma de 1ª ordem (sistema hiperbólico):
#     u = h_k
#     v = h_k'
#     u' = v
#     v' =  2/η v - k^2 u
#
# - Integração numérica via RK4 em tempo conforme η < 0
# - Condições iniciais de Bunch–Davies (vácuo quântico)
# - Cálculo do espectro tensorial aproximado:
#     P_T(k) ≈ (2 k^3 / π^2) |h_k(η_fim)|^2
# - Impressão de resultados com ~21 dígitos de precisão
# - Gráficos:
#     • |h_k(η)| vs -η (escala positiva de "tempo")
#     • P_T(k) vs k (log-log)
# ============================================================

import math
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp


# ------------------------------------------------------------
# 1. Configurações gerais do modelo
# ------------------------------------------------------------
@dataclass
class ConfiguracoesCosmologicas:
    # Parâmetros de inflação (natural units, H ~ constante)
    H_inflacao: float = 1.0  # valor arbitrário (unidades naturais)

    # Intervalo de tempo conforme (η < 0):
    # η_inicial: muito negativo (dentro do horizonte)
    # η_final: próximo de 0- (fora do horizonte)
    eta_inicial: float = -100.0
    eta_final: float = -0.1

    # Número de passos da integração RK4
    numero_passos: int = 20000

    # Modos de onda (números de onda k) a serem simulados
    lista_modos_k: Tuple[float, ...] = (0.1, 0.3, 1.0, 3.0, 10.0)

    # Precisão decimal (número de dígitos significativos)
    precisao_decimal: int = 30  # >= 21 dígitos, com margem


# ------------------------------------------------------------
# 2. Funções auxiliares: Hubble conforme, sistema de EDOs
# ------------------------------------------------------------
def hubble_conforme(eta: mp.mpf) -> mp.mpf:
    """
    Hubble conforme em de Sitter:
    ℋ = a'/a = -1/η
    """
    return -1 / eta


def derivadas_sistema_tensorial(
    eta: mp.mpf,
    vetor_estado: Tuple[mp.mpc, mp.mpc],
    modo_k: mp.mpf
) -> Tuple[mp.mpc, mp.mpc]:
    """
    Sistema de 1ª ordem para o modo tensorial h_k(η):
    u = h_k
    v = h_k'

    u' = v
    v' = -k^2 u - 2ℋ v   (forma geral)
       = -k^2 u + (2/η) v  em de Sitter, pois ℋ = -1/η
    """
    u, v = vetor_estado
    eta_mp = mp.mpf(eta)
    k_mp = mp.mpf(modo_k)

    Hc = hubble_conforme(eta_mp)  # ℋ(η) = -1/η

    du = v
    dv = - (k_mp ** 2) * u - 2 * Hc * v  # -k^2 u - 2 ℋ v

    return du, dv


# ------------------------------------------------------------
# 3. Condições iniciais de Bunch–Davies para h_k(η)
# ------------------------------------------------------------
def condicoes_iniciais_bunch_davies(
    modo_k: float,
    eta_inicial: float
) -> Tuple[mp.mpc, mp.mpc]:
    """
    Condições iniciais aproximadas de Bunch–Davies no limite η → -∞:
      h_k(η) ~ e^{-i k η} / sqrt(2k)
      h_k'(η) ~ -i sqrt(k/2) e^{-i k η}

    Usamos isso em um η_inicial grande e negativo.
    """
    k_mp = mp.mpf(modo_k)
    eta_mp = mp.mpf(eta_inicial)

    fase = mp.e ** (-1j * k_mp * eta_mp)
    u0 = fase / mp.sqrt(2 * k_mp)
    v0 = -1j * mp.sqrt(k_mp / 2) * fase

    return u0, v0


# ------------------------------------------------------------
# 4. Integrador RK4 em tempo conforme
# ------------------------------------------------------------
def passo_rk4(
    eta: mp.mpf,
    vetor_estado: Tuple[mp.mpc, mp.mpc],
    passo_eta: mp.mpf,
    modo_k: mp.mpf
) -> Tuple[mp.mpc, mp.mpc]:
    """
    Aplica um passo de Runge–Kutta de 4ª ordem (RK4)
    ao sistema de EDOs tensorial.
    """
    u, v = vetor_estado

    du1, dv1 = derivadas_sistema_tensorial(eta, (u, v), modo_k)
    du2, dv2 = derivadas_sistema_tensorial(
        eta + passo_eta / 2,
        (u + passo_eta * du1 / 2, v + passo_eta * dv1 / 2),
        modo_k
    )
    du3, dv3 = derivadas_sistema_tensorial(
        eta + passo_eta / 2,
        (u + passo_eta * du2 / 2, v + passo_eta * dv2 / 2),
        modo_k
    )
    du4, dv4 = derivadas_sistema_tensorial(
        eta + passo_eta,
        (u + passo_eta * du3, v + passo_eta * dv3),
        modo_k
    )

    u_novo = u + (passo_eta / 6) * (du1 + 2 * du2 + 2 * du3 + du4)
    v_novo = v + (passo_eta / 6) * (dv1 + 2 * dv2 + 2 * dv3 + dv4)

    return u_novo, v_novo


def integrar_modo_tensorial(
    cfg: ConfiguracoesCosmologicas,
    modo_k: float
) -> Dict[str, np.ndarray]:
    """
    Integra numericamente o modo tensorial h_k(η) usando RK4
    entre η_inicial e η_final.
    Retorna dicionário com:
      - eta_array: array de η (float)
      - h_array: array de h_k(η) (complex)
      - h_modulo: |h_k(η)|
    """
    # Configuração de precisão alta
    mp.mp.dps = cfg.precisao_decimal

    eta_inicial_mp = mp.mpf(cfg.eta_inicial)
    eta_final_mp = mp.mpf(cfg.eta_final)
    numero_passos = cfg.numero_passos

    # Passo em tempo conforme
    passo_eta = (eta_final_mp - eta_inicial_mp) / numero_passos

    # Condições iniciais (Bunch–Davies)
    u, v = condicoes_iniciais_bunch_davies(modo_k, cfg.eta_inicial)

    # Armazenar resultados
    lista_eta: List[float] = []
    lista_h: List[complex] = []

    eta_atual = eta_inicial_mp

    for _ in range(numero_passos + 1):
        # Guardar valores atuais
        lista_eta.append(float(eta_atual))
        lista_h.append(complex(u))  # converter mpc -> complex para numpy

        # Avançar um passo de RK4
        u, v = passo_rk4(eta_atual, (u, v), passo_eta, mp.mpf(modo_k))
        eta_atual += passo_eta

    eta_array = np.array(lista_eta, dtype=float)
    h_array = np.array(lista_h, dtype=complex)
    h_modulo = np.abs(h_array)

    return {
        "eta_array": eta_array,
        "h_array": h_array,
        "h_modulo": h_modulo,
    }


# ------------------------------------------------------------
# 5. Cálculo do espectro tensorial primordial aproximado
# ------------------------------------------------------------
def espectro_tensorial(
    modos_k: np.ndarray,
    amplitudes_finais: np.ndarray
) -> np.ndarray:
    """
    Espectro tensorial aproximado:
      P_T(k) ≈ (2 k^3 / π^2) |h_k(η_fim)|^2

    Aqui é um modelo didático (normalização arbitrária).
    """
    modulo2 = amplitudes_finais ** 2
    return (2.0 * modos_k ** 3 / (math.pi ** 2)) * modulo2


# ------------------------------------------------------------
# 6. Função principal
# ------------------------------------------------------------
def main():
    cfg = ConfiguracoesCosmologicas()

    # Ajustar precisão da mpmath
    mp.mp.dps = cfg.precisao_decimal

    # Listas para espectro
    lista_k = []
    lista_amp_final = []

    resultados_por_modo = {}

    print("=================================================")
    print("SIMULAÇÃO — ONDAS GRAVITACIONAIS PRIMORDIAIS")
    print("Modelo: de Sitter + modos tensoriais h_k(η)")
    print("Integração RK4 com alta precisão (mpmath)")
    print("=================================================\n")

    # Integrar cada modo k
    for k in cfg.lista_modos_k:
        print(f"Integrando modo k = {k:.6f} ...")

        resultado = integrar_modo_tensorial(cfg, k)
        resultados_por_modo[k] = resultado

        # Amplitude final |h_k(η_fim)|
        h_modulo = resultado["h_modulo"]
        amp_final = h_modulo[-1]

        lista_k.append(k)
        lista_amp_final.append(amp_final)

        # Impressão com alta precisão (~21 dígitos)
        amp_final_mp = mp.mpf(amp_final)
        texto_amp = mp.nstr(amp_final_mp, 21)

        print(f"  |h_k(η_final)| ≈ {texto_amp}  (≈ 21 dígitos)")
        print()

    # Converter para numpy
    modos_k_np = np.array(lista_k, dtype=float)
    amp_final_np = np.array(lista_amp_final, dtype=float)

    # Calcular espectro tensorial
    P_T = espectro_tensorial(modos_k_np, amp_final_np)

    print("=================================================")
    print("ESPECTRO TENSORIAL PRIMORDIAL (modelo didático)")
    print("P_T(k) ≈ (2 k^3 / π^2) |h_k(η_final)|^2")
    print("=================================================\n")

    for k, pt in zip(modos_k_np, P_T):
        pt_mp = mp.mpf(pt)
        texto_pt = mp.nstr(pt_mp, 21)
        print(f"  k = {k:.6f}  ->  P_T(k) ≈ {texto_pt}")

    # --------------------------------------------------------
    # 7. Gráficos
    # --------------------------------------------------------
    plt.figure(figsize=(10, 6))
    for k in cfg.lista_modos_k:
        resultado = resultados_por_modo[k]
        eta = resultado["eta_array"]
        h_mod = resultado["h_modulo"]

        # Usamos -η para ter "tempo positivo" na horizontal
        plt.plot(-eta, h_mod, label=f"k = {k:.2f}")

    plt.xlabel(r"$-\,\eta$  (tempo conforme positivo)")
    plt.ylabel(r"$|h_k(\eta)|$")
    plt.title("Evolução temporal das ondas gravitacionais primordiais (modos tensoriais)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Gráfico do espectro tensorial
    plt.figure(figsize=(8, 6))
    plt.loglog(modos_k_np, P_T, marker="o")
    plt.xlabel(r"$k$  (número de onda)")
    plt.ylabel(r"$\mathcal{P}_T(k)$  (unidades arbitrárias)")
    plt.title("Espectro tensorial primordial — modelo inflacionário didático")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
