# ============================================================
# FALÊNCIA BANCÁRIA — MODELO DINÂMICO COM DASHBOARD INTERATIVO
# Autor: Luiz Tiago Wilcke (LT)
# ============================================================
# Ideia do modelo:
#   Estados:
#       A(t)   : Ativos totais
#       D(t)   : Depósitos
#       L(t)   : Liquidez
#       p_NPL  : proporção de crédito inadimplente
#       P_fal  : probabilidade acumulada de falência
#
#   Capital adequacy ratio (CAR):
#       CAR(t) = (A(t) - D(t)) / A(t)
#
#   Regras de falência:
#       - A(t) - D(t) <= 0   (patrimônio líquido negativo), OU
#       - CAR(t) < CAR_minimo, OU
#       - L(t) <= 0 (sem liquidez)
#
#   Sistema de EDOs (cenário estressado):
#
#       dA/dt = g_A * A
#               - gamma * p_NPL * A            (perdas de crédito)
#               - phi * max(CAR_min - CAR, 0) * D   (fuga de depósitos)
#
#       dD/dt = g_D * D
#               - phi * max(CAR_min - CAR, 0) * D
#
#       dL/dt = - phi * max(CAR_min - CAR, 0) * D
#               + theta_L * max(CAR - CAR_min, 0) * A
#
#       dp_NPL/dt = k_NPL * (p_NPL_eq - p_NPL) + stress(t)
#
#       dP_fal/dt = k_fal * max(CAR_min - CAR, 0) * (1 - P_fal)
#
# Dashboard:
#   - Tela azul escura
#   - Gráficos de A, D, L, CAR e P_fal
#   - Sliders para:
#       * Choque de crédito (gamma)
#       * Sensibilidade à fuga de depósitos (phi)
#       * CAR mínimo regulatório
#       * Ativos iniciais
#   - Equações renderizadas em LaTeX (subset compatível com Matplotlib)
# ============================================================

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.integrate import solve_ivp


# ------------------------------------------------------------
# Parâmetros do modelo
# ------------------------------------------------------------
@dataclass
class ParametrosModelo:
    # Horizonte temporal (dias) e malha de tempo
    horizonte_dias: int = 365
    pontos_tempo: int = 400

    # Condições iniciais (em bilhões de unidades monetárias)
    ativos_iniciais: float = 100.0
    depositos_iniciais: float = 92.0
    liquidez_inicial: float = 15.0
    p_npl_inicial: float = 0.03      # 3% de inadimplência
    prob_falencia_inicial: float = 0.0

    # Crescimento "normal" (aprox. anual convertido para diário)
    taxa_crescimento_ativos: float = 0.04 / 365.0
    taxa_crescimento_depositos: float = 0.03 / 365.0

    # Risco de crédito e corrida bancária
    choque_credito: float = 0.35     # severidade da perda de crédito (gamma)
    sensibilidade_fuga: float = 2.0  # reação dos depositantes ao CAR baixo (phi)
    car_minimo: float = 0.08         # 8% de capital mínimo regulatório
    theta_liquidez: float = 0.05 / 365.0

    # Dinâmica de NPL
    p_npl_equilibrio: float = 0.04
    taxa_reversao_npl: float = 0.50 / 365.0   # velocidade de reversão ao equilíbrio
    instante_stress: float = 60.0             # início do stress (dias)
    tamanho_stress: float = 0.15 / 365.0      # intensidade do stress em dp_NPL/dt

    # Dinâmica de probabilidade de falência
    velocidade_prob_falencia: float = 3.0 / 365.0

    # Limites numéricos
    p_npl_max: float = 0.6


# ------------------------------------------------------------
# Funções auxiliares
# ------------------------------------------------------------
def calcular_car(ativos: float, depositos: float) -> float:
    """Calcula o CAR = (A - D) / A."""
    if ativos <= 0.0:
        return -1.0
    return (ativos - depositos) / max(ativos, 1e-9)


def stress_npl(t: float, params: ParametrosModelo) -> float:
    """Choque exógeno no NPL após um instante de stress."""
    if t < params.instante_stress:
        return 0.0
    return params.tamanho_stress


def sistema_edo(t: float, y: np.ndarray, params: ParametrosModelo) -> np.ndarray:
    """
    Sistema de EDO:
        y[0] = A(t)       ativos
        y[1] = D(t)       depósitos
        y[2] = L(t)       liquidez
        y[3] = p_NPL(t)   inadimplência
        y[4] = P_fal(t)   probabilidade acumulada de falência
    """
    A, D, L, p_npl, P_fal = y

    # Garante valores em faixas razoáveis
    p_npl = float(np.clip(p_npl, 0.0, params.p_npl_max))
    P_fal = float(np.clip(P_fal, 0.0, 1.0))

    CAR = calcular_car(A, D)
    shortfall = max(params.car_minimo - CAR, 0.0)

    # Perda de crédito proporcional a NPL
    perda_credito = params.choque_credito * p_npl * A

    # Fuga de depósitos se CAR < CAR_minimo
    fuga_depositos = params.sensibilidade_fuga * shortfall * D

    # Reforço de liquidez se CAR > CAR_minimo
    reforco_liquidez = params.theta_liquidez * max(CAR - params.car_minimo, 0.0) * A

    # Dinâmica dos estados
    dA_dt = (
        params.taxa_crescimento_ativos * A
        - perda_credito
        - fuga_depositos
    )

    dD_dt = (
        params.taxa_crescimento_depositos * D
        - fuga_depositos
    )

    dL_dt = (
        - fuga_depositos
        + reforco_liquidez
    )

    dp_npl_dt = (
        params.taxa_reversao_npl * (params.p_npl_equilibrio - p_npl)
        + stress_npl(t, params)
    )

    dP_fal_dt = (
        params.velocidade_prob_falencia * shortfall * (1.0 - P_fal)
    )

    return np.array([dA_dt, dD_dt, dL_dt, dp_npl_dt, dP_fal_dt], dtype=float)


def simular_modelo(params: ParametrosModelo) -> Dict[str, np.ndarray]:
    """Resolve o sistema de EDO ao longo do horizonte especificado."""
    t0 = 0.0
    tf = float(params.horizonte_dias)
    t_eval = np.linspace(t0, tf, params.pontos_tempo)

    y0 = np.array(
        [
            params.ativos_iniciais,
            params.depositos_iniciais,
            params.liquidez_inicial,
            params.p_npl_inicial,
            params.prob_falencia_inicial,
        ],
        dtype=float,
    )

    sol = solve_ivp(
        fun=lambda t, y: sistema_edo(t, y, params),
        t_span=(t0, tf),
        y0=y0,
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-8,
    )

    A = sol.y[0]
    D = sol.y[1]
    L = sol.y[2]
    p_npl = np.clip(sol.y[3], 0.0, params.p_npl_max)
    P_fal = np.clip(sol.y[4], 0.0, 1.0)

    CAR = np.array([calcular_car(a, d) for a, d in zip(A, D)], dtype=float)

    # Detecta instante de falência (qualquer regra disparando)
    falencias = (CAR <= 0.0) | (L <= 0.0) | (A <= D)
    if np.any(falencias):
        indice_fal = int(np.argmax(falencias))
        tempo_fal = sol.t[indice_fal]
    else:
        indice_fal = None
        tempo_fal = None

    return {
        "t": sol.t,
        "ativos": A,
        "depositos": D,
        "liquidez": L,
        "p_npl": p_npl,
        "prob_falencia": P_fal,
        "CAR": CAR,
        "indice_falencia": indice_fal,
        "tempo_falencia": tempo_fal,
    }


# ------------------------------------------------------------
# Dashboard interativo (tela azul, sliders, equações)
# ------------------------------------------------------------
def criar_dashboard():
    params = ParametrosModelo()

    resultados = simular_modelo(params)

    t = resultados["t"]
    A = resultados["ativos"]
    D = resultados["depositos"]
    L = resultados["liquidez"]
    CAR = resultados["CAR"]
    P_fal = resultados["prob_falencia"]
    tempo_falencia: Optional[float] = resultados["tempo_falencia"]

    # Figura principal
    fig = plt.figure(figsize=(12, 7))
    azul_fundo = "#021B3D"
    fig.patch.set_facecolor(azul_fundo)

    gs = fig.add_gridspec(2, 2, height_ratios=[3, 2])

    # --------------------------------------------------------
    # Painel 1: Ativos, Depósitos, Liquidez
    # --------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#041F4F")

    linha_A, = ax1.plot(t, A, label="Ativos A(t)", linewidth=2)
    linha_D, = ax1.plot(t, D, label="Depósitos D(t)", linewidth=2)
    linha_L, = ax1.plot(t, L, label="Liquidez L(t)", linewidth=2, linestyle="--")

    ax1.set_xlabel("Tempo (dias)", color="white")
    ax1.set_ylabel("Valor (unidades)", color="white")
    ax1.set_title("Dinâmica de Ativos, Depósitos e Liquidez", color="white")
    ax1.tick_params(colors="white")

    leg1 = ax1.legend(facecolor="#021B3D", edgecolor="white")
    for text in leg1.get_texts():
        text.set_color("white")

    # Linha vertical para falência (inicialmente invisível)
    linha_fal = ax1.axvline(
        x=0.0,
        color="red",
        linestyle=":",
        linewidth=2,
        visible=False,
    )
    if tempo_falencia is not None:
        linha_fal.set_xdata([tempo_falencia, tempo_falencia])
        linha_fal.set_visible(True)

    # --------------------------------------------------------
    # Painel 2: CAR e Probabilidade de Falência
    # --------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#041F4F")

    linha_CAR, = ax2.plot(t, CAR, label="CAR(t)", linewidth=2)
    linha_Pf, = ax2.plot(t, P_fal, label="Prob. de falência P_fal(t)", linewidth=2)

    linha_car_min = ax2.axhline(
        params.car_minimo,
        color="orange",
        linestyle="--",
        linewidth=1.8,
        label="CAR mínimo",
    )

    ax2.set_xlabel("Tempo (dias)", color="white")
    ax2.set_ylabel("Proporção", color="white")
    ax2.set_title("Risco de Capital e Probabilidade de Falência", color="white")
    ax2.tick_params(colors="white")

    leg2 = ax2.legend(facecolor="#021B3D", edgecolor="white")
    for text in leg2.get_texts():
        text.set_color("white")

    # --------------------------------------------------------
    # Painel 3: Equações diferenciais
    # --------------------------------------------------------
    ax_eq = fig.add_subplot(gs[1, :])
    ax_eq.set_facecolor("#021B3D")
    ax_eq.axis("off")

    texto_eq = (
        r"$\frac{dA}{dt} = g_A A - \gamma p_{NPL} A - \phi \max(CAR_{min} - CAR, 0)\,D$" "\n"
        r"$\frac{dD}{dt} = g_D D - \phi \max(CAR_{min} - CAR, 0)\,D$" "\n"
        r"$\frac{dL}{dt} = - \phi \max(CAR_{min} - CAR, 0)\,D + \theta_L \max(CAR - CAR_{min}, 0)\,A$" "\n"
        r"$\frac{dp_{NPL}}{dt} = k_{NPL}(p_{NPL}^{eq} - p_{NPL}) + s(t)$" "\n"
        r"$\frac{dP_{fal}}{dt} = k_{fal}\max(CAR_{min} - CAR, 0)(1 - P_{fal})$"
    )

    ax_eq.text(
        0.01,
        0.97,
        texto_eq,
        color="white",
        fontsize=11,
        va="top",
    )

    # --------------------------------------------------------
    # Título global
    # --------------------------------------------------------
    if tempo_falencia is not None:
        titulo_global = (
            f"Modelo Dinâmico de Falência Bancária — Falência em {tempo_falencia:.1f} dias"
        )
    else:
        titulo_global = (
            "Modelo Dinâmico de Falência Bancária — Sem falência no horizonte simulado"
        )

    fig.suptitle(
        titulo_global,
        color="white",
        fontsize=14,
        y=0.98,
        fontweight="bold",
    )

    # --------------------------------------------------------
    # Sliders (dashboard interativo)
    # --------------------------------------------------------
    plt.subplots_adjust(left=0.07, right=0.97, top=0.92, bottom=0.15, hspace=0.35)

    eixo_choque = plt.axes([0.07, 0.08, 0.25, 0.02], facecolor="#041F4F")
    eixo_fuga = plt.axes([0.07, 0.05, 0.25, 0.02], facecolor="#041F4F")
    eixo_carmin = plt.axes([0.40, 0.08, 0.25, 0.02], facecolor="#041F4F")
    eixo_ativos0 = plt.axes([0.40, 0.05, 0.25, 0.02], facecolor="#041F4F")

    slider_choque = Slider(
        ax=eixo_choque,
        label="Choque crédito γ",
        valmin=0.05,
        valmax=0.80,
        valinit=params.choque_credito,
        valstep=0.01,
    )

    slider_fuga = Slider(
        ax=eixo_fuga,
        label="Sensib. fuga φ",
        valmin=0.2,
        valmax=5.0,
        valinit=params.sensibilidade_fuga,
        valstep=0.1,
    )

    slider_carmin = Slider(
        ax=eixo_carmin,
        label="CAR mínimo",
        valmin=0.04,
        valmax=0.20,
        valinit=params.car_minimo,
        valstep=0.005,
    )

    slider_ativos0 = Slider(
        ax=eixo_ativos0,
        label="Ativos iniciais A0",
        valmin=50.0,
        valmax=200.0,
        valinit=params.ativos_iniciais,
        valstep=1.0,
    )

    # Deixa texto dos sliders mais legível no fundo escuro
    for s in [slider_choque, slider_fuga, slider_carmin, slider_ativos0]:
        s.label.set_color("white")
        s.valtext.set_color("white")

    # --------------------------------------------------------
    # Função de atualização (recalcula tudo ao mexer nos sliders)
    # --------------------------------------------------------
    def atualizar(_):
        # Atualiza parâmetros
        params.choque_credito = slider_choque.val
        params.sensibilidade_fuga = slider_fuga.val
        params.car_minimo = slider_carmin.val
        params.ativos_iniciais = slider_ativos0.val

        res = simular_modelo(params)

        t_new = res["t"]
        A_new = res["ativos"]
        D_new = res["depositos"]
        L_new = res["liquidez"]
        CAR_new = res["CAR"]
        P_fal_new = res["prob_falencia"]
        tempo_fal_new = res["tempo_falencia"]

        # Atualiza linhas
        linha_A.set_xdata(t_new)
        linha_A.set_ydata(A_new)

        linha_D.set_xdata(t_new)
        linha_D.set_ydata(D_new)

        linha_L.set_xdata(t_new)
        linha_L.set_ydata(L_new)

        linha_CAR.set_xdata(t_new)
        linha_CAR.set_ydata(CAR_new)

        linha_Pf.set_xdata(t_new)
        linha_Pf.set_ydata(P_fal_new)

        linha_car_min.set_ydata([params.car_minimo, params.car_minimo])

        # Linha de falência
        if tempo_fal_new is not None:
            linha_fal.set_xdata([tempo_fal_new, tempo_fal_new])
            linha_fal.set_visible(True)
            novo_titulo = (
                f"Modelo Dinâmico de Falência Bancária — Falência em {tempo_fal_new:.1f} dias"
            )
        else:
            linha_fal.set_visible(False)
            novo_titulo = (
                "Modelo Dinâmico de Falência Bancária — Sem falência no horizonte simulado"
            )

        fig.suptitle(
            novo_titulo,
            color="white",
            fontsize=14,
            y=0.98,
            fontweight="bold",
        )

        # Reajusta e redesenha
        ax1.relim()
        ax1.autoscale_view()

        ax2.relim()
        ax2.autoscale_view()

        fig.canvas.draw_idle()

    # Conecta sliders
    slider_choque.on_changed(atualizar)
    slider_fuga.on_changed(atualizar)
    slider_carmin.on_changed(atualizar)
    slider_ativos0.on_changed(atualizar)

    plt.show()


# ------------------------------------------------------------
# Execução
# ------------------------------------------------------------
if __name__ == "__main__":
    criar_dashboard()
