# ============================================================
# MODELO DE SALÁRIO DE EFICIÊNCIA — SHAPIRO–STIGLITZ
# Autor: Luiz Tiago Wilcke (LT)
# ============================================================
# Ideia central:
#   - As firmas não conseguem monitorar esforço o tempo todo.
#   - Trabalhador pode:
#       * Fazer esforço (não fraudar)  -> custo de esforço 'custo_esforco'
#       * Fraudar (não se esforçar)    -> evita o custo, mas corre risco de ser demitido.
#   - Para induzir esforço, a firma paga um SALÁRIO DE EFICIÊNCIA acima do de mercado.
#   - Isso gera desemprego em equilíbrio -> ameaça de ficar desempregado disciplina o trabalhador.
#
# Estados do trabalhador:
#   - Empregado e se esforçando:        V_e
#   - Empregado e fraudando (shirking): V_s
#   - Desempregado:                     V_u
#
# Bellman (tempo contínuo):
#   r V_e = w - c + s (V_u - V_e)
#   r V_s = w     + (s + q)(V_u - V_s)
#   r V_u = b     + f (V_e - V_u)
#
# onde:
#   r  : taxa de desconto intertemporal
#   s  : taxa de separação exógena do emprego
#   q  : taxa de monitoramento (descoberta de fraude)
#   f  : taxa de encontro de emprego (depende do desemprego u)
#   w  : salário pago pela firma
#   c  : custo de esforço (custo_esforco)
#   b  : benefício no desemprego (beneficio_desemprego)
#
# Condição de Não-Fraude (NSC): V_e >= V_s
#   -> resolvendo V_e = V_s obtemos o salário mínimo de eficiência:
#
#   w_NSC(u) = beneficio_desemprego
#              + custo_esforco * [f(u) + taxa_monitoramento + taxa_desconto + taxa_separacao]
#                                                / taxa_monitoramento
#
#   com f(u) = taxa_separacao * (1 - u) / u (equilíbrio do fluxo: f u = s (1-u))
#
# Equilíbrio:
#   - Curva NSC: w_NSC(u) (crescente em (1/u))
#   - Demanda de trabalho: aqui tomamos produtividade 'produtividade' por trabalhador,
#     zero-lucro competitivo => w <= produtividade.
#   - Em equilíbrio: w = produtividade = w_NSC(u*)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable, Tuple, Optional


# ------------------------------------------------------------
# 1) Parâmetros do modelo (todos em português)
# ------------------------------------------------------------
@dataclass
class ParametrosModelo:
    taxa_desconto: float = 0.05          # r
    taxa_separacao: float = 0.02         # s
    taxa_monitoramento: float = 0.10     # q
    beneficio_desemprego: float = 0.40   # b
    custo_esforco: float = 0.40          # c
    produtividade: float = 1.20          # y (produto por trabalhador)
    populacao_trabalhadores: int = 10000 # N (para simulações, se quiser)


# ------------------------------------------------------------
# 2) Funções auxiliares do modelo
# ------------------------------------------------------------

def taxa_encontro_emprego(desemprego: float, p: ParametrosModelo) -> float:
    """
    f(u) = s (1 - u) / u  (fluxo de separações = fluxo de contratações)
    """
    u = np.clip(desemprego, 1e-6, 1 - 1e-6)
    return p.taxa_separacao * (1.0 - u) / u


def salario_nsc(desemprego: float, p: ParametrosModelo) -> float:
    """
    Salário mínimo de eficiência imposto pela condição de não-fraude (NSC).
    w_NSC(u) = b + c * [f(u) + q + r + s] / q
    """
    f_u = taxa_encontro_emprego(desemprego, p)
    return (
        p.beneficio_desemprego
        + p.custo_esforco * (f_u + p.taxa_monitoramento + p.taxa_desconto + p.taxa_separacao)
        / p.taxa_monitoramento
    )


def excesso_salario(desemprego: float, p: ParametrosModelo) -> float:
    """
    Função para achar o desemprego de equilíbrio:
        excesso_salario(u) = w_NSC(u) - produtividade
    Procuramos u* tal que excesso_salario(u*) = 0
    """
    return salario_nsc(desemprego, p) - p.produtividade


def encontrar_raiz_bissecao(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-6,
    max_iter: int = 10_000
) -> Optional[float]:
    """
    Método de bisseção simples para encontrar raiz de f no intervalo [a, b].
    Retorna None se não houver mudança de sinal.
    """
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        return None  # sem raiz no intervalo

    esquerda, direita = a, b
    for _ in range(max_iter):
        meio = 0.5 * (esquerda + direita)
        fm = f(meio)
        if abs(fm) < tol:
            return meio
        if fa * fm < 0:
            direita = meio
            fb = fm
        else:
            esquerda = meio
            fa = fm
    return meio


# ------------------------------------------------------------
# 3) Valores presentes: V_e (não fraude) e V_u (desempregado)
# ------------------------------------------------------------

def valor_presente_empregado_nao_fraudando(
    salario: float,
    desemprego: float,
    p: ParametrosModelo
) -> float:
    """
    Fórmula fechada obtida das equações de Bellman:

    r V_e = w - c + s (V_u - V_e)
    r V_u = b + f (V_e - V_u)

    Resolvendo o sistema, temos:

      V_e = (b*s - c*f - c*r + f*w + r*w) / (r * (f + r + s))

    onde:
      f = taxa_encontro_emprego(u)
      b = beneficio_desemprego
      c = custo_esforco
    """
    r = p.taxa_desconto
    s = p.taxa_separacao
    b = p.beneficio_desemprego
    c = p.custo_esforco
    f = taxa_encontro_emprego(desemprego, p)

    numerador = b * s - c * f - c * r + f * salario + r * salario
    denominador = r * (f + r + s)
    return numerador / denominador


def valor_presente_desempregado(
    salario: float,
    desemprego: float,
    p: ParametrosModelo
) -> float:
    """
    Também da solução do sistema:

      V_u = (b*r + b*s - c*f + f*w) / (r * (f + r + s))
    """
    r = p.taxa_desconto
    s = p.taxa_separacao
    b = p.beneficio_desemprego
    c = p.custo_esforco
    f = taxa_encontro_emprego(desemprego, p)

    numerador = b * r + b * s - c * f + f * salario
    denominador = r * (f + r + s)
    return numerador / denominador


# ------------------------------------------------------------
# 4) Simulação de transição emprego-desemprego (discreta)
# ------------------------------------------------------------

def simular_dinamica_emprego(
    p: ParametrosModelo,
    salario: float,
    desemprego_inicial: float = 0.2,
    dt: float = 0.1,
    horizonte: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simula a fração de empregados ao longo do tempo com dinâmica aproximada:

        dE/dt = f(u) * u - s * E
        u = 1 - E

    onde E é a proporção empregada.
    Discretização de Euler: E_{t+dt} = E_t + dE/dt * dt
    """
    num_passos = int(horizonte / dt)
    tempo = np.linspace(0.0, horizonte, num_passos + 1)
    E = np.empty(num_passos + 1)
    E[0] = 1.0 - desemprego_inicial

    for t in range(num_passos):
        u_t = 1.0 - E[t]
        f_t = taxa_encontro_emprego(u_t, p)
        dE_dt = f_t * u_t - p.taxa_separacao * E[t]
        E[t + 1] = np.clip(E[t] + dE_dt * dt, 0.0, 1.0)

    return tempo, E


# ------------------------------------------------------------
# 5) Rotina principal: calcula equilíbrio e plota curva NSC
# ------------------------------------------------------------

def executar_modelo():
    # Parâmetros padrão (podem ser alterados)
    p = ParametrosModelo()

    # Achar desemprego de equilíbrio (u*) tal que w_NSC(u*) = produtividade
    f_para_raiz = lambda u: excesso_salario(u, p)
    u_min, u_max = 1e-4, 0.999
    u_estrela = encontrar_raiz_bissecao(f_para_raiz, u_min, u_max)

    if u_estrela is None:
        print("Não foi possível encontrar um desemprego de equilíbrio no intervalo (0,1).")
        print("Tente ajustar os parâmetros do modelo.")
        return

    w_estrela = salario_nsc(u_estrela, p)
    f_estrela = taxa_encontro_emprego(u_estrela, p)
    Ve = valor_presente_empregado_nao_fraudando(w_estrela, u_estrela, p)
    Vu = valor_presente_desempregado(w_estrela, u_estrela, p)

    # --------------------------------------------------------
    # Saída numérica básica
    # --------------------------------------------------------
    print("===============================================")
    print("  MODELO DE SALÁRIO DE EFICIÊNCIA (Shapiro–Stiglitz)")
    print("===============================================")
    print(f"Produtividade (y):              {p.produtividade:.4f}")
    print(f"Desemprego de equilíbrio (u*):  {u_estrela:.4f}")
    print(f"Emprego de equilíbrio (1-u*):   {1.0 - u_estrela:.4f}")
    print(f"Salário de eficiência (w*):     {w_estrela:.4f}")
    print(f"Taxa de encontro de emprego f*: {f_estrela:.4f}")
    print("-----------------------------------------------")
    print(f"Valor presente V_e (empregado, esforço):   {Ve:.4f}")
    print(f"Valor presente V_u (desempregado):         {Vu:.4f}")
    print("OBS: Em equilíbrio com NSC, V_e = V_s (indiferença entre fraudar e não).")
    print("===============================================")

    # --------------------------------------------------------
    # Gráfico da curva NSC e da produtividade
    # --------------------------------------------------------
    grades_u = np.linspace(0.02, 0.98, 300)
    salarios_nsc = np.array([salario_nsc(u, p) for u in grades_u])

    plt.figure(figsize=(8, 5))
    plt.plot(grades_u, salarios_nsc, label="Salário mínimo de eficiência w_NSC(u)")
    plt.axhline(p.produtividade, linestyle="--", label=f"Produtividade y = {p.produtividade:.2f}")
    plt.axvline(u_estrela, linestyle=":", label=f"u* = {u_estrela:.2f}")
    plt.scatter([u_estrela], [w_estrela], zorder=5, label="Equilíbrio (u*, w*)")

    plt.xlabel("Taxa de desemprego u")
    plt.ylabel("Salário real w")
    plt.title("Curva de Salário de Eficiência (NSC) — Shapiro–Stiglitz")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # Simulação opcional da dinâmica de emprego (com w = w*)
    # --------------------------------------------------------
    tempo, E = simular_dinamica_emprego(
        p,
        salario=w_estrela,
        desemprego_inicial=0.5  # começa em 50% e converge para 1-u*
    )

    plt.figure(figsize=(8, 4))
    plt.plot(tempo, 1.0 - E, label="Desemprego simulado u(t)")
    plt.axhline(u_estrela, linestyle="--", label=f"u* teórico = {u_estrela:.2f}")
    plt.xlabel("Tempo (unidades arbitrárias)")
    plt.ylabel("Taxa de desemprego u(t)")
    plt.title("Dinâmica aproximada emprego-desemprego")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    executar_modelo()
