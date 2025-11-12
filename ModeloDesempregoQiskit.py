# ============================================================
# MODELO QUÂNTICO-ESTOCÁSTICO DO DESEMPREGO
#   — Ornstein–Uhlenbeck em logit + Qiskit (distribuição quântica)
#   — Autor: Luiz Tiago Wilcke (LT)
# ============================================================

import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# Importações Qiskit (compatíveis com 1.x)
# ------------------------------------------------------------
TEM_QISKIT = True
MOTIVO_QISKIT_INDISPONIVEL = ""

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import Aer
    from qiskit.circuit.library import StatePreparation
except Exception as e:
    TEM_QISKIT = False
    MOTIVO_QISKIT_INDISPONIVEL = str(e)

# ============================================================
# 1) Modelo Estocástico do Desemprego
#    dX_t = κ(θ - X_t) dt + σ dW_t   (processo OU)
#    U_t = 1 / (1 + exp(-X_t))       (taxa de desemprego)
# ============================================================

@dataclass
class ParametrosDesemprego:
    taxa_inicial: float = 0.11           # 11% de desemprego no início
    taxa_longo_prazo: float = 0.09       # nível de equilíbrio de longo prazo (~9%)
    kappa: float = 1.2                   # velocidade de reversão à média
    sigma: float = 0.35                  # volatilidade da dinâmica em logit
    horizonte_meses: int = 24            # horizonte de previsão (em meses)
    caminhos: int = 30000                # número de trajetórias de Monte Carlo
    limiar_desemprego_alto: float = 0.14 # 14%: desemprego considerado "alto"


@dataclass
class ParametrosQuanticos:
    n_qubits: int = 7            # 2^7 = 128 pontos na grade
    shots: int = 4096            # número de medições no simulador
    u_min: Optional[float] = None  # mínimo da taxa na grade (se None, é estimado)
    u_max: Optional[float] = None  # máximo da taxa na grade (se None, é estimado)


def logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def simular_caminhos_desemprego(param: ParametrosDesemprego) -> np.ndarray:
    """
    Simula a dinâmica do desemprego:
      - Processo OU em X_t (logit da taxa)
      - U_t = logistic(X_t)
    Retorna matriz [caminhos x (passos+1)] com trajetórias de U_t.
    """
    dt = 1.0 / 12.0  # passo mensal em "anos"
    passos = param.horizonte_meses
    n = param.caminhos

    caminhos = np.zeros((n, passos + 1))

    # Estado inicial em logit
    x0 = logit(param.taxa_inicial)
    theta_x = logit(param.taxa_longo_prazo)

    x = np.full(n, x0)
    u = 1.0 / (1.0 + np.exp(-x))
    caminhos[:, 0] = u

    for t in range(1, passos + 1):
        z = np.random.normal(size=n)
        x = x + param.kappa * (theta_x - x) * dt + param.sigma * math.sqrt(dt) * z
        u = 1.0 / (1.0 + np.exp(-x))
        caminhos[:, t] = u

    return caminhos

# ============================================================
# 2) Preparação do Estado Quântico da Distribuição de U_T
# ============================================================

def preparar_estado_desemprego(
    u_T: np.ndarray,
    param_q: ParametrosQuanticos
) -> Tuple["StatePreparation", np.ndarray, np.ndarray]:
    """
    A partir dos valores simulados de U_T (taxa de desemprego no horizonte),
    constrói uma distribuição discreta em 2^n_qubits pontos e prepara
    o circuito de StatePreparation correspondente.
    """
    n_qubits = param_q.n_qubits
    N = 2 ** n_qubits

    # Definição da faixa de desemprego na grade (se não fornecida)
    if param_q.u_min is not None:
        u_min = param_q.u_min
    else:
        u_min = max(0.01, float(np.quantile(u_T, 0.001)))

    if param_q.u_max is not None:
        u_max = param_q.u_max
    else:
        u_max = min(0.40, float(np.quantile(u_T, 0.999)))

    # Grade de desemprego
    grade = np.linspace(u_min, u_max, N)

    # Histograma discreto com N bins
    hist, bordas = np.histogram(u_T, bins=N, range=(u_min, u_max), density=False)
    pdf = hist.astype(float) + 1e-12  # evitar zeros
    pdf /= pdf.sum()

    # Amplitudes quânticas: |ψ> = Σ sqrt(pdf_i) |i>
    amplitudes = np.sqrt(pdf)
    circuito_estado = StatePreparation(amplitudes)

    return circuito_estado, grade, pdf

# ============================================================
# 3) Estimação Quântica: E[U_T] e P(U_T >= limiar)
# ============================================================

def estimar_quanticamente_desemprego(
    u_T: np.ndarray,
    param_des: ParametrosDesemprego,
    param_q: ParametrosQuanticos
):
    if not TEM_QISKIT:
        raise RuntimeError(
            f"Qiskit/Aer indisponível neste ambiente: {MOTIVO_QISKIT_INDISPONIVEL}"
        )

    estado, grade, pdf_discreta = preparar_estado_desemprego(u_T, param_q)

    n_qubits = param_q.n_qubits
    shots = param_q.shots

    qc = QuantumCircuit(n_qubits)
    qc.append(estado, range(n_qubits))
    qc.measure_all()

    backend = Aer.get_backend("qasm_simulator")
    job = backend.run(transpile(qc, backend), shots=shots)
    resultado = job.result()
    contagens = resultado.get_counts()

    # Reconstrução de probabilidades respeitando a convenção de bits do Qiskit
    N = 2 ** n_qubits
    probs = np.zeros(N)
    for bitstring, c in contagens.items():
        # Qiskit usa ordem little-endian; invertemos a string para indexar
        idx = int(bitstring[::-1], 2)
        probs[idx] = c / shots

    # E[U_T] quântico
    E_quantico = float(np.sum(probs * grade))

    # P(U_T >= limiar) quântico
    limiar = param_des.limiar_desemprego_alto
    mascara_alto = grade >= limiar
    P_alto_quantic = float(np.sum(probs[mascara_alto]))

    return E_quantico, P_alto_quantic, grade, probs, contagens

# ============================================================
# 4) Execução Principal: Clássico + Quântico
# ============================================================

def executar_modelo():
    np.random.seed(42)

    # --------------------------
    # 4.1 Parâmetros
    # --------------------------
    param_des = ParametrosDesemprego()
    param_q = ParametrosQuanticos()

    # --------------------------
    # 4.2 Simulação clássica
    # --------------------------
    caminhos = simular_caminhos_desemprego(param_des)
    U_T = caminhos[:, -1]

    E_classico = float(np.mean(U_T))
    IC95 = np.quantile(U_T, [0.025, 0.975])
    P_alto_classico = float(np.mean(U_T >= param_des.limiar_desemprego_alto))

    # --------------------------
    # 4.3 Estimação quântica
    # --------------------------
    if TEM_QISKIT:
        try:
            E_quantico, P_alto_quantic, grade, probs, contagens = \
                estimar_quanticamente_desemprego(U_T, param_des, param_q)
            parte_quantica_ok = True
        except Exception as e:
            parte_quantica_ok = False
            motivo_erro = str(e)
    else:
        parte_quantica_ok = False
        motivo_erro = MOTIVO_QISKIT_INDISPONIVEL

    # --------------------------
    # 4.4 Impressão dos resultados
    # --------------------------
    print("\n================== MODELO QUÂNTICO-ESTOCÁSTICO DO DESEMPREGO ==================")
    print(f"Taxa inicial de desemprego (U0):     {param_des.taxa_inicial*100:.4f}%")
    print(f"Taxa de longo prazo (θ):             {param_des.taxa_longo_prazo*100:.4f}%")
    print(f"Velocidade de reversão (κ):          {param_des.kappa:.4f}")
    print(f"Volatilidade (σ):                    {param_des.sigma:.4f}")
    print(f"Horizonte:                           {param_des.horizonte_meses} meses")
    print(f"Caminhos Monte Carlo:                {param_des.caminhos}")
    print(f"Limiar desemprego alto:              {param_des.limiar_desemprego_alto*100:.4f}%")
    print("-------------------------------------------------------------------------------")
    print(f"E[U_T] clássico (MC):                {E_classico*100:.4f}%")
    print(f"IC95% clássico:                      [{IC95[0]*100:.4f}% ; {IC95[1]*100:.4f}%]")
    print(f"P(U_T >= limiar)_clássico:           {P_alto_classico*100:.4f}%")

    if parte_quantica_ok:
        print("-------------------------------------------------------------------------------")
        print(f"E[U_T] quântico (Qiskit):            {E_quantico*100:.4f}%")
        print(f"P(U_T >= limiar)_quântico:           {P_alto_quantic*100:.4f}%")
        print(f"Número de qubits:                    {param_q.n_qubits}")
        print(f"Shots no simulador:                  {param_q.shots}")
    else:
        print("-------------------------------------------------------------------------------")
        print("[Aviso] Parte quântica indisponível neste ambiente.")
        print(f"Motivo (Qiskit/Aer): {motivo_erro}")

    # --------------------------
    # 4.5 Gráficos
    # --------------------------
    plt.figure(figsize=(11, 5))

    # Histograma clássico
    plt.subplot(1, 2, 1)
    plt.hist(U_T * 100, bins=60, density=True, alpha=0.7, label="Monte Carlo clássico")
    plt.axvline(E_classico * 100, color="blue", linestyle="--",
                label=f"Média clássica = {E_classico*100:.2f}%")
    plt.xlabel("Taxa de desemprego no horizonte (%)")
    plt.ylabel("Densidade de probabilidade")
    plt.title("Distribuição clássica — U_T (desemprego)")
    plt.legend()

    # Distribuição quântica (se disponível)
    plt.subplot(1, 2, 2)
    if parte_quantica_ok:
        plt.bar(grade * 100, probs, width=(grade[1] - grade[0]) * 100,
                alpha=0.8, label="Distribuição quântica (simulada)")
        plt.axvline(E_quantico * 100, color="red", linestyle=":",
                    label=f"Média quântica = {E_quantico*100:.2f}%")
        plt.axvline(param_des.limiar_desemprego_alto * 100,
                    color="black", linestyle="--",
                    label=f"Limiar desemprego alto = {param_des.limiar_desemprego_alto*100:.2f}%")
        plt.title("Distribuição quântica — U_T (desemprego)")
        plt.xlabel("Taxa de desemprego no horizonte (%)")
        plt.ylabel("Probabilidade (aprox.)")
        plt.legend()
    else:
        plt.text(0.5, 0.5,
                 "Parte quântica indisponível\n(Apenas Monte Carlo clássico)",
                 ha="center", va="center", transform=plt.gca().transAxes)
        plt.axis("off")

    plt.suptitle("Modelo Quântico-Estocástico do Desemprego — Luiz Tiago Wilcke (LT)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    executar_modelo()
