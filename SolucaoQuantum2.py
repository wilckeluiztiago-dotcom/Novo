# ============================================================
# QUANTUM MACHINE LEARNING (VQLS) PARA EDO/PDE
# Exemplo: Poisson 1D com contorno (u'' = -pi^2 sin(pi x))
# Autor: Luiz Tiago Wilcke (LT)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

from math import pi
from scipy.optimize import minimize

# --- Qiskit (API moderna >= 1.0/2.x)
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector

# ============================================================
# 1) Montar o sistema linear A u = b via diferenças finitas
# ============================================================

def montar_sistema_poisson_1d(n_pontos_internos: int):
    """
    Discretiza u''(x) = f(x) em [0,1], u(0)=u(1)=0
    com diferenças finitas centrais.
    """
    n = n_pontos_internos
    h = 1.0 / (n + 1)

    # Matriz tridiagonal do laplaciano 1D
    diag_principal = 2.0 * np.ones(n)
    diag_sub = -1.0 * np.ones(n - 1)
    A = (1.0 / h**2) * (np.diag(diag_principal) + np.diag(diag_sub, -1) + np.diag(diag_sub, 1))

    # RHS b_i = f(x_i) = -pi^2 sin(pi x_i)
    xs = np.array([(i + 1) * h for i in range(n)])
    f = -(pi**2) * np.sin(pi * xs)
    b = f.copy()

    return A, b, xs, h

# ============================================================
# 2) Converter b em estado quântico |b>
# ============================================================

def normalizar_vetor(v):
    norma = np.linalg.norm(v)
    if norma < 1e-12:
        return v, 1.0
    return v / norma, norma

# ============================================================
# 3) Ansatz variacional (QNN) para aproximar |x(theta)>
# ============================================================

def criar_ansatz_qnn(n_qubits: int, profundidade: int):
    """
    Ansatz tipo hardware-efficient:
    camadas de RY-RZ + entanglement CNOT.
    """
    parametros = ParameterVector("theta", length=n_qubits * 2 * profundidade)
    qc = QuantumCircuit(n_qubits)

    k = 0
    for _ in range(profundidade):
        # rotações locais
        for q in range(n_qubits):
            qc.ry(parametros[k], q); k += 1
            qc.rz(parametros[k], q); k += 1
        # emaranhamento linear
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)

    return qc, parametros

def estado_do_ansatz(qc, parametros, theta_valores):
    bind = dict(zip(parametros, theta_valores))
    qc_b = qc.assign_parameters(bind)
    psi = Statevector.from_instruction(qc_b)
    return np.array(psi.data, dtype=np.complex128)

# ============================================================
# 4) Função custo VQLS:
#     C(theta) = 1 - | <b | A | x(theta)> |^2 / ||A x||^2
# ============================================================

def custo_vqls(theta_valores, qc, parametros, A, b_estado):
    x_estado = estado_do_ansatz(qc, parametros, theta_valores)
    x_real = np.real(x_estado)

    Ax = A @ x_real
    norma_Ax = np.linalg.norm(Ax)
    if norma_Ax < 1e-12:
        return 1.0

    numerador = np.dot(b_estado, Ax)
    custo = 1.0 - (numerador**2) / (norma_Ax**2)
    return float(custo)

# ============================================================
# 5) Treino variacional
# ============================================================

def treinar_vqls(A, b, n_qubits, profundidade, max_iter=200):
    b_estado, norma_b = normalizar_vetor(b)

    qc, parametros = criar_ansatz_qnn(n_qubits, profundidade)

    theta_inicial = 2 * pi * np.random.rand(len(parametros))

    resultado = minimize(
        fun=lambda th: custo_vqls(th, qc, parametros, A, b_estado),
        x0=theta_inicial,
        method="COBYLA",
        options={"maxiter": max_iter, "disp": True}
    )

    theta_otimo = resultado.x
    x_estado = estado_do_ansatz(qc, parametros, theta_otimo)
    x_real = np.real(x_estado)

    # Escala ótima alpha para aproximar solução real u:
    Ax = A @ x_real
    alpha = np.dot(b_estado, Ax) / np.dot(Ax, Ax)
    u_quantico = alpha * x_real

    return u_quantico, theta_otimo, resultado

# ============================================================
# 6) Rodar experimento (4 pontos internos -> 4x4 -> 2 qubits)
# ============================================================

if __name__ == "__main__":
    np.random.seed(0)

    n_pontos_internos = 4  # 4 incógnitas => 2 qubits
    A, b, xs, h = montar_sistema_poisson_1d(n_pontos_internos)

    n_qubits = int(np.log2(n_pontos_internos))
    profundidade = 3

    u_quantico, theta_otimo, resultado = treinar_vqls(
        A, b, n_qubits=n_qubits, profundidade=profundidade, max_iter=250
    )

    # --- solução clássica de referência
    u_classico = np.linalg.solve(A, b)

    # --- solução exata
    u_exato = np.sin(pi * xs)

    print("\n================ RESULTADOS NUMÉRICOS ================")
    for i, x in enumerate(xs):
        print(f"x={x:.6f} | u_quantico={u_quantico[i]: .8f} | "
              f"u_classico={u_classico[i]: .8f} | u_exato={u_exato[i]: .8f}")

    erro_q = np.linalg.norm(u_quantico - u_exato) / np.linalg.norm(u_exato)
    erro_c = np.linalg.norm(u_classico - u_exato) / np.linalg.norm(u_exato)

    print("\nErro relativo (quântico): ", f"{erro_q:.6e}")
    print("Erro relativo (clássico): ", f"{erro_c:.6e}")

    # --- gráfico
    plt.figure()
    plt.plot(xs, u_exato, "o-", label="Exato")
    plt.plot(xs, u_classico, "s--", label="Clássico (FD)")
    plt.plot(xs, u_quantico, "d-.", label="Quântico (VQLS/QNN)")
    plt.title("Poisson 1D: solução por QML (VQLS)")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.legend()
    plt.grid(True)
    plt.show()
