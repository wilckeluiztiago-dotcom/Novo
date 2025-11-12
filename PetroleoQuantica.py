# ============================================================
# Previsão Quântica do Preço do Petróleo (WTI/Brent)
#   — Modelo Estocástico GBM + Difusão Quântica via Qiskit 1.x
#   — Autor: Luiz Tiago Wilcke (LT)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import math
from dataclasses import dataclass
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.circuit.library import StatePreparation, RYGate

# ------------------------------------------------------------
# 1) Modelo Estocástico (GBM)
# ------------------------------------------------------------
@dataclass
class ParametrosSDE:
    preco_inicial: float = 82.5
    mu: float = 0.08
    sigma: float = 0.32
    dias_uteis_ano: int = 252
    horizonte_dias: int = 20
    amostras: int = 20000

def simular_gbm(param: ParametrosSDE):
    dt = 1.0 / param.dias_uteis_ano
    passos = param.horizonte_dias
    caminhos = np.zeros((param.amostras, passos + 1))
    caminhos[:, 0] = param.preco_inicial
    ln_S = np.full(param.amostras, math.log(param.preco_inicial))
    for t in range(1, passos + 1):
        z = np.random.normal(size=param.amostras)
        ln_S += (param.mu - 0.5 * param.sigma**2) * dt + param.sigma * math.sqrt(dt) * z
        caminhos[:, t] = np.exp(ln_S)
    return caminhos

# ------------------------------------------------------------
# 2) Preparação do Estado Quântico
# ------------------------------------------------------------
def preparar_estado_lognormal(n_qubits, preco0, mu, sigma, dias, s_min, s_max):
    T = dias / 252
    mu_log = math.log(preco0) + (mu - 0.5 * sigma**2) * T
    sigma_log = sigma * math.sqrt(T)

    N = 2 ** n_qubits
    s_vals = np.linspace(s_min, s_max, N)
    pdf = np.exp(-((np.log(s_vals + 1e-9) - mu_log)**2) / (2 * sigma_log**2))
    pdf /= np.sum(pdf)
    amplitudes = np.sqrt(pdf)
    circuito_estado = StatePreparation(amplitudes)
    return circuito_estado, s_vals, pdf

# ------------------------------------------------------------
# 3) Operador de Difusão Quântica
# ------------------------------------------------------------
def operador_difusao(n_qubits, sigma, tempo):
    qc = QuantumCircuit(n_qubits)
    for q in range(n_qubits):
        ang = 2 * np.arctan(np.tanh(sigma * tempo / (q + 1)))
        qc.ry(ang, q)
    return qc

# ------------------------------------------------------------
# 4) Estimação Quântica
# ------------------------------------------------------------
def estimar_quanticamente(preco0, mu, sigma, dias, n_qubits=7):
    s_min, s_max = 30.0, 160.0
    estado, s_vals, pdf = preparar_estado_lognormal(n_qubits, preco0, mu, sigma, dias, s_min, s_max)
    difusao = operador_difusao(n_qubits, sigma, dias / 252)

    qc = QuantumCircuit(n_qubits)
    qc.append(estado, range(n_qubits))
    qc.append(difusao, range(n_qubits))
    qc.measure_all()

    backend = Aer.get_backend("qasm_simulator")
    job = backend.run(transpile(qc, backend), shots=4096)
    result = job.result()
    counts = result.get_counts()
    probs = np.array([counts.get(f"{i:0{n_qubits}b}", 0) for i in range(2**n_qubits)]) / 4096

    E_quantico = np.sum(probs * s_vals)
    return E_quantico, counts, s_vals, probs

# ------------------------------------------------------------
# 5) Execução Principal
# ------------------------------------------------------------
def executar_modelo():
    ps = ParametrosSDE()
    caminhos = simular_gbm(ps)
    ST = caminhos[:, -1]
    E_classico = np.mean(ST)
    IC95 = np.quantile(ST, [0.025, 0.975])

    E_quantico, counts, s_vals, probs = estimar_quanticamente(ps.preco_inicial, ps.mu, ps.sigma, ps.horizonte_dias)

    print("\n================== RESULTADOS ==================")
    print(f"Preço inicial (S0):            {ps.preco_inicial:.8f}")
    print(f"Drift anual (μ):               {ps.mu:.8f}")
    print(f"Volatilidade anual (σ):        {ps.sigma:.8f}")
    print(f"Horizonte (dias úteis):        {ps.horizonte_dias}")
    print("------------------------------------------------")
    print(f"E[S_T] clássico (Monte Carlo): {E_classico:.8f}")
    print(f"IC95% clássico: [{IC95[0]:.8f} , {IC95[1]:.8f}]")
    print(f"E[S_T] quântico (Difusão Qiskit): {E_quantico:.8f}")

    plt.figure(figsize=(10,5))
    plt.hist(ST, bins=80, density=True, alpha=0.5, label="Monte Carlo Clássico")
    plt.plot(s_vals, probs/np.max(probs)*np.max(np.histogram(ST, bins=80, density=True)[0]),
             color="red", label="Distribuição Quântica (simulada)")
    plt.axvline(E_classico, color="blue", linestyle="--", label=f"E[S_T] clássico={E_classico:.2f}")
    plt.axvline(E_quantico, color="red", linestyle=":", label=f"E[S_T] quântico={E_quantico:.2f}")
    plt.legend()
    plt.title("Previsão Estocástica e Quântica do Preço do Petróleo — Luiz Tiago Wilcke (LT)")
    plt.xlabel("Preço final simulado (USD)")
    plt.ylabel("Densidade de probabilidade")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    executar_modelo()
