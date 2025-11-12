# ============================================================
# Precificação Quântica — Opção Europeia 
# Autor: Luiz Tiago Wilcke (LT)
# ============================================================

import warnings, inspect
warnings.filterwarnings("ignore")

from dataclasses import dataclass
from typing import Optional, Tuple, Literal
import numpy as np
from math import log, sqrt, exp

# ---------------- Primitives: detectar V2 vs V1 ----------------
_PRIMS = "v1"
SamplerV1 = None
try:
    # Terra novas: SamplerV2
    from qiskit.primitives import SamplerV2 as _SamplerV2
    _PRIMS = "v2"
except Exception:
    _SamplerV2 = None

if _PRIMS == "v2":
    Sampler = _SamplerV2
else:
    try:
        from qiskit.primitives import Sampler as SamplerV1  # V1 base
    except Exception:
        from qiskit_aer.primitives import Sampler as SamplerV1  # fallback Aer
    Sampler = SamplerV1

# ---------------- Algorithms (usaremos se PRIMS==v2) -----------
_USE_ALGOS = True
try:
    from qiskit_algorithms import (
        IterativeAmplitudeEstimation as IAE,
        EstimationProblem,
    )
    from qiskit_algorithms import MaximumLikelihoodAmplitudeEstimation as MLAE
except Exception:
    _USE_ALGOS = False
    EstimationProblem = None  # type: ignore
    MLAE = None  # type: ignore
    IAE = None   # type: ignore

# ---------------- Finance ----------------
from qiskit_finance.circuit.library import LogNormalDistribution
try:
    from qiskit_finance.circuit.library.payoff_functions import EuropeanCallExpectedValue
    _PAYOFF_MODE = "finance"
except Exception:
    EuropeanCallExpectedValue = None  # type: ignore
    _PAYOFF_MODE = "manual"

# ============================================================
# Parâmetros
# ============================================================

@dataclass
class ParametrosMercado:
    S0: float = 100.0
    K: float = 100.0
    r: float = 0.05
    sigma: float = 0.2
    T: float = 1.0

@dataclass
class ParametrosQAE:
    n_qubits: int = 4
    low: float = 0.0
    high: float = 200.0
    fator_reescala: float = 200.0
    tiros: Optional[int] = 20_000
    algoritmo: Literal["MLAE", "IAE"] = "MLAE"
    avaliacao_mlae: int = 4
    epsilon_rel_iae: float = 0.01
    alpha_iae: float = 0.05

# ============================================================
# Black-Scholes (checagem)
# ============================================================

def d1_d2(S0, K, r, sigma, T):
    d1 = (log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    return d1, d2

def call_bs(S0, K, r, sigma, T):
    from scipy.stats import norm
    d1, d2 = d1_d2(S0, K, r, sigma, T)
    return S0*norm.cdf(d1) - K*exp(-r*T)*norm.cdf(d2)

# ============================================================
# Precificador
# ============================================================

class PrecificadorQuantico:
    def __init__(self, pm: ParametrosMercado, pq: ParametrosQAE):
        from qiskit import QuantumCircuit
        self.pm, self.pq = pm, pq
        self.sampler = Sampler()

        mu = log(self.pm.S0) + (self.pm.r - 0.5*self.pm.sigma**2) * self.pm.T
        sigma_ln = sqrt((self.pm.sigma**2) * self.pm.T)

        # Compatibilidade num_qubits / num_state_qubits
        try:
            self.dist = LogNormalDistribution(
                num_qubits=self.pq.n_qubits,
                mu=mu, sigma=sigma_ln,
                bounds=(self.pq.low, self.pq.high)
            )
        except TypeError:
            self.dist = LogNormalDistribution(
                num_state_qubits=self.pq.n_qubits,
                mu=mu, sigma=sigma_ln,
                bounds=(self.pq.low, self.pq.high)
            )

        self._QuantumCircuit = QuantumCircuit

    # --------- Construção do circuito A + qubit objetivo ----------
    def circuito_payoff(self):
        qc = self._QuantumCircuit(self.pq.n_qubits + 1)
        qc.compose(self.dist, range(self.pq.n_qubits), inplace=True)

        # Payoff pronto se disponível; senão LinearAmplitudeFunction
        if EuropeanCallExpectedValue is not None and _PAYOFF_MODE == "finance":
            try:
                payoff = EuropeanCallExpectedValue(
                    num_state_qubits=self.pq.n_qubits,
                    strike_price=self.pm.K,
                    rescaling_factor=self.pq.fator_reescala,
                    bounds=(self.pq.low, self.pq.high),
                )
            except TypeError:
                payoff = EuropeanCallExpectedValue(
                    strike_price=self.pm.K,
                    rescaling_factor=self.pq.fator_reescala,
                    bounds=(self.pq.low, self.pq.high),
                )
            qc.compose(payoff, range(self.pq.n_qubits + 1), inplace=True)
        else:
            try:
                from qiskit.circuit.library import LinearAmplitudeFunction
            except Exception:
                # último fallback: step linear manual simples (digital)
                # marca S>=K com prob ~ min((S-K)/R,1)
                grid = np.linspace(self.pq.low, self.pq.high, 2**self.pq.n_qubits)
                k_idx = np.searchsorted(grid, self.pm.K, side="right")
                alvo = self.pq.n_qubits
                for idx, s_val in enumerate(grid):
                    if idx < k_idx:
                        continue
                    ganho = max(s_val - self.pm.K, 0.0) / max(self.pq.fator_reescala, 1e-9)
                    ganho = max(0.0, min(ganho, 1.0))
                    theta = 2*np.arcsin(np.sqrt(ganho))
                    bits = format(idx, f"0{self.pq.n_qubits}b")
                    for q, b in enumerate(bits):
                        if b == "0": qc.x(q)
                    qc.mcry(float(theta), list(range(self.pq.n_qubits)), alvo)
                    for q, b in enumerate(bits):
                        if b == "0": qc.x(q)
            else:
                slope = 1.0 / max(self.pq.fator_reescala, 1e-9)
                offset = -self.pm.K / max(self.pq.fator_reescala, 1e-9)
                f = LinearAmplitudeFunction(
                    num_state_qubits=self.pq.n_qubits,
                    slope=slope,
                    offset=offset,
                    domain=(self.pq.low, self.pq.high),
                    image=(0, 1),
                )
                qc.compose(f, range(self.pq.n_qubits + 1), inplace=True)

        alvo = self.pq.n_qubits
        return qc, alvo

    # --------- Estimação de amplitude (V2 com algos; V1 fallback) ---------
    def estimar_amplitude(self) -> float:
        qc, alvo = self.circuito_payoff()

        # Caminho “bonito” com MLAE/IAE só se PRIMS==v2 e algos disponíveis
        if _PRIMS == "v2" and _USE_ALGOS and EstimationProblem is not None:
            problema = EstimationProblem(state_preparation=qc, objective_qubits=[alvo])
            try:
                if self.pq.algoritmo == "MLAE":
                    ae = MLAE(evaluation_schedule=self.pq.avaliacao_mlae, sampler=self.sampler)
                    res = ae.estimate(problema)  # V2 não usa 'shots' aqui
                    return float(res.estimation)
                else:
                    ae = IAE(epsilon_rel=self.pq.epsilon_rel_iae, alpha=self.pq.alpha_iae, sampler=self.sampler)
                    res = ae.estimate(problema)
                    return float(res.estimation)
            except Exception:
                # cai para amostragem direta
                pass

        # ---------- Fallback robusto: amostragem direta do qubit-objetivo ----------
        # Precisamos medir o qubit-objetivo.
        qc_meas = qc.copy()
        from qiskit.circuit import ClassicalRegister
        cr = ClassicalRegister(1, "c_obj")
        qc_meas.add_register(cr)
        qc_meas.measure(alvo, cr[0])

        shots = int(self.pq.tiros or 10_000)
        job = self.sampler.run([qc_meas], shots=shots)
        res = job.result()
        # V1 retorna quasi_dists ou counts; padronizamos:
        try:
            # Terra novas: .quasi_dists[0] -> dict bitstring->prob
            qd = res.quasi_dists[0]
            # bit do alvo é o único medido; contar '1'
            p1 = float(qd.get(1, 0.0) if isinstance(list(qd.keys())[0], int) else qd.get("1", 0.0))
        except Exception:
            # Aer antigo: .metadata / .meas.get_counts?
            try:
                counts = res.metadata[0]["counts"]
            except Exception:
                counts = res.results[0].data.counts  # type: ignore
            total = sum(counts.values())
            p1 = counts.get("1", counts.get(1, 0)) / max(total, 1)

        return float(p1)

    def precificar_call(self) -> Tuple[float, float]:
        a = self.estimar_amplitude()
        esperado_T = self.pq.fator_reescala * a
        preco_0 = exp(-self.pm.r * self.pm.T) * esperado_T
        return preco_0, esperado_T

# ============================================================
# Demo
# ============================================================

def executar_demo():
    pm = ParametrosMercado(S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0)
    pq = ParametrosQAE(n_qubits=4, low=0.0, high=200.0, fator_reescala=200.0,
                       tiros=20_000, algoritmo="MLAE", avaliacao_mlae=4)

    q = PrecificadorQuantico(pm, pq)
    preco_0, payoff_T = q.precificar_call()
    bs = call_bs(pm.S0, pm.K, pm.r, pm.sigma, pm.T)

    print(f"[Primitives={_PRIMS}] Preço_0 ≈ {preco_0:.6f} | E[payoff_T] ≈ {payoff_T:.6f}")
    print(f"[Black-Sch]       Preço_0 = {bs:.6f}")

if __name__ == "__main__":
    executar_demo()
