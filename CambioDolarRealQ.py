# ============================================================
# MODELO QUÂNTICO-ESTOCÁSTICO DO CÂMBIO USD/BRL
#   — Duas taxas (Brasil/EUA) + SDE para câmbio
#   — Ruído clássico vs ruído quântico (Qiskit)
#   — Autor: Luiz Tiago Wilcke (LT)
# ============================================================

import math
import warnings
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ============================================================
# 0) Modelo contínuo (ideia teórica)
# ============================================================
"""
Modelo contínuo (tempo t em anos, câmbio C(t) = USD/BRL):

1) Taxa doméstica (Brasil): r_d(t)
   Processo de Ornstein–Uhlenbeck:
       d r_d(t) = κ_d (θ_d − r_d(t)) dt + σ_d dW_d(t)

2) Taxa estrangeira (EUA): r_f(t)
       d r_f(t) = κ_f (θ_f − r_f(t)) dt + σ_f dW_f(t)

3) Câmbio USD/BRL: C(t)
   Versão tipo Garman–Kohlhagen:
       dC(t) / C(t) = [ r_d(t) − r_f(t) ] dt + σ_C dW_C(t)

Discretização (Euler–Maruyama) com passo Δt:

   r_d(t+Δt) = r_d(t) + κ_d (θ_d − r_d(t)) Δt + σ_d √Δt * Z_d
   r_f(t+Δt) = r_f(t) + κ_f (θ_f − r_f(t)) Δt + σ_f √Δt * Z_f
   C(t+Δt)   = C(t) * exp( [r_d(t) − r_f(t) − 0.5 σ_C²] Δt + σ_C √Δt * Z_C )

onde Z_d, Z_f, Z_C ~ N(0,1) independentes.

Neste arquivo, os Z's serão gerados:
   (a) de forma clássica (NumPy)
   (b) via amostras quânticas (Qiskit + AerSimulator).
"""

# ============================================================
# 1) Importações Qiskit (se disponíveis)
# ============================================================

TEM_QISKIT = True
MOTIVO_QISKIT_INDISPONIVEL = ""

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
except Exception as e:
    TEM_QISKIT = False
    MOTIVO_QISKIT_INDISPONIVEL = str(e)


# ============================================================
# 2) Parâmetros do modelo
# ============================================================

@dataclass
class ParametrosEconomicos:
    # Câmbio inicial (USD/BRL)
    cambio_inicial: float = 5.10

    # Taxa doméstica (Brasil) — OU
    taxa_domestica_inicial: float = 0.11  # 11% a.a.
    kappa_domestica: float = 1.5
    nivel_medio_domestico: float = 0.10   # 10% a.a.
    volatilidade_domestica: float = 0.03  # 3 p.p. a.a.

    # Taxa estrangeira (EUA) — OU
    taxa_estrangeira_inicial: float = 0.05  # 5% a.a.
    kappa_estrangeira: float = 1.2
    nivel_medio_estrangeiro: float = 0.04   # 4% a.a.
    volatilidade_estrangeira: float = 0.015 # 1.5 p.p. a.a.

    # Câmbio — volatilidade
    volatilidade_cambio: float = 0.20       # 20% a.a.

    # Horizonte e discretização
    anos: float = 1.0          # horizonte em anos
    passos_por_ano: int = 252  # diário útil
    quantidade_caminhos: int = 2000

    # Semente para replicabilidade
    semente_classica: int = 42


@dataclass
class ParametrosQuanticos:
    # Número de bits por amostra (resolução do U(0,1))
    bits_por_amostra: int = 16

    # Tamanho do "lote" de amostras quânticas geradas de uma vez
    tamanho_lote: int = 4096


# ============================================================
# 3) Gerador de ruído quântico (U(0,1) → N(0,1))
# ============================================================

class GeradorRuidoQuantico:
    """
    Usa um circuito de Hadamards em n_qubits, mede no AerSimulator,
    obtém bitstrings ~ U({0,...,2^n - 1}) e transforma em U(0,1).
    Depois aplica Box–Muller para gerar N(0,1).
    """

    def __init__(self,
                 parametros_q: ParametrosQuanticos,
                 semente: Optional[int] = None):
        if not TEM_QISKIT:
            raise RuntimeError(f"Qiskit indisponível: {MOTIVO_QISKIT_INDISPONIVEL}")

        self.pq = parametros_q
        self.n_qubits = self.pq.bits_por_amostra

        # Circuito base: Hadamard em todos os qubits + medição
        self.circuito = QuantumCircuit(self.n_qubits)
        self.circuito.h(range(self.n_qubits))
        self.circuito.measure_all()

        # Simulador
        self.simulador = AerSimulator()
        self.circuito_compilado = transpile(self.circuito, self.simulador)

        # Semente clássica apenas para embaralhar/ordem de uso dos resultados
        self.rng_local = np.random.default_rng(semente)

    def _gerar_uniformes(self, quantidade: int) -> np.ndarray:
        """
        Gera 'quantidade' amostras ~ U(0,1) usando medidas quânticas.

        Estratégia:
            - Roda o mesmo circuito H^{⊗n} com 'shots' medidas.
            - get_counts() retorna um dict {bitstring: contagem}.
            - Reconstrói um vetor com 'quantidade' inteiros em [0, 2^n - 1].
            - Normaliza para (0,1).
        """
        bits = self.pq.bits_por_amostra
        max_val = 2 ** bits - 1

        # Para garantir que temos pelo menos 'quantidade' amostras:
        shots = max(quantidade, self.pq.tamanho_lote)
        resultado = self.simulador.run(self.circuito_compilado,
                                       shots=shots).result()
        contagens = resultado.get_counts()

        # Reconstrói as amostras individuais
        valores = []
        for bitstring, freq in contagens.items():
            # remove espaços se houver
            bs = bitstring.replace(" ", "")
            valor_int = int(bs, 2)
            valores.extend([valor_int] * freq)

        valores = np.array(valores[:quantidade], dtype=float)

        # Normaliza para (0,1) evitando extremos 0 e 1
        u = (valores + 0.5) / (max_val + 1.0)
        # Embaralha um pouco a ordem (quase irrelevante, só pra estética)
        self.rng_local.shuffle(u)
        return u

    def gerar_normais(self, quantidade: int) -> np.ndarray:
        """
        Gera 'quantidade' amostras N(0,1) usando Box–Muller
        a partir de uniformes quânticos.
        """
        # Precisamos de quantidade/2 pares (u1, u2)
        n_pares = (quantidade + 1) // 2
        u1 = self._gerar_uniformes(n_pares)
        u2 = self._gerar_uniformes(n_pares)

        # Box–Muller
        z1 = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * math.pi * u2)
        z2 = np.sqrt(-2.0 * np.log(u1)) * np.sin(2.0 * math.pi * u2)

        z = np.concatenate([z1, z2])[:quantidade]
        return z


# ============================================================
# 4) Simulação clássica (NumPy) do modelo de câmbio
# ============================================================

def simular_cambio_classico(param: ParametrosEconomicos) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simula caminhos do câmbio USD/BRL usando ruído clássico (NumPy).

    Retorna:
        tempos: vetor de tempos (anos)
        media_caminhos: valor médio de C_t ao longo dos caminhos.
    """
    rng = np.random.default_rng(param.semente_classica)

    passos_totais = int(param.anos * param.passos_por_ano)
    dt = 1.0 / param.passos_por_ano
    raiz_dt = math.sqrt(dt)

    # Matrizes: (caminho, tempo)
    r_d = np.zeros((param.quantidade_caminhos, passos_totais + 1))
    r_f = np.zeros((param.quantidade_caminhos, passos_totais + 1))
    C = np.zeros((param.quantidade_caminhos, passos_totais + 1))

    # Condições iniciais
    r_d[:, 0] = param.taxa_domestica_inicial
    r_f[:, 0] = param.taxa_estrangeira_inicial
    C[:, 0] = param.cambio_inicial

    # Simulação
    for t in range(passos_totais):
        Z_d = rng.normal(size=param.quantidade_caminhos)
        Z_f = rng.normal(size=param.quantidade_caminhos)
        Z_c = rng.normal(size=param.quantidade_caminhos)

        # Taxa doméstica
        r_d[:, t+1] = (
            r_d[:, t]
            + param.kappa_domestica * (param.nivel_medio_domestico - r_d[:, t]) * dt
            + param.volatilidade_domestica * raiz_dt * Z_d
        )

        # Taxa estrangeira
        r_f[:, t+1] = (
            r_f[:, t]
            + param.kappa_estrangeira * (param.nivel_medio_estrangeiro - r_f[:, t]) * dt
            + param.volatilidade_estrangeira * raiz_dt * Z_f
        )

        # Câmbio
        drift_inst = r_d[:, t] - r_f[:, t]
        C[:, t+1] = C[:, t] * np.exp(
            (drift_inst - 0.5 * param.volatilidade_cambio**2) * dt
            + param.volatilidade_cambio * raiz_dt * Z_c
        )

    tempos = np.linspace(0.0, param.anos, passos_totais + 1)
    media_caminhos = C.mean(axis=0)
    return tempos, media_caminhos


# ============================================================
# 5) Simulação "quântica" (ruído via Qiskit)
# ============================================================

def simular_cambio_quantico(param: ParametrosEconomicos,
                            param_q: ParametrosQuanticos) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simula caminhos do câmbio USD/BRL usando ruído N(0,1) gerado a partir
    de medições quânticas (Qiskit + AerSimulator).

    Se Qiskit não estiver disponível, levanta RuntimeError.
    """
    if not TEM_QISKIT:
        raise RuntimeError(f"Qiskit não disponível neste ambiente: {MOTIVO_QISKIT_INDISPONIVEL}")

    gerador = GeradorRuidoQuantico(param_q, semente=param.semente_classica)

    passos_totais = int(param.anos * param.passos_por_ano)
    dt = 1.0 / param.passos_por_ano
    raiz_dt = math.sqrt(dt)

    r_d = np.zeros((param.quantidade_caminhos, passos_totais + 1))
    r_f = np.zeros((param.quantidade_caminhos, passos_totais + 1))
    C = np.zeros((param.quantidade_caminhos, passos_totais + 1))

    r_d[:, 0] = param.taxa_domestica_inicial
    r_f[:, 0] = param.taxa_estrangeira_inicial
    C[:, 0] = param.cambio_inicial

    total_ruidos = 3 * param.quantidade_caminhos  # Z_d, Z_f, Z_c por passo

    for t in range(passos_totais):
        # Gera todos os N(0,1) necessários para este passo, de uma vez
        Z_t = gerador.gerar_normais(total_ruidos)
        Z_d = Z_t[0:param.quantidade_caminhos]
        Z_f = Z_t[param.quantidade_caminhos:2*param.quantidade_caminhos]
        Z_c = Z_t[2*param.quantidade_caminhos:3*param.quantidade_caminhos]

        # Taxa doméstica
        r_d[:, t+1] = (
            r_d[:, t]
            + param.kappa_domestica * (param.nivel_medio_domestico - r_d[:, t]) * dt
            + param.volatilidade_domestica * raiz_dt * Z_d
        )

        # Taxa estrangeira
        r_f[:, t+1] = (
            r_f[:, t]
            + param.kappa_estrangeira * (param.nivel_medio_estrangeiro - r_f[:, t]) * dt
            + param.volatilidade_estrangeira * raiz_dt * Z_f
        )

        # Câmbio
        drift_inst = r_d[:, t] - r_f[:, t]
        C[:, t+1] = C[:, t] * np.exp(
            (drift_inst - 0.5 * param.volatilidade_cambio**2) * dt
            + param.volatilidade_cambio * raiz_dt * Z_c
        )

    tempos = np.linspace(0.0, param.anos, passos_totais + 1)
    media_caminhos = C.mean(axis=0)
    return tempos, media_caminhos


# ============================================================
# 6) Função de demonstração
# ============================================================

def executar_demo():
    param = ParametrosEconomicos()
    param_q = ParametrosQuanticos()

    print("========== MODELO QUÂNTICO-ESTOCÁSTICO USD/BRL ==========")
    print(f"Câmbio inicial (USD/BRL):   {param.cambio_inicial:.4f}")
    print(f"Horizonte:                  {param.anos:.2f} anos")
    print(f"Passos por ano:            {param.passos_por_ano}")
    print(f"Caminhos simulados:        {param.quantidade_caminhos}")
    print("--------------------------------------------------------")

    # Simulação clássica
    tempos, media_classica = simular_cambio_classico(param)
    C_T_classico = media_classica[-1]

    if TEM_QISKIT:
        # Simulação com ruído quântico
        tempos_q, media_quantica = simular_cambio_quantico(param, param_q)
        C_T_quantico = media_quantica[-1]
    else:
        media_quantica = None
        C_T_quantico = float("nan")

    print("Resultados médios no horizonte (E[C_T])")
    print(f"  Clássico (NumPy):    {C_T_classico:.4f} BRL/USD")
    if TEM_QISKIT:
        print(f"  Quântico (Qiskit):   {C_T_quantico:.4f} BRL/USD")
        print(f"  Diferença relativa:  {(C_T_quantico/C_T_classico - 1)*100:.3f}%")
    else:
        print("  Parte quântica indisponível neste ambiente.")
        print("  Motivo:", MOTIVO_QISKIT_INDISPONIVEL)

    # --------------------------------------------------------
    # Plotagem
    # --------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(tempos, media_classica, label="Média câmbio — ruído clássico")
    if media_quantica is not None:
        plt.plot(tempos_q, media_quantica, "--", label="Média câmbio — ruído quântico")

    plt.title("USD/BRL — Modelo Estocástico com Ruído Clássico vs Quântico\nLuiz Tiago Wilcke (LT)")
    plt.xlabel("Tempo (anos)")
    plt.ylabel("Câmbio USD/BRL esperado")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 7) Execução direta
# ============================================================

if __name__ == "__main__":
    executar_demo()
