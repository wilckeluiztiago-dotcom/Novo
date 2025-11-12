# ============================================================
# Deutsch–Jozsa (Qiskit)
# Autor: Luiz Tiago Wilcke (LT)
# ============================================================


from __future__ import annotations
from typing import Callable, Iterable, Dict
import random

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer

# ------------------------------------------------------------
# Tipagem: todo oráculo aplica portas no circuito fornecido
# ------------------------------------------------------------
# Assinatura: aplicar_oraculo(qc, q_entrada, q_ancila) -> None
TipoOraculo = Callable[[QuantumCircuit, QuantumRegister, QuantumRegister], None]

# ------------------------------------------------------------
# Utilidades de oráculo
# ------------------------------------------------------------
def construir_oraculo_constante(
    n_qubits_entrada: int,
    bit_constante: int
) -> TipoOraculo:
    """
    Oráculo constante: f(x) = bit_constante (0 ou 1).
    Implementação DJ: se f(x)=1, aplicar X na ancila (independe de x).
    """
    b = 1 if bit_constante else 0

    def aplicar(qc: QuantumCircuit, q_entrada: QuantumRegister, q_ancila: QuantumRegister) -> None:
        if b == 1:
            qc.x(q_ancila[0])

    return aplicar


def construir_oraculo_balanceado_linear(
    n_qubits_entrada: int,
    vetor_a: Iterable[int],
    termo_b: int = 0
) -> TipoOraculo:
    """
    Oráculo balanceado linear:
        f(x) = a · x  ⊕  b,  com a != 0 (produto escalar mod 2).
    Implementação: CX dos qubits com a_i=1 para a ancila; X na ancila se b=1.
    """
    a = [1 if int(v) & 1 else 0 for v in vetor_a]
    if len(a) != n_qubits_entrada:
        raise ValueError("vetor_a com tamanho incompatível com n_qubits_entrada")
    if sum(a) == 0:
        raise ValueError("Para ser balanceado, pelo menos um elemento de 'a' deve ser 1")
    b = 1 if termo_b else 0

    def aplicar(qc: QuantumCircuit, q_entrada: QuantumRegister, q_ancila: QuantumRegister) -> None:
        for i, ai in enumerate(a):
            if ai:
                qc.cx(q_entrada[i], q_ancila[0])
        if b == 1:
            qc.x(q_ancila[0])

    return aplicar


def construir_oraculo_por_tabela_verdade(
    n_qubits_entrada: int,
    mintermos_f_igual_1: Iterable[str]
) -> TipoOraculo:
    """
    Constrói Uf a partir dos mintermos (strings binárias de tamanho n) onde f(x)=1.
    Estratégia: para cada mintermo m, inverter entradas com '0' -> aplica MCX na ancila -> desfazer inversões.
    Adequado para n pequeno (<= 5-6), pois cresce exponencialmente no nº de mintermos.
    """
    mintermos = list(mintermos_f_igual_1)
    for m in mintermos:
        if len(m) != n_qubits_entrada or any(ch not in "01" for ch in m):
            raise ValueError("Cada mintermo deve ser string binária de tamanho n_qubits_entrada")

    def aplicar(qc: QuantumCircuit, q_entrada: QuantumRegister, q_ancila: QuantumRegister) -> None:
        for m in mintermos:
            # preparar |x>=|11...1| no padrão do mintermo
            for i, ch in enumerate(m):
                if ch == '0':
                    qc.x(q_entrada[i])
            qc.mcx(list(q_entrada), q_ancila[0])  # multi-controlled X
            for i, ch in enumerate(m):
                if ch == '0':
                    qc.x(q_entrada[i])

    return aplicar

# ------------------------------------------------------------
# Núcleo de Deutsch–Jozsa
# ------------------------------------------------------------
def circuito_deutsch_jozsa(
    n_qubits_entrada: int,
    aplicar_oraculo: TipoOraculo,
    medir: bool = True,
    desenhar: bool = False
) -> QuantumCircuit:
    """
    Fluxo:
      1) |0>^{⊗n} |1>
      2) H^{⊗(n+1)}
      3) Uf
      4) H^{⊗n} nas entradas
      5) Medição das entradas (opcional)
    Interpretação:
      - "000...0" -> f constante
      - outro padrão -> f balanceada
    """
    q_entrada = QuantumRegister(n_qubits_entrada, "q_entrada")
    q_ancila = QuantumRegister(1, "q_ancila")
    if medir:
        c_entrada = ClassicalRegister(n_qubits_entrada, "c_entrada")
        qc = QuantumCircuit(q_entrada, q_ancila, c_entrada)
    else:
        qc = QuantumCircuit(q_entrada, q_ancila)

    # |0...0>|1>
    qc.x(q_ancila[0])

    # H em todos
    qc.h(q_entrada)
    qc.h(q_ancila[0])

    # Oráculo Uf no MESMO circuito
    aplicar_oraculo(qc, q_entrada, q_ancila)

    # H nas entradas
    qc.h(q_entrada)

    # Medição
    if medir:
        qc.measure(q_entrada, c_entrada)

    if desenhar:
        try:
            print(qc.draw("mpl"))
        except Exception:
            print(qc.draw("text"))

    return qc


def executar_deutsch_jozsa(
    n_qubits_entrada: int,
    aplicar_oraculo: TipoOraculo,
    tiros: int = 1000,
    backend_name: str = "aer_simulator"
) -> Dict[str, int]:
    """
    Executa no Aer e retorna contagens.
    """
    qc = circuito_deutsch_jozsa(n_qubits_entrada, aplicar_oraculo, medir=True)
    backend = Aer.get_backend(backend_name)
    qc_t = transpile(qc, backend)
    resultado = backend.run(qc_t, shots=tiros).result()
    contagens = resultado.get_counts()

    bitstring_zero = "0" * n_qubits_entrada
    constante = contagens.get(bitstring_zero, 0) == max(contagens.values())

    print("Contagens:", dict(contagens))
    print("Decisão Deutsch–Jozsa:", "CONSTANTE" if constante else "BALANCEADA")
    return contagens


# ------------------------------------------------------------
# Demonstrações e testes
# ------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)

    # Exemplo A: oráculo CONSTANTE (f(x)=1)
    nA = 4
    oraculo_const_1 = construir_oraculo_constante(nA, bit_constante=1)
    print("\n=== Exemplo A: Constante (f(x)=1) ===")
    executar_deutsch_jozsa(nA, oraculo_const_1, tiros=2048)

    # Exemplo B: oráculo BALANCEADO linear (a·x ⊕ b) com a != 0
    nB = 5
    a = [0, 1, 0, 1, 1]  # a·x = x1 ⊕ x3 ⊕ x4 (índices 1,3,4) -> balanceado
    b = 0
    oraculo_linear = construir_oraculo_balanceado_linear(nB, a, termo_b=b)
    print("\n=== Exemplo B: Balanceado linear (a·x ⊕ b) ===")
    executar_deutsch_jozsa(nB, oraculo_linear, tiros=2048)

    # Exemplo C: oráculo por TABELA-VERDADE (n pequeno)
    nC = 3
    mintermos = ["001", "010", "111"]  # 3/8 -> balanceado
    oraculo_tv = construir_oraculo_por_tabela_verdade(nC, mintermos)
    print("\n=== Exemplo C: Balanceado por tabela-verdade ===")
    executar_deutsch_jozsa(nC, oraculo_tv, tiros=4096)
