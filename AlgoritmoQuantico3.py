# AlgoritmoQuantico3.py
# ============================================================
# Algoritmo de Grover 
# Autor: Luiz Tiago Wilcke (LT)
# ------------------------------------------------------------
# O que faz:
#   • Monta o oráculo que marca um bitstring "alvo"
#   • Monta o difusor (inversão sobre a média)
#   • Define número de iterações ~ floor(pi/4 * sqrt(2^n))
#   • Executa no AerSimulator e imprime as contagens
# ============================================================

from math import pi, sqrt, floor
from typing import Dict, Tuple

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator


# ------------------------------------------------------------
# 0) Configurações do usuário
# ------------------------------------------------------------
# Tamanho do registro e alvo desejado (string binária do mesmo tamanho).
N_QUBITS: int = 5
ALVO_BITS: str = "10101"   # Ex.: "00011", "01100", etc. Deve ter N_QUBITS dígitos.


# ------------------------------------------------------------
# 1) Utilidades
# ------------------------------------------------------------
def validar_parametros(n_qubits: int, alvo_bits: str) -> None:
    if n_qubits <= 0:
        raise ValueError("n_qubits deve ser >= 1.")
    if len(alvo_bits) != n_qubits:
        raise ValueError(f"ALVO_BITS ('{alvo_bits}') deve ter {n_qubits} bits.")
    if any(c not in "01" for c in alvo_bits):
        raise ValueError("ALVO_BITS deve conter apenas '0' e '1'.")


def bitstring_little_endian_para_big_endian(bits: str) -> str:
    """Qiskit mede little-endian; esta função inverte p/ leitura humana tradicional."""
    return bits[::-1]


# ------------------------------------------------------------
# 2) Oráculo e Difusor
# ------------------------------------------------------------
def construir_oraculo(alvo_bits: str) -> QuantumCircuit:
    """
    Oráculo que aplica fase -1 no estado |alvo>.
    Estratégia: X nas posições onde o alvo tem '0' para mapear |alvo> -> |11..1|,
    depois CZ(11..1) implementado como H + MCX + H no primeiro qubit, e desfaz X.
    Nota: usamos convenção little-endian do Qiskit (qubit 0 é o menos significativo).
    """
    n = len(alvo_bits)
    dados = QuantumRegister(n, "dados")
    qc = QuantumCircuit(dados, name="oraculo")

    # Pré-inversões (X) onde alvo possui '0', considerando endianness do Qiskit.
    for i, b in enumerate(reversed(alvo_bits)):
        if b == '0':
            qc.x(dados[i])

    # CZ com n-1 controles via H-MCX-H no qubit 0 (alvo)
    qc.h(dados[0])
    qc.mcx(dados[1:], dados[0])
    qc.h(dados[0])

    # Desfazer pré-inversões
    for i, b in enumerate(reversed(alvo_bits)):
        if b == '0':
            qc.x(dados[i])

    return qc


def construir_difusor(n_qubits: int) -> QuantumCircuit:
    """
    Difusor (inversão sobre a média):
        H^⊗n -> X^⊗n -> CZ(11..1) -> X^⊗n -> H^⊗n
    Implementação do CZ(11..1) igual ao oráculo (H-MCX-H no qubit 0).
    """
    dados = QuantumRegister(n_qubits, "dados")
    qc = QuantumCircuit(dados, name="difusor")

    qc.h(dados)
    qc.x(dados)

    qc.h(dados[0])
    qc.mcx(dados[1:], dados[0])
    qc.h(dados[0])

    qc.x(dados)
    qc.h(dados)
    return qc


# ------------------------------------------------------------
# 3) Montagem do circuito de Grover
# ------------------------------------------------------------
def montar_circuito_grover(alvo_bits: str, iteracoes: int | None = None) -> QuantumCircuit:
    """
    Constrói o circuito completo do Grover.
    Se 'iteracoes' for None, usa k ≈ floor(pi/4 * sqrt(2^n)).
    """
    n = len(alvo_bits)
    if iteracoes is None:
        iteracoes = max(1, floor((pi / 4) * sqrt(2 ** n)))

    dados = QuantumRegister(n, "dados")
    medidas = ClassicalRegister(n, "medidas")
    qc = QuantumCircuit(dados, medidas, name="grover")

    # Estado uniforme
    qc.h(dados)

    # Oráculo e difusor em forma de gates reutilizáveis
    oraculo = construir_oraculo(alvo_bits).to_gate(label="O")
    difusor = construir_difusor(n).to_gate(label="D")

    # Iterações de Grover
    for _ in range(iteracoes):
        qc.append(oraculo, dados[:])
        qc.append(difusor, dados[:])

    # Medição
    qc.measure(dados, medidas)
    return qc


# ------------------------------------------------------------
# 4) Execução e relatório
# ------------------------------------------------------------
def executar_grover(alvo_bits: str, tiros: int = 4096) -> Dict[str, int]:
    simulador = AerSimulator()
    circuito = montar_circuito_grover(alvo_bits)
    circuito_transpilado = transpile(circuito, simulador, optimization_level=1)
    resultado = simulador.run(circuito_transpilado, shots=tiros).result()
    return resultado.get_counts()


def imprimir_relatorio(contagens: Dict[str, int], alvo_bits: str, tiros: int) -> None:
    # Ordenar por frequência
    topo = sorted(contagens.items(), key=lambda kv: kv[1], reverse=True)

    print("\n=========== RELATÓRIO — GROVER ===========")
    print(f"Qubits: {len(alvo_bits)}   Alvo (big-endian): {alvo_bits}")
    print("------------------------------------------")
    print("Top estados medidos (little-endian → big-endian):")
    for estado_le, cont in topo[:8]:
        estado_be = bitstring_little_endian_para_big_endian(estado_le)
        print(f"  estado LE={estado_le}  (BE={estado_be})   contagens={cont}   prob≈{cont/tiros:.4f}")

    # Checar alvo (big-endian) → converter para little-endian para buscar em 'contagens'
    alvo_le = bitstring_little_endian_para_big_endian(alvo_bits)
    acertos = contagens.get(alvo_le, 0)
    print("------------------------------------------")
    print(f"Alvo (LE) = {alvo_le}  →  contagens={acertos}  prob≈{acertos/tiros:.4f}")
    print("==========================================\n")


# ------------------------------------------------------------
# 5) Main
# ------------------------------------------------------------
if __name__ == "__main__":
    # Validar entrada
    validar_parametros(N_QUBITS, ALVO_BITS)

    # Executar
    TIROS = 4096
    contagens = executar_grover(ALVO_BITS, tiros=TIROS)

    # Relatório amigável
    imprimir_relatorio(contagens, ALVO_BITS, TIROS)
