# ============================================================
# QNN Binária em Qiskit — Classificação XOR
# Autor: Luiz Tiago Wilcke (LT)
# ============================================================

import numpy as np
from dataclasses import dataclass
from typing import Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp


# ------------------------------------------------------------
# 1) Geração de dados sintéticos (XOR 2D)
# ------------------------------------------------------------

def gerar_dados_xor(n_por_quadrante: int = 20,
                    semente: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gera um conjunto de pontos 2D tipo XOR:
      - Quadrantes (−,−) e (+,+) => classe 0
      - Quadrantes (−,+) e (+,−) => classe 1
    """
    rng = np.random.default_rng(semente)
    dados = []
    rotulos = []

    desvio = 0.3

    # Quadrante (-1, -1) -> classe 0
    for _ in range(n_por_quadrante):
        x1 = rng.normal(-1.0, desvio)
        x2 = rng.normal(-1.0, desvio)
        dados.append([x1, x2])
        rotulos.append(0)

    # Quadrante (+1, +1) -> classe 0
    for _ in range(n_por_quadrante):
        x1 = rng.normal(+1.0, desvio)
        x2 = rng.normal(+1.0, desvio)
        dados.append([x1, x2])
        rotulos.append(0)

    # Quadrante (-1, +1) -> classe 1
    for _ in range(n_por_quadrante):
        x1 = rng.normal(-1.0, desvio)
        x2 = rng.normal(+1.0, desvio)
        dados.append([x1, x2])
        rotulos.append(1)

    # Quadrante (+1, -1) -> classe 1
    for _ in range(n_por_quadrante):
        x1 = rng.normal(+1.0, desvio)
        x2 = rng.normal(-1.0, desvio)
        dados.append([x1, x2])
        rotulos.append(1)

    dados = np.array(dados, dtype=float)
    rotulos = np.array(rotulos, dtype=int)
    return dados, rotulos


def normalizar_para_intervalo(dados: np.ndarray,
                              limite_max: float = np.pi) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Normaliza cada coluna de 'dados' para o intervalo [0, limite_max].
    Retorna os dados normalizados e (minimos, maximos) para referência.
    """
    minimos = dados.min(axis=0)
    maximos = dados.max(axis=0)
    intervalo = maximos - minimos + 1e-9  # evitar divisão por zero

    dados_norm = (dados - minimos) / intervalo * limite_max
    return dados_norm, (minimos, maximos)


# ------------------------------------------------------------
# 2) Configuração da QNN
# ------------------------------------------------------------

@dataclass
class ConfiguracaoQNN:
    n_qubits: int = 2
    n_caracteristicas: int = 2          # dimensão do vetor de entrada
    n_pesos: int = 6                    # nº de parâmetros treináveis do ansatz
    taxa_aprendizado: float = 0.4
    epocas: int = 60
    semente: int = 42
    passo_shift: float = np.pi / 2.0    # passo do parameter shift


class QNNBinaria:
    """
    Implementa uma QNN simples:
      - Entrada: vetor R^2 -> ângulos em RY
      - Circuito: feature map + ansatz variacional com emaranhamento
      - Saída: valor esperado de Z no qubit 0, convertido em probabilidade
    """

    def __init__(self, config: ConfiguracaoQNN):
        self.config = config

        # Parâmetros simbólicos: dados e pesos
        self.param_dados = ParameterVector("x", config.n_caracteristicas)
        self.param_pesos = ParameterVector("θ", config.n_pesos)

        # Circuito quântico parametrizado
        self.circuito = self._construir_circuito()

        # Observável: Z no primeiro qubit (Z ⊗ I)
        self.observavel = SparsePauliOp("Z" + "I" * (config.n_qubits - 1))

        # Estimador de estado vetorial (simulação exata) :contentReference[oaicite:1]{index=1}
        self.estimador = StatevectorEstimator()

        # Inicialização aleatória dos pesos
        rng = np.random.default_rng(config.semente)
        self.pesos = rng.uniform(-np.pi, np.pi, size=config.n_pesos)

    # --------------------------------------------------------
    # Construção do circuito
    # --------------------------------------------------------
    def _construir_circuito(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.config.n_qubits)

        # --- 1) Feature map (codificação dos dados clássicos) ---
        qc.ry(self.param_dados[0], 0)
        qc.ry(self.param_dados[1], 1)
        qc.cx(0, 1)

        # --- 2) Ansatz variacional (camadas treináveis) ---
        # Camada 1
        qc.ry(self.param_pesos[0], 0)
        qc.rz(self.param_pesos[1], 0)
        qc.ry(self.param_pesos[2], 1)
        qc.rz(self.param_pesos[3], 1)
        qc.cx(0, 1)

        # Camada 2 (rotações adicionais)
        qc.ry(self.param_pesos[4], 0)
        qc.ry(self.param_pesos[5], 1)

        # Sem medidas: o Estimator trabalha com circuitos sem medida.
        return qc

    # --------------------------------------------------------
    # Forward: expectativa de Z e probabilidade de classe
    # --------------------------------------------------------
    def _expectativa_z(self, x: np.ndarray, pesos: np.ndarray | None = None) -> float:
        """
        Calcula <Z> no primeiro qubit para um dado vetor x e pesos.
        """
        if pesos is None:
            pesos = self.pesos

        # Ordem dos parâmetros no circuito: [x0, x1, θ0,...,θ5]
        valores_param = list(x) + list(pesos)

        # Estimator V2: run([(circuito, observavel, [parametros])]) :contentReference[oaicite:2]{index=2}
        job = self.estimador.run([
            (self.circuito, self.observavel, [valores_param])
        ])
        resultado = job.result()
        expectativa = float(resultado[0].data.evs[0])
        return expectativa

    def probabilidade_classe_1(self, x: np.ndarray, pesos: np.ndarray | None = None) -> float:
        """
        Converte <Z> em probabilidade de classe 1.
        Para Z: autovalores +1 (|0>) e -1 (|1>).
        P(classe=1) ≈ P(|1>) = (1 - <Z>) / 2.
        """
        expectativa_z = self._expectativa_z(x, pesos)
        p1 = (1.0 - expectativa_z) / 2.0
        return float(np.clip(p1, 1e-6, 1 - 1e-6))

    def prever_classe(self, x: np.ndarray) -> int:
        """
        Retorna 0 ou 1 a partir da probabilidade.
        """
        p1 = self.probabilidade_classe_1(x)
        return int(p1 >= 0.5)

    # --------------------------------------------------------
    # Funções de perda e gradiente (parameter shift)
    # --------------------------------------------------------
    def _perda_media(self, dados: np.ndarray, rotulos: np.ndarray,
                     pesos: np.ndarray | None = None) -> float:
        """
        Binary cross-entropy média:
          L = - 1/N Σ [ y log p + (1-y) log(1-p) ].
        """
        if pesos is None:
            pesos = self.pesos

        perdas = []
        for x, y in zip(dados, rotulos):
            p1 = self.probabilidade_classe_1(x, pesos)
            perda = -(y * np.log(p1) + (1 - y) * np.log(1 - p1))
            perdas.append(perda)
        return float(np.mean(perdas))

    def _gradiente_parameter_shift(self, dados: np.ndarray,
                                   rotulos: np.ndarray) -> np.ndarray:
        """
        Calcula o gradiente da perda em relação a cada peso usando
        uma regra de parameter shift "estilo" derivada:
            dL/dθ_k ≈ [L(θ_k + s) - L(θ_k - s)] / (2),
        com s = passo_shift.
        """
        grad = np.zeros_like(self.pesos)
        s = self.config.passo_shift

        for k in range(len(self.pesos)):
            desloc = np.zeros_like(self.pesos)
            desloc[k] = s

            perda_mais = self._perda_media(dados, rotulos, self.pesos + desloc)
            perda_menos = self._perda_media(dados, rotulos, self.pesos - desloc)

            grad[k] = 0.5 * (perda_mais - perda_menos)

        return grad

    # --------------------------------------------------------
    # Treinamento
    # --------------------------------------------------------
    def treinar(self, dados: np.ndarray, rotulos: np.ndarray) -> None:
        """
        Loop de treinamento simples com gradient descent.
        """
        for epoca in range(1, self.config.epocas + 1):
            grad = self._gradiente_parameter_shift(dados, rotulos)
            self.pesos -= self.config.taxa_aprendizado * grad

            perda = self._perda_media(dados, rotulos)
            # Acurácia no conjunto de treino
            previsoes = [self.prever_classe(x) for x in dados]
            acuracia = np.mean(np.array(previsoes) == rotulos)

            if epoca % 10 == 0 or epoca == 1:
                print(
                    f"[Época {epoca:3d}] "
                    f"Perda = {perda:.4f} | Acurácia treino = {acuracia*100:5.1f}%"
                )

    def avaliar(self, dados: np.ndarray, rotulos: np.ndarray) -> float:
        """
        Retorna a acurácia em %.
        """
        previsoes = [self.prever_classe(x) for x in dados]
        acuracia = np.mean(np.array(previsoes) == rotulos)
        return float(acuracia * 100.0)



if __name__ == "__main__":
    # 1) Gerar dados XOR
    dados_brutos, rotulos = gerar_dados_xor(n_por_quadrante=25, semente=123)
    print(f"Total de amostras: {len(dados_brutos)}")

    # 2) Normalizar para [0, π] (compatível com rotações RY)
    dados_norm, (mins, maxs) = normalizar_para_intervalo(dados_brutos, limite_max=np.pi)

    # 3) Instanciar e treinar a QNN
    config = ConfiguracaoQNN(
        n_qubits=2,
        n_caracteristicas=2,
        n_pesos=6,
        taxa_aprendizado=0.4,
        epocas=60,
        semente=7,
    )

    qnn = QNNBinaria(config)
    qnn.treinar(dados_norm, rotulos)

    # 4) Acurácia final no treino
    acc_final = qnn.avaliar(dados_norm, rotulos)
    print(f"\nAcurácia final no conjunto inteiro: {acc_final:.2f}%")

    # 5) Mostrar algumas previsões exemplo
    print("\nAlgumas previsões da QNN (pontos originais):")
    for i in range(5):
        x = dados_brutos[i]
        x_norm = dados_norm[i]
        p1 = qnn.probabilidade_classe_1(x_norm)
        y_pred = qnn.prever_classe(x_norm)
        print(
            f"x = {x}, y_real = {rotulos[i]}, "
            f"p(classe=1) = {p1:.3f}, y_pred = {y_pred}"
        )
