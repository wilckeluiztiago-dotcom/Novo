# ============================================================
# REDE DE TRANSPORTE — Modelo Matemático de Fluxo de Mínimo Custo
# Autor: Luiz Tiago Wilcke (LT)
# ============================================================
#
# Modelo:
#   min  Σ_{(i,j)∈E} c_ij * x_ij
#   s.a. Σ_{j} x_ji - Σ_{j} x_ij = b_i   ∀ i ∈ V
#        0 ≤ x_ij ≤ u_ij                 ∀ (i,j) ∈ E
# ============================================================

from dataclasses import dataclass
from typing import Dict, Tuple, List

import pulp
import networkx as nx
import matplotlib.pyplot as plt


# -----------------------
# 1) Dados do modelo
# -----------------------

@dataclass
class Aresta:
    origem: str
    destino: str
    capacidade: float
    custo: float


def criar_dados_rede() -> Tuple[List[str], Dict[str, float], List[Aresta]]:
    """
    Cria um exemplo de rede de transporte para o modelo matemático.
    Vértices: A, B, C, D, E, F
      - A, B: nós de oferta (b_i < 0)
      - E, F: nós de demanda (b_i > 0)
      - C, D: nós de transbordo (b_i = 0)
    """
    lista_nos = ["A", "B", "C", "D", "E", "F"]

    # b_i (demanda):
    #   negativo = oferta
    #   positivo = demanda
    #   zero     = transbordo
    demandas: Dict[str, float] = {
        "A": -15.0,
        "B": -10.0,
        "C": 0.0,
        "D": 0.0,
        "E": 12.0,
        "F": 13.0,
    }

    # Arestas (origem, destino, capacidade u_ij, custo c_ij)
    lista_arestas: List[Aresta] = [
        Aresta("A", "C", capacidade=10, custo=2.0),
        Aresta("A", "D", capacidade=8, custo=4.0),
        Aresta("B", "C", capacidade=5, custo=3.0),
        Aresta("B", "D", capacidade=12, custo=2.5),
        Aresta("C", "D", capacidade=6, custo=1.0),
        Aresta("C", "E", capacidade=10, custo=2.0),
        Aresta("C", "F", capacidade=5, custo=3.0),
        Aresta("D", "E", capacidade=5, custo=2.5),
        Aresta("D", "F", capacidade=12, custo=1.5),
    ]

    # Checagem rápida: soma das demandas deve ser 0
    soma_demandas = sum(demandas.values())
    if abs(soma_demandas) > 1e-6:
        raise ValueError(f"Soma das demandas != 0 (atual = {soma_demandas})")

    return lista_nos, demandas, lista_arestas


# -----------------------
# 2) Modelo matemático
# -----------------------

def resolver_fluxo_minimo_custo(
    lista_nos: List[str],
    demandas: Dict[str, float],
    lista_arestas: List[Aresta],
):
    """
    Monta e resolve o modelo de fluxo de mínimo custo via programação linear.

    Retorna:
      - dicionario_fluxo[(i, j)] = valor ótimo de x_ij
      - custo_total_minimo
    """

    # Cria problema de PL: minimização
    problema = pulp.LpProblem("Fluxo_Minimo_Custo", pulp.LpMinimize)

    # Variáveis x_ij para cada aresta (i,j)
    variaveis_fluxo: Dict[Tuple[str, str], pulp.LpVariable] = {}
    for aresta in lista_arestas:
        nome_var = f"x_{aresta.origem}_{aresta.destino}"
        variavel = pulp.LpVariable(
            nome_var,
            lowBound=0,
            upBound=aresta.capacidade,
            cat=pulp.LpContinuous,
        )
        variaveis_fluxo[(aresta.origem, aresta.destino)] = variavel

    # Função objetivo: min Σ c_ij * x_ij
    problema += pulp.lpSum(
        aresta.custo * variaveis_fluxo[(aresta.origem, aresta.destino)]
        for aresta in lista_arestas
    ), "Custo_Total"

    # Restrições de conservação de fluxo em cada nó i
    for no in lista_nos:
        fluxo_entrando = []
        fluxo_saindo = []
        for aresta in lista_arestas:
            if aresta.destino == no:
                fluxo_entrando.append(variaveis_fluxo[(aresta.origem, aresta.destino)])
            if aresta.origem == no:
                fluxo_saindo.append(variaveis_fluxo[(aresta.origem, aresta.destino)])

        problema += (
            pulp.lpSum(fluxo_entrando) - pulp.lpSum(fluxo_saindo)
            == demandas[no]
        ), f"Conservacao_{no}"

    # Resolver
    status = problema.solve(pulp.PULP_CBC_CMD(msg=False))

    if status != pulp.LpStatusOptimal:
        raise RuntimeError(f"Modelo não encontrou solução ótima. Status: {pulp.LpStatus[status]}")

    # Extrair solução
    dicionario_fluxo: Dict[Tuple[str, str], float] = {}
    for chave, var in variaveis_fluxo.items():
        dicionario_fluxo[chave] = var.value()

    custo_total_minimo = pulp.value(problema.objective)

    return dicionario_fluxo, custo_total_minimo


# -----------------------
# 3) Desenho da rede
# -----------------------

def desenhar_rede(
    lista_nos: List[str],
    demandas: Dict[str, float],
    lista_arestas: List[Aresta],
    dicionario_fluxo: Dict[Tuple[str, str], float],
):
    """
    Desenha a rede de transporte e destaca as arestas com fluxo > 0.
    """

    # Construir grafo direcionado
    grafo = nx.DiGraph()
    for no in lista_nos:
        grafo.add_node(no, demand=demandas[no])

    for aresta in lista_arestas:
        fluxo = dicionario_fluxo[(aresta.origem, aresta.destino)]
        grafo.add_edge(
            aresta.origem,
            aresta.destino,
            capacidade=aresta.capacidade,
            custo=aresta.custo,
            fluxo=fluxo,
        )

    # Posições apenas ilustrativas (poderiam vir de coordenadas reais)
    posicoes = {
        "A": (-1.5, 0.5),
        "B": (-1.5, -0.5),
        "C": (0.0, 0.7),
        "D": (0.0, -0.7),
        "E": (1.5, 0.7),
        "F": (1.5, -0.7),
    }

    plt.figure(figsize=(10, 6))
    plt.axis("off")
    plt.title("Rede de Transporte — Solução do Modelo de Fluxo de Mínimo Custo", fontsize=13)

    # Cores dos nós de acordo com demanda/oferta
    cores_nos = []
    rotulos_nos = {}
    for no in lista_nos:
        demanda = demandas[no]
        if demanda < 0:
            cores_nos.append("#22c55e")  # oferta
            rotulos_nos[no] = f"{no}\nOferta {abs(demanda):.0f}"
        elif demanda > 0:
            cores_nos.append("#ef4444")  # demanda
            rotulos_nos[no] = f"{no}\nDemanda {demanda:.0f}"
        else:
            cores_nos.append("#0ea5e9")  # transbordo
            rotulos_nos[no] = no

    nx.draw_networkx_nodes(
        grafo, posicoes,
        node_color=cores_nos,
        node_size=1300,
        edgecolors="black",
        alpha=0.9,
    )
    nx.draw_networkx_labels(grafo, posicoes, labels=rotulos_nos, font_size=9)

    # Separar arestas com e sem fluxo
    arestas_fluxo_pos = []
    arestas_fluxo_zero = []
    for aresta in lista_arestas:
        fluxo = dicionario_fluxo[(aresta.origem, aresta.destino)]
        if fluxo > 1e-6:
            arestas_fluxo_pos.append((aresta.origem, aresta.destino))
        else:
            arestas_fluxo_zero.append((aresta.origem, aresta.destino))

    # Desenhar arestas sem fluxo (transparentes)
    nx.draw_networkx_edges(
        grafo,
        posicoes,
        edgelist=arestas_fluxo_zero,
        width=1.0,
        alpha=0.25,
        arrows=True,
        arrowstyle="->",
        connectionstyle="arc3,rad=0.05",
    )

    # Desenhar arestas com fluxo (destacadas)
    nx.draw_networkx_edges(
        grafo,
        posicoes,
        edgelist=arestas_fluxo_pos,
        width=3.0,
        alpha=0.9,
        arrows=True,
        arrowstyle="->",
        connectionstyle="arc3,rad=0.05",
    )

    # Rótulos de arestas: fluxo/capacidade (custo)
    rotulos_arestas = {}
    for aresta in lista_arestas:
        chave = (aresta.origem, aresta.destino)
        fluxo = dicionario_fluxo[chave]
        rotulos_arestas[chave] = f"{fluxo:.1f}/{aresta.capacidade:.0f}  (c={aresta.custo})"

    nx.draw_networkx_edge_labels(
        grafo, posicoes, edge_labels=rotulos_arestas, font_size=8
    )

    plt.tight_layout()
    plt.show()


# -----------------------
# 4) Execução principal
# -----------------------

def main():
    lista_nos, demandas, lista_arestas = criar_dados_rede()

    dicionario_fluxo, custo_total_minimo = resolver_fluxo_minimo_custo(
        lista_nos, demandas, lista_arestas
    )

    print("===== SOLUÇÃO DO MODELO DE FLUXO DE MÍNIMO CUSTO =====")
    for (origem, destino), fluxo in dicionario_fluxo.items():
        if fluxo > 1e-6:
            print(f"{origem} -> {destino}: fluxo = {fluxo:.2f}")

    print(f"\nCusto total mínimo: {custo_total_minimo:.2f}")

    desenhar_rede(lista_nos, demandas, lista_arestas, dicionario_fluxo)


if __name__ == "__main__":
    main()
