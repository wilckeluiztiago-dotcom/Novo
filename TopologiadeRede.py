# ============================================================
# Modelo Topológico de Rede Financeira (TDA em Correlações)
# Autor: Luiz Tiago Wilcke (LT)
# ============================================================
r"""
IDEIA MATEMÁTICA (FORMAÇÃO DA REDE E DA ESTRUTURA TOPOLÓGICA)
--------------------------------------------------------------

1) SÉRIES DE PREÇOS E REDE FINANCEIRA
-------------------------------------
Seja um conjunto de ativos financeiros
    \mathcal{A} = \{A_1, A_2, \dots, A_N\}.

Para cada ativo A_i temos uma série temporal de preços de fechamento
    S_i(t), \quad t = 0,1,\dots,T.

Definimos o retorno logarítmico diário:
    r_i(t) = \log\frac{S_i(t)}{S_i(t-1)},  \quad t = 1,\dots,T.

A partir dos retornos, construímos a matriz de correlações
    \rho_{ij} = \mathrm{Corr}(r_i, r_j),  \quad 1 \le i,j \le N,

onde \rho_{ij} \in [-1,1]. Interpretando \rho_{ij} como proximidade
financeira, temos um grafo ponderado completo G = (V,E),
onde V = \mathcal{A} e a aresta (i,j) tem peso w_{ij} definido via
uma métrica derivada da correlação.

2) MÉTRICA DE MANTEGNA (ESPAÇO MÉTRICO FINANCEIRO)
---------------------------------------------------
Uma métrica clássica em redes financeiras é a métrica de Mantegna:

    d_{ij} = \sqrt{2(1 - \rho_{ij})}.

Propriedades:
- d_{ij} \in [0, 2].
- d_{ij} = 0 se \rho_{ij} = 1 (ativos perfeitamente correlacionados).
- d_{ij} cresce conforme a correlação diminui.

Assim, obtemos um espaço métrico
    (X, d),  X = \{1,\dots,N\},  d = (d_{ij})_{i,j}.

3) COMPLEXO DE VIETORIS–RIPS
----------------------------
Dada uma métrica d e um parâmetro de escala \varepsilon \ge 0,
o complexo simplicial de Vietoris–Rips é definido por:

    K_\varepsilon(X) = \{ \sigma \subseteq X \;:\;
                         \mathrm{diam}(\sigma) \le 2\varepsilon \},

onde \mathrm{diam}(\sigma) = \max_{i,j \in \sigma} d_{ij}.

Intuição:
- Para \varepsilon pequeno, apenas pares muito próximos (altamente
  correlacionados) formam arestas e simpleses.
- À medida que \varepsilon cresce, formam-se triângulos, buracos,
  cavidades etc., capturando clusters e ciclos da rede financeira.

4) HOMOLOGIA E HOMOLOGIA PERSISTENTE
------------------------------------
Para cada \varepsilon, temos um complexo K_\varepsilon. A homologia
em dimensão k é:

    H_k(K_\varepsilon; \mathbb{F}) =
        \ker(\partial_k) / \mathrm{im}(\partial_{k+1}),

onde \partial_k é o operador de fronteira e \mathbb{F} é um corpo
(geralmente \mathbb{Z}_2). Cada H_k mede k-buracos independentes:
- H_0: componentes conexas (clusters).
- H_1: ciclos (loops) na rede.
- H_2: cavidades, etc.

Uma filtração é uma família crescente de complexos:

    K_{\varepsilon_0} \subseteq K_{\varepsilon_1} \subseteq
    \dots \subseteq K_{\varepsilon_L},

com 0 = \varepsilon_0 < \varepsilon_1 < \dots < \varepsilon_L.
A homologia persistente rastreia o nascimento e morte de classes de
homologia ao longo da filtração. Para uma classe em dimensão k, temos
um par (b,d) com:

- b = parâmetro de nascimento (quando a classe aparece),
- d = parâmetro de morte (quando ela desaparece).

Isso gera o diagrama de persistência:

    D_k = \{(b_i, d_i)\}_i.

5) CURVAS DE BETTI E ENTROPIA PERSISTENTE
-----------------------------------------
Dada a coleção de intervalos de persistência (b_i, d_i), definimos a
função característica de cada intervalo e construímos a curva de Betti:

    \beta_k(\varepsilon)
      = \#\{ i : b_i \le \varepsilon < d_i \},

que conta quantos k-buracos estão "ativos" no nível \varepsilon.

Definimos também o comprimento de cada intervalo:

    \ell_i = d_i - b_i, \quad \ell_i > 0.

Normalizamos:

    p_i = \frac{\ell_i}{\sum_j \ell_j},

e obtemos a entropia persistente em dimensão k:

    H_k = - \sum_i p_i \log p_i.

Interpretação:
- H_k alto -> estrutura topológica mais "dispersa" em escalas.
- H_k baixo -> poucos ciclos dominantes ou clusters bem definidos.

APLICAÇÃO FINANCEIRA
--------------------
- Estrutura de clusters H_0: identifica blocos de ativos fortemente
  correlacionados (setores, regimes de mercado).
- Ciclos H_1: padrões de redundância/hedge na rede de ativos.
- Entropia persistente H_k: mede complexidade estrutural da rede
  financeira ao longo das escalas de correlação.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

from scipy.spatial.distance import squareform

# (Opcional) rede financeira explícita em grafo
try:
    import networkx as nx
    TEM_NETWORKX = True
except ImportError:
    TEM_NETWORKX = False

# TDA (homologia persistente)
try:
    from ripser import ripser
    from persim import plot_diagrams
    TEM_TDA = True
except ImportError:
    TEM_TDA = False


# ------------------------------------------------------------
# 1) Parâmetros do modelo topológico
# ------------------------------------------------------------
@dataclass
class ParametrosTopologicos:
    max_dim: int = 1            # dimensão máxima da homologia (0,1,2,...)
    max_eps: float = 1.5        # raio máximo da filtração (escala Vietoris–Rips)
    n_pontos_curva_betti: int = 100  # resolução da curva de Betti


# ------------------------------------------------------------
# 2) Classe principal do modelo topológico financeiro
# ------------------------------------------------------------
class ModeloTopologicoRedeFinanceira:
    def __init__(
        self,
        precos: pd.DataFrame,
        parametros: Optional[ParametrosTopologicos] = None
    ) -> None:
        """
        precos: DataFrame com índice temporal e colunas = ativos.
        """
        if parametros is None:
            parametros = ParametrosTopologicos()
        self.parametros = parametros

        # Garantir ordenação temporal
        self.precos = precos.sort_index()
        self.ativos = list(precos.columns)

        # Placeholders
        self.retornos_log = None
        self.matriz_correlacao = None
        self.matriz_distancia = None
        self._resultado_ripser: Optional[Dict[str, Any]] = None
        self.curvas_betti: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None
        self.entropias_persistentes: Optional[Dict[int, float]] = None

        self._preprocessar()

    # ------------------------ Pré-processamento ------------------------
    def _preprocessar(self) -> None:
        self._calcular_retornos_log()
        self._calcular_matriz_correlacao()
        self._calcular_metrica_mantegna()

    def _calcular_retornos_log(self) -> None:
        """
        r_i(t) = log(S_i(t) / S_i(t-1))
        """
        self.retornos_log = np.log(self.precos / self.precos.shift(1)).dropna()

    def _calcular_matriz_correlacao(self) -> None:
        """
        Matriz de correlação dos retornos logarítmicos.
        """
        self.matriz_correlacao = self.retornos_log.corr().values

    def _calcular_metrica_mantegna(self) -> None:
        """
        d_ij = sqrt(2(1 - rho_ij))
        """
        rho = self.matriz_correlacao
        self.matriz_distancia = np.sqrt(2 * (1.0 - rho))

    # ------------------------ Estrutura de rede ------------------------
    def construir_grafo_financeiro(self) -> Optional["nx.Graph"]:
        """
        Constrói um grafo ponderado com pesos = d_ij (distância de Mantegna).
        Retorna None se networkx não estiver disponível.
        """
        if not TEM_NETWORKX:
            print("[AVISO] networkx não instalado; grafo não será construído.")
            return None

        n = len(self.ativos)
        G = nx.Graph()
        for i, ativo in enumerate(self.ativos):
            G.add_node(ativo)

        for i in range(n):
            for j in range(i + 1, n):
                d_ij = float(self.matriz_distancia[i, j])
                G.add_edge(self.ativos[i], self.ativos[j], weight=d_ij)

        return G

    # ------------------------ Homologia persistente ------------------------
    def calcular_homologia_persistente(self) -> None:
        """
        Aplica Vietoris–Rips via 'ripser' sobre a matriz de distâncias.
        """
        if not TEM_TDA:
            raise RuntimeError(
                "ripser/persim não instalados. "
                "Instale com: pip install ripser persim"
            )

        # ripser aceita vetor condensado ou matriz completa.
        # Aqui usamos a matriz completa indicando distance_matrix=True.
        print("[INFO] Calculando homologia persistente (Vietoris–Rips)...")
        res = ripser(
            self.matriz_distancia,
            maxdim=self.parametros.max_dim,
            thresh=self.parametros.max_eps,
            distance_matrix=True,
        )
        self._resultado_ripser = res
        print("[INFO] Homologia persistente calculada.")

    def plotar_diagramas_persistencia(self) -> None:
        """
        Plota diagramas de persistência para cada dimensão.
        """
        if not TEM_TDA:
            print("[AVISO] ripser/persim não instalados.")
            return
        if self._resultado_ripser is None:
            print("[AVISO] Primeiro chame calcular_homologia_persistente().")
            return

        dgms = self._resultado_ripser["dgms"]
        plt.figure(figsize=(8, 4))
        plot_diagrams(dgms, show=True)

    # ------------------------ Curvas de Betti ------------------------
    def _construir_curvas_betti(self) -> None:
        """
        A partir dos intervalos (b,d) de cada dimensão k, constrói
        a curva de Betti beta_k(eps) em uma grade de eps.
        """
        if self._resultado_ripser is None:
            raise RuntimeError(
                "Resultado de homologia inexistente. "
                "Chame calcular_homologia_persistente() primeiro."
            )

        dgms = self._resultado_ripser["dgms"]
        eps_grid = np.linspace(0, self.parametros.max_eps,
                               self.parametros.n_pontos_curva_betti)

        curvas: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        for k, dgm in enumerate(dgms):
            # dgm é um array de pares [b,d]
            betti = np.zeros_like(eps_grid)
            for (b, d) in dgm:
                # Para cada epsilon entre b e d soma 1
                betti += ((eps_grid >= b) & (eps_grid < d)).astype(float)
            curvas[k] = (eps_grid, betti)

        self.curvas_betti = curvas

    def plotar_curvas_betti(self) -> None:
        """
        Plota as curvas de Betti beta_k(eps).
        """
        if self.curvas_betti is None:
            self._construir_curvas_betti()

        plt.figure(figsize=(8, 4))
        for k, (eps, betti) in self.curvas_betti.items():
            plt.plot(eps, betti, label=f"β_{k}(ε)")
        plt.xlabel("ε (escala na filtração de Vietoris–Rips)")
        plt.ylabel("β_k(ε)")
        plt.title("Curvas de Betti — Rede Financeira")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # ------------------------ Entropia persistente ------------------------
    def _calcular_entropias_persistentes(self) -> None:
        """
        H_k = -∑ p_i log p_i, com p_i = ℓ_i / ∑ ℓ_j e ℓ_i = d_i - b_i.
        """
        if self._resultado_ripser is None:
            raise RuntimeError(
                "Resultado de homologia inexistente. "
                "Chame calcular_homologia_persistente() primeiro."
            )

        dgms = self._resultado_ripser["dgms"]
        entropias: Dict[int, float] = {}

        for k, dgm in enumerate(dgms):
            comprimentos = []
            for (b, d) in dgm:
                l = d - b
                if l > 0:
                    comprimentos.append(l)
            if not comprimentos:
                entropias[k] = 0.0
                continue
            comprimentos = np.array(comprimentos)
            p = comprimentos / comprimentos.sum()
            # Evitar log(0)
            p = p[p > 0]
            H_k = float(-(p * np.log(p)).sum())
            entropias[k] = H_k

        self.entropias_persistentes = entropias

    def diagnostico_topologico(self) -> None:
        """
        Imprime resumo das entropias e estatísticas dos diagramas.
        """
        if self.entropias_persistentes is None:
            self._calcular_entropias_persistentes()

        print("=========== DIAGNÓSTICO TOPOLÓGICO DA REDE FINANCEIRA ===========")
        print(f"Ativos: {self.ativos}")
        print("Entropias persistentes H_k:")
        for k, Hk in self.entropias_persistentes.items():
            print(f"  H_{k} = {Hk:.4f}")

        if self._resultado_ripser is not None:
            dgms = self._resultado_ripser["dgms"]
            for k, dgm in enumerate(dgms):
                n_classes = len(dgm)
                print(f"Dimensão {k}: {n_classes} classes de homologia (intervalos).")
        print("==================================================================")


# ------------------------------------------------------------
# 3) Geração de dados sintéticos (caso não haja CSV real)
# ------------------------------------------------------------
def gerar_precos_sinteticos(
    n_ativos: int = 8,
    n_passos: int = 500,
    seed: int = 42
) -> pd.DataFrame:
    """
    Gera uma matriz de preços sintéticos com correlação não trivial
    entre ativos via covariância aleatória positiva-definida.

    1) Gera matriz de covariância Σ positiva-definida.
    2) Simula retornos gaussianos multivariados ~ N(0, Σ).
    3) Constrói preços: S_i(t) = S_i(0) * exp(cumsum(retornos)).
    """
    rng = np.random.default_rng(seed)

    # Matriz aleatória -> covariância
    A = rng.normal(size=(n_ativos, n_ativos))
    Sigma = np.dot(A, A.T)   # PSD
    # Normalizar variâncias (opcional)
    d = np.sqrt(np.diag(Sigma))
    Sigma = Sigma / np.outer(d, d)

    # Simula retornos multivariados
    retornos = rng.multivariate_normal(
        mean=np.zeros(n_ativos),
        cov=Sigma,
        size=n_passos
    )

    # Constrói preços (base 100)
    S0 = 100.0 * np.ones(n_ativos)
    log_precos = np.cumsum(retornos, axis=0) + np.log(S0)
    precos = np.exp(log_precos)

    datas = pd.date_range("2020-01-01", periods=n_passos, freq="B")
    colunas = [f"Ativo_{i+1}" for i in range(n_ativos)]
    df_precos = pd.DataFrame(precos, index=datas, columns=colunas)
    return df_precos

if __name__ == "__main__":
    # 1) Gera dados sintéticos (substitua por dados reais via CSV/Yahoo etc.)
    precos = gerar_precos_sinteticos(n_ativos=10, n_passos=400)

    # 2) Define parâmetros topológicos
    parametros = ParametrosTopologicos(
        max_dim=1,
        max_eps=1.5,
        n_pontos_curva_betti=200
    )

    # 3) Instancia o modelo
    modelo = ModeloTopologicoRedeFinanceira(precos, parametros)

    # 4) Calcula homologia persistente
    if TEM_TDA:
        modelo.calcular_homologia_persistente()

        # 5) Plota diagramas e curvas de Betti
        modelo.plotar_diagramas_persistencia()
        modelo.plotar_curvas_betti()

        # 6) Diagnóstico (entropias persistentes, número de classes etc.)
        modelo.diagnostico_topologico()
    else:
        print(
            "[AVISO] Bibliotecas de TDA (ripser, persim) não instaladas. "
            "Instale para habilitar a parte topológica."
        )

    # 7) (Opcional) Construir grafo de correlações / distâncias
    if TEM_NETWORKX:
        G = modelo.construir_grafo_financeiro()
        print(f"Grafo financeiro: {G.number_of_nodes()} nós, {G.number_of_edges()} arestas.")
    else:
        print(
            "[AVISO] networkx não instalado; o grafo financeiro "
            "não foi explicitamente construído."
        )
