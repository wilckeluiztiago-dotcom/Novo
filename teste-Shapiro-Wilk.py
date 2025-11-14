# ============================================================
# Teste de Normalidade de Shapiro–Wilk em Modelo Complexo
# Autor: Luiz Tiago Wilcke (LT)
# ============================================================

import math
import warnings
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import shapiro
import statsmodels.api as sm

warnings.filterwarnings("ignore")


# ------------------------------------------------------------
# 1) Configurações do experimento
# ------------------------------------------------------------
@dataclass
class ConfiguracaoExperimento:
    n_observacoes: int = 500     # tamanho da amostra
    semente: int = 42            # reprodutibilidade
    proporcao_outliers: float = 0.05  # fração de outliers na resposta
    escala_ruido_normal: float = 1.0  # desvio padrão do ruído "normal"
    escala_ruido_pesado: float = 4.0  # desvio padrão do ruído pesado
    gerar_graficos: bool = True


# ------------------------------------------------------------
# 2) Geração de dados "complexos"
#    (regressão múltipla com não linearidade + heterocedasticidade)
# ------------------------------------------------------------
def gerar_dados_complexos(cfg: ConfiguracaoExperimento) -> pd.DataFrame:
    """
    Gera um dataset artificial para um problema de regressão
    com estrutura mais realista e ruído não-normal.

    Variáveis:
      - idade: anos (18 a 70)
      - renda: renda mensal (em milhares)
      - anos_estudo: anos de escolaridade
      - indice_saude: índice latente de saúde (0–1)
      - y: variável resposta (ex.: custo anual com saúde)
    """
    rng = np.random.default_rng(cfg.semente)
    n = cfg.n_observacoes

    idade = rng.integers(18, 71, size=n)
    renda = rng.lognormal(mean=1.5, sigma=0.6, size=n)  # assimétrica
    anos_estudo = rng.integers(4, 21, size=n)

    # Índice de saúde entre 0 e 1 (quanto maior, melhor saúde)
    indice_saude = np.clip(
        1
        - 0.015 * (idade - 30) / 40
        + 0.01 * (anos_estudo - 8) / 12
        + rng.normal(0, 0.1, size=n),
        0.0,
        1.0,
    )

    # Heterocedasticidade: variância do erro aumenta com a idade
    desvio_ruido_base = cfg.escala_ruido_normal * (1 + 0.03 * (idade - 30) ** 2 / 1000)

    # Mistura de erros normais e "pesados" (tipo t-Student)
    indicador_pesado = rng.random(n) < cfg.proporcao_outliers
    ruido = rng.normal(0, desvio_ruido_base)
    ruido[indicador_pesado] += rng.normal(
        0, cfg.escala_ruido_pesado, size=indicador_pesado.sum()
    )

    # Relação não-linear para a resposta (ex.: custo anual com saúde)
    # y = β0 + β1*idade + β2*idade^2 + β3*renda + β4*anos_estudo + β5*indice_saude + erro
    beta0 = 5.0
    beta1 = 0.3
    beta2 = 0.005
    beta3 = -0.4
    beta4 = -0.1
    beta5 = -8.0

    custo_saude = (
        beta0
        + beta1 * idade
        + beta2 * (idade ** 2)
        + beta3 * renda
        + beta4 * anos_estudo
        + beta5 * indice_saude
        + ruido
    )

    dados = pd.DataFrame(
        {
            "idade": idade,
            "renda": renda,
            "anos_estudo": anos_estudo,
            "indice_saude": indice_saude,
            "custo_saude": custo_saude,
        }
    )
    return dados


# ------------------------------------------------------------
# 3) Classe para modelo de regressão + Shapiro–Wilk
# ------------------------------------------------------------
@dataclass
class ResultadoShapiroWilk:
    estatistica_W: float
    valor_p: float
    tamanho_amostra: int
    conclusao_5pct: str


class TesteNormalidadeShapiroWilk:
    """
    Envolve o teste de Shapiro–Wilk para ser usado de forma
    organizada em pipelines de modelagem.
    """

    def __init__(self, nivel_significancia: float = 0.05) -> None:
        self.nivel_significancia = nivel_significancia

    def aplicar(self, amostra: np.ndarray) -> ResultadoShapiroWilk:
        """
        Aplica o teste de Shapiro–Wilk em uma amostra univariada.
        """
        amostra = np.asarray(amostra, dtype=float)
        amostra = amostra[~np.isnan(amostra)]
        n = len(amostra)

        if n < 3:
            raise ValueError("Shapiro–Wilk requer pelo menos 3 observações.")

        estatistica_W, valor_p = shapiro(amostra)

        if valor_p < self.nivel_significancia:
            conclusao = (
                f"Rejeita H0 (não-normalidade) ao nível de {self.nivel_significancia:.2%}."
            )
        else:
            conclusao = (
                f"Não rejeita H0 (compatível com normalidade) ao nível de "
                f"{self.nivel_significancia:.2%}."
            )

        return ResultadoShapiroWilk(
            estatistica_W=float(estatistica_W),
            valor_p=float(valor_p),
            tamanho_amostra=n,
            conclusao_5pct=conclusao,
        )


class ModeloRegressaoComplexo:
    """
    Ajusta um modelo de regressão (OLS) aos dados complexos
    e aplica o teste de Shapiro–Wilk aos resíduos.
    """

    def __init__(self, dados: pd.DataFrame) -> None:
        self.dados = dados.copy()
        self.modelo_ajustado = None
        self.residuos = None

    def ajustar_modelo(self) -> None:
        """
        Ajusta um modelo OLS com especificação deliberadamente
        simplificada (sem o termo quadrático) para induzir
        não-normalidade e autocorrelação residual.
        """
        y = self.dados["custo_saude"].values
        X = self.dados[["idade", "renda", "anos_estudo", "indice_saude"]]
        X = sm.add_constant(X)

        self.modelo_ajustado = sm.OLS(y, X).fit()
        self.residuos = self.modelo_ajustado.resid

    def relatorio_regressao(self) -> str:
        if self.modelo_ajustado is None:
            raise RuntimeError("Modelo ainda não foi ajustado.")
        return self.modelo_ajustado.summary().as_text()

    def aplicar_shapiro_residuos(
        self, nivel_significancia: float = 0.05
    ) -> ResultadoShapiroWilk:
        if self.residuos is None:
            raise RuntimeError("Resíduos ainda não foram calculados.")
        teste = TesteNormalidadeShapiroWilk(nivel_significancia=nivel_significancia)
        return teste.aplicar(self.residuos)

    def qqplot_residuos(self) -> None:
        """
        Gera QQ-plot dos resíduos para avaliação visual da normalidade.
        """
        if self.residuos is None:
            raise RuntimeError("Resíduos ainda não foram calculados.")

        sm.qqplot(self.residuos, line="s")
        plt.title("QQ-plot dos resíduos (modelo complexo)")
        plt.tight_layout()
        plt.show()


# ------------------------------------------------------------
# 4) Função principal de demonstração
# ------------------------------------------------------------
def executar_experimento() -> None:
    cfg = ConfiguracaoExperimento()
    dados = gerar_dados_complexos(cfg)

    print("\n================= VISÃO GERAL DOS DADOS =================")
    print(dados.describe().T)

    modelo = ModeloRegressaoComplexo(dados)
    modelo.ajustar_modelo()

    print("\n================= RESUMO DO MODELO OLS ==================")
    print(modelo.relatorio_regressao())

    # Teste de Shapiro–Wilk nos resíduos
    resultado_shapiro = modelo.aplicar_shapiro_residuos(nivel_significancia=0.05)

    print("\n============= TESTE DE SHAPIRO–WILK (RESÍDUOS) ==========")
    print(f"Tamanho da amostra (resíduos): {resultado_shapiro.tamanho_amostra}")
    print(f"Estatística W:                 {resultado_shapiro.estatistica_W:.6f}")
    print(f"Valor-p:                       {resultado_shapiro.valor_p:.6e}")
    print(f"Conclusão (5%):                {resultado_shapiro.conclusao_5pct}")

    # QQ-plot dos resíduos
    modelo.qqplot_residuos()


# ------------------------------------------------------------
# Execução direta
# ------------------------------------------------------------
if __name__ == "__main__":
    executar_experimento()
