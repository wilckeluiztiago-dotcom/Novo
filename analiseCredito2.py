# ============================================================
# MODELO DE CRÉDITO
# Equações Diferenciais + Processo de Lévy (α-estável)
# Autor: Luiz Tiago Wilcke (LT)
# ============================================================
# Ideia geral:
#   - Modelar a "saúde de crédito" de cada empresa via SDE com saltos de Lévy
#   - Resolver numericamente EDOs associadas à intensidade de default
#   - Simular uma carteira de crédito (N empresas)
#   - Gerar indicadores avançados: PD(t), curva de sobrevivência, LGD, EAD,
#     perda esperada, perda inesperada, VaR, ES, contribuições marginais
#     e gráficos para análise de risco de crédito.
#
# Observação:
#   - Código focado em ser didático e complexo. Ajuste tamanhos (N, caminhos)
#     se ficar pesado.
# ============================================================

import math
import random
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import levy_stable, norm
from scipy.integrate import solve_ivp

# ============================================================
# 1. CONFIGURAÇÕES GERAIS DO MODELO
# ============================================================

@dataclass
class ParametrosGerais:
    # Horizonte temporal
    horizonte_anos: float = 5.0
    passos_por_ano: int = 252        # dias úteis

    # Tamanho da carteira
    numero_empresas: int = 250

    # Semente de aleatoriedade
    seed: int = 42

    # Número de cenários Monte Carlo
    numero_cenarios: int = 2000

    # Taxa livre de risco (para descontar perdas)
    taxa_livre_risco_anual: float = 0.08

    # Nível de confiança para VaR / ES
    nivel_confianca: float = 0.99


@dataclass
class ParametrosLevy:
    # Parâmetros da distribuição α-estável (Lévy)
    alpha: float = 1.5   # 0 < alpha <= 2 (1.5 gera caudas pesadas)
    beta: float = 0.0    # simetria
    scale: float = 1.0   # escala
    loc: float = 0.0     # deslocamento

    # Intensidade de saltos (frequência) ao ano
    intensidade_saltos: float = 5.0

    # Fator de escala para impacto do salto na saúde de crédito
    fator_impacto: float = 0.5


@dataclass
class ParametrosSDE:
    # SDE da saúde de crédito X_t:
    #   dX_t = kappa * (media_reversao - X_t) dt
    #          + sigma * dW_t
    #          + fator_impacto * dL_t
    #
    # X_t alto = saúde boa (menor prob de default)
    # X_t baixo = saúde ruim (maior prob de default)

    kappa: float = 1.0            # velocidade de reversão
    media_reversao: float = 1.0   # nível de longo prazo
    sigma: float = 0.6            # volatilidade Browniana

    # Barreira de default
    barreira_default: float = -2.0

    # Valor inicial de saúde de crédito típico por rating
    # será amostrado a partir de distribuições por faixa de rating


@dataclass
class ParametrosEmpresas:
    # Distribuição de rating inicial da carteira
    proporcao_AAA: float = 0.05
    proporcao_AA: float = 0.10
    proporcao_A: float = 0.20
    proporcao_BBB: float = 0.30
    proporcao_BB: float = 0.20
    proporcao_B: float = 0.10
    proporcao_CCC: float = 0.05

    # Exposição a cada crédito (EAD inicial)
    ead_min: float = 1e5    # 100k
    ead_max: float = 1e7    # 10M

    # LGD média por rating (em %)
    lgd_rating: Dict[str, float] = None

    def __post_init__(self):
        if self.lgd_rating is None:
            self.lgd_rating = {
                "AAA": 0.25,
                "AA": 0.30,
                "A": 0.35,
                "BBB": 0.40,
                "BB": 0.45,
                "B": 0.55,
                "CCC": 0.65,
            }


# ============================================================
# 2. GERADOR DE CAMINHOS DO PROCESSO DE LÉVY
# ============================================================

class GeradorLevy:
    def __init__(self, parametros: ParametrosLevy, parametros_gerais: ParametrosGerais):
        self.parametros = parametros
        self.parametros_gerais = parametros_gerais
        self._rng = np.random.default_rng(parametros_gerais.seed)

    def gerar_incrementos_levy(self, numero_passos: int) -> np.ndarray:
        """
        Gera uma sequência de incrementos de um processo de Lévy α-estável
        (sem drift) ajustado para o passo de tempo Δt.
        """
        dt = 1.0 / self.parametros_gerais.passos_por_ano
        # Intensidade de saltos: poisson para saber quantos saltos no intervalo
        # mas aqui vamos simplificar: um incremento α-estável por passo
        # com escala ajustada por sqrt(dt) ou dt^(1/alpha)
        escala_dt = self.parametros.scale * (dt ** (1.0 / self.parametros.alpha))
        inc = levy_stable.rvs(
            self.parametros.alpha,
            self.parametros.beta,
            loc=self.parametros.loc,
            scale=escala_dt,
            size=numero_passos,
            random_state=self._rng.integers(0, 1_000_000),
        )
        return inc

    def gerar_incrementos_poisson(self, numero_passos: int) -> np.ndarray:
        """
        Gera número de saltos em cada passo via processo de Poisson com
        intensidade especificada.
        """
        dt = 1.0 / self.parametros_gerais.passos_por_ano
        lambda_dt = self.parametros.intensidade_saltos * dt
        return self._rng.poisson(lambda_dt, size=numero_passos)

    def gerar_caminho_levy_composto(self, numero_passos: int) -> np.ndarray:
        """
        Combina Poisson + incrementos α-estáveis: cada salto de Poisson
        gera um incremento α-estável; soma todos.
        """
        num_saltos = self.gerar_incrementos_poisson(numero_passos)
        caminho = np.zeros(numero_passos)
        for t in range(numero_passos):
            if num_saltos[t] > 0:
                # soma num_saltos réplicas independentes
                inc = levy_stable.rvs(
                    self.parametros.alpha,
                    self.parametros.beta,
                    loc=self.parametros.loc,
                    scale=self.parametros.scale,
                    size=num_saltos[t],
                    random_state=self._rng.integers(0, 1_000_000),
                )
                caminho[t] = inc.sum()
        return caminho


# ============================================================
# 3. SDE DA SAÚDE DE CRÉDITO COM SALTOS DE LÉVY
# ============================================================

class SimuladorSaudeCredito:
    def __init__(
        self,
        parametros_gerais: ParametrosGerais,
        parametros_sde: ParametrosSDE,
        gerador_levy: GeradorLevy,
    ):
        self.pg = parametros_gerais
        self.ps = parametros_sde
        self.gerador_levy = gerador_levy
        self.dt = 1.0 / self.pg.passos_por_ano

        self._rng = np.random.default_rng(self.pg.seed)

    def _incrementos_brownianos(self, numero_passos: int) -> np.ndarray:
        return self._rng.normal(
            loc=0.0,
            scale=math.sqrt(self.dt),
            size=numero_passos
        )

    def simular_trajetoria(
        self,
        x0: float,
        numero_passos: int,
        usar_levy_composto: bool = True,
    ) -> np.ndarray:
        """
        Simula uma trajetória de X_t (saúde de crédito) usando esquema de Euler.
        """
        X = np.zeros(numero_passos + 1)
        X[0] = x0

        inc_brown = self._incrementos_brownianos(numero_passos)

        if usar_levy_composto:
            inc_levy = self.gerador_levy.gerar_caminho_levy_composto(numero_passos)
        else:
            inc_levy = self.gerador_levy.gerar_incrementos_levy(numero_passos)

        for t in range(numero_passos):
            drift = self.ps.kappa * (self.ps.media_reversao - X[t])
            difusao = self.ps.sigma * inc_brown[t]
            salto = self.gerador_levy.parametros.fator_impacto * inc_levy[t]
            X[t + 1] = X[t] + drift * self.dt + difusao + salto

        return X

    def simular_varias_trajetorias(
        self,
        x0: float,
        numero_passos: int,
        numero_cenarios: int,
        usar_levy_composto: bool = True,
    ) -> np.ndarray:
        """
        Retorna matriz [cenário, tempo] com trajetórias.
        """
        caminhos = np.zeros((numero_cenarios, numero_passos + 1))
        for c in range(numero_cenarios):
            caminhos[c] = self.simular_trajetoria(
                x0=x0,
                numero_passos=numero_passos,
                usar_levy_composto=usar_levy_composto,
            )
        return caminhos


# ============================================================
# 4. MAPEAMENTO DE SAÚDE DE CRÉDITO -> INTENSIDADE DE DEFAULT
# ============================================================

def intensidade_default_de_saude(saude: np.ndarray, escala: float = 0.5) -> np.ndarray:
    """
    Transforma saúde de crédito X_t em intensidade de default λ_t.
    Use uma função decrescente em X_t (melhor saúde -> menor λ).
    Exemplo: λ_t = exp( - escala * X_t ).
    """
    return np.exp(-escala * saude)


def resolver_edo_sobrevivencia(
    tempos: np.ndarray,
    intensidade: np.ndarray,
    s0: float = 1.0,
) -> np.ndarray:
    """
    Resolve EDO dS/dt = -λ(t) S(t) com condição inicial S(0) = s0.
    Usa solve_ivp para maior generalidade, embora a solução tenha forma fechada.
    """

    def rhs(t, s):
        # interpolar intensidade λ(t)
        # tempos está em anos, intensidade é array discreto
        if t <= tempos[0]:
            lam = intensidade[0]
        elif t >= tempos[-1]:
            lam = intensidade[-1]
        else:
            lam = np.interp(t, tempos, intensidade)
        return -lam * s

    sol = solve_ivp(
        fun=rhs,
        t_span=(tempos[0], tempos[-1]),
        y0=[s0],
        t_eval=tempos,
        method="RK45",
        rtol=1e-6,
        atol=1e-9,
    )

    return sol.y[0]


# ============================================================
# 5. GERAÇÃO DE CARTEIRA SINTÉTICA DE CRÉDITO
# ============================================================

class GeradorCarteiraCredito:
    def __init__(
        self,
        parametros_gerais: ParametrosGerais,
        parametros_empresas: ParametrosEmpresas,
        parametros_sde: ParametrosSDE,
    ):
        self.pg = parametros_gerais
        self.pe = parametros_empresas
        self.ps = parametros_sde
        self._rng = np.random.default_rng(self.pg.seed)

        # mapa rating -> média e desvio padrão de X0
        self.map_x0_rating = {
            "AAA": (2.5, 0.3),
            "AA": (2.2, 0.3),
            "A": (1.8, 0.4),
            "BBB": (1.5, 0.4),
            "BB": (1.0, 0.5),
            "B": (0.5, 0.6),
            "CCC": (0.0, 0.8),
        }

    def gerar_ratings(self) -> List[str]:
        n = self.pg.numero_empresas
        props = [
            self.pe.proporcao_AAA,
            self.pe.proporcao_AA,
            self.pe.proporcao_A,
            self.pe.proporcao_BBB,
            self.pe.proporcao_BB,
            self.pe.proporcao_B,
            self.pe.proporcao_CCC,
        ]
        ratings = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]
        probs = np.array(props) / sum(props)
        amostra = self._rng.choice(ratings, size=n, p=probs)
        return list(amostra)

    def gerar_ead(self, n: int) -> np.ndarray:
        return self._rng.uniform(self.pe.ead_min, self.pe.ead_max, size=n)

    def gerar_lgd(self, ratings: List[str]) -> np.ndarray:
        lgd = np.zeros(len(ratings))
        for i, r in enumerate(ratings):
            media = self.pe.lgd_rating.get(r, 0.45)
            # pequena variação em torno da média
            lgd[i] = np.clip(
                self._rng.normal(loc=media, scale=0.05),
                0.05,
                0.95
            )
        return lgd

    def gerar_x0(self, ratings: List[str]) -> np.ndarray:
        x0 = np.zeros(len(ratings))
        for i, r in enumerate(ratings):
            m, s = self.map_x0_rating[r]
            x0[i] = self._rng.normal(m, s)
        return x0

    def gerar_carteira(self) -> pd.DataFrame:
        ratings = self.gerar_ratings()
        ead = self.gerar_ead(len(ratings))
        lgd = self.gerar_lgd(ratings)
        x0 = self.gerar_x0(ratings)

        df = pd.DataFrame({
            "id_emprestimo": np.arange(len(ratings)),
            "rating_inicial": ratings,
            "ead_inicial": ead,
            "lgd": lgd,
            "saude_inicial": x0,
        })
        return df


# ============================================================
# 6. SIMULAÇÃO DA CARTEIRA AO LONGO DO TEMPO
# ============================================================

class SimuladorCarteira:
    def __init__(
        self,
        parametros_gerais: ParametrosGerais,
        parametros_sde: ParametrosSDE,
        carteira: pd.DataFrame,
        simulador_saude: SimuladorSaudeCredito,
    ):
        self.pg = parametros_gerais
        self.ps = parametros_sde
        self.carteira = carteira.copy()
        self.simulador_saude = simulador_saude

        self.numero_passos = int(self.pg.horizonte_anos * self.pg.passos_por_ano)
        self.tempos_anos = np.linspace(
            0.0,
            self.pg.horizonte_anos,
            self.numero_passos + 1,
        )

        self._rng = np.random.default_rng(self.pg.seed + 10)

    def simular_saude_empresas(self) -> np.ndarray:
        """
        Simula a saúde de crédito X_t para cada empresa e cenário.
        Retorna array [empresa, cenário, tempo].
        """
        n_emp = len(self.carteira)
        n_cen = self.pg.numero_cenarios
        n_passos = self.numero_passos

        saude = np.zeros((n_emp, n_cen, n_passos + 1))

        for i in range(n_emp):
            x0 = float(self.carteira.loc[i, "saude_inicial"])
            # para cada cenário da empresa i
            caminhos = self.simulador_saude.simular_varias_trajetorias(
                x0=x0,
                numero_passos=n_passos,
                numero_cenarios=n_cen,
                usar_levy_composto=True,
            )
            saude[i] = caminhos

        return saude

    def calcular_default_e_perda(
        self,
        saude: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        A partir do campo de saúde de crédito, determina o tempo de default
        e a perda de cada empresa cenário. Usa barreira de default e LGD.
        Retorna:
            - matriz_default: [empresa, cenário] com tempo (em anos) do default
                              ou NaN se nunca defaulta
            - matriz_perda: [empresa, cenário] com perda (descontada) no evento
        """
        n_emp, n_cen, n_t = saude.shape
        dt = 1.0 / self.pg.passos_por_ano
        taxa_continua = math.log(1 + self.pg.taxa_livre_risco_anual)

        matriz_default = np.full((n_emp, n_cen), np.nan)
        matriz_perda = np.zeros((n_emp, n_cen))

        ead = self.carteira["ead_inicial"].values
        lgd = self.carteira["lgd"].values

        for i in range(n_emp):
            for c in range(n_cen):
                # localizar primeiro cruzamento da barreira de default
                indices = np.where(
                    saude[i, c, :] <= self.ps.barreira_default
                )[0]
                if len(indices) > 0:
                    idx_default = indices[0]
                    tempo_default = idx_default * dt
                    matriz_default[i, c] = tempo_default

                    # perda = LGD * EAD descontada
                    perda_bruta = lgd[i] * ead[i]
                    fator_desconto = math.exp(-taxa_continua * tempo_default)
                    matriz_perda[i, c] = perda_bruta * fator_desconto
                else:
                    matriz_perda[i, c] = 0.0

        return matriz_default, matriz_perda

    def construir_curva_pd_portfolio(
        self,
        matriz_default: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Constrói curva PD(t) do portfólio: probabilidade cumulativa de pelo menos
        um default até o tempo t, ou probabilidade marginal por intervalo.
        Aqui calculamos PD média de cada empresa (default até t) e fazemos média.
        """
        n_emp, n_cen = matriz_default.shape
        n_passos = self.numero_passos
        dt = 1.0 / self.pg.passos_por_ano

        pd_tempo = np.zeros(n_passos + 1)

        for k in range(n_passos + 1):
            t_atual = k * dt
            # empresa defaulta até t_atual se tempo_default <= t_atual
            indicadores = (matriz_default <= t_atual)
            prob_empresas = indicadores.sum(axis=1) / n_cen
            pd_tempo[k] = prob_empresas.mean()

        tempos = np.linspace(0.0, self.pg.horizonte_anos, n_passos + 1)
        return tempos, pd_tempo

    def simular(self) -> Dict[str, object]:
        """
        Pipeline completo da simulação da carteira:
            - simula saude
            - calcula default/perda
            - constrói curva PD
            - monta várias estatísticas
        """
        print("Simulando saúde de crédito das empresas...")
        saude = self.simular_saude_empresas()

        print("Calculando tempos de default e perdas...")
        matriz_default, matriz_perda = self.calcular_default_e_perda(saude)

        print("Construindo curva de probabilidade de default do portfólio...")
        tempos_pd, pd_tempo = self.construir_curva_pd_portfolio(matriz_default)

        resultados = {
            "saude": saude,
            "matriz_default": matriz_default,
            "matriz_perda": matriz_perda,
            "tempos_pd": tempos_pd,
            "pd_tempo": pd_tempo,
        }
        return resultados


# ============================================================
# 7. INDICADORES AVANÇADOS DE RISCO DE CRÉDITO
# ============================================================

class CalculadoraIndicadoresRisco:
    def __init__(
        self,
        parametros_gerais: ParametrosGerais,
        carteira: pd.DataFrame,
        resultados_simulacao: Dict[str, object],
    ):
        self.pg = parametros_gerais
        self.carteira = carteira
        self.res = resultados_simulacao

        self.matriz_perda = self.res["matriz_perda"]
        self.matriz_default = self.res["matriz_default"]

        self.numero_empresas = len(self.carteira)
        self.numero_cenarios = self.pg.numero_cenarios

    def perdas_totais_cenario(self) -> np.ndarray:
        """
        Soma as perdas (descontadas) de todas as empresas em cada cenário.
        """
        return self.matriz_perda.sum(axis=0)

    def perda_esperada_total(self) -> float:
        return self.perdas_totais_cenario().mean()

    def perda_inesperada_total(self) -> float:
        perdas = self.perdas_totais_cenario()
        return perdas.std(ddof=1)

    def var_portfolio(self) -> float:
        """
        Calcula VaR no nível de confiança especificado.
        VaR = quantil_{α} da distribuição de perdas.
        """
        perdas = self.perdas_totais_cenario()
        alpha = self.pg.nivel_confianca
        return np.quantile(perdas, alpha)

    def es_portfolio(self) -> float:
        """
        Calcula ES (Expected Shortfall) no nível de confiança especificado:
        média condicional das perdas acima do VaR.
        """
        perdas = self.perdas_totais_cenario()
        alpha = self.pg.nivel_confianca
        var = np.quantile(perdas, alpha)
        perdas_extremas = perdas[perdas >= var]
        if len(perdas_extremas) == 0:
            return var
        return perdas_extremas.mean()

    def pd_marginal_anual_media(self) -> float:
        """
        Estima PD marginal anual média da carteira (1º ano).
        """
        matriz_default = self.matriz_default
        dt = 1.0 / self.pg.passos_por_ano
        indice_1ano = int(1.0 / dt)

        indicadores = (matriz_default <= 1.0)
        prob_empresas = indicadores.sum(axis=1) / matriz_default.shape[1]
        return prob_empresas.mean()

    def pd_termino_horizonte(self) -> float:
        """
        Probabilidade média de default até o fim do horizonte.
        """
        matriz_default = self.matriz_default
        horizontes = np.isfinite(matriz_default)
        prob_empresas = horizontes.sum(axis=1) / matriz_default.shape[1]
        return prob_empresas.mean()

    def contribuicao_marginal_var(
        self,
        numero_empresas_amostrar: int = 30,
    ) -> pd.DataFrame:
        """
        Cálculo simplificado de contribuição marginal ao VaR:
        - Amostra subconjunto de empresas.
        - Para cada uma, "remove" da carteira (perdas = 0) e recalcula VaR.
        - Marginal_i = VaR_total - VaR_sem_i
        """
        perdas_total = self.perdas_totais_cenario()
        var_total = np.quantile(perdas_total, self.pg.nivel_confianca)

        n_emp = self.numero_empresas
        indices = np.arange(n_emp)
        if numero_empresas_amostrar < n_emp:
            indices = np.random.choice(
                indices,
                size=numero_empresas_amostrar,
                replace=False,
            )

        lista_ids = []
        lista_marginal = []

        for i in indices:
            perdas_sem_i = perdas_total - self.matriz_perda[i, :]
            var_sem_i = np.quantile(perdas_sem_i, self.pg.nivel_confianca)
            marginal = var_total - var_sem_i
            lista_ids.append(i)
            lista_marginal.append(marginal)

        df = pd.DataFrame({
            "id_emprestimo": lista_ids,
            "marginal_var": lista_marginal,
        }).merge(
            self.carteira,
            on="id_emprestimo",
            how="left",
        ).sort_values("marginal_var", ascending=False)

        return df

    def indicadores_resumidos(self) -> Dict[str, float]:
        perda_esp = self.perda_esperada_total()
        perda_inesp = self.perda_inesperada_total()
        var = self.var_portfolio()
        es = self.es_portfolio()
        pd_anual = self.pd_marginal_anual_media()
        pd_horizonte = self.pd_termino_horizonte()

        return {
            "perda_esperada_total": perda_esp,
            "perda_inesperada_total": perda_inesp,
            "var_portfolio": var,
            "es_portfolio": es,
            "pd_marginal_anual_media": pd_anual,
            "pd_acumulada_horizonte": pd_horizonte,
        }


# ============================================================
# 8. FUNÇÕES DE GRÁFICOS E DIAGNÓSTICOS
# ============================================================

def plotar_caminhos_saude(
    tempos: np.ndarray,
    saude: np.ndarray,
    numero_empresas_plot: int = 5,
    numero_cenarios_plot: int = 5,
):
    """
    Plota alguns caminhos da saúde de crédito.
    saude: [empresa, cenário, tempo]
    """
    n_emp, n_cen, _ = saude.shape
    empresas = np.arange(n_emp)
    cenarios = np.arange(n_cen)

    empresas_plot = np.random.choice(
        empresas,
        size=min(numero_empresas_plot, n_emp),
        replace=False,
    )
    cenarios_plot = np.random.choice(
        cenarios,
        size=min(numero_cenarios_plot, n_cen),
        replace=False,
    )

    plt.figure(figsize=(12, 6))
    for i in empresas_plot:
        for c in cenarios_plot:
            plt.plot(
                tempos,
                saude[i, c, :],
                alpha=0.4,
            )
    plt.axhline(
        y=-2.0,
        color="red",
        linestyle="--",
        label="Barreira de Default",
    )
    plt.title("Trajetórias da Saúde de Crédito (amostra)")
    plt.xlabel("Tempo (anos)")
    plt.ylabel("Saúde de crédito X_t")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plotar_curva_pd(tempos: np.ndarray, pd_tempo: np.ndarray):
    plt.figure(figsize=(10, 5))
    plt.plot(tempos, pd_tempo, lw=2)
    plt.title("Curva de Probabilidade de Default da Carteira")
    plt.xlabel("Tempo (anos)")
    plt.ylabel("PD acumulada média")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plotar_histograma_perdas(perdas_totais: np.ndarray):
    plt.figure(figsize=(10, 5))
    plt.hist(perdas_totais, bins=40, alpha=0.7, density=False)
    plt.title("Distribuição de Perdas da Carteira (descontadas)")
    plt.xlabel("Perda")
    plt.ylabel("Frequência")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plotar_curva_capital(perdas: np.ndarray):
    """
    Curva quantil x perda (capital econômico).
    """
    quantis = np.linspace(0.90, 0.999, 30)
    valores = np.quantile(perdas, quantis)

    plt.figure(figsize=(10, 5))
    plt.plot(quantis, valores, marker="o")
    plt.title("Curva de Capital Econômico (quantis da perda)")
    plt.xlabel("Quantil")
    plt.ylabel("Perda")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plotar_marginais_var(df_marginal: pd.DataFrame, top_k: int = 20):
    """
    Plota top_k empréstimos que mais contribuem para o VaR.
    """
    df_top = df_marginal.head(top_k)
    plt.figure(figsize=(12, 6))
    plt.bar(
        df_top["id_emprestimo"].astype(str),
        df_top["marginal_var"],
    )
    plt.title(f"Top {top_k} contribuições marginais ao VaR")
    plt.xlabel("ID Empréstimo")
    plt.ylabel("Marginal VaR")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


# ============================================================
# 9. FUNÇÃO PRINCIPAL PARA RODAR O MODELO
# ============================================================

def executar_modelagem_avancada_credito():
    # ---------------------------
    # Configuração dos parâmetros
    # ---------------------------
    parametros_gerais = ParametrosGerais(
        horizonte_anos=5.0,
        passos_por_ano=252,
        numero_empresas=200,        # pode aumentar, mas fica pesado
        seed=123,
        numero_cenarios=1000,       # pode aumentar
        taxa_livre_risco_anual=0.08,
        nivel_confianca=0.99,
    )

    parametros_levy = ParametrosLevy(
        alpha=1.5,
        beta=0.0,
        scale=1.0,
        loc=0.0,
        intensidade_saltos=3.0,
        fator_impacto=0.4,
    )

    parametros_sde = ParametrosSDE(
        kappa=1.2,
        media_reversao=1.0,
        sigma=0.7,
        barreira_default=-2.0,
    )

    parametros_empresas = ParametrosEmpresas(
        proporcao_AAA=0.05,
        proporcao_AA=0.10,
        proporcao_A=0.20,
        proporcao_BBB=0.30,
        proporcao_BB=0.20,
        proporcao_B=0.10,
        proporcao_CCC=0.05,
    )

    # ---------------------------
    # 1) Gerar carteira sintética
    # ---------------------------
    gerador_carteira = GeradorCarteiraCredito(
        parametros_gerais=parametros_gerais,
        parametros_empresas=parametros_empresas,
        parametros_sde=parametros_sde,
    )
    carteira = gerador_carteira.gerar_carteira()

    print("Amostra da carteira de crédito:")
    print(carteira.head())

    # ---------------------------
    # 2) Criar gerador de Lévy e simulador de saúde
    # ---------------------------
    gerador_levy = GeradorLevy(
        parametros=parametros_levy,
        parametros_gerais=parametros_gerais,
    )

    simulador_saude = SimuladorSaudeCredito(
        parametros_gerais=parametros_gerais,
        parametros_sde=parametros_sde,
        gerador_levy=gerador_levy,
    )

    # ---------------------------
    # 3) Simular carteira
    # ---------------------------
    simulador_carteira = SimuladorCarteira(
        parametros_gerais=parametros_gerais,
        parametros_sde=parametros_sde,
        carteira=carteira,
        simulador_saude=simulador_saude,
    )

    resultados = simulador_carteira.simular()

    # ---------------------------
    # 4) Calcular indicadores
    # ---------------------------
    calc = CalculadoraIndicadoresRisco(
        parametros_gerais=parametros_gerais,
        carteira=carteira,
        resultados_simulacao=resultados,
    )

    indicadores = calc.indicadores_resumidos()

    print("\n================ INDICADORES RESUMIDOS ================")
    for k, v in indicadores.items():
        print(f"{k}: {v:,.2f}")

    perdas_totais = calc.perdas_totais_cenario()

    # contribuição marginal ao VaR
    df_marginal = calc.contribuicao_marginal_var(numero_empresas_amostrar=40)

    print("\nTop empréstimos por contribuição marginal ao VaR:")
    print(df_marginal.head(10))

    # ---------------------------
    # 5) Gráficos
    # ---------------------------
    saude = resultados["saude"]
    tempos = simulador_carteira.tempos_anos
    tempos_pd = resultados["tempos_pd"]
    pd_tempo = resultados["pd_tempo"]

    print("\nGerando gráficos (podem ser muitos, feche as janelas para seguir)...")

    plotar_caminhos_saude(tempos, saude, numero_empresas_plot=5, numero_cenarios_plot=5)
    plotar_curva_pd(tempos_pd, pd_tempo)
    plotar_histograma_perdas(perdas_totais)
    plotar_curva_capital(perdas_totais)
    plotar_marginais_var(df_marginal, top_k=20)

    # Retorno para uso em outros scripts, se necessário
    return {
        "parametros_gerais": parametros_gerais,
        "parametros_levy": parametros_levy,
        "parametros_sde": parametros_sde,
        "parametros_empresas": parametros_empresas,
        "carteira": carteira,
        "resultados": resultados,
        "indicadores": indicadores,
        "df_marginal": df_marginal,
        "perdas_totais": perdas_totais,
    }


# ============================================================
# 10. EXECUÇÃO DIRETA
# ============================================================

if __name__ == "__main__":
    # Rodar modelagem avançada de crédito
    resultados_completos = executar_modelagem_avancada_credito()
