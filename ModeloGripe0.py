# ============================================================
# Modelo EDP da Propagação da Gripe em 1D (SIR com Difusão)
# Autor: Luiz Tiago Wilcke 
# ============================================================

import math
from dataclasses import dataclass
from typing import Tuple, Dict, List

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. Parâmetros do modelo epidemiológico
# ============================================================

@dataclass
class ParametrosEpidemia:
    """
    Parâmetros epidemiológicos para o modelo SIR com difusão.
    Todas as taxas estão em unidades compatíveis com 'dias' e 'km'.
    """
    taxa_transmissao_base: float = 0.8      # β0  (contato médio por dia)
    taxa_recuperacao: float = 0.2           # γ   (1 / duração média da infecção)
    taxa_mortalidade_natural: float = 1e-4  # μ   (muito pequena)
    taxa_natalidade: float = 1e-4           # ν   (mantém população ~ constante)
    intensidade_sazonal: float = 0.2        # ε   (variação sazonal da transmissão)
    periodo_sazonal_dias: float = 365.0     # T   (sazonalidade anual)
    difusao_suscetiveis: float = 0.05       # D_S (km² / dia)
    difusao_infectados: float = 0.08        # D_I (km² / dia)
    difusao_recuperados: float = 0.02       # D_R (km² / dia)


# ============================================================
# 2. Parâmetros numéricos (malha espacial e temporal)
# ============================================================

@dataclass
class ParametrosNumericos:
    """
    Parâmetros numéricos para o esquema de diferenças finitas.
    """
    comprimento_km: float = 100.0    # intervalo espacial [0, L]
    pontos_espaciais: int = 201      # número de pontos de grade em x
    tempo_final_dias: float = 120.0  # simulação em dias
    dt_inicial: float = 0.01         # passo de tempo inicial (ajustado pela estabilidade)
    salvar_cada_n_passos: int = 50   # frequência de salvamento de snapshots


# ============================================================
# 3. Funções auxiliares — sazonalidade e derivadas
# ============================================================

def taxa_transmissao_sazonal(t: float, params: ParametrosEpidemia) -> float:
    """
    Taxa de transmissão β(t) com sazonalidade:
        β(t) = β0 * (1 + ε * cos(2π t / T))
    """
    beta0 = params.taxa_transmissao_base
    eps = params.intensidade_sazonal
    T = params.periodo_sazonal_dias
    return beta0 * (1.0 + eps * math.cos(2.0 * math.pi * t / T))


def calcular_laplaciano_1d(
    campo: np.ndarray,
    dx: float
) -> np.ndarray:
    """
    Calcula o Laplaciano 1D usando diferenças centrais de segunda ordem.
    Condições de contorno: Neumann (derivada espacial zero nas bordas).
    """
    n = campo.size
    lap = np.zeros_like(campo)

    # Condições de contorno de Neumann aproximadas:
    # ∂u/∂x = 0 -> u_(-1) = u_1  e u_(N) = u_(N-2)
    # Implementação explícita nas bordas:
    lap[0] = (campo[1] - campo[0]) / (dx ** 2)
    lap[-1] = (campo[-2] - campo[-1]) / (dx ** 2)

    # Pontos internos
    for i in range(1, n - 1):
        lap[i] = (campo[i + 1] - 2.0 * campo[i] + campo[i - 1]) / (dx ** 2)

    return lap


# ============================================================
# 4. Inicialização da malha espacial e condições iniciais
# ============================================================

@dataclass
class MalhaEspacial1D:
    """
    Representa uma malha espacial 1D uniforme.
    """
    x_min: float
    x_max: float
    n_pontos: int

    def __post_init__(self):
        self.x = np.linspace(self.x_min, self.x_max, self.n_pontos)
        self.dx = (self.x_max - self.x_min) / (self.n_pontos - 1)


def condicao_inicial_populacoes(
    malha: MalhaEspacial1D,
    populacao_total: float = 1.0,
    largura_foco_inicial_km: float = 5.0,
    posicao_foco_km: float = 50.0,
    fracao_infectados_inicial: float = 0.01
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Constrói condições iniciais S(x,0), I(x,0), R(x,0).
    - Suscetíveis quase toda a população
    - Infectados concentrados em um foco Gaussiano
    - Recuperados inicialmente zero
    """
    x = malha.x
    S = np.full_like(x, populacao_total, dtype=float)
    I = np.zeros_like(x, dtype=float)
    R = np.zeros_like(x, dtype=float)

    # Foco Gaussiano de infectados
    sigma = largura_foco_inicial_km / 2.0
    gauss = np.exp(-0.5 * ((x - posicao_foco_km) / sigma) ** 2)

    # Normaliza para que a fração infectada inicial seja aproximadamente fracao_infectados_inicial
    gauss = gauss / gauss.max()
    I0 = populacao_total * fracao_infectados_inicial
    I = I0 * gauss

    # Ajusta S para manter S + I + R ~ populacao_total (em cada ponto)
    S = np.clip(populacao_total - I, 0.0, None)
    R[:] = 0.0

    return S, I, R


# ============================================================
# 5. Núcleo do modelo EDP SIR com difusão
# ============================================================

@dataclass
class EstadoSIR:
    """
    Estrutura para armazenar os campos S(x), I(x), R(x).
    """
    S: np.ndarray
    I: np.ndarray
    R: np.ndarray


@dataclass
class ModeloGripeEDP:
    """
    Modelo de propagação da gripe em 1D usando um sistema de EDPs SIR com difusão.
    """
    params_epi: ParametrosEpidemia
    params_num: ParametrosNumericos
    malha: MalhaEspacial1D
    estado: EstadoSIR
    tempo_atual: float = 0.0
    dt_real: float = 0.0

    def __post_init__(self):
        # Ajusta dt pela condição de estabilidade do termo difusivo
        self.ajustar_dt_por_estabilidade()

    # --------------------------------------------------------
    # Estabilidade numérica (critério CFL para difusão)
    # --------------------------------------------------------

    def ajustar_dt_por_estabilidade(self):
        """
        Garante que o passo de tempo dt satisfaça aproximadamente
        a condição de estabilidade para o esquema explícito de difusão:
            dt <= dx² / (2 * D_max)
        """
        D_max = max(
            self.params_epi.difusao_suscetiveis,
            self.params_epi.difusao_infectados,
            self.params_epi.difusao_recuperados
        )

        dx = self.malha.dx

        if D_max > 0:
            dt_limite = dx ** 2 / (2.0 * D_max)
            # Aplica um fator de segurança
            self.dt_real = min(self.params_num.dt_inicial, 0.5 * dt_limite)
        else:
            self.dt_real = self.params_num.dt_inicial

    # --------------------------------------------------------
    # Cálculo das derivadas temporais
    # --------------------------------------------------------

    def derivadas_temporais(self, t: float, estado: EstadoSIR) -> EstadoSIR:
        """
        Calcula as derivadas temporais dS/dt, dI/dt, dR/dt em cada ponto da malha.
        """
        S = estado.S
        I = estado.I
        R = estado.R

        beta_t = taxa_transmissao_sazonal(t, self.params_epi)
        gamma = self.params_epi.taxa_recuperacao
        mu = self.params_epi.taxa_mortalidade_natural
        nu = self.params_epi.taxa_natalidade

        # Laplacianos
        lap_S = calcular_laplaciano_1d(S, self.malha.dx)
        lap_I = calcular_laplaciano_1d(I, self.malha.dx)
        lap_R = calcular_laplaciano_1d(R, self.malha.dx)

        # Termos de reação (SIR)
        infeccao = beta_t * S * I
        recuperacao = gamma * I

        # EDPs:
        # dS/dt = -β(t) S I + D_S ∆S + ν (S+I+R) - μ S
        # dI/dt =  β(t) S I - γ I + D_I ∆I - μ I
        # dR/dt =  γ I + D_R ∆R - μ R
        populacao_total = S + I + R

        dSdt = (
            -infeccao
            + self.params_epi.difusao_suscetiveis * lap_S
            + nu * populacao_total
            - mu * S
        )

        dIdt = (
            infeccao
            - recuperacao
            + self.params_epi.difusao_infectados * lap_I
            - mu * I
        )

        dRdt = (
            recuperacao
            + self.params_epi.difusao_recuperados * lap_R
            - mu * R
        )

        return EstadoSIR(dSdt, dIdt, dRdt)

    # --------------------------------------------------------
    # Passo de integração temporal (método de Heun / RK2)
    # --------------------------------------------------------

    def passo_tempo(self):
        """
        Avança o estado em um passo de tempo usando método de Heun (RK2 explícito),
        que é mais estável e preciso do que Euler explícito simples.
        """
        dt = self.dt_real
        t = self.tempo_atual
        S0 = self.estado.S
        I0 = self.estado.I
        R0 = self.estado.R

        # Estágio 1 (Euler)
        k1 = self.derivadas_temporais(t, self.estado)

        S1 = S0 + dt * k1.S
        I1 = I0 + dt * k1.I
        R1 = R0 + dt * k1.R

        # Impõe não negatividade
        S1 = np.clip(S1, 0.0, None)
        I1 = np.clip(I1, 0.0, None)
        R1 = np.clip(R1, 0.0, None)

        estado_intermediario = EstadoSIR(S1, I1, R1)

        # Estágio 2 (previsão em t + dt)
        k2 = self.derivadas_temporais(t + dt, estado_intermediario)

        # Combinação final (média de Euler e previsão)
        S_novo = S0 + 0.5 * dt * (k1.S + k2.S)
        I_novo = I0 + 0.5 * dt * (k1.I + k2.I)
        R_novo = R0 + 0.5 * dt * (k1.R + k2.R)

        # Impõe não negatividade e atualiza o estado
        self.estado.S = np.clip(S_novo, 0.0, None)
        self.estado.I = np.clip(I_novo, 0.0, None)
        self.estado.R = np.clip(R_novo, 0.0, None)

        # Atualiza tempo
        self.tempo_atual += dt

    # --------------------------------------------------------
    # Quantidades agregadas (estatísticas globais)
    # --------------------------------------------------------

    def totais_populacao(self) -> Dict[str, float]:
        """
        Retorna os totais espaciais de S, I, R (integrais aproximadas via soma).
        """
        dx = self.malha.dx
        S_total = float(np.sum(self.estado.S) * dx)
        I_total = float(np.sum(self.estado.I) * dx)
        R_total = float(np.sum(self.estado.R) * dx)
        return {"S_total": S_total, "I_total": I_total, "R_total": R_total}

    # --------------------------------------------------------
    # Rotina principal de simulação
    # --------------------------------------------------------

    def simular(self) -> Dict[str, np.ndarray]:
        """
        Executa a simulação completa, armazenando séries temporais e
        alguns snapshots espaciais.
        """
        n_passos = int(self.params_num.tempo_final_dias / self.dt_real) + 1

        # Histórico temporal de totais
        tempos = np.zeros(n_passos)
        S_totais = np.zeros(n_passos)
        I_totais = np.zeros(n_passos)
        R_totais = np.zeros(n_passos)

        # Snapshots espaciais em alguns tempos representativos
        snapshots_tempos: List[float] = []
        snapshots_S: List[np.ndarray] = []
        snapshots_I: List[np.ndarray] = []
        snapshots_R: List[np.ndarray] = []

        for passo in range(n_passos):
            # Armazena estatísticas atuais
            totais = self.totais_populacao()
            tempos[passo] = self.tempo_atual
            S_totais[passo] = totais["S_total"]
            I_totais[passo] = totais["I_total"]
            R_totais[passo] = totais["R_total"]

            # Armazena snapshot a cada N passos
            if passo % self.params_num.salvar_cada_n_passos == 0:
                snapshots_tempos.append(self.tempo_atual)
                snapshots_S.append(self.estado.S.copy())
                snapshots_I.append(self.estado.I.copy())
                snapshots_R.append(self.estado.R.copy())

            # Avança um passo no tempo (exceto após o último armazenamento)
            if passo < n_passos - 1:
                self.passo_tempo()

        resultados = {
            "tempos": tempos,
            "S_totais": S_totais,
            "I_totais": I_totais,
            "R_totais": R_totais,
            "snapshots_tempos": np.array(snapshots_tempos),
            "snapshots_S": np.array(snapshots_S),
            "snapshots_I": np.array(snapshots_I),
            "snapshots_R": np.array(snapshots_R),
            "x": self.malha.x,
        }

        return resultados


# ============================================================
# 6. Rotinas de visualização
# ============================================================

def plotar_series_temporais(resultados: Dict[str, np.ndarray]):
    """
    Plota as séries temporais dos totais de S, I, R.
    """
    tempos = resultados["tempos"]
    S_totais = resultados["S_totais"]
    I_totais = resultados["I_totais"]
    R_totais = resultados["R_totais"]

    plt.figure(figsize=(10, 6))
    plt.plot(tempos, S_totais, label="Suscetíveis (total)")
    plt.plot(tempos, I_totais, label="Infectados (total)")
    plt.plot(tempos, R_totais, label="Recuperados (total)")
    plt.xlabel("Tempo (dias)")
    plt.ylabel("População (integrada no espaço)")
    plt.title("Evolução Temporal dos Totais S, I, R (Modelo EDP Gripe)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def plotar_perfis_espaciais(resultados: Dict[str, np.ndarray], indices_snapshot: List[int] = None):
    """
    Plota perfis espaciais de infectados em alguns tempos.
    """
    x = resultados["x"]
    snapshots_tempos = resultados["snapshots_tempos"]
    snapshots_I = resultados["snapshots_I"]

    if indices_snapshot is None:
        # seleciona até 4 snapshots uniformemente espaçados
        n_snapshots = snapshots_I.shape[0]
        if n_snapshots <= 4:
            indices_snapshot = list(range(n_snapshots))
        else:
            indices_snapshot = [
                0,
                n_snapshots // 3,
                2 * n_snapshots // 3,
                n_snapshots - 1,
            ]

    plt.figure(figsize=(10, 6))
    for idx in indices_snapshot:
        t = snapshots_tempos[idx]
        I_perfil = snapshots_I[idx]
        plt.plot(x, I_perfil, label=f"t = {t:.1f} dias")

    plt.xlabel("Posição (km)")
    plt.ylabel("Infectados I(x,t)")
    plt.title("Perfis Espaciais de Infectados ao Longo do Tempo")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


# ============================================================
# 7. Impressão de estatísticas com 6 dígitos de precisão
# ============================================================

def imprimir_estatisticas(resultados: Dict[str, np.ndarray]):
    """
    Imprime estatísticas chave com 6 dígitos de precisão.
    """
    tempos = resultados["tempos"]
    I_totais = resultados["I_totais"]
    S_totais = resultados["S_totais"]
    R_totais = resultados["R_totais"]

    # Pico de infectados
    indice_pico = int(np.argmax(I_totais))
    t_pico = tempos[indice_pico]
    I_pico = I_totais[indice_pico]

    S_final = S_totais[-1]
    I_final = I_totais[-1]
    R_final = R_totais[-1]

    print("===== Estatísticas da Simulação =====")
    print(f"Tempo do pico de infectados (dias): {t_pico:.6f}")
    print(f"Total de infectados no pico:       {I_pico:.6f}")
    print("----------------------------------------")
    print(f"Suscetíveis finais (total):         {S_final:.6f}")
    print(f"Infectados finais (total):          {I_final:.6f}")
    print(f"Recuperados finais (total):         {R_final:.6f}")
    print("========================================")


# ============================================================
# 8. Função principal para rodar o modelo
# ============================================================

def executar_modelo_gripe():
    """
    Constrói o modelo de EDP da gripe, executa a simulação e
    gera gráficos e estatísticas.
    """
    # -----------------------------
    # 8.1 Definição dos parâmetros
    # -----------------------------
    params_epi = ParametrosEpidemia(
        taxa_transmissao_base=0.9,
        taxa_recuperacao=0.25,
        taxa_mortalidade_natural=1e-4,
        taxa_natalidade=1e-4,
        intensidade_sazonal=0.3,
        periodo_sazonal_dias=365.0,
        difusao_suscetiveis=0.02,
        difusao_infectados=0.08,
        difusao_recuperados=0.01,
    )

    params_num = ParametrosNumericos(
        comprimento_km=100.0,
        pontos_espaciais=201,
        tempo_final_dias=180.0,
        dt_inicial=0.05,
        salvar_cada_n_passos=80,
    )

    # ------------------------------------------------
    # 8.2 Malha espacial e condições iniciais (S,I,R)
    # ------------------------------------------------
    malha = MalhaEspacial1D(0.0, params_num.comprimento_km, params_num.pontos_espaciais)

    S0, I0, R0 = condicao_inicial_populacoes(
        malha=malha,
        populacao_total=1.0,
        largura_foco_inicial_km=5.0,
        posicao_foco_km=40.0,
        fracao_infectados_inicial=0.02,
    )

    estado_inicial = EstadoSIR(S=S0, I=I0, R=R0)

    # ------------------------------------
    # 8.3 Construção do modelo EDP completo
    # ------------------------------------
    modelo = ModeloGripeEDP(
        params_epi=params_epi,
        params_num=params_num,
        malha=malha,
        estado=estado_inicial,
    )

    print("Passo de tempo real utilizado (ajustado pela estabilidade): "
          f"{modelo.dt_real:.6f} dias")

    # ------------------------------------
    # 8.4 Execução da simulação
    # ------------------------------------
    resultados = modelo.simular()

    # ------------------------------------
    # 8.5 Impressão de estatísticas
    # ------------------------------------
    imprimir_estatisticas(resultados)

    # ------------------------------------
    # 8.6 Geração de gráficos
    # ------------------------------------
    plotar_series_temporais(resultados)
    plotar_perfis_espaciais(resultados)

    plt.show()


# ============================================================
# 9. Execução direta
# ============================================================

if __name__ == "__main__":
    executar_modelo_gripe()
