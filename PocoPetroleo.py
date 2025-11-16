# ============================================================
# Dinâmica de Poço de Petróleo — EDOs + Rede Neural + Pygame
# Autor: Luiz Tiago Wilcke (LT)
# ============================================================
# Ideia:
#   - Modelo físico simplificado em EDOs:
#       x = (pressao_reservatorio, pressao_cabeca, fracao_agua, vazao_oleo)
#
#     dP_r/dt = -a1 * vazao_oleo + a2*(P_ext - P_r)
#     dP_wh/dt = b1*(vazao_oleo - vazao_alvo) - b2*(P_wh - P_sep)
#     dW/dt    = c1*(W_eq(P_r) - W)
#     dQ/dt    = d1*(Q_teorico(P_r, P_wh, u) - Q)
#
#   - Controle: abertura_valvula u ∈ [0,1]
#   - Rede neural aprende uma "política" u(x) aproximando uma regra heurística.
#
#   - Pygame: simulação em tempo "quase real" com barras e indicadores.
# ============================================================

import math
import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.integrate import solve_ivp

import torch
import torch.nn as nn
import torch.optim as optim

import pygame


# ------------------------------------------------------------
# 1) Parâmetros físicos e de simulação
# ------------------------------------------------------------
@dataclass
class ParametrosPoco:
    # Pressões (unidades arbitrárias, ex.: bar)
    pressao_reservatorio_inicial: float = 280.0
    pressao_cabeca_inicial: float = 80.0
    pressao_externa_aquifero: float = 300.0
    pressao_separador: float = 60.0

    # Estados adicionais
    fracao_agua_inicial: float = 0.1
    vazao_oleo_inicial: float = 500.0  # m3/dia (escala genérica)

    # Parâmetros das EDOs
    a1_deplecao: float = 1.0e-3
    a2_recarga: float = 5.0e-4
    b1_acoplamento_vazao_pressao: float = 1.0e-3
    b2_relaxamento_cabeca: float = 5.0e-3
    c1_dinamica_agua: float = 1.0e-3
    d1_relaxamento_vazao: float = 1.0e-2

    # Parâmetros de vazão teórica
    ganho_produtividade: float = 3.0  # Q_teorico ≈ k * (P_r - P_wh) * u

    # Alvos operacionais
    vazao_alvo: float = 600.0
    pressao_cabeca_max: float = 120.0
    fracao_agua_max: float = 0.6

    # Tempo de simulação para gerar dados (dias)
    tempo_total_dados: float = 500.0
    dt_dados: float = 0.5

    # Tempo de passo da simulação Pygame (em "dias simulados" por frame)
    dt_simulacao: float = 0.25


# ------------------------------------------------------------
# 2) Dinâmica do poço (EDOs)
# ------------------------------------------------------------
def fracao_agua_equilibrio(pressao_reservatorio: float) -> float:
    """
    Fração de água de equilíbrio W_eq(P_r): aumenta quando a pressão cai
    (entrada de água do aquífero / conificação).
    """
    # Função logística simples entre 0.05 e 0.8, com ponto de transição em 200 bar
    W_min, W_max = 0.05, 0.8
    p0 = 200.0
    k = 0.03
    return W_min + (W_max - W_min) / (1.0 + math.exp(-k * (p0 - pressao_reservatorio)))


def vazao_teorica(pressao_reservatorio: float,
                  pressao_cabeca: float,
                  abertura_valvula: float,
                  params: ParametrosPoco) -> float:
    """
    Vazão teórica de óleo (sem água) em função da drawdown (P_r - P_wh) e abertura da válvula.
    """
    delta_p = max(pressao_reservatorio - pressao_cabeca, 0.0)
    return params.ganho_produtividade * delta_p * max(min(abertura_valvula, 1.0), 0.0)


def dinamica_poco(
    t: float,
    estado: np.ndarray,
    abertura_valvula: float,
    params: ParametrosPoco
) -> np.ndarray:
    """
    Sistema de EDOs d/dt [P_r, P_wh, W, Q].
    """
    pressao_reservatorio = estado[0]
    pressao_cabeca = estado[1]
    fracao_agua = estado[2]
    vazao_oleo = estado[3]

    # 1) Dinâmica da pressão do reservatório
    dP_r_dt = (
        -params.a1_deplecao * vazao_oleo
        + params.a2_recarga * (params.pressao_externa_aquifero - pressao_reservatorio)
    )

    # 2) Dinâmica da pressão na cabeça do poço
    dP_wh_dt = (
        params.b1_acoplamento_vazao_pressao * (vazao_oleo - params.vazao_alvo)
        - params.b2_relaxamento_cabeca * (pressao_cabeca - params.pressao_separador)
    )

    # 3) Dinâmica da fração de água
    W_eq = fracao_agua_equilibrio(pressao_reservatorio)
    dW_dt = params.c1_dinamica_agua * (W_eq - fracao_agua)

    # 4) Dinâmica da vazão de óleo
    Q_teo = vazao_teorica(pressao_reservatorio, pressao_cabeca, abertura_valvula, params)
    dQ_dt = params.d1_relaxamento_vazao * (Q_teo * (1.0 - fracao_agua) - vazao_oleo)

    return np.array([dP_r_dt, dP_wh_dt, dW_dt, dQ_dt], dtype=float)


# ------------------------------------------------------------
# 3) Política heurística (controle "ideal" para gerar dados)
# ------------------------------------------------------------
def politica_heuristica(
    estado: np.ndarray,
    params: ParametrosPoco
) -> float:
    """
    Regra "ideal" (inventada) para abertura da válvula:
      - Aumenta se vazão está abaixo do alvo e pressões seguras.
      - Diminui se P_wh está alta ou fração de água está alta.
    """
    P_r, P_wh, W, Q = estado

    termo_vazao = 0.004 * (params.vazao_alvo - Q)           # quer puxar Q -> alvo
    termo_pressao = -0.01 * max(P_wh - params.pressao_cabeca_max, 0.0)
    termo_agua = -0.5 * max(W - params.fracao_agua_max, 0.0)

    u_base = 0.5 + termo_vazao + termo_pressao + termo_agua

    # Saturação em [0, 1]
    u = max(0.0, min(1.0, u_base))
    return u


# ------------------------------------------------------------
# 4) Geração de dados sintéticos (EDO + política heurística)
# ------------------------------------------------------------
def gerar_dados(params: ParametrosPoco) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gera trajetória longa com solve_ivp e coleta pares (estado -> abertura_ideal).
    """
    t0 = 0.0
    tf = params.tempo_total_dados
    n_passos = int((tf - t0) / params.dt_dados) + 1
    t_eval = np.linspace(t0, tf, n_passos)

    estado0 = np.array([
        params.pressao_reservatorio_inicial,
        params.pressao_cabeca_inicial,
        params.fracao_agua_inicial,
        params.vazao_oleo_inicial
    ], dtype=float)

    estados = []
    controles = []

    estado_atual = estado0.copy()
    tempo_atual = t0

    for k in range(n_passos):
        u = politica_heuristica(estado_atual, params)

        # Integra um passo de dt_dados com solve_ivp usando u fixo
        sol = solve_ivp(
            fun=lambda t, y: dinamica_poco(t, y, u, params),
            t_span=(tempo_atual, tempo_atual + params.dt_dados),
            y0=estado_atual,
            method="RK45",
            t_eval=[tempo_atual + params.dt_dados]
        )
        estado_novo = sol.y[:, -1]

        estados.append(estado_atual.copy())
        controles.append(u)

        estado_atual = estado_novo
        tempo_atual += params.dt_dados

    X = np.vstack(estados)      # [N, 4]
    y = np.array(controles)     # [N]

    return X, y


# ------------------------------------------------------------
# 5) Rede Neural: política u(x) ≈ u_heurística(x)
# ------------------------------------------------------------
class PoliticaNN(nn.Module):
    def __init__(self, dim_entrada: int = 4, dim_oculto: int = 64):
        super().__init__()
        self.rede = nn.Sequential(
            nn.Linear(dim_entrada, dim_oculto),
            nn.Tanh(),
            nn.Linear(dim_oculto, dim_oculto),
            nn.Tanh(),
            nn.Linear(dim_oculto, 1),
            nn.Sigmoid()  # garante saída em [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rede(x)


def treinar_rede(
    X: np.ndarray,
    y: np.ndarray,
    epocas: int = 300,
    taxa_aprendizado: float = 1e-3,
    batch_size: int = 256
) -> PoliticaNN:
    """
    Treina uma pequena rede neural para aproximar u_ideal(x).
    """
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    modelo = PoliticaNN(dim_entrada=X.shape[1], dim_oculto=64).to(dispositivo)
    otimizador = optim.Adam(modelo.parameters(), lr=taxa_aprendizado)
    criterio = nn.MSELoss()

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y.reshape(-1, 1), dtype=torch.float32)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoca in range(epocas):
        modelo.train()
        perda_total = 0.0
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(dispositivo)
            batch_y = batch_y.to(dispositivo)

            otimizador.zero_grad()
            saida = modelo(batch_X)
            perda = criterio(saida, batch_y)
            perda.backward()
            otimizador.step()

            perda_total += perda.item() * batch_X.size(0)

        if (epoca + 1) % 50 == 0:
            print(f"[Treino] Época {epoca+1}/{epocas} | Loss média: {perda_total / len(dataset):.6f}")

    return modelo


# ------------------------------------------------------------
# 6) Simulação em Pygame
# ------------------------------------------------------------
class SimulacaoPygame:
    def __init__(self, params: ParametrosPoco, modelo_nn: PoliticaNN):
        self.params = params
        self.modelo_nn = modelo_nn.eval()
        self.dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.modelo_nn.to(self.dispositivo)

        # Estado
        self.estado = np.array([
            params.pressao_reservatorio_inicial,
            params.pressao_cabeca_inicial,
            params.fracao_agua_inicial,
            params.vazao_oleo_inicial
        ], dtype=float)
        self.tempo = 0.0
        self.abertura_valvula = 0.5
        self.controle_manual = False

        # Pygame
        pygame.init()
        self.largura = 900
        self.altura = 600
        self.tela = pygame.display.set_mode((self.largura, self.altura))
        pygame.display.set_caption("Dinâmica de Poço de Petróleo — EDO + Rede Neural")
        self.clock = pygame.time.Clock()
        self.fonte = pygame.font.SysFont("consolas", 18)

    def _calcular_abertura(self) -> float:
        if self.controle_manual:
            # Em modo manual, self.abertura_valvula é alterada pelas teclas
            return self.abertura_valvula

        # Usa a rede neural
        x = torch.tensor(self.estado, dtype=torch.float32).unsqueeze(0).to(self.dispositivo)
        with torch.no_grad():
            u = self.modelo_nn(x).cpu().numpy()[0, 0]
        self.abertura_valvula = float(u)
        return self.abertura_valvula

    def _passo_dinamico(self):
        dt = self.params.dt_simulacao
        u = self._calcular_abertura()
        derivadas = dinamica_poco(self.tempo, self.estado, u, self.params)
        # Integrador simples de Euler explícito
        self.estado = self.estado + dt * derivadas
        # Pequenas proteções
        self.estado[2] = max(0.0, min(1.0, self.estado[2]))  # fração de água ∈ [0, 1]
        self.estado[3] = max(0.0, self.estado[3])            # vazão não negativa
        self.tempo += dt

    def _desenhar_barras(self):
        # Desenha barras para P_r, P_wh, W, Q, u
        self.tela.fill((240, 245, 255))

        P_r, P_wh, W, Q = self.estado
        u = self.abertura_valvula

        # Escalas aproximadas
        max_P_r = 350.0
        max_P_wh = 200.0
        max_Q = 1000.0

        def barra(x, y, largura, altura, valor_norm, cor):
            valor_norm = max(0.0, min(1.0, valor_norm))
            h = int(altura * valor_norm)
            pygame.draw.rect(self.tela, (220, 220, 220), (x, y, largura, altura), border_radius=6)
            pygame.draw.rect(self.tela, cor,
                             (x, y + (altura - h), largura, h),
                             border_radius=6)

        base_y = 120
        altura_barra = 350
        largura_barra = 60
        espacamento = 80
        x0 = 80

        # P_r
        barra(x0, base_y, largura_barra, altura_barra, P_r / max_P_r, (40, 90, 180))
        # P_wh
        barra(x0 + espacamento, base_y, largura_barra, altura_barra, P_wh / max_P_wh, (200, 80, 80))
        # W
        barra(x0 + 2 * espacamento, base_y, largura_barra, altura_barra, W, (80, 160, 80))
        # Q
        barra(x0 + 3 * espacamento, base_y, largura_barra, altura_barra, Q / max_Q, (200, 160, 60))
        # u
        barra(x0 + 4 * espacamento, base_y, largura_barra, altura_barra, u, (120, 60, 200))

        # Textos
        textos = [
            ("P_res (bar)", x0),
            ("P_cab (bar)", x0 + espacamento),
            ("Frac_agua", x0 + 2 * espacamento),
            ("Q_oleo", x0 + 3 * espacamento),
            ("Abertura", x0 + 4 * espacamento),
        ]
        for txt, x in textos:
            s = self.fonte.render(txt, True, (20, 20, 40))
            self.tela.blit(s, (x - 10, base_y + altura_barra + 10))

        # Infos numéricas
        info_linhas = [
            f"Tempo simulado: {self.tempo:7.1f} dias",
            f"Pressao reservatorio: {P_r:7.1f} bar",
            f"Pressao cabeca:      {P_wh:7.1f} bar (max alvo {self.params.pressao_cabeca_max:.1f})",
            f"Fracao de agua:      {W:7.3f}",
            f"Vazao oleo:          {Q:7.1f} m3/d (alvo {self.params.vazao_alvo:.1f})",
            f"Abertura valvula u:  {u:7.3f}",
            f"Modo controle:       {'MANUAL (setas ↑/↓)' if self.controle_manual else 'REDE NEURAL'}"
        ]
        ytxt = 20
        for linha in info_linhas:
            s = self.fonte.render(linha, True, (10, 10, 30))
            self.tela.blit(s, (420, ytxt))
            ytxt += 24

        # Instruções
        instr = [
            "Teclas:",
            "[N] - alternar NN / manual",
            "[↑] / [↓] - ajuste da abertura em modo manual",
            "[R] - resetar estado do poço",
            "[ESC] - sair"
        ]
        ytxt = 20
        for linha in instr:
            s = self.fonte.render(linha, True, (30, 30, 60))
            self.tela.blit(s, (40, ytxt))
            ytxt += 24

    def _processar_eventos(self) -> bool:
        """
        Processa eventos. Retorna False se for para encerrar.
        """
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                return False
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_ESCAPE:
                    return False
                if evento.key == pygame.K_n:
                    self.controle_manual = not self.controle_manual
                if evento.key == pygame.K_r:
                    # Reseta o poço
                    self.estado = np.array([
                        self.params.pressao_reservatorio_inicial,
                        self.params.pressao_cabeca_inicial,
                        self.params.fracao_agua_inicial,
                        self.params.vazao_oleo_inicial
                    ], dtype=float)
                    self.tempo = 0.0
                if self.controle_manual:
                    if evento.key == pygame.K_UP:
                        self.abertura_valvula = min(1.0, self.abertura_valvula + 0.05)
                    if evento.key == pygame.K_DOWN:
                        self.abertura_valvula = max(0.0, self.abertura_valvula - 0.05)
        return True

    def rodar(self):
        rodando = True
        while rodando:
            rodando = self._processar_eventos()
            self._passo_dinamico()
            self._desenhar_barras()
            pygame.display.flip()
            self.clock.tick(30)  # ~30 FPS


# ------------------------------------------------------------
# 7) Main
# ------------------------------------------------------------
def main():
    params = ParametrosPoco()

    print("=== Gerando dados sintéticos via EDO + política heurística ===")
    X, y = gerar_dados(params)
    print(f"Dados gerados: {X.shape[0]} amostras.")

    print("=== Treinando rede neural de política u(x) ===")
    modelo_nn = treinar_rede(X, y, epocas=300, taxa_aprendizado=1e-3, batch_size=256)

    print("=== Iniciando simulação interativa em Pygame ===")
    simulacao = SimulacaoPygame(params, modelo_nn)
    simulacao.rodar()
    pygame.quit()


if __name__ == "__main__":
    main()
