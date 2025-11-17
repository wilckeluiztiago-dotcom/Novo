# ============================================================
# TORNADOS SUL DO BRASIL — MODELO HÍBRIDO EDO + REDE NEURAL
# Autor: Luiz Tiago Wilcke (LT)
# ============================================================
# - Modelo físico: EDOs não lineares para CAPE, cisalhamento,
#   helicidade, umidade e vorticidade na baixa camada.
# - Modelo de risco: índice físico de tornadogênese + rede LSTM.
# - Saída: séries sintéticas, treinamento de rede, gráficos
#   e animação em pygame de uma supercélula sobre o Sul do Brasil.
# ============================================================

import math
import random
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pygame

# ------------------------------------------------------------
# 1) Configurações
# ------------------------------------------------------------

@dataclass
class ParametrosFisicos:
    dt_horas: float = 0.25          # passo de tempo (0.25 h = 15 min)
    C0: float = 600.0               # escala de CAPE para I(C)
    alpha_C: float = 1/6            # relaxação em ~6 h
    alpha_S: float = 1/6
    alpha_H: float = 1/6
    alpha_q: float = 1/12
    alpha_V: float = 1/3

    beta_C: float = 1/3
    beta_S: float = 1/6
    beta_H: float = 1/6
    beta_q: float = 1/4
    beta_V: float = 1/3

    gamma_C: float = 1.5e-1
    gamma_S: float = 0.5
    gamma_H: float = 2.0e-4
    gamma_q: float = 0.0
    gamma_V: float = 1.2e-4

    k_V: float = 800.0              # sensibilidade da vorticidade no índice
    a0: float = -3.0                # parâmetros logísticos p_fis
    a1: float = 2.0
    a2: float = 1.0

    # Ruído estocástico
    sigma_ruido: float = 0.02


@dataclass
class ParametrosCenarios:
    n_cenarios: int = 200          # número de dias/casos sintéticos
    horas_por_cenario: int = 24    # duração de cada cenário
    prob_tornado_condicional: float = 0.4  # chance de tornado se T_fis alto


@dataclass
class ParametrosTreino:
    tamanho_janela: int = 12       # passos na janela LSTM (12*15min = 3h)
    proporcao_treino: float = 0.8
    epocas: int = 15
    batch: int = 64
    lr: float = 1e-3
    usar_cuda: bool = True


# Fixar semente para reprodutibilidade
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)


# ------------------------------------------------------------
# 2) Modelo físico — EDOs + integração RK4
# ------------------------------------------------------------

def intensidade_convectiva(cape: float, C0: float) -> float:
    return cape / (cape + C0 + 1e-6)


def rhs_edo(estado: np.ndarray,
            forca: np.ndarray,
            pf: ParametrosFisicos) -> np.ndarray:
    """
    estado = [C, S, H, q, V]
    forca  = [C_env, S_env, H_env, q_env, dU]
    """
    C, S, H, q, V = estado
    C_env, S_env, H_env, q_env, dU = forca

    I = intensidade_convectiva(C, pf.C0)

    dC = (pf.alpha_C * (C_env - C)
          - pf.beta_C * I * C
          + pf.gamma_C * q * S)

    dS = (pf.alpha_S * (S_env - S)
          - pf.beta_S * I * S
          + pf.gamma_S * dU)

    dH = (pf.alpha_H * (H_env - H)
          - pf.beta_H * H
          + pf.gamma_H * S * V)

    dq = (pf.alpha_q * (q_env - q)
          - pf.beta_q * I * q
          + pf.gamma_q * 0.0)

    dV = (pf.alpha_V * (dU - V)
          - pf.beta_V * V
          + pf.gamma_V * S * q)

    return np.array([dC, dS, dH, dq, dV], dtype=float)


def rk4_passo(estado: np.ndarray,
              forca: np.ndarray,
              dt: float,
              pf: ParametrosFisicos) -> np.ndarray:
    k1 = rhs_edo(estado, forca, pf)
    k2 = rhs_edo(estado + 0.5 * dt * k1, forca, pf)
    k3 = rhs_edo(estado + 0.5 * dt * k2, forca, pf)
    k4 = rhs_edo(estado + dt * k3,      forca, pf)
    return estado + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def indice_tornadogenese_fisico(C: float, S: float, H: float,
                                q: float, V: float,
                                pf: ParametrosFisicos) -> float:
    # Normalizações suaves (evitar explosão)
    term_C = math.log1p(max(C, 0.0) / 1500.0)   # log(1 + C/1500)
    term_S = math.log1p(max(S, 0.0) / 20.0)
    term_H = math.log1p(max(H, 0.0) / 150.0)
    term_q = 2.0 * (q - 0.8)                    # úmido favorece
    term_V = pf.k_V * V                         # vorticidade baixa camada

    T_fis = term_C + term_S + term_H + term_q + term_V
    return T_fis


def prob_tornado_fisica(T_fis: float, pf: ParametrosFisicos) -> float:
    z = pf.a0 + pf.a1 * T_fis + pf.a2 * (T_fis ** 2)
    return 1.0 / (1.0 + math.exp(-z))


def gerar_forcantes_ambiente(n_passos: int,
                             dt: float) -> np.ndarray:
    """
    Gera séries suaves de C_env, S_env, H_env, q_env, dU
    para o Sul do Brasil (valores típicos, sintéticos).
    """
    t = np.arange(n_passos) * dt

    C_env = 1000 + 600 * np.sin(2 * np.pi * t / 24) + 200 * np.random.randn(n_passos)
    C_env = np.clip(C_env, 0, 3500)

    S_env = 15 + 5 * np.sin(2 * np.pi * (t - 3) / 24) + 2 * np.random.randn(n_passos)
    S_env = np.clip(S_env, 0, 35)

    H_env = 120 + 60 * np.sin(2 * np.pi * (t - 5) / 24) + 40 * np.random.randn(n_passos)
    H_env = np.clip(H_env, 0, 400)

    q_env = 0.7 + 0.1 * np.sin(2 * np.pi * (t - 2) / 24) + 0.05 * np.random.randn(n_passos)
    q_env = np.clip(q_env, 0.4, 0.99)

    dU = 15 + 8 * np.sin(2 * np.pi * (t - 1) / 24) + 3 * np.random.randn(n_passos)
    dU = np.clip(dU, 0, 40)

    forcantes = np.stack([C_env, S_env, H_env, q_env, dU], axis=-1)
    return forcantes


def simular_cenario(pf: ParametrosFisicos,
                    pc: ParametrosCenarios) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simula UM cenário de 24h: retorna
    - estados: [n_passos, 5]
    - T_fis:   [n_passos]
    - p_fis:   [n_passos]
    """
    n_passos = int(pc.horas_por_cenario / pf.dt_horas)
    forcantes = gerar_forcantes_ambiente(n_passos, pf.dt_horas)

    # Condições iniciais "quase ambiente"
    estado = forcantes[0].copy()
    C0, S0, H0, q0, dU0 = estado
    q0 = np.clip(q0, 0.6, 0.95)
    V0 = 0.0
    estado = np.array([C0, S0, H0, q0, V0], dtype=float)

    estados = []
    T_fis_lista = []
    p_fis_lista = []

    for i in range(n_passos):
        C, S, H, q, V = estado
        T_fis = indice_tornadogenese_fisico(C, S, H, q, V, pf)
        p_fis = prob_tornado_fisica(T_fis, pf)

        estados.append(estado.copy())
        T_fis_lista.append(T_fis)
        p_fis_lista.append(p_fis)

        # Integra próximo passo com RK4
        estado = rk4_passo(estado, forcantes[i], pf.dt_horas, pf)

        # Ruído estocástico leve (representa turbulência não resolvida)
        estado = estado + pf.sigma_ruido * np.random.randn(5)
        # Limites físicos aproximados
        estado[0] = np.clip(estado[0], 0, 4000)  # CAPE
        estado[1] = np.clip(estado[1], 0, 40)    # shear
        estado[2] = np.clip(estado[2], 0, 600)   # helicity
        estado[3] = np.clip(estado[3], 0.3, 1.0) # umidade
        estado[4] = np.clip(estado[4], -0.01, 0.01)  # vorticidade

    estados = np.array(estados)
    T_fis = np.array(T_fis_lista)
    p_fis = np.array(p_fis_lista)
    return estados, T_fis, p_fis


def gerar_base_sintetica(pf: ParametrosFisicos,
                         pc: ParametrosCenarios,
                         pt: ParametrosTreino) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gera base de treinamento:
    X: [N_amostras, janela, 5]   (sequência de estados)
    y: [N_amostras]             (0/1 tornado)
    """
    todas_janelas = []
    todos_labels = []

    for _ in range(pc.n_cenarios):
        estados, T_fis, p_fis = simular_cenario(pf, pc)

        n_passos = estados.shape[0]
        limiar_T = 1.0  # limiar de índice para considerar "ambiente favorável"

        # Rótulos binários com base no índice físico, + ruído
        labels_inst = np.zeros(n_passos, dtype=int)
        for t in range(n_passos):
            if T_fis[t] > limiar_T:
                if random.random() < pc.prob_tornado_condicional:
                    labels_inst[t] = 1

        # Construir janelas temporais para a LSTM
        L = pt.tamanho_janela
        for t in range(L, n_passos):
            janela = estados[t-L:t, :]   # [L, 5]
            label = int(labels_inst[t])
            todas_janelas.append(janela)
            todos_labels.append(label)

    X = np.stack(todas_janelas, axis=0)          # [N, L, 5]
    y = np.array(todos_labels, dtype=int)        # [N]
    return X, y


# ------------------------------------------------------------
# 3) Rede neural LSTM para risco de tornado
# ------------------------------------------------------------

class DatasetTornado(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float().unsqueeze(-1)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class RedeTornadoLSTM(nn.Module):
    def __init__(self, dim_entrada: int, dim_oculta: int = 64,
                 num_camadas_lstm: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=dim_entrada,
            hidden_size=dim_oculta,
            num_layers=num_camadas_lstm,
            batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(dim_oculta, dim_oculta),
            nn.ReLU(),
            nn.Linear(dim_oculta, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [batch, janela, dim]
        saida_seq, _ = self.lstm(x)
        # pega o último passo temporal
        h_final = saida_seq[:, -1, :]
        prob = self.mlp(h_final)
        return prob


def treinar_rede(X: np.ndarray,
                 y: np.ndarray,
                 pt: ParametrosTreino) -> Tuple[RedeTornadoLSTM, torch.device]:
    N = X.shape[0]
    idx = np.arange(N)
    np.random.shuffle(idx)
    N_treino = int(pt.proporcao_treino * N)
    idx_treino = idx[:N_treino]
    idx_teste  = idx[N_treino:]

    X_treino, y_treino = X[idx_treino], y[idx_treino]
    X_teste,  y_teste  = X[idx_teste],  y[idx_teste]

    ds_treino = DatasetTornado(X_treino, y_treino)
    ds_teste  = DatasetTornado(X_teste,  y_teste)

    loader_treino = DataLoader(ds_treino, batch_size=pt.batch, shuffle=True)
    loader_teste  = DataLoader(ds_teste,  batch_size=pt.batch, shuffle=False)

    device = torch.device("cuda" if (pt.usar_cuda and torch.cuda.is_available()) else "cpu")

    modelo = RedeTornadoLSTM(dim_entrada=X.shape[-1]).to(device)
    criterio = nn.BCELoss()
    otimizador = torch.optim.Adam(modelo.parameters(), lr=pt.lr)

    for ep in range(1, pt.epocas + 1):
        modelo.train()
        perdas = []
        for xb, yb in loader_treino:
            xb = xb.to(device)
            yb = yb.to(device)
            otimizador.zero_grad()
            pb = modelo(xb)
            loss = criterio(pb, yb)
            loss.backward()
            otimizador.step()
            perdas.append(loss.item())

        modelo.eval()
        with torch.no_grad():
            # treino
            Xtr = torch.from_numpy(X_treino).float().to(device)
            ytr = torch.from_numpy(y_treino).float().unsqueeze(-1).to(device)
            ptr = modelo(Xtr)
            ytr_pred = (ptr >= 0.5).float()
            acc_tr = (ytr_pred.eq(ytr).float().mean().item())

            # teste
            Xte = torch.from_numpy(X_teste).float().to(device)
            yte = torch.from_numpy(y_teste).float().unsqueeze(-1).to(device)
            pte = modelo(Xte)
            yte_pred = (pte >= 0.5).float()
            acc_te = (yte_pred.eq(yte).float().mean().item())

        print(f"Época {ep:02d} | Loss médio: {np.mean(perdas):.4f} | "
              f"Acc treino: {acc_tr:.3f} | Acc teste: {acc_te:.3f}")

    return modelo, device


# ------------------------------------------------------------
# 4) Gráficos de um cenário (e probabilidade da rede)
# ------------------------------------------------------------

def avaliar_cenario_com_rede(modelo: RedeTornadoLSTM,
                             device: torch.device,
                             estados: np.ndarray,
                             pt: ParametrosTreino) -> np.ndarray:
    """
    Calcula a probabilidade da rede ao longo de um cenário,
    usando janelas deslizantes.
    """
    n_passos = estados.shape[0]
    L = pt.tamanho_janela
    probs = np.zeros(n_passos)

    modelo.eval()
    with torch.no_grad():
        for t in range(L, n_passos):
            janela = estados[t-L:t, :]      # [L, 5]
            x = torch.from_numpy(janela).float().unsqueeze(0).to(device)
            p = modelo(x).item()
            probs[t] = p

    return probs


def plotar_cenario(estados: np.ndarray,
                   T_fis: np.ndarray,
                   p_fis: np.ndarray,
                   p_nn: np.ndarray,
                   pf: ParametrosFisicos):
    t_horas = np.arange(estados.shape[0]) * pf.dt_horas
    C = estados[:, 0]
    S = estados[:, 1]
    H = estados[:, 2]
    q = estados[:, 3]
    V = estados[:, 4]

    fig, axs = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
    fig.suptitle("Cenário sintético — Sul do Brasil (EDO + Rede Neural)")

    axs[0].plot(t_horas, C)
    axs[0].set_ylabel("CAPE (J/kg)")

    axs[1].plot(t_horas, S)
    axs[1].set_ylabel("Shear 0–6km (m/s)")

    axs[2].plot(t_horas, H)
    axs[2].set_ylabel("SRH 0–3km (m²/s²)")

    axs[3].plot(t_horas, q)
    axs[3].set_ylabel("Umidade baixa (adim.)")

    axs[4].plot(t_horas, p_fis, label="p_física")
    axs[4].plot(t_horas, p_nn, label="p_rede", linestyle="--")
    axs[4].set_ylabel("Prob. tornado")
    axs[4].set_xlabel("Tempo (h)")
    axs[4].legend()

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 5) Animação em pygame
# ------------------------------------------------------------

def rodar_animacao_pygame(p_nn: np.ndarray,
                          estados: np.ndarray,
                          pf: ParametrosFisicos):
    """
    Anima uma supercélula cruzando o Sul do Brasil.
    - Cor e tamanho indicam CAPE.
    - Intensidade do funil depende de p_nn.
    """

    pygame.init()
    largura, altura = 900, 600
    tela = pygame.display.set_mode((largura, altura))
    pygame.display.set_caption("Animação — Tornados no Sul do Brasil (modelo LT)")
    clock = pygame.time.Clock()

    fonte = pygame.font.SysFont("Arial", 20)

    n_passos = len(p_nn)
    idx = 0

    rodando = True
    while rodando:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                rodando = False

        tela.fill((220, 235, 255))

        # Desenha "mapa" simplificado do Sul do Brasil
        # (um retângulo representando RS/SC/PR)
        margem = 80
        mapa_rect = pygame.Rect(margem, margem,
                                largura - 2 * margem,
                                altura - 2 * margem)
        pygame.draw.rect(tela, (190, 210, 230), mapa_rect, border_radius=15)

        # Passo atual
        t = idx % n_passos
        prob = float(p_nn[t])
        C = float(estados[t, 0])

        # Posição da tempestade: anda da esquerda para a direita
        x = margem + (mapa_rect.width) * (t / max(1, n_passos - 1))
        y = margem + mapa_rect.height * 0.4

        # Cor da tempestade: quanto maior CAPE, mais escuro
        intensidade_cape = min(1.0, C / 3500.0)
        cor_nuvem = (
            int(150 + 80 * intensidade_cape),
            int(150 + 40 * intensidade_cape),
            int(160 + 30 * intensidade_cape),
        )

        # Desenha nuvem
        raio_nuvem = 40 + 20 * intensidade_cape
        pygame.draw.circle(tela, cor_nuvem, (int(x), int(y)), int(raio_nuvem))

        # Desenha "funil" se probabilidade alta
        if prob > 0.4:
            intensidade = (prob - 0.4) / 0.6
            intensidade = max(0.0, min(1.0, intensidade))
            altura_funil = 120 + 80 * intensidade
            larg_topo = 30 + 20 * intensidade
            larg_base = 6 + 10 * intensidade
            x_top = x
            y_top = y + raio_nuvem * 0.8
            x_base = x_top + 20 * (intensidade - 0.5)
            y_base = y_top + altura_funil

            pontos_funil = [
                (x_top - larg_topo, y_top),
                (x_top + larg_topo, y_top),
                (x_base + larg_base, y_base),
                (x_base - larg_base, y_base),
            ]
            pygame.draw.polygon(tela, (130, 130, 150), pontos_funil)

        # Texto com probabilidades
        texto1 = fonte.render(f"Passo: {t}  (hora ~ {t * pf.dt_horas:.2f})", True, (10, 20, 60))
        texto2 = fonte.render(f"Prob. tornado (rede): {prob:.3f}", True, (120, 10, 10))
        tela.blit(texto1, (20, 20))
        tela.blit(texto2, (20, 45))

        pygame.display.flip()
        clock.tick(20)  # FPS

        idx += 1

    pygame.quit()


# ------------------------------------------------------------
# 6) Execução principal
# ------------------------------------------------------------

def main():
    pf = ParametrosFisicos()
    pc = ParametrosCenarios()
    pt = ParametrosTreino()

    print("Gerando base sintética com EDOs meteorológicas...")
    X, y = gerar_base_sintetica(pf, pc, pt)
    print(f"Base gerada: X = {X.shape}, y = {y.shape}, proporção positivos = {y.mean():.3f}")

    print("Treinando rede LSTM de risco de tornado...")
    modelo, device = treinar_rede(X, y, pt)

    # Escolher um cenário novo para visualização
    print("Simulando cenário para visualização...")
    estados, T_fis, p_fis = simular_cenario(pf, pc)
    p_nn = avaliar_cenario_com_rede(modelo, device, estados, pt)

    print("Gerando gráficos...")
    plotar_cenario(estados, T_fis, p_fis, p_nn, pf)

    print("Iniciando animação pygame (feche a janela para encerrar)...")
    rodar_animacao_pygame(p_nn, estados, pf)


if __name__ == "__main__":
    main()
