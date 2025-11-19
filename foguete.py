# ============================================================
# SIMULADOR AVANÇADO DE FOGUETE — EDOs + REDE NEURAL + PYGAME
# Autor: Luiz Tiago Wilcke (LT)
# ============================================================
# - Modelo físico (sistema de equações diferenciais):
#     • x(t), z(t)  -> posição horizontal e vertical
#     • vx(t), vz(t)-> velocidades
#     • m(t)        -> massa (queima de combustível)
#     • theta(t)    -> ângulo da trajetória (controlado pela rede)
#
#   Equações (simplificadas, mas realistas):
#
#   dx/dt    = vx
#   dz/dt    = vz
#   dvx/dt   = (T*cos(theta) - D * vx/v) / m
#   dvz/dt   = (T*sin(theta) - D * vz/v) / m - g(z)
#   dm/dt    = -taxa_queima (enquanto houver combustível)
#   dtheta/dt= (theta_cmd(t) - theta) / tau_controle
#
#   com:
#       T      = empuxo do motor (constante na queima)
#       D      = 0.5 * rho(z) * v^2 * Cd * A
#       g(z)   = g0 * (R/(R+z))^2
#       rho(z) = rho0 * exp(-z/H)
#
# - Rede Neural:
#     • Entrada: [t_normalizado, z_normalizado]
#     • Saída:   valor escalar → mapeado para theta_cmd(t) em [0, pi/2]
#     • Treinada para imitar um perfil ideal de inclinação (gravity turn)
#
# - Pygame:
#     • Animação 2D da trajetória
#     • Desenho do foguete com rotação conforme theta(t)
#     • HUD com tempo, altitude, velocidade, ângulo
# ============================================================

import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pygame

import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# PARÂMETROS DO FOGUETE E SIMULAÇÃO
# ============================================================

@dataclass
class ParametrosFoguete:
    massa_inicial: float = 50000.0       # kg (foguete + combustível)
    massa_estrutura: float = 15000.0     # kg (estrutura seca)
    empuxo_maximo: float = 1.2e6         # N (empuxo total do motor)
    tempo_queima: float = 120.0          # s (duração da queima principal)
    taxa_queima: float = 290.0           # kg/s (consumo de combustível)
    area_referencia: float = 10.0        # m² (área frontal)
    coef_arrasto: float = 0.4            # Cd (aprox. foguete esguio)
    tau_controle: float = 10.0           # s (constante de tempo do controle de atitude)
    gravidade_nivel_mar: float = 9.80665 # m/s²
    raio_terra: float = 6_371_000.0      # m
    densidade_nivel_mar: float = 1.225   # kg/m³
    altura_escala_atmosfera: float = 8500.0  # m
    dt: float = 0.2                      # passo temporal da integração
    tempo_total: float = 300.0           # tempo total de simulação


@dataclass
class EstadoFoguete:
    x: float
    z: float
    vx: float
    vz: float
    massa: float
    theta: float  # ângulo em rad (0 = horizontal, pi/2 = vertical)


# ============================================================
# REDE NEURAL DE CONTROLE DE INCLINAÇÃO
# ============================================================

class RedeNeuralControle(nn.Module):
    """
    Pequena MLP que aprende um perfil de inclinação ideal (gravity turn).
    Entrada:  [t_normalizado, z_normalizado]
    Saída:    escalar → mapeado para ângulo desejado em [0, pi/2].
    """
    def __init__(self):
        super().__init__()
        self.rede = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, t_norm: torch.Tensor, z_norm: torch.Tensor) -> torch.Tensor:
        entrada = torch.stack([t_norm, z_norm], dim=-1)
        y = self.rede(entrada)
        # Mapeia saída para [0, pi/2] usando sigmoide
        theta_min = 0.0
        theta_max = math.pi / 2
        return theta_min + (theta_max - theta_min) * torch.sigmoid(y)


def perfil_inclinacao_ideal(t_norm: np.ndarray) -> np.ndarray:
    """
    Perfil ideal "analítico" de ângulo:
    - t_norm=0   → 90° (vertical)
    - t_norm=0.3 → 70°
    - t_norm=0.6 → 30°
    - t_norm=1.0 → 5°
    Interpolação suave com função polinomial.
    """
    # Vamos usar um decaimento suave tipo (1 - t_norm^p)
    p = 2.5
    frac = (1.0 - np.clip(t_norm, 0, 1) ** p)
    # converte de [0,1] em [5°, 90°]
    angulo_graus = 5.0 + 85.0 * frac
    return np.deg2rad(angulo_graus)


def treinar_rede_controle(
    rede: RedeNeuralControle,
    parametros: ParametrosFoguete,
    epocas: int = 800,
    tamanho_lote: int = 256,
    lr: float = 1e-2,
    dispositivo: str = "cpu"
) -> None:
    """
    Treina a rede para aproximar o perfil analítico de inclinação.
    Não usa dados "reais" da simulação, mas um perfil alvo sofisticado.
    """
    rede.to(dispositivo)
    otimizador = optim.Adam(rede.parameters(), lr=lr)
    criterio = nn.MSELoss()

    # Gera dados sintéticos: t_norm uniforme, z_norm aproximado
    for epoca in range(epocas):
        t_norm_np = np.random.rand(tamanho_lote).astype(np.float32)
        # altura típica de lançamento: vamos simular um crescimento ~ t^2
        z_aprox = 120_000.0 * (t_norm_np ** 2)
        z_norm_np = (z_aprox / 120_000.0).astype(np.float32)

        alvo_theta_np = perfil_inclinacao_ideal(t_norm_np).astype(np.float32)

        t_norm = torch.from_numpy(t_norm_np).to(dispositivo)
        z_norm = torch.from_numpy(z_norm_np).to(dispositivo)
        alvo_theta = torch.from_numpy(alvo_theta_np).unsqueeze(-1).to(dispositivo)

        otimizador.zero_grad()
        saida_theta = rede(t_norm, z_norm)
        perda = criterio(saida_theta, alvo_theta)
        perda.backward()
        otimizador.step()

        if (epoca + 1) % 200 == 0:
            print(f"[TREINO REDE] Época {epoca+1}/{epocas} - Perda: {perda.item():.6f}")


# ============================================================
# DINÂMICA DO FOGUETE (EQUAÇÕES DIFERENCIAIS)
# ============================================================

def gravidade(param: ParametrosFoguete, z: float) -> float:
    return param.gravidade_nivel_mar * (param.raio_terra / (param.raio_terra + max(z, 0.0))) ** 2


def densidade_atmosfera(param: ParametrosFoguete, z: float) -> float:
    return param.densidade_nivel_mar * math.exp(-max(z, 0.0) / param.altura_escala_atmosfera)


def dinamica_foguete(
    t: float,
    estado: np.ndarray,
    rede: RedeNeuralControle,
    param: ParametrosFoguete
) -> np.ndarray:
    """
    Calcula d(estado)/dt para o foguete.
    estado = [x, z, vx, vz, m, theta]
    """
    x, z, vx, vz, massa, theta = estado

    # Gravidade e atmosfera
    g = gravidade(param, z)
    rho = densidade_atmosfera(param, z)

    # Velocidade total
    v = math.sqrt(vx * vx + vz * vz) + 1e-6

    # Arrasto aerodinâmico
    arrasto = 0.5 * rho * v * v * param.coef_arrasto * param.area_referencia

    # Empuxo e queima
    if (t <= param.tempo_queima) and (massa > param.massa_estrutura):
        empuxo = param.empuxo_maximo
        dm_dt = -param.taxa_queima
    else:
        empuxo = 0.0
        dm_dt = 0.0

    # Rede neural de controle: calcula ângulo desejado
    t_norm = float(t / max(param.tempo_queima, 1.0))
    z_norm = float(z / 120_000.0)  # normaliza por ~120 km

    with torch.no_grad():
        t_tensor = torch.tensor([t_norm], dtype=torch.float32)
        z_tensor = torch.tensor([z_norm], dtype=torch.float32)
        theta_cmd = rede(t_tensor, z_tensor).item()

    # Dinâmica da atitude: primeira ordem em direção ao theta_cmd
    dtheta_dt = (theta_cmd - theta) / param.tau_controle

    # Decomposição do arrasto na direção da velocidade
    ax_arrasto = -arrasto * vx / (v * massa)
    az_arrasto = -arrasto * vz / (v * massa)

    # Acelerações
    ax = (empuxo * math.cos(theta)) / massa + ax_arrasto
    az = (empuxo * math.sin(theta)) / massa + az_arrasto - g

    dx_dt = vx
    dz_dt = vz
    dvx_dt = ax
    dvz_dt = az

    return np.array([dx_dt, dz_dt, dvx_dt, dvz_dt, dm_dt, dtheta_dt], dtype=float)


def integrar_trajetoria(
    estado_inicial: EstadoFoguete,
    rede: RedeNeuralControle,
    param: ParametrosFoguete
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Integra o sistema de EDOs usando método de Runge-Kutta 4 (RK4).
    Retorna:
        tempos:  vetor (N,)
        estados: matriz (N, 6) com [x, z, vx, vz, m, theta]
    """
    dt = param.dt
    n_passos = int(param.tempo_total / dt) + 1

    tempos = np.zeros(n_passos, dtype=float)
    estados = np.zeros((n_passos, 6), dtype=float)

    y = np.array([
        estado_inicial.x,
        estado_inicial.z,
        estado_inicial.vx,
        estado_inicial.vz,
        estado_inicial.massa,
        estado_inicial.theta
    ], dtype=float)

    for i in range(n_passos):
        t = i * dt
        tempos[i] = t
        estados[i] = y

        # Para de integrar se o foguete "cair" de volta no solo
        if y[1] <= 0 and i > 10 and y[3] < 0:
            estados = estados[: i + 1]
            tempos = tempos[: i + 1]
            break

        # RK4
        k1 = dinamica_foguete(t, y, rede, param)
        k2 = dinamica_foguete(t + 0.5 * dt, y + 0.5 * dt * k1, rede, param)
        k3 = dinamica_foguete(t + 0.5 * dt, y + 0.5 * dt * k2, rede, param)
        k4 = dinamica_foguete(t + dt, y + dt * k3, rede, param)

        y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return tempos, estados


# ============================================================
# ANIMAÇÃO COM PYGAME
# ============================================================

def desenhar_foguete(
    tela: pygame.Surface,
    x_px: float,
    y_px: float,
    angulo_rad: float,
    escala_foguete: float = 30.0
) -> None:
    """
    Desenha um foguete simples como um triângulo, rotacionado por angulo_rad.
    """
    # Forma básica do foguete em coordenadas locais
    corpo = [
        (0, -1.5),  # ponta
        (-0.5, 0.7),
        (0.5, 0.7),
    ]
    # Escala
    corpo_escalado = [(x * escala_foguete, y * escala_foguete) for (x, y) in corpo]

    # Rotação (angulação: 0 = horizontal direita, pi/2 = vertical para cima)
    cos_a = math.cos(angulo_rad - math.pi / 2)  # ajustando para "apontar" para cima
    sin_a = math.sin(angulo_rad - math.pi / 2)
    corpo_rotacionado = []
    for (x, y) in corpo_escalado:
        xr = x * cos_a - y * sin_a
        yr = x * sin_a + y * cos_a
        corpo_rotacionado.append((x_px + xr, y_px + yr))

    pygame.draw.polygon(tela, (255, 255, 255), corpo_rotacionado)


def animar_trajetoria(
    tempos: np.ndarray,
    estados: np.ndarray,
    param: ParametrosFoguete
) -> None:
    pygame.init()
    largura = 1000
    altura = 600
    tela = pygame.display.set_mode((largura, altura))
    pygame.display.set_caption("Simulação de Foguete — LT")

    fonte = pygame.font.SysFont("consolas", 16)

    # Escalas de desenho
    x_max = max(estados[:, 0].max(), 1.0)
    z_max = max(estados[:, 1].max(), 1.0)

    margem = 60
    escala_x = (largura - 2 * margem) / x_max
    escala_z = (altura - 2 * margem) / z_max

    relogio = pygame.time.Clock()
    indice = 0
    rodando = True

    while rodando:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                rodando = False

        tela.fill((5, 10, 30))

        # Desenha grade simples
        cor_grade = (40, 60, 90)
        for i in range(6):
            y = altura - margem - i * (z_max * escala_z / 5)
            pygame.draw.line(tela, cor_grade, (margem, y), (largura - margem, y), 1)

        # Desenha trajetória
        pontos = []
        for j in range(indice + 1):
            x, z = estados[j, 0], estados[j, 1]
            x_px = margem + x * escala_x
            y_px = altura - margem - z * escala_z
            pontos.append((x_px, y_px))
        if len(pontos) > 1:
            pygame.draw.lines(tela, (80, 180, 255), False, pontos, 2)

        # Estado atual
        x, z, vx, vz, massa, theta = estados[indice]
        x_px = margem + x * escala_x
        y_px = altura - margem - z * escala_z

        # Desenha foguete
        desenhar_foguete(tela, x_px, y_px, theta)

        # HUD
        v = math.sqrt(vx * vx + vz * vz)
        texto1 = f"t = {tempos[indice]:6.1f} s"
        texto2 = f"alt = {z/1000:6.2f} km"
        texto3 = f"vel = {v:7.1f} m/s"
        texto4 = f"theta = {math.degrees(theta):6.1f} deg"
        texto5 = f"massa = {massa:8.1f} kg"

        huds = [texto1, texto2, texto3, texto4, texto5]
        for k, txt in enumerate(huds):
            surf = fonte.render(txt, True, (230, 240, 255))
            tela.blit(surf, (20, 20 + 20 * k))

        pygame.display.flip()

        indice += 1
        if indice >= len(tempos):
            indice = len(tempos) - 1  # para no final

        relogio.tick(30)  # FPS da animação

    pygame.quit()


# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================

def main():
    # 1. Parâmetros e rede
    parametros = ParametrosFoguete()

    print("== Treinando rede neural de controle de inclinação ==")
    rede = RedeNeuralControle()
    treinar_rede_controle(rede, parametros, epocas=800, tamanho_lote=256, lr=1e-2)

    # 2. Estado inicial do foguete (em repouso na plataforma, vertical)
    estado_inicial = EstadoFoguete(
        x=0.0,
        z=1.0,                   # ligeiramente acima do solo
        vx=0.0,
        vz=0.0,
        massa=parametros.massa_inicial,
        theta=math.pi / 2        # 90 graus (vertical)
    )

    print("== Integrando equações diferenciais do foguete ==")
    tempos, estados = integrar_trajetoria(estado_inicial, rede, parametros)
    print(f"Simulação concluída com {len(tempos)} passos.")
    print(f"Altura máxima: {estados[:,1].max()/1000:.2f} km")
    print(f"Alcance horizontal: {estados[:,0].max()/1000:.2f} km")

    print("== Iniciando animação no Pygame ==")
    animar_trajetoria(tempos, estados, parametros)


if __name__ == "__main__":
    main()
