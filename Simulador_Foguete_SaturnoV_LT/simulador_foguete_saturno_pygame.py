# ============================================================
# SIMULADOR DE LANÇAMENTO — FOGUETE TIPO SATURNO V (NASA)
# Autor: Luiz Tiago Wilcke (LT)
# ============================================================
# - Modelo físico com EDOs: dinâmica orbital 2D com massa variável
# - 3 estágios (parâmetros inspirados no Saturn V)
# - Gravidade variável com altitude
# - Atmosfera exponencial + arrasto
# - Integração RK4
# - Animação em tempo real com Pygame
#
# Teclas:
#   ESPAÇO  -> Pausar/Continuar
#   R       -> Reiniciar
#   + / -   -> Acelerar/Desacelerar simulação
#   ESC     -> Sair
# ============================================================

import math, os, sys, time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

# Pygame (animação)
try:
    import pygame
except Exception as e:
    raise RuntimeError(
        "Pygame não encontrado. Instale com:\n"
        "pip install pygame"
    ) from e


# ============================================================
# 1. Constantes físicas
# ============================================================
MU_TERRA = 3.986004418e14     # parâmetro gravitacional (m^3/s^2)
RAIO_TERRA = 6_371_000.0      # m
G0 = 9.80665                 # m/s^2

RHO0 = 1.225                 # kg/m^3 (nível do mar)
ALT_ESCALA = 8500.0          # m (atmosfera exponencial)


# ============================================================
# 2. Estruturas de estágio e configurações
# ============================================================
@dataclass
class Estagio:
    nome: str
    massa_estrutura: float     # kg
    massa_prop: float          # kg
    empuxo_sl: float           # N ao nível do mar
    empuxo_vac: float          # N no vácuo
    isp_sl: float              # s ao nível do mar
    isp_vac: float             # s no vácuo
    tempo_queima: float        # s
    diametro: float            # m (para área de arrasto)
    cd: float                  # coef. de arrasto


@dataclass
class Configuracoes:
    dt: float = 0.05
    duracao_max: float = 800.0
    passo_log: float = 0.5

    # programa de inclinação (gravity turn)
    t_inicio_pitch: float = 10.0
    t_fim_pitch: float = 160.0
    angulo_final_deg: float = 2.0  # próximo de horizontal

    # visual
    largura: int = 1000
    altura: int = 700
    fps: int = 60
    fator_tempo_inicial: float = 1.0


def estagios_saturno_v() -> List[Estagio]:
    # Valores aproximados inspirados no Saturn V
    return [
        Estagio(
            nome="S-IC (1º estágio)",
            massa_estrutura=131_000.0,
            massa_prop=2_130_000.0,
            empuxo_sl=33.4e6,
            empuxo_vac=35.1e6,
            isp_sl=263.0,
            isp_vac=304.0,
            tempo_queima=150.0,
            diametro=10.1,
            cd=0.5
        ),
        Estagio(
            nome="S-II (2º estágio)",
            massa_estrutura=36_000.0,
            massa_prop=456_000.0,
            empuxo_sl=5.1e6,
            empuxo_vac=5.7e6,
            isp_sl=396.0,
            isp_vac=421.0,
            tempo_queima=360.0,
            diametro=10.1,
            cd=0.4
        ),
        Estagio(
            nome="S-IVB (3º estágio)",
            massa_estrutura=13_300.0,
            massa_prop=109_500.0,
            empuxo_sl=0.9e6,
            empuxo_vac=1.0e6,
            isp_sl=421.0,
            isp_vac=421.0,
            tempo_queima=165.0,
            diametro=6.6,
            cd=0.3
        )
    ]


# ============================================================
# 3. Ambiente (gravidade + atmosfera + arrasto)
# ============================================================
def gravidade(alt: float) -> float:
    r = RAIO_TERRA + max(0.0, alt)
    return MU_TERRA / (r*r)

def densidade_ar(alt: float) -> float:
    return RHO0 * math.exp(-max(0.0, alt)/ALT_ESCALA)

def area_frontal(diametro: float) -> float:
    return math.pi*(diametro/2.0)**2

def arrasto(rho: float, v: float, cd: float, A: float) -> float:
    return 0.5 * rho * v*v * cd * A


# ============================================================
# 4. Programa de pitch (ângulo de empuxo)
# ============================================================
def angulo_empuxo(t: float, cfg: Configuracoes) -> float:
    # 90° = vertical, 0° = horizontal
    if t < cfg.t_inicio_pitch:
        return math.radians(90.0)
    if t > cfg.t_fim_pitch:
        return math.radians(cfg.angulo_final_deg)

    frac = (t - cfg.t_inicio_pitch) / (cfg.t_fim_pitch - cfg.t_inicio_pitch)
    ang_deg = 90.0 + frac*(cfg.angulo_final_deg - 90.0)
    return math.radians(ang_deg)


# ============================================================
# 5. Dinâmica do foguete
#    Estado: [x, y, vx, vy, m, idx_estagio, tempo_estagio]
# ============================================================
class SimuladorFoguete:
    def __init__(self, cfg: Configuracoes, estagios: List[Estagio]):
        self.cfg = cfg
        self.estagios = estagios
        self.reiniciar()

    def reiniciar(self):
        self.t = 0.0
        self.x = 0.0
        self.y = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.idx = 0
        self.tempo_estagio = 0.0

        self.massa_total_estagios = sum(e.massa_estrutura + e.massa_prop for e in self.estagios)
        self.m = self.massa_total_estagios

        self.log = []
        self._prox_log = 0.0

    def estagio_atual(self) -> Estagio:
        return self.estagios[self.idx]

    def empuxo_e_isp(self, alt: float) -> Tuple[float, float]:
        e = self.estagio_atual()
        rho = densidade_ar(alt)
        # interpola entre SL e VAC pelo decaimento de densidade
        w = math.exp(-alt/30_000.0)
        emp = e.empuxo_vac*(1-w) + e.empuxo_sl*w
        isp = e.isp_vac*(1-w) + e.isp_sl*w
        return emp, isp

    def vazao_massa(self, emp: float, isp: float) -> float:
        return emp / (isp*G0)

    def derivadas(self, estado: np.ndarray) -> np.ndarray:
        x, y, vx, vy, m, idx, te = estado
        idx = int(idx)
        if idx >= len(self.estagios):
            return np.zeros_like(estado)

        e = self.estagios[idx]
        alt = y
        g = gravidade(alt)
        rho = densidade_ar(alt)
        A = area_frontal(e.diametro)

        v = math.hypot(vx, vy)
        # direção da velocidade (para drag)
        if v > 1e-6:
            ux, uy = vx/v, vy/v
        else:
            ux, uy = 0.0, 1.0

        emp, isp = self.empuxo_e_isp(alt)
        mdot = self.vazao_massa(emp, isp) if te < e.tempo_queima else 0.0

        # direção do empuxo (pitch program)
        ang = angulo_empuxo(self.t, self.cfg)
        tx, ty = math.cos(ang), math.sin(ang)

        # forças
        F_emp_x = emp * tx if te < e.tempo_queima else 0.0
        F_emp_y = emp * ty if te < e.tempo_queima else 0.0

        D = arrasto(rho, v, e.cd, A)
        F_drag_x = -D * ux
        F_drag_y = -D * uy

        # aceleração
        ax = (F_emp_x + F_drag_x) / m
        ay = (F_emp_y + F_drag_y) / m - g

        # der estado
        dx = vx
        dy = vy
        dvx = ax
        dvy = ay
        dm = -mdot
        didx = 0.0
        dte = 1.0

        return np.array([dx, dy, dvx, dvy, dm, didx, dte], dtype=float)

    def passo_rk4(self):
        estado = np.array([self.x, self.y, self.vx, self.vy, self.m, float(self.idx), self.tempo_estagio])
        dt = self.cfg.dt

        k1 = self.derivadas(estado)
        k2 = self.derivadas(estado + 0.5*dt*k1)
        k3 = self.derivadas(estado + 0.5*dt*k2)
        k4 = self.derivadas(estado + dt*k3)

        novo = estado + dt*(k1 + 2*k2 + 2*k3 + k4)/6.0
        self.x, self.y, self.vx, self.vy, self.m, _, self.tempo_estagio = novo.tolist()

        # consumo de propelente do estágio
        e = self.estagio_atual()
        massa_estagio_antes = e.massa_estrutura + e.massa_prop
        massa_restante_estagio = self.m - sum(
            self.estagios[j].massa_estrutura + self.estagios[j].massa_prop
            for j in range(self.idx+1, len(self.estagios))
        )

        # se queima terminou ou propelente acabou, faz separação
        if (self.tempo_estagio >= e.tempo_queima) or (massa_restante_estagio <= e.massa_estrutura + 5.0):
            # remove massa estrutural do estágio atual
            self.m -= e.massa_estrutura
            self.idx += 1
            self.tempo_estagio = 0.0
            if self.idx < len(self.estagios):
                print(f">>> Separação de estágio: agora {self.estagio_atual().nome}")
            else:
                print(">>> Motor desligado — todos os estágios consumidos")

        self.t += dt

        # log
        if self.t >= self._prox_log:
            self.log.append({
                "t": self.t, "x": self.x, "y": self.y,
                "vx": self.vx, "vy": self.vy,
                "v": math.hypot(self.vx, self.vy),
                "m": self.m, "estagio": self.idx
            })
            self._prox_log += self.cfg.passo_log


# ============================================================
# 6. Visualização Pygame
# ============================================================
class Visualizador:
    def __init__(self, cfg: Configuracoes):
        pygame.init()
        self.cfg = cfg
        self.tela = pygame.display.set_mode((cfg.largura, cfg.altura))
        pygame.display.set_caption("Simulador de Lançamento — Saturno V (2D)")
        self.clock = pygame.time.Clock()
        self.fonte = pygame.font.SysFont("consolas", 18)
        self.fonte_mini = pygame.font.SysFont("consolas", 14)

        self.camera_zoom = 1.0
        self.trajeto = []  # posições anteriores

    def coord_para_tela(self, x, y):
        # mapa simples: origem no centro inferior
        cx = self.cfg.largura*0.5
        cy = self.cfg.altura*0.85
        escala = 1/300.0 * self.camera_zoom  # ajuste visual
        sx = cx + x*escala
        sy = cy - y*escala
        return int(sx), int(sy)

    def desenhar_foguete(self, pos, ang_rad):
        x, y = pos
        # foguete estilizado (polígono)
        corpo = [(0, -30), (8, -10), (8, 20), (-8, 20), (-8, -10)]
        nariz = [(0, -42), (8, -30), (-8, -30)]
        asa_esq = [(-8, 10), (-18, 22), (-8, 22)]
        asa_dir = [(8, 10), (18, 22), (8, 22)]

        def rot(p):
            px, py = p
            rx = px*math.cos(ang_rad) - py*math.sin(ang_rad)
            ry = px*math.sin(ang_rad) + py*math.cos(ang_rad)
            return (x+rx, y+ry)

        pygame.draw.polygon(self.tela, (240,240,240), list(map(rot, corpo)))
        pygame.draw.polygon(self.tela, (220,30,30), list(map(rot, nariz)))
        pygame.draw.polygon(self.tela, (40,40,200), list(map(rot, asa_esq)))
        pygame.draw.polygon(self.tela, (40,40,200), list(map(rot, asa_dir)))

    def desenhar(self, sim: SimuladorFoguete, fator_tempo, pausado):
        self.tela.fill((10, 12, 25))

        # gradiente simples do céu
        for i in range(0, self.cfg.altura, 8):
            c = 20 + int(40*i/self.cfg.altura)
            pygame.draw.rect(self.tela, (10, c, 40), (0, i, self.cfg.largura, 8))

        # chão/curvatura da terra
        pygame.draw.rect(self.tela, (15, 80, 15), (0, int(self.cfg.altura*0.85), self.cfg.largura, int(self.cfg.altura*0.15)))

        # trajeto
        if len(self.trajeto) > 2:
            pygame.draw.lines(self.tela, (255, 220, 100), False, self.trajeto[-1800:], 2)

        # foguete
        sx, sy = self.coord_para_tela(sim.x, sim.y)
        self.trajeto.append((sx, sy))

        # ângulo do foguete baseado na velocidade
        ang = math.atan2(sim.vy, sim.vx) if (sim.vx**2+sim.vy**2)>1e-6 else math.radians(90)

        self.desenhar_foguete((sx, sy), -ang + math.pi/2)

        # HUD
        v = math.hypot(sim.vx, sim.vy)
        alt_km = sim.y/1000
        t = sim.t
        est = sim.idx
        est_nome = sim.estagio_atual().nome if est < len(sim.estagios) else "Sem motor"

        texto = [
            f"t = {t:6.1f} s   | fator tempo = {fator_tempo:.1f}x   | {'PAUSADO' if pausado else ''}",
            f"altitude = {alt_km:7.2f} km",
            f"velocidade = {v:7.1f} m/s",
            f"vx = {sim.vx:7.1f} m/s | vy = {sim.vy:7.1f} m/s",
            f"massa = {sim.m:,.0f} kg",
            f"estágio = {est+1 if est<len(sim.estagios) else '-'}  ({est_nome})",
        ]

        y0 = 10
        for line in texto:
            surf = self.fonte.render(line, True, (230,230,230))
            self.tela.blit(surf, (10, y0))
            y0 += 22

        # mini legenda teclas
        leg = "ESPAÇO pausa | R reinicia | +/- tempo | ESC sai"
        surf = self.fonte_mini.render(leg, True, (200,200,200))
        self.tela.blit(surf, (10, self.cfg.altura-24))

        pygame.display.flip()


# ============================================================
# 7. Execução
# ============================================================
def main():
    cfg = Configuracoes()
    estagios = estagios_saturno_v()
    sim = SimuladorFoguete(cfg, estagios)
    vis = Visualizador(cfg)

    fator_tempo = cfg.fator_tempo_inicial
    pausado = False
    rodando = True
    acumulador = 0.0

    while rodando:
        dt_real = vis.clock.tick(cfg.fps)/1000.0
        acumulador += dt_real * fator_tempo

        # eventos
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                rodando = False
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    rodando = False
                elif ev.key == pygame.K_SPACE:
                    pausado = not pausado
                elif ev.key == pygame.K_r:
                    sim.reiniciar()
                    vis.trajeto.clear()
                elif ev.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    fator_tempo = min(20.0, fator_tempo + 0.5)
                elif ev.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    fator_tempo = max(0.5, fator_tempo - 0.5)

        # simulação
        if not pausado:
            while acumulador >= cfg.dt and sim.t < cfg.duracao_max:
                sim.passo_rk4()
                acumulador -= cfg.dt

        # zoom automático conforme altitude
        vis.camera_zoom = 1.0 + sim.y/120_000.0

        vis.desenhar(sim, fator_tempo, pausado)

        if sim.t >= cfg.duracao_max:
            pausado = True

    pygame.quit()

    # salva log em CSV
    if len(sim.log) > 0:
        import pandas as pd
        df_log = pd.DataFrame(sim.log)
        os.makedirs("saidas", exist_ok=True)
        df_log.to_csv("saidas/trajetoria.csv", index=False)
        print("Log salvo em saidas/trajetoria.csv")


if __name__ == "__main__":
    main()