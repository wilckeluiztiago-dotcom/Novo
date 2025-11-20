# ============================================================
# SIMULADOR DIDÁTICO DE FISSÃO DO URÂNIO-235 (TOY MODEL)
# Autor: Luiz Tiago Wilcke
# ============================================================
# - Simulação numérica (Monte Carlo) de reação em cadeia toy
# - Modelo 2D simplificado com nêutrons e núcleos U-235
# - Animação em Pygame: absorção -> fissão -> emissão de nêutrons
# - Estatísticas: número de fissões, nêutrons ao longo do tempo,
#   energia liberada (unidades arbitrárias)
# ============================================================

import math, random, sys
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

# Pygame pode não estar instalado no seu ambiente.
# Instale com:
#   python -m pip install pygame numpy matplotlib
try:
    import pygame
except ImportError:
    pygame = None

# ------------------------------------------------------------
# 1. CONFIGURAÇÕES DO MODELO
# ------------------------------------------------------------

@dataclass
class ConfiguracoesSimulacao:
    # domínio 2D (unidades arbitrárias)
    largura: float = 1.0
    altura: float = 1.0

    # quantidade inicial de núcleos U-235
    n_nucleos: int = 220

    # nêutrons iniciais (gatilho)
    n_neutrons_iniciais: int = 3

    # passos de tempo da simulação
    passos: int = 600

    # passo temporal (afeta deslocamento)
    dt: float = 0.015

    # velocidade média dos nêutrons (unidades arbitrárias)
    vel_neutron: float = 0.38

    # raio de captura/fissão efetiva (toy)
    raio_interacao: float = 0.018

    # probabilidade de um núcleo capturar um nêutron quando perto
    prob_captura: float = 0.42

    # probabilidade de gerar fissão ao capturar (toy)
    prob_fissao: float = 0.90

    # número médio de nêutrons produzidos por fissão (toy)
    # (na realidade varia ~2-3, mas aqui é só conceitual)
    media_neutrons_fissao: float = 2.6
    desvio_neutrons_fissao: float = 0.6

    # energia liberada por fissão (unidade arbitrária)
    energia_por_fissao: float = 1.0

    # espalhamento: chance de mudar direção aleatoriamente
    prob_espalhamento: float = 0.08

    # absorção sem fissão (perda): chance do nêutron sumir
    prob_absorcao_perda: float = 0.02

@dataclass
class ConfiguracoesPygame:
    largura_px: int = 1000
    altura_px: int = 700
    fps: int = 60
    escala: float = 600  # converte unidade do domínio em pixels
    raio_nucleo_px: int = 4
    raio_neutron_px: int = 2


# ------------------------------------------------------------
# 2. OBJETOS DO MUNDO (NUCLEO E NEUTRON)
# ------------------------------------------------------------

@dataclass
class NucleoU235:
    x: float
    y: float
    vivo: bool = True
    # para animar fragmentos após fissão
    fragmento_a: Optional[Tuple[float, float]] = None
    fragmento_b: Optional[Tuple[float, float]] = None
    t_fissao: float = 0.0  # tempo desde a fissão (pra animar)

@dataclass
class Neutron:
    x: float
    y: float
    vx: float
    vy: float
    vivo: bool = True


# ------------------------------------------------------------
# 3. FUNÇÕES AUXILIARES
# ------------------------------------------------------------

def limitar_reflexao(x: float, y: float, vx: float, vy: float, cfg: ConfiguracoesSimulacao):
    # Reflete nas bordas do domínio.
    if x < 0:
        x = -x
        vx = -vx
    if x > cfg.largura:
        x = 2*cfg.largura - x
        vx = -vx
    if y < 0:
        y = -y
        vy = -vy
    if y > cfg.altura:
        y = 2*cfg.altura - y
        vy = -vy
    return x, y, vx, vy

def amostrar_neutrons_fissao(cfg: ConfiguracoesSimulacao) -> int:
    n = int(round(random.gauss(cfg.media_neutrons_fissao, cfg.desvio_neutrons_fissao)))
    return max(1, min(5, n))  # limita pra ficar estável/visual

def direcao_aleatoria(vel: float):
    ang = random.random() * 2*math.pi
    return vel*math.cos(ang), vel*math.sin(ang)

def distancia2(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    dx = a[0]-b[0]
    dy = a[1]-b[1]
    return dx*dx + dy*dy


# ------------------------------------------------------------
# 4. SIMULAÇÃO MONTE CARLO (TOY)
# ------------------------------------------------------------

class SimuladorFissaoU235:
    def __init__(self, cfg: ConfiguracoesSimulacao):
        self.cfg = cfg
        self.nucleos: List[NucleoU235] = []
        self.neutrons: List[Neutron] = []

        self.tempo = 0.0
        self.fissoes_total = 0
        self.energia_total = 0.0

        self.historico_neutrons = []
        self.historico_fissoes = []

        self._inicializar_mundo()

    def _inicializar_mundo(self):
        # espalha núcleos no domínio
        for _ in range(self.cfg.n_nucleos):
            x = random.random()*self.cfg.largura
            y = random.random()*self.cfg.altura
            self.nucleos.append(NucleoU235(x, y))

        # adiciona nêutrons iniciais
        for _ in range(self.cfg.n_neutrons_iniciais):
            x = random.random()*self.cfg.largura
            y = random.random()*self.cfg.altura
            vx, vy = direcao_aleatoria(self.cfg.vel_neutron)
            self.neutrons.append(Neutron(x, y, vx, vy))

    def passo(self):
        cfg = self.cfg
        self.tempo += cfg.dt

        # move nêutrons
        for n in self.neutrons:
            if not n.vivo:
                continue

            # espalhamento aleatório simples
            if random.random() < cfg.prob_espalhamento:
                n.vx, n.vy = direcao_aleatoria(cfg.vel_neutron)

            n.x += n.vx * cfg.dt
            n.y += n.vy * cfg.dt
            n.x, n.y, n.vx, n.vy = limitar_reflexao(n.x, n.y, n.vx, n.vy, cfg)

            # absorção "perda"
            if random.random() < cfg.prob_absorcao_perda:
                n.vivo = False

        # checa interações nêutron-núcleo
        r2 = cfg.raio_interacao**2
        for n in self.neutrons:
            if not n.vivo:
                continue

            for nuc in self.nucleos:
                if (not nuc.vivo):
                    continue

                if distancia2((n.x,n.y),(nuc.x,nuc.y)) <= r2:
                    # tentativa de captura
                    if random.random() < cfg.prob_captura:
                        n.vivo = False

                        # decide se fissiona
                        if random.random() < cfg.prob_fissao:
                            self._fissionar(nuc)
                        break

        # atualiza animação de fragmentos
        for nuc in self.nucleos:
            if nuc.fragmento_a is not None:
                nuc.t_fissao += cfg.dt

        # limpa mortos
        self.neutrons = [n for n in self.neutrons if n.vivo]

        # registra histórico
        self.historico_neutrons.append(len(self.neutrons))
        self.historico_fissoes.append(self.fissoes_total)

    def _fissionar(self, nuc: NucleoU235):
        cfg = self.cfg
        nuc.vivo = False

        # cria dois fragmentos que "explodem" em direções opostas (toy)
        ang = random.random()*2*math.pi
        dir_a = (math.cos(ang), math.sin(ang))
        dir_b = (-dir_a[0], -dir_a[1])

        # posição inicial dos fragmentos é o núcleo
        nuc.fragmento_a = (nuc.x + 0.003*dir_a[0], nuc.y + 0.003*dir_a[1])
        nuc.fragmento_b = (nuc.x + 0.003*dir_b[0], nuc.y + 0.003*dir_b[1])
        nuc.t_fissao = 0.0

        # estatísticas
        self.fissoes_total += 1
        self.energia_total += cfg.energia_por_fissao

        # cria nêutrons novos emitidos
        k = amostrar_neutrons_fissao(cfg)
        for _ in range(k):
            vx, vy = direcao_aleatoria(cfg.vel_neutron)
            self.neutrons.append(Neutron(nuc.x, nuc.y, vx, vy))

    def rodar(self):
        for _ in range(self.cfg.passos):
            if len(self.neutrons) == 0:
                self.historico_neutrons.append(0)
                self.historico_fissoes.append(self.fissoes_total)
                self.tempo += self.cfg.dt
                continue
            self.passo()


# ------------------------------------------------------------
# 5. ANIMAÇÃO PYGAME
# ------------------------------------------------------------

class VisualizadorPygame:
    def __init__(self, simulador: SimuladorFissaoU235, cfg_pg: ConfiguracoesPygame):
        if pygame is None:
            raise RuntimeError("pygame não está instalado. Instale com: pip install pygame")
        self.sim = simulador
        self.cfg_pg = cfg_pg

        pygame.init()
        self.tela = pygame.display.set_mode((cfg_pg.largura_px, cfg_pg.altura_px))
        pygame.display.set_caption("Fissão U-235 (Toy Model) — Luiz Tiago Wilcke")
        self.relogio = pygame.time.Clock()
        self.fonte = pygame.font.SysFont("consolas", 18)
        self.fonte_peq = pygame.font.SysFont("consolas", 14)

        self.rodando = True
        self.pausado = False

    def _to_px(self, x, y):
        return int(x*self.cfg_pg.escala + 50), int(y*self.cfg_pg.escala + 50)

    def loop(self):
        passo_idx = 0
        while self.rodando:
            self._eventos()
            if (not self.pausado) and passo_idx < self.sim.cfg.passos:
                self.sim.passo()
                passo_idx += 1
            self._desenhar()
            self.relogio.tick(self.cfg_pg.fps)
        pygame.quit()

    def _eventos(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self.rodando = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_SPACE:
                    self.pausado = not self.pausado
                if ev.key == pygame.K_r:
                    self.sim.__init__(self.sim.cfg)

    def _desenhar(self):
        cfg_pg = self.cfg_pg

        self.tela.fill((10, 12, 16))
        pygame.draw.rect(self.tela, (25, 30, 40),
                         (30, 30, int(cfg_pg.escala+40), int(cfg_pg.escala+40)),
                         border_radius=8)

        # núcleos vivos e fragmentos
        for nuc in self.sim.nucleos:
            if nuc.vivo:
                xpx, ypx = self._to_px(nuc.x, nuc.y)
                pygame.draw.circle(self.tela, (60, 200, 120), (xpx, ypx), cfg_pg.raio_nucleo_px)
            else:
                if nuc.fragmento_a is not None and nuc.t_fissao < 0.6:
                    xa, ya = nuc.fragmento_a
                    xb, yb = nuc.fragmento_b
                    fator = nuc.t_fissao * 0.10
                    ax = nuc.x + (xa - nuc.x) + fator*(xa - nuc.x)
                    ay = nuc.y + (ya - nuc.y) + fator*(ya - nuc.y)
                    bx = nuc.x + (xb - nuc.x) + fator*(xb - nuc.x)
                    by = nuc.y + (yb - nuc.y) + fator*(yb - nuc.y)
                    axp, ayp = self._to_px(ax, ay)
                    bxp, byp = self._to_px(bx, by)
                    pygame.draw.circle(self.tela, (230, 180, 60), (axp, ayp), cfg_pg.raio_nucleo_px+1)
                    pygame.draw.circle(self.tela, (230, 120, 60), (bxp, byp), cfg_pg.raio_nucleo_px+1)

        # nêutrons
        for n in self.sim.neutrons:
            xpx, ypx = self._to_px(n.x, n.y)
            pygame.draw.circle(self.tela, (150, 180, 255), (xpx, ypx), cfg_pg.raio_neutron_px)

        # painel lateral
        x0 = int(cfg_pg.escala + 90)
        pygame.draw.rect(self.tela, (18, 20, 26), (x0, 30, 330, cfg_pg.altura_px-60), border_radius=10)

        texto = [
            "Fissão U-235 — Toy Model",
            f"Tempo simulado: {self.sim.tempo:7.3f}",
            f"Neutrons ativos: {len(self.sim.neutrons):5d}",
            f"Núcleos restantes: {sum(1 for k in self.sim.nucleos if k.vivo):5d}",
            f"Fissões totais: {self.sim.fissoes_total:5d}",
            f"Energia total (arb): {self.sim.energia_total:8.3f}",
            "",
            "Controles:",
            "  SPACE  -> pausar/continuar",
            "  R      -> reiniciar",
            "  fechar -> sair",
        ]

        y = 50
        for t in texto:
            surf = self.fonte.render(t, True, (220, 225, 235))
            self.tela.blit(surf, (x0+20, y))
            y += 25

        # histórico de nêutrons (mini gráfico)
        hist = self.sim.historico_neutrons[-200:]
        if len(hist) > 2:
            hmax = max(1, max(hist))
            base_y = cfg_pg.altura_px - 90
            base_x = x0 + 20
            w, h = 290, 120
            pygame.draw.rect(self.tela, (12, 14, 18), (base_x, base_y-h, w, h), border_radius=8)
            pts = []
            for i, val in enumerate(hist):
                px = base_x + int(w*i/(len(hist)-1))
                py = base_y - int(h*val/hmax)
                pts.append((px, py))
            if len(pts) >= 2:
                pygame.draw.lines(self.tela, (120, 200, 255), False, pts, 2)
            lab = self.fonte_peq.render("Histórico de nêutrons ativos", True, (180, 185, 195))
            self.tela.blit(lab, (base_x, base_y-h-20))

        pygame.display.flip()


def main():
    sem_animacao = ("--sem-animacao" in sys.argv)

    cfg = ConfiguracoesSimulacao()
    sim = SimuladorFissaoU235(cfg)

    if sem_animacao or pygame is None:
        sim.rodar()
        print("\n=== RELATÓRIO FINAL (Toy Model) ===")
        print(f"Tempo total simulado: {sim.tempo:.4f}")
        print(f"Fissões totais:       {sim.fissoes_total}")
        print(f"Energia total (arb):  {sim.energia_total:.4f}")
        print(f"Nêutrons finais:      {len(sim.neutrons)}")
        print("\n--- README (cole no GitHub) ---\n")
        return

    cfg_pg = ConfiguracoesPygame()
    vis = VisualizadorPygame(sim, cfg_pg)
    vis.loop()

    print("\n--- README (cole no GitHub) ---\n")
    print(README_GITHUB)


if __name__ == "__main__":
    main()
