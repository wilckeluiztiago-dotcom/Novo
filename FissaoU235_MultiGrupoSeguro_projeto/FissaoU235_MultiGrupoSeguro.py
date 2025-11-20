# ============================================================
# PROJETO AVANÇADO (SEGURO) — FISSÃO U-235 COM EDP MULTIGRUPO
# Autor: Luiz Tiago Wilcke
# ============================================================
# Objetivo:
#  - Implementar a FORMA REAL das equações de difusão/reatividade
#    de nêutrons (multi-grupo) + cinética pontual com nêutrons atrasados.
#  - Resolver numericamente em 2D com diferenças finitas.
#  - Animar no Pygame a densidade de fluxo nos grupos e fissões toy.
# ============================================================

import sys, math, random
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

try:
    import pygame
except ImportError:
    pygame = None


# ------------------------------------------------------------
# 1. CONFIGURAÇÕES DO DOMÍNIO/GRADES
# ------------------------------------------------------------

@dataclass
class ConfigDominio:
    nx: int = 140
    ny: int = 100
    largura: float = 1.0
    altura: float = 1.0
    dt: float = 8e-5        # passo temporal (toy)
    passos: int = 7000      # passos de simulação

    def __post_init__(self):
        self.dx = self.largura/(self.nx-1)
        self.dy = self.altura/(self.ny-1)


# ------------------------------------------------------------
# 2. PARÂMETROS MULTIGRUPO (FORMA REAL, VALORES FICTÍCIOS)
# ------------------------------------------------------------

@dataclass
class ParametrosMultiGrupo:
    # Grupos:
    # g=0: rápido (fast)
    # g=1: térmico (thermal)
    n_grupos: int = 2

    # Coeficientes de difusão D_g  (placeholders)
    D: Tuple[float, float] = (0.25, 0.12)

    # Seções de choque de absorção Σ_a,g  (placeholders)
    Sigma_a: Tuple[float, float] = (0.35, 0.55)

    # Seções de choque de fissão νΣ_f,g (placeholders)
    nu_Sigma_f: Tuple[float, float] = (0.62, 0.88)

    # Espalhamento Σ_s,g→g' (placeholders)
    # matriz 2x2 com diagonal zero (toy)
    Sigma_s: Tuple[Tuple[float, float], Tuple[float, float]] = (
        (0.0, 0.30),  # fast -> thermal
        (0.04, 0.0)   # thermal -> fast (upscatter toy)
    )

    # Espectro de nêutrons produzidos por fissão χ_g (placeholders)
    chi: Tuple[float, float] = (0.9, 0.1)

    # Fator de saturação não linear para estabilidade (toy)
    k_nl: float = 2.0


# ------------------------------------------------------------
# 3. CINÉTICA PONTUAL (NEUTRONS ATRASADOS) — FORMA REAL
# ------------------------------------------------------------

@dataclass
class ParametrosAtrasados:
    # Número de grupos de precursores (comum em livros)
    m: int = 6

    # Frações β_i (placeholders) e total β
    beta_i: Tuple[float, ...] = (0.0004, 0.0010, 0.0011, 0.0025, 0.0008, 0.0002)
    # Constantes de decaimento λ_i (placeholders)
    lambda_i: Tuple[float, ...] = (0.012, 0.030, 0.111, 0.301, 1.14, 3.01)

    # Tempo de geração Λ (placeholder)
    Lambda: float = 0.0001

    def __post_init__(self):
        self.beta = float(sum(self.beta_i))


# ------------------------------------------------------------
# 4. REATIVIDADE E FONTE INICIAL (TOY)
# ------------------------------------------------------------

@dataclass
class ParametrosReatividade:
    # Reatividade ρ(t) (placeholder)
    rho0: float = 0.0015
    # feedback simples (toy)
    k_feedback: float = 0.0003


@dataclass
class ConfigNucleosToy:
    n_nucleos: int = 300
    limiar_fluxo: float = 1.6      # integra fluxo local até fissionar (toy)
    fonte_fissao: float = 2.8      # força da fonte local após fissão (toy)
    raio_fonte: float = 0.016      # raio de influência da fonte


@dataclass
class ConfigPygame:
    largura_px: int = 1200
    altura_px: int = 760
    fps: int = 60
    escala: int = 650
    raio_nucleo_px: int = 4


# ------------------------------------------------------------
# 5. OBJETO NÚCLEO (TOY, SÓ VISUAL/ESTOCÁSTICO)
# ------------------------------------------------------------

@dataclass
class NucleoToy:
    x: float
    y: float
    vivo: bool = True
    fissao_ativa: bool = False
    t_fissao: float = 0.0
    fluxo_acum: float = 0.0

    def pos_grade(self, dom: ConfigDominio):
        i = int(self.x/dom.dx)
        j = int(self.y/dom.dy)
        return max(0,min(dom.nx-1,i)), max(0,min(dom.ny-1,j))


# ------------------------------------------------------------
# 6. SIMULADOR EDP MULTIGRUPO + CINÉTICA
# ------------------------------------------------------------

class SimuladorMultiGrupoSeguro:
    def __init__(self,
                 dom: ConfigDominio,
                 mg: ParametrosMultiGrupo,
                 atras: ParametrosAtrasados,
                 reat: ParametrosReatividade,
                 nuc_cfg: ConfigNucleosToy):
        self.dom = dom
        self.mg = mg
        self.atras = atras
        self.reat = reat
        self.nuc_cfg = nuc_cfg

        # Fluxos por grupo φ_g(x,y)
        self.phi = np.zeros((mg.n_grupos, dom.nx, dom.ny), dtype=np.float64)

        # Fontes totais por grupo S_g
        self.S_base = np.zeros_like(self.phi)
        self.S_dyn  = np.zeros_like(self.phi)

        # Potência/amplitude global P(t) (cinética pontual)
        self.P = 1.0
        self.C = np.zeros(atras.m, dtype=np.float64)

        self.tempo = 0.0
        self.fissoes_total = 0
        self.energia_total = 0.0

        self.h_phi_total = []
        self.h_P = []
        self.h_keff = []
        self.h_fissoes = []

        self.nucleos: List[NucleoToy] = []
        self._inic()

    def _inic(self):
        dom = self.dom

        # Fonte inicial gaussiana no centro no grupo rápido
        cx, cy = dom.largura/2, dom.altura/2
        for i in range(dom.nx):
            for j in range(dom.ny):
                x = i*dom.dx
                y = j*dom.dy
                r2 = (x-cx)**2 + (y-cy)**2
                self.S_base[0,i,j] = 7.0*math.exp(-r2/(2*(0.05**2)))
                self.S_base[1,i,j] = 1.5*math.exp(-r2/(2*(0.06**2)))

        # Núcleos toy aleatórios
        for _ in range(self.nuc_cfg.n_nucleos):
            self.nucleos.append(NucleoToy(random.random()*dom.largura,
                                          random.random()*dom.altura))

        # Precursores iniciais em equilíbrio toy
        for k in range(self.atras.m):
            self.C[k] = (self.atras.beta_i[k]/self.atras.Lambda)*self.P/self.atras.lambda_i[k]

    def _laplaciano(self, campo):
        dom = self.dom
        dx2 = dom.dx*dom.dx
        dy2 = dom.dy*dom.dy
        lap = np.zeros_like(campo)
        lap[1:-1,1:-1] = (
            (campo[2:,1:-1] - 2*campo[1:-1,1:-1] + campo[:-2,1:-1])/dx2 +
            (campo[1:-1,2:] - 2*campo[1:-1,1:-1] + campo[1:-1,:-2])/dy2
        )
        # Neumann nas bordas
        lap[0,:] = lap[1,:]; lap[-1,:] = lap[-2,:]
        lap[:,0] = lap[:,1]; lap[:,-1] = lap[:,-2]
        return lap

    def _keff_simbolico(self):
        """k_eff toy simbólico: produção / perdas."""
        mg = self.mg
        dom = self.dom

        prod = 0.0
        abs_ = 0.0
        for g in range(mg.n_grupos):
            phi_g = self.phi[g]
            prod += mg.nu_Sigma_f[g]*phi_g.sum()
            abs_ += mg.Sigma_a[g]*phi_g.sum()
        prod *= dom.dx*dom.dy
        abs_ *= dom.dx*dom.dy
        return prod/(abs_+1e-12)

    def _aplicar_fontes_fissao_toy(self):
        """Núcleo toy vira fonte local em ambos grupos."""
        dom = self.dom
        cfg = self.nuc_cfg
        raio2 = cfg.raio_fonte**2

        self.S_dyn[:] = 0.0
        for nuc in self.nucleos:
            if nuc.fissao_ativa and nuc.t_fissao < 0.10:
                i0,j0 = nuc.pos_grade(dom)
                x0,y0 = nuc.x,nuc.y
                for di in range(-6,7):
                    for dj in range(-6,7):
                        i=i0+di; j=j0+dj
                        if 0<=i<dom.nx and 0<=j<dom.ny:
                            x=i*dom.dx; y=j*dom.dy
                            if (x-x0)**2+(y-y0)**2 <= raio2:
                                self.S_dyn[0,i,j] += cfg.fonte_fissao*0.8
                                self.S_dyn[1,i,j] += cfg.fonte_fissao*0.2

    def passo(self):
        dom, mg, atras, reat = self.dom, self.mg, self.atras, self.reat

        self.tempo += dom.dt

        # (A) Fontes toy de fissões
        self._aplicar_fontes_fissao_toy()

        # (B) Calcula termo de produção por fissão multigrupo
        producao_total = np.zeros((dom.nx, dom.ny), dtype=np.float64)
        for g in range(mg.n_grupos):
            producao_total += mg.nu_Sigma_f[g]*self.phi[g]

        # (C) Atualiza EDP por grupo:
        # dφ_g/dt = D_g ∇²φ_g - Σ_a,g φ_g + Σ_s,g'→g φ_g'
        #           + χ_g * (1-β)/Λ * P * (producao_total / normalização)
        #
        # Forma real mantida; termos numéricos são toys.

        normalizacao = producao_total.max() + 1e-9

        for g in range(mg.n_grupos):
            phi_g = self.phi[g]
            lap = self._laplaciano(phi_g)

            absorcao = - mg.Sigma_a[g]*phi_g

            espalhamento = np.zeros_like(phi_g)
            for gp in range(mg.n_grupos):
                if gp != g:
                    espalhamento += mg.Sigma_s[gp][g]*self.phi[gp]

            fonte_fissao = mg.chi[g] * ((1.0 - atras.beta)/atras.Lambda) * self.P * (producao_total/normalizacao)

            # saturação toy
            nao_linear = - mg.k_nl * (phi_g**2)

            self.phi[g] += dom.dt*( mg.D[g]*lap + absorcao + espalhamento + fonte_fissao + self.S_base[g] + self.S_dyn[g] + nao_linear )

            np.maximum(self.phi[g], 0.0, out=self.phi[g])

        # (D) Cinética pontual real com nêutrons atrasados:
        # dP/dt = ((ρ-β)/Λ) P + Σ λ_i C_i
        # dC_i/dt = (β_i/Λ) P - λ_i C_i

        # reatividade toy com feedback (não físico)
        keff = self._keff_simbolico()
        rho = reat.rho0 - reat.k_feedback*(keff-1.0)

        soma_lamb_C = float(np.dot(atras.lambda_i, self.C))
        dP = ((rho - atras.beta)/atras.Lambda)*self.P + soma_lamb_C
        self.P += dom.dt*dP
        self.P = max(1e-8, self.P)

        for i in range(atras.m):
            dC = (atras.beta_i[i]/atras.Lambda)*self.P - atras.lambda_i[i]*self.C[i]
            self.C[i] += dom.dt*dC
            self.C[i] = max(0.0, self.C[i])

        # (E) Núcleos toy acumulam fluxo térmico
        phi_termico = self.phi[1]
        for nuc in self.nucleos:
            if nuc.vivo:
                i,j = nuc.pos_grade(dom)
                nuc.fluxo_acum += phi_termico[i,j]*dom.dt
                if nuc.fluxo_acum >= self.nuc_cfg.limiar_fluxo:
                    nuc.vivo = False
                    nuc.fissao_ativa = True
                    nuc.t_fissao = 0.0
                    self.fissoes_total += 1
                    self.energia_total += 1.0  # arb

        for nuc in self.nucleos:
            if nuc.fissao_ativa:
                nuc.t_fissao += dom.dt
                if nuc.t_fissao >= 0.10:
                    nuc.fissao_ativa = False

        # históricos
        phi_total = float(self.phi.sum())*dom.dx*dom.dy
        self.h_phi_total.append(phi_total)
        self.h_P.append(self.P)
        self.h_keff.append(keff)
        self.h_fissoes.append(self.fissoes_total)

    def rodar(self):
        for _ in range(self.dom.passos):
            self.passo()


# ------------------------------------------------------------
# 7. VISUALIZAÇÃO PYGAME
# ------------------------------------------------------------

class VisualizadorMultiGrupo:
    def __init__(self, sim: SimuladorMultiGrupoSeguro, cfg_pg: ConfigPygame):
        if pygame is None:
            raise RuntimeError("pygame não instalado. Use pip install pygame.")
        self.sim = sim
        self.cfg_pg = cfg_pg

        pygame.init()
        self.tela = pygame.display.set_mode((cfg_pg.largura_px, cfg_pg.altura_px))
        pygame.display.set_caption("Fissão U-235 — Multigrupo + Atrasados (Toy Seguro) — Luiz Tiago Wilcke")
        self.rel = pygame.time.Clock()
        self.fonte = pygame.font.SysFont("consolas", 18)
        self.fonte_peq = pygame.font.SysFont("consolas", 14)

        self.rodando = True
        self.pausado = False
        self.idx = 0
        self.mostrar_grupo = 1  # 0 rápido / 1 térmico

    def _eventos(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self.rodando = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_SPACE:
                    self.pausado = not self.pausado
                if ev.key == pygame.K_r:
                    self.sim.__init__(self.sim.dom, self.sim.mg, self.sim.atras, self.sim.reat, self.sim.nuc_cfg)
                    self.idx=0
                if ev.key == pygame.K_TAB:
                    self.mostrar_grupo = 1 - self.mostrar_grupo

    def _to_px(self, x,y):
        return int(x*self.cfg_pg.escala+35), int(y*self.cfg_pg.escala+35)

    def _heatmap(self, campo2d):
        campo = campo2d.T  # ny x nx
        vmax = max(1e-9, float(campo.max()))
        img = np.clip(campo/vmax, 0, 1)

        r = (img**0.6)*255
        g = (img**0.9)*255
        b = (1-img**1.8)*255
        rgb = np.stack([r,g,b], axis=-1).astype(np.uint8)

        surf = pygame.surfarray.make_surface(rgb)
        surf = pygame.transform.smoothscale(surf, (self.cfg_pg.escala, int(self.cfg_pg.escala*0.72)))
        return surf

    def _desenhar(self):
        self.tela.fill((8,10,14))

        # painel heatmap
        pygame.draw.rect(self.tela, (18,22,30),
                         (20,20, self.cfg_pg.escala+20, int(self.cfg_pg.escala*0.72)+20),
                         border_radius=10)

        surf = self._heatmap(self.sim.phi[self.mostrar_grupo])
        self.tela.blit(surf, (30,30))

        # núcleos toy
        for nuc in self.sim.nucleos:
            xpx, ypx = self._to_px(nuc.x, nuc.y)
            if nuc.vivo:
                pygame.draw.circle(self.tela, (70,220,140), (xpx, ypx), self.cfg_pg.raio_nucleo_px)
            elif nuc.fissao_ativa:
                pygame.draw.circle(self.tela, (240,160,60), (xpx, ypx), self.cfg_pg.raio_nucleo_px+2)

        # painel lateral
        x0 = self.cfg_pg.escala + 80
        pygame.draw.rect(self.tela, (14,16,22),
                         (x0,20,380,self.cfg_pg.altura_px-40),
                         border_radius=12)

        dom=self.sim.dom
        texto = [
            "Multigrupo Difusão-Reação",
            f"Grupo exibido: {'TERMICO' if self.mostrar_grupo==1 else 'RAPIDO'} (TAB)",
            f"t = {self.sim.tempo:8.5f}",
            f"keff(t) simb.: {self.sim.h_keff[-1]:7.4f}" if self.sim.h_keff else "keff: -",
            f"P(t) (ampl.): {self.sim.P:8.3f}",
            f"Fissões toy: {self.sim.fissoes_total:6d}",
            f"Energia arb: {self.sim.energia_total:8.3f}",
            "",
            "Controles:",
            "SPACE pausar",
            "TAB troca grupo",
            "R reiniciar",
        ]

        y=40
        for t in texto:
            self.tela.blit(self.fonte.render(t, True, (220,225,235)), (x0+20,y))
            y+=24

        # sparkline P(t)
        hist = self.sim.h_P[-260:]
        if len(hist)>2:
            w,h=340,120
            bx,by=x0+20,self.cfg_pg.altura_px-80
            pygame.draw.rect(self.tela,(10,12,16),(bx,by-h,w,h),border_radius=8)
            hm=max(1e-6,max(hist))
            pts=[]
            for i,val in enumerate(hist):
                px=bx+int(w*i/(len(hist)-1))
                py=by-int(h*val/hm)
                pts.append((px,py))
            pygame.draw.lines(self.tela,(150,210,255),False,pts,2)
            self.tela.blit(self.fonte_peq.render("Histórico P(t)",True,(180,185,195)),(bx,by-h-20))

        pygame.display.flip()

    def loop(self):
        while self.rodando:
            self._eventos()
            if not self.pausado and self.idx < self.sim.dom.passos:
                self.sim.passo()
                self.idx+=1
            self._desenhar()
            self.rel.tick(self.cfg_pg.fps)
        pygame.quit()


# ------------------------------------------------------------
# 8. README
# ------------------------------------------------------------

README_GITHUB = r"""
# Projeto Avançado (Seguro) — Fissão U-235 com EDP Multigrupo + Nêutrons Atrasados

**Autor:** Luiz Tiago Wilcke

Este projeto implementa a **forma real das equações diferenciais**
usadas em física de reatores (difusão multigrupo) e a **cinética
pontual com nêutrons atrasados**, porém com **parâmetros fictícios/
normalizados**, por segurança.

> **Aviso de segurança**
> - Não há seções de choque reais, enriquecimento real, geometria de reator,
>   dados de criticidade operacional, nem validação para uso industrial.
> - Toy model para estudo e portfólio.

---

## Equações implementadas

### Difusão multigrupo (2 grupos)
Para cada grupo g:

\[
\frac{\partial \phi_g}{\partial t}
= D_g\nabla^2\phi_g
- \Sigma_{a,g}\phi_g
+ \sum_{g'\neq g}\Sigma_{s,g'\to g}\phi_{g'}
+ \chi_g \frac{1-\beta}{\Lambda}P(t)\,F(\phi)
\]

onde o termo de fissão total é:

\[
F(\phi)=\sum_{g}\nu\Sigma_{f,g}\phi_g
\]

### Cinética pontual com nêutrons atrasados (6 grupos)

\[
\frac{dP}{dt}=\frac{\rho-\beta}{\Lambda}P+\sum_{i=1}^6\lambda_i C_i
\]

\[
\frac{dC_i}{dt}=\frac{\beta_i}{\Lambda}P-\lambda_i C_i
\]

### k_eff monitorado (simbólico toy)

\[
k_{eff}(t)=\frac{\int \nu\Sigma_f \phi\,dV}{\int\Sigma_a\phi\,dV}
\]

---

## Como rodar

### Dependências
```bash
python -m pip install numpy pygame
```

### Executar
```bash
python FissaoU235_MultiGrupoSeguro.py
```

Sem pygame:
```bash
python FissaoU235_MultiGrupoSeguro.py --sem-animacao
```

---

## Ajustando parâmetros (didático)

No bloco `ParametrosMultiGrupo` você encontra:
- `D_g`
- `Sigma_a,g`
- `nu_Sigma_f,g`
- `Sigma_s,g->g'`
- `chi_g`

Você pode substituir por valores de material didático **sob sua responsabilidade**.

---

## O que você vê

- Heatmap do fluxo de nêutrons por grupo (TAB troca rápido/térmico).
- Núcleos toy que acumulam fluxo térmico e “fissionam” visualmente.
- Painel com P(t), k_eff simbólico, energia toy.

---

## Licença
Livre para estudo e portfólio, com atribuição ao autor.
"""


# ------------------------------------------------------------
# 9. MAIN
# ------------------------------------------------------------

def main():
    sem_animacao = ("--sem-animacao" in sys.argv)

    dom = ConfigDominio()
    mg = ParametrosMultiGrupo()
    atras = ParametrosAtrasados()
    reat = ParametrosReatividade()
    nuc_cfg = ConfigNucleosToy()

    sim = SimuladorMultiGrupoSeguro(dom, mg, atras, reat, nuc_cfg)

    if sem_animacao or pygame is None:
        sim.rodar()
        print("\n=== RELATÓRIO FINAL (Toy Seguro) ===")
        print(f"Tempo total: {sim.tempo:.6f}")
        print(f"Fissões toy: {sim.fissoes_total}")
        print(f"Energia arb: {sim.energia_total:.3f}")
        print(f"P final:     {sim.P:.3f}")
        print(f"keff final:  {sim.h_keff[-1]:.4f}")
        print("\n--- README ---\n")
        print(README_GITHUB)
        return

    vis = VisualizadorMultiGrupo(sim, ConfigPygame())
    vis.loop()
    print("\n--- README ---\n")
    print(README_GITHUB)


if __name__ == "__main__":
    main()
