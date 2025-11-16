# ============================================================
# Modelo Estatístico da Morte do Universo — Cosmologia FLRW
# Autor: Luiz Tiago Wilcke (LT)
# ============================================================
# Idéia física (em unidades físicas aproximadas):
#
#  H(a) = H0 * sqrt( Omega_r * a^-4 +
#                    Omega_m * a^-3 +
#                    Omega_k * a^-2 +
#                    Omega_de * a^(-3 (1 + w_de)) )
#
#  da/dt = a * H(a)
#
# Onde:
#  - a(t): fator de escala (a = 1 hoje).
#  - H0: constante de Hubble hoje.
#  - Omega_m: densidade de matéria (bariônica + escura).
#  - Omega_r: densidade de radiação.
#  - Omega_de: densidade de energia escura.
#  - Omega_k: curvatura efetiva = 1 - Omega_m - Omega_r - Omega_de.
#  - w_de: parâmetro de estado da energia escura (p = w rho).
#
# "Morte do universo" (cenários):
#  - Big Rip: w_de < -1 e a(t) explode em tempo finito (tomamos a > a_limite).
#  - Morte térmica: expansão eterna com T_CMB -> quase 0 K.
#  - Big Crunch: o radicando de H(a) zera => recoloapso gravitacional.
#
# Estatística:
#  - Sorteamos N universos com parâmetros (Omega_m0, w_de, Omega_k0).
#  - Integramos cada um, classificamos o cenário e o tempo até a morte.
#  - Mostramos probabilidades estimadas e simulamos visualmente 1 universo.
# ============================================================

import math
import random
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import pygame

# ============================================================
# 1) Parâmetros globais do modelo
# ============================================================

@dataclass
class ParametrosCosmologicos:
    # Constante de Hubble hoje (unidades astronômicas clássicas)
    H0_km_s_Mpc: float = 70.0

    # Valores médios típicos de densidades adimensionais
    omega_m0_media: float = 0.3
    omega_m0_desvio: float = 0.05

    omega_r0: float = 8.4e-5  # radiação ~ valor fixo pequeno

    omega_de0_media: float = 0.7  # será ajustado para fechar Omega_total ~ 1
    omega_k_desvio: float = 0.02  # pequena variação de curvatura

    # Energia escura (w ~ -1 com dispersão)
    w_de_media: float = -1.0
    w_de_desvio: float = 0.15

    # Integração temporal
    t_max_Gyr: float = 2000.0      # horizonte de simulação (em Gyr)
    n_passos: int = 4000          # número de passos de integração

    # Critérios de "morte"
    limite_a_big_rip: float = 1e3  # se w < -1 e a > limite -> Big Rip
    temperatura_limite_k: float = 1e-3  # T_CMB abaixo disso ~ morte térmica

    # Estatística
    n_universos: int = 200        # quantos universos aleatórios simular


@dataclass
class UniversoSimulado:
    tempos_Gyr: np.ndarray
    escalas: np.ndarray
    tipo_morte: str
    tempo_morte_Gyr: float
    parametros_locais: Dict[str, float]


# ============================================================
# 2) Núcleo cosmológico estatístico
# ============================================================

class CosmologiaEstatistica:
    def __init__(self, parametros: ParametrosCosmologicos):
        self.parametros = parametros

        # Conversão de H0 para unidades 1/Gyr
        Mpc_m = 3.085677581e22        # 1 Mpc em metros
        H0_s = self.parametros.H0_km_s_Mpc * 1000.0 / Mpc_m  # s^-1
        seg_por_ano = 365.25 * 24 * 3600.0
        seg_por_Gyr = 1e9 * seg_por_ano
        self.H0_Gyr = H0_s * seg_por_Gyr  # ~ 0.0716 1/Gyr
        self.tempo_Hubble_Gyr = 1.0 / self.H0_Gyr

        self.universos: List[UniversoSimulado] = []
        self.prob_cenarios: Dict[str, float] = {}
        self.tempo_morte_medio_Gyr: float = 0.0

    # -------------------------
    # Funções auxiliares
    # -------------------------
    def gerar_parametros_aleatorios(self, rng: np.random.Generator) -> Dict[str, float]:
        """Sorteia um conjunto de parâmetros cosmológicos plausíveis."""
        omega_m0 = max(
            0.01,
            rng.normal(self.parametros.omega_m0_media,
                       self.parametros.omega_m0_desvio)
        )
        # Curvatura pequena, média ~0
        omega_k0 = rng.normal(0.0, self.parametros.omega_k_desvio)

        # Radiação fixa
        omega_r0 = self.parametros.omega_r0

        # Energia escura ajustada para fechar aproximadamente Omega_total ~ 1
        omega_de0 = 1.0 - omega_m0 - omega_r0 - omega_k0

        # Se isso ficar muito estranho, recortamos para faixas físicas
        omega_de0 = max(-1.5, min(omega_de0, 2.0))

        # Parâmetro de estado w_de
        w_de = rng.normal(self.parametros.w_de_media,
                          self.parametros.w_de_desvio)

        return {
            "omega_m0": omega_m0,
            "omega_r0": omega_r0,
            "omega_de0": omega_de0,
            "omega_k0": 1.0 - omega_m0 - omega_r0 - omega_de0,
            "w_de": w_de
        }

    def _E_de_a(self, a: float, p: Dict[str, float]) -> float:
        """Função E(a) adimensional tal que H(a) = H0 * E(a)."""
        omega_r = p["omega_r0"]
        omega_m = p["omega_m0"]
        omega_de = p["omega_de0"]
        omega_k = p["omega_k0"]
        w_de = p["w_de"]

        if a <= 0.0:
            return 0.0

        termo_r = omega_r / (a ** 4)
        termo_m = omega_m / (a ** 3)
        termo_k = omega_k / (a ** 2)
        termo_de = omega_de * (a ** (-3.0 * (1.0 + w_de)))

        radicando = termo_r + termo_m + termo_k + termo_de

        # Pode ficar muito pequeno/negativo numericamente -> tratamos
        if radicando <= 0.0:
            return 0.0
        return math.sqrt(radicando)

    def _da_dt(self, a: float, t_Gyr: float, p: Dict[str, float]) -> float:
        """Equação diferencial da escala: da/dt (t em Gyr)."""
        E = self._E_de_a(a, p)
        H_Gyr = self.H0_Gyr * E
        return a * H_Gyr

    def _temperatura_cmb(self, a: float) -> float:
        """Temperatura aproximada da CMB em Kelvin."""
        T0 = 2.725  # K hoje
        return T0 / max(a, 1e-6)

    def evoluir_universo(self, parametros_local: Dict[str, float]) -> UniversoSimulado:
        """Integra um único universo até a 'morte'."""
        t_max = self.parametros.t_max_Gyr
        n_passos = self.parametros.n_passos
        dt = t_max / float(n_passos - 1)

        tempos: List[float] = []
        escalas: List[float] = []

        # Condições iniciais: a(0) = 1 (hoje)
        t = 0.0
        a = 1.0

        tipo_morte = "Desconhecido"
        tempo_morte = t_max

        for _ in range(n_passos):
            tempos.append(t)
            escalas.append(a)

            # Verificações de "morte" em cada passo
            E = self._E_de_a(a, parametros_local)
            radicando = E * E

            if radicando <= 0.0:
                # H(a) ~ 0 => ponto de retorno / Big Crunch
                tipo_morte = "Big Crunch (colapso gravitacional)"
                tempo_morte = t
                break

            T_cmb = self._temperatura_cmb(a)
            if T_cmb <= self.parametros.temperatura_limite_k:
                tipo_morte = "Morte térmica (expansão eterna)"
                tempo_morte = t
                break

            if (parametros_local["w_de"] < -1.0 and
                    a >= self.parametros.limite_a_big_rip):
                tipo_morte = "Big Rip (energia escura fantasma)"
                tempo_morte = t
                break

            if t >= t_max:
                tipo_morte = "Expansão eterna (incompleta no horizonte)"
                tempo_morte = t
                break

            # Integração RK4 para da/dt
            def f(a_local: float, t_local: float) -> float:
                return self._da_dt(a_local, t_local, parametros_local)

            k1 = f(a, t)
            k2 = f(a + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = f(a + 0.5 * dt * k2, t + 0.5 * dt)
            k4 = f(a + dt * k3, t + dt)

            a = a + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            t = t + dt

            # Evitar valores absurdos ou negativos
            if a <= 0.0:
                tipo_morte = "Big Crunch (colapso total)"
                tempo_morte = t
                break
            if a > 1e6:
                # Universo expandiu demais, consideramos morte térmica
                tipo_morte = "Morte térmica (expansão extrema)"
                tempo_morte = t
                break

        tempos_arr = np.array(tempos, dtype=float)
        escalas_arr = np.array(escalas, dtype=float)

        return UniversoSimulado(
            tempos_Gyr=tempos_arr,
            escalas=escalas_arr,
            tipo_morte=tipo_morte,
            tempo_morte_Gyr=tempo_morte,
            parametros_locais=parametros_local
        )

    def simular_muitos_universos(self):
        """Roda o modelo estatístico para muitos universos."""
        rng = np.random.default_rng()
        self.universos = []

        contagem: Dict[str, int] = {}
        tempos_morte = []

        for _ in range(self.parametros.n_universos):
            p_local = self.gerar_parametros_aleatorios(rng)
            universo = self.evoluir_universo(p_local)
            self.universos.append(universo)

            contagem[universo.tipo_morte] = contagem.get(universo.tipo_morte, 0) + 1
            tempos_morte.append(universo.tempo_morte_Gyr)

        # Probabilidades empíricas dos cenários
        total = float(len(self.universos))
        self.prob_cenarios = {
            k: v / total for k, v in contagem.items()
        }
        self.tempo_morte_medio_Gyr = float(np.mean(tempos_morte))

    def interpolar_escala(self, universo: UniversoSimulado, t_Gyr: float) -> float:
        """Interpolação linear de a(t) em um universo simulado."""
        tempos = universo.tempos_Gyr
        escalas = universo.escalas

        if t_Gyr <= tempos[0]:
            return escalas[0]
        if t_Gyr >= tempos[-1]:
            return escalas[-1]

        idx = int(np.searchsorted(tempos, t_Gyr))
        i0 = max(0, idx - 1)
        i1 = min(len(tempos) - 1, idx)

        t0, t1 = tempos[i0], tempos[i1]
        a0, a1 = escalas[i0], escalas[i1]

        if t1 == t0:
            return a0
        frac = (t_Gyr - t0) / (t1 - t0)
        return a0 + frac * (a1 - a0)


# ============================================================
# 3) Visualização em Pygame
# ============================================================

class SimulacaoPygame:
    def __init__(self, cosmo: CosmologiaEstatistica):
        self.cosmo = cosmo

        pygame.init()
        pygame.display.set_caption("Morte do Universo — Modelo Cosmológico Estatístico (LT)")
        self.largura = 1200
        self.altura = 700
        self.tela = pygame.display.set_mode((self.largura, self.altura))
        self.clock = pygame.time.Clock()

        pygame.font.init()
        self.fonte_pequena = pygame.font.SysFont("consolas", 16)
        self.fonte_media = pygame.font.SysFont("consolas", 20)
        self.fonte_grande = pygame.font.SysFont("consolas", 26)

        # Universo atualmente destacado
        self.indice_universo = 0
        self.t_atual_Gyr = 0.0
        self.velocidade_tempo = 3.0  # Gyr por frame (escala acelerada)
        self.pausado = False

        # Estrelas de fundo (cenário)
        self.estrelas = self._gerar_estrelas(300)

        # Cores
        self.cor_fundo = (5, 5, 20)
        self.cor_caixa = (20, 25, 50)
        self.cor_borda = (80, 90, 140)
        self.cor_texto = (225, 235, 255)

        self.cores_cenarios = {
            "Big Rip (energia escura fantasma)": (220, 60, 60),
            "Morte térmica (expansão eterna)": (80, 170, 255),
            "Morte térmica (expansão extrema)": (80, 170, 255),
            "Expansão eterna (incompleta no horizonte)": (160, 200, 255),
            "Big Crunch (colapso gravitacional)": (255, 120, 60),
            "Big Crunch (colapso total)": (255, 120, 60),
            "Desconhecido": (200, 200, 200),
        }

    def _gerar_estrelas(self, n: int):
        estrelas = []
        for _ in range(n):
            x = random.randint(0, self.largura)
            y = random.randint(0, self.altura)
            brilho = random.randint(80, 255)
            estrelas.append((x, y, brilho))
        return estrelas

    def _desenhar_estrelas(self, escala_atual: float):
        # Brilho decai com a expansão do universo
        fator = 1.0 / (1.0 + escala_atual ** 2)
        for (x, y, brilho) in self.estrelas:
            b = int(max(10, min(255, brilho * fator)))
            pygame.draw.circle(self.tela, (b, b, b), (x, y), 1)

    def _desenhar_paineis(self):
        # Painel principal (esquerda)
        pygame.draw.rect(self.tela, self.cor_caixa,
                         pygame.Rect(20, 20, 700, 500))
        pygame.draw.rect(self.tela, self.cor_borda,
                         pygame.Rect(20, 20, 700, 500), 2)

        # Painel de gráfico (baixo)
        pygame.draw.rect(self.tela, self.cor_caixa,
                         pygame.Rect(20, 540, 700, 140))
        pygame.draw.rect(self.tela, self.cor_borda,
                         pygame.Rect(20, 540, 700, 140), 2)

        # Painel de info (direita)
        pygame.draw.rect(self.tela, self.cor_caixa,
                         pygame.Rect(740, 20, 440, 660))
        pygame.draw.rect(self.tela, self.cor_borda,
                         pygame.Rect(740, 20, 440, 660), 2)

    def _desenhar_texto(self, texto: str, x: int, y: int, fonte=None, cor=None):
        if fonte is None:
            fonte = self.fonte_pequena
        if cor is None:
            cor = self.cor_texto
        surf = fonte.render(texto, True, cor)
        self.tela.blit(surf, (x, y))

    def _desenhar_universo_bolha(self, universo: UniversoSimulado, escala_atual: float):
        # Centro da "bolha" do universo
        cx = 370
        cy = 270

        # Raio proporcional ao fator de escala (limitado)
        a = max(0.2, min(escala_atual, 10.0))
        raio_base = 40
        raio = int(raio_base + (a - 1.0) * 30.0)
        raio = max(20, min(raio, 260))

        # Cor varia com tipo de morte
        tipo = universo.tipo_morte
        cor_base = self.cores_cenarios.get(tipo, (200, 200, 200))

        # Preenchimento gradiente simples
        pygame.draw.circle(self.tela, (20, 30, 60), (cx, cy), raio + 5)
        pygame.draw.circle(self.tela, cor_base, (cx, cy), raio, 3)

        # Pequenas "galáxias" dentro da bolha
        random.seed(42)  # fixo para aparência estável
        for _ in range(50):
            ang = random.random() * 2 * math.pi
            r = random.random() * raio * 0.9
            gx = int(cx + r * math.cos(ang))
            gy = int(cy + r * math.sin(ang))
            tamanho = 2
            brilho = int(180 / (1.0 + a ** 0.5))
            cor_gala = (brilho, brilho, 255)
            pygame.draw.circle(self.tela, cor_gala, (gx, gy), tamanho)

        # Texto no centro
        self._desenhar_texto("Universo", cx - 50, cy - 10, self.fonte_media)

    def _desenhar_grafico_escala(self, universo: UniversoSimulado):
        # Retângulo do gráfico
        x0, y0, w, h = 30, 550, 680, 120

        tempos = universo.tempos_Gyr
        escalas = universo.escalas
        if len(tempos) < 2:
            return

        t_min, t_max = tempos[0], tempos[-1]
        a_min, a_max = float(np.min(escalas)), float(np.max(escalas))
        a_min = min(a_min, 1.0)
        a_max = max(a_max, 1.5)

        # Margem
        t_min_plot = t_min
        t_max_plot = t_max
        a_min_plot = a_min
        a_max_plot = a_max

        def conv_t(t):
            if t_max_plot == t_min_plot:
                return x0
            return int(x0 + (t - t_min_plot) / (t_max_plot - t_min_plot) * w)

        def conv_a(a):
            if a_max_plot == a_min_plot:
                return y0 + h // 2
            frac = (a - a_min_plot) / (a_max_plot - a_min_plot)
            return int(y0 + h - frac * h)

        # Eixo de fundo
        pygame.draw.line(self.tela, (100, 110, 160),
                         (x0, y0 + h - 1), (x0 + w, y0 + h - 1), 1)

        # Curva a(t)
        pontos = []
        for t, a in zip(tempos, escalas):
            px = conv_t(t)
            py = conv_a(a)
            pontos.append((px, py))

        if len(pontos) >= 2:
            pygame.draw.lines(self.tela, (120, 200, 255), False, pontos, 2)

        # Marcador do tempo atual
        t_atual = self.t_atual_Gyr
        if t_atual >= t_min_plot and t_atual <= t_max_plot:
            a_atual = escala_atual = self.cosmo.interpolar_escala(universo, t_atual)
            px = conv_t(t_atual)
            py = conv_a(a_atual)
            pygame.draw.circle(self.tela, (255, 255, 255), (px, py), 4)

    def _desenhar_painel_info(self, universo: UniversoSimulado):
        x_base = 750
        y = 30

        self._desenhar_texto("Modelo Estatístico da Morte do Universo", x_base, y,
                             self.fonte_grande)
        y += 40

        # Info geral
        t_hubble = self.cosmo.tempo_Hubble_Gyr
        self._desenhar_texto(f"Tempo de Hubble ~ {t_hubble:5.2f} Gyr", x_base, y)
        y += 22
        self._desenhar_texto(f"Universos simulados: {len(self.cosmo.universos)}",
                             x_base, y)
        y += 22
        self._desenhar_texto(f"Tempo médio até 'morte': {self.cosmo.tempo_morte_medio_Gyr:6.1f} Gyr",
                             x_base, y)
        y += 30

        # Cenário do universo atual
        tipo = universo.tipo_morte
        cor_tipo = self.cores_cenarios.get(tipo, self.cor_texto)
        self._desenhar_texto("Universo destacado:", x_base, y, self.fonte_media)
        y += 25
        self._desenhar_texto(f"ID = {self.indice_universo}", x_base + 10, y)
        y += 22
        self._desenhar_texto(f"Tipo de morte: {tipo}", x_base + 10, y,
                             self.fonte_pequena, cor_tipo)
        y += 22
        self._desenhar_texto(f"Tempo até a morte ~ {universo.tempo_morte_Gyr:6.1f} Gyr",
                             x_base + 10, y)
        y += 30

        # Parâmetros cosmológicos locais
        p = universo.parametros_locais
        self._desenhar_texto("Parâmetros cosmológicos (amostra):", x_base, y,
                             self.fonte_media)
        y += 25
        self._desenhar_texto(f"Omega_m0 = {p['omega_m0']:+.3f}", x_base + 10, y)
        y += 20
        self._desenhar_texto(f"Omega_r0 = {p['omega_r0']:+.3e}", x_base + 10, y)
        y += 20
        self._desenhar_texto(f"Omega_de0 = {p['omega_de0']:+.3f}", x_base + 10, y)
        y += 20
        self._desenhar_texto(f"Omega_k0 = {p['omega_k0']:+.3f}", x_base + 10, y)
        y += 20
        self._desenhar_texto(f"w_de     = {p['w_de']:+.3f}", x_base + 10, y)
        y += 30

        # Probabilidades dos cenários
        self._desenhar_texto("Probabilidades empíricas dos cenários:", x_base,
                             y, self.fonte_media)
        y += 25

        # Barras simples
        x_barra = x_base + 10
        largura_max = 250
        altura_barra = 14
        espaco = 5

        for nome, prob in self.cosmo.prob_cenarios.items():
            largura = int(largura_max * prob)
            cor = self.cores_cenarios.get(nome, (180, 180, 200))
            pygame.draw.rect(self.tela, cor,
                             pygame.Rect(x_barra, y, largura, altura_barra))
            self._desenhar_texto(f"{nome}: {prob*100:4.1f}%",
                                 x_barra + largura_max + 10, y - 3,
                                 self.fonte_pequena)
            y += altura_barra + espaco

        y += 20
        self._desenhar_texto("Controles:", x_base, y, self.fonte_media)
        y += 22
        self._desenhar_texto("Espaço: Pausar/continuar", x_base + 10, y)
        y += 18
        self._desenhar_texto("← / → : Trocar universo", x_base + 10, y)
        y += 18
        self._desenhar_texto("ESC: Sair", x_base + 10, y)

    def loop_principal(self):
        rodando = True
        while rodando:
            dt_ms = self.clock.tick(60)
            dt = dt_ms / 1000.0  # segundos reais, só pra animação

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    rodando = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        rodando = False
                    elif event.key == pygame.K_SPACE:
                        self.pausado = not self.pausado
                    elif event.key == pygame.K_RIGHT:
                        self.indice_universo = (self.indice_universo + 1) % len(self.cosmo.universos)
                        self.t_atual_Gyr = 0.0
                    elif event.key == pygame.K_LEFT:
                        self.indice_universo = (self.indice_universo - 1) % len(self.cosmo.universos)
                        self.t_atual_Gyr = 0.0

            universo = self.cosmo.universos[self.indice_universo]

            if not self.pausado:
                self.t_atual_Gyr += self.velocidade_tempo * dt
                if self.t_atual_Gyr > universo.tempo_morte_Gyr:
                    # Reinicia o filme desse universo
                    self.t_atual_Gyr = 0.0

            escala_atual = self.cosmo.interpolar_escala(universo, self.t_atual_Gyr)

            # Desenho
            self.tela.fill(self.cor_fundo)
            self._desenhar_estrelas(escala_atual)
            self._desenhar_paineis()
            self._desenhar_universo_bolha(universo, escala_atual)
            self._desenhar_grafico_escala(universo)
            self._desenhar_painel_info(universo)

            pygame.display.flip()

        pygame.quit()


# ============================================================
# 4) Execução
# ============================================================

def main():
    parametros = ParametrosCosmologicos()
    cosmo = CosmologiaEstatistica(parametros)
    cosmo.simular_muitos_universos()

    print("=== Resumo Estatístico dos Cenários de Morte do Universo ===")
    print(f"Tempo de Hubble ~ {cosmo.tempo_Hubble_Gyr:.2f} Gyr")
    print("Probabilidades empíricas (aproximadas):")
    for nome, prob in cosmo.prob_cenarios.items():
        print(f"  {nome:40s}: {prob*100:5.2f}%")
    print(f"Tempo médio até a morte (entre universos simulados): "
          f"{cosmo.tempo_morte_medio_Gyr:.2f} Gyr")

    simulacao = SimulacaoPygame(cosmo)
    simulacao.loop_principal()


if __name__ == "__main__":
    main()
