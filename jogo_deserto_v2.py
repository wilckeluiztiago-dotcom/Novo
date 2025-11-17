"""
Autor Luiz Tiago Wilcke
"""

import sys
import math
import random
import pygame

# ------------------------------------------------------------
# Configurações gerais
# ------------------------------------------------------------
LARGURA_TELA = 960
ALTURA_TELA = 540
FPS = 60

GRAVIDADE = 0.75
VELOCIDADE_ANDAR = 3.0
VELOCIDADE_CORRER = 5.2
IMPULSO_PULO = -14.5
LIMITE_QUEDA = 22

COR_CEU = (30, 25, 70)
COR_FUNDO_LONGE = (110, 80, 40)
COR_FUNDO_PERTO = (170, 120, 60)
COR_PLATAFORMA = (205, 155, 95)
COR_PLATAFORMA_ESCURA = (160, 120, 80)
COR_JOGADOR_ROUPA = (240, 230, 90)
COR_JOGADOR_CALCA = (130, 110, 40)
COR_JOGADOR_TURBANTE = (250, 250, 250)
COR_INIMIGO = (210, 60, 60)
COR_COLETAVEL = (40, 220, 220)
COR_TEXTO = (245, 245, 245)
COR_VIDA = (20, 220, 80)
COR_VIDA_FUNDO = (80, 35, 35)


# ------------------------------------------------------------
# Classes de entidades
# ------------------------------------------------------------

class Jogador(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.largura = 32
        self.altura = 52

        # imagem base (será animada trocando superfícies)
        self.image = pygame.Surface((self.largura, self.altura), pygame.SRCALPHA)
        self.rect = self.image.get_rect(topleft=(x, y))

        # física
        self.pos = pygame.Vector2(x, y)
        self.vel = pygame.Vector2(0, 0)
        self.no_chao = False
        self.direcao = 1  # 1 direita, -1 esquerda

        # estado
        self.estado = "parado"
        self.tempo_estado = 0.0
        self.vida_maxima = 100
        self.vida = 100
        self.invulneravel = 0.0
        self.tempo_recarga_faca = 0.0

        # animação
        self.quadro = 0
        self.tempo_quadro = 0.0
        self.frames_parado = self._criar_frames(cor_roupa=(245, 235, 110))
        self.frames_correndo = self._criar_frames(cor_roupa=(240, 225, 90))
        self.frames_pulando = self._criar_frames(cor_roupa=(250, 245, 160))
        self.frames_caindo = self._criar_frames(cor_roupa=(220, 210, 70))
        self.frames_atacando = self._criar_frames(cor_roupa=(255, 255, 180))

        self.facas = pygame.sprite.Group()

    def _criar_frames(self, cor_roupa):
        frames = []
        for i in range(4):
            surf = pygame.Surface((self.largura, self.altura), pygame.SRCALPHA)
            # túnica
            pygame.draw.rect(surf, cor_roupa, (6, 14, self.largura - 12, self.altura - 22))
            # calça
            pygame.draw.rect(surf, COR_JOGADOR_CALCA, (6, self.altura - 18, self.largura - 12, 16))
            # turbante
            pygame.draw.rect(surf, COR_JOGADOR_TURBANTE, (8, 0, self.largura - 16, 12))
            # detalhe animado
            if i % 2 == 0:
                pygame.draw.rect(surf, (255, 220, 0), (self.largura - 8, 4, 4, 4))
            else:
                pygame.draw.rect(surf, (255, 220, 0), (self.largura - 12, 4, 4, 4))
            frames.append(surf)
        return frames

    # -------------------------- entrada -----------------------

    def processar_entrada(self, teclas):
        self.vel.x = 0
        velocidade_alvo = VELOCIDADE_ANDAR

        if teclas[pygame.K_LSHIFT] or teclas[pygame.K_RSHIFT]:
            velocidade_alvo = VELOCIDADE_CORRER

        if teclas[pygame.K_LEFT] or teclas[pygame.K_a]:
            self.vel.x = -velocidade_alvo
            self.direcao = -1
        elif teclas[pygame.K_RIGHT] or teclas[pygame.K_d]:
            self.vel.x = velocidade_alvo
            self.direcao = 1

        # pulo
        if (teclas[pygame.K_SPACE] or teclas[pygame.K_z] or teclas[pygame.K_UP]) and self.no_chao:
            self.vel.y = IMPULSO_PULO
            self.no_chao = False
            self.estado = "pulando"
            self.tempo_estado = 0.0

        # arremesso de faca
        if (teclas[pygame.K_x] or teclas[pygame.K_LCTRL]) and self.tempo_recarga_faca <= 0:
            self.lancar_faca()
            self.estado = "atacando"
            self.tempo_estado = 0.0
            self.tempo_recarga_faca = 0.35

    def lancar_faca(self):
        origem_x = self.rect.centerx + self.direcao * 14
        origem_y = self.rect.centery - 4
        vel_inicial = pygame.Vector2(10 * self.direcao, -3.0)
        faca = Faca(origem_x, origem_y, vel_inicial)
        self.facas.add(faca)

    # ------------------------- física -------------------------

    def aplicar_fisica(self, dt, plataformas):
        # gravidade
        self.vel.y += GRAVIDADE
        if self.vel.y > LIMITE_QUEDA:
            self.vel.y = LIMITE_QUEDA

        # movimento horizontal
        self.pos.x += self.vel.x
        self.rect.x = int(self.pos.x)
        self._resolver_colisoes(plataformas, eixo="x")

        # movimento vertical
        self.pos.y += self.vel.y
        self.rect.y = int(self.pos.y)
        self.no_chao = False
        self._resolver_colisoes(plataformas, eixo="y")

    def _resolver_colisoes(self, plataformas, eixo):
        for p in plataformas:
            if self.rect.colliderect(p.rect):
                if eixo == "x":
                    if self.vel.x > 0:
                        self.rect.right = p.rect.left
                    elif self.vel.x < 0:
                        self.rect.left = p.rect.right
                    self.pos.x = self.rect.x
                    self.vel.x = 0
                else:
                    if self.vel.y > 0:
                        self.rect.bottom = p.rect.top
                        # pequeno amortecimento de queda alta
                        if self.vel.y > 10:
                            self.vel.y = -self.vel.y * 0.25
                        else:
                            self.vel.y = 0
                        self.no_chao = True
                        self.estado = "parado" if abs(self.vel.x) < 0.1 else "correndo"
                    elif self.vel.y < 0:
                        self.rect.top = p.rect.bottom
                        self.vel.y = 0
                    self.pos.y = self.rect.y

    # ------------------------ estado / anim --------------------

    def atualizar_estado(self, dt):
        self.tempo_estado += dt
        self.tempo_quadro += dt
        self.tempo_recarga_faca = max(0.0, self.tempo_recarga_faca - dt)
        if self.invulneravel > 0:
            self.invulneravel -= dt

        if not self.no_chao:
            if self.vel.y < -1:
                self.estado = "pulando"
            elif self.vel.y > 1:
                self.estado = "caindo"
        else:
            if abs(self.vel.x) > 0.2:
                self.estado = "correndo"
            else:
                if self.estado != "atacando":
                    self.estado = "parado"

        if self.estado == "parado":
            frames = self.frames_parado
            dur = 0.25
        elif self.estado == "correndo":
            frames = self.frames_correndo
            dur = 0.09
        elif self.estado == "pulando":
            frames = self.frames_pulando
            dur = 0.1
        elif self.estado == "caindo":
            frames = self.frames_caindo
            dur = 0.12
        elif self.estado == "atacando":
            frames = self.frames_atacando
            dur = 0.07
            if self.tempo_estado > 0.28:
                self.estado = "parado"
        else:
            frames = self.frames_parado
            dur = 0.25

        if self.tempo_quadro > dur:
            self.tempo_quadro = 0.0
            self.quadro = (self.quadro + 1) % len(frames)

        self.image = frames[self.quadro]

        # atualizar facas
        self.facas.update(dt)

    def desenhar(self, superficie, deslocamento_x):
        img = self.image
        if self.direcao == -1:
            img = pygame.transform.flip(img, True, False)

        if self.invulneravel > 0 and int(self.invulneravel * 20) % 2 == 0:
            # pisca quando toma dano
            pass
        else:
            superficie.blit(img, (self.rect.x - deslocamento_x, self.rect.y))

        # facas
        for faca in self.facas:
            faca.desenhar(superficie, deslocamento_x)

    def levar_dano(self, quantidade):
        if self.invulneravel > 0:
            return
        self.vida -= quantidade
        if self.vida < 0:
            self.vida = 0
        self.invulneravel = 1.0
        self.vel.y = -8
        self.vel.x = -4 * self.direcao
        self.estado = "caindo"


class Faca(pygame.sprite.Sprite):
    def __init__(self, x, y, velocidade):
        super().__init__()
        self.image = pygame.Surface((14, 4), pygame.SRCALPHA)
        pygame.draw.rect(self.image, (220, 220, 220), (0, 0, 10, 4))
        pygame.draw.rect(self.image, (160, 160, 160), (8, 1, 4, 2))
        self.rect = self.image.get_rect(center=(x, y))
        self.vel = pygame.Vector2(velocidade)

    def update(self, dt):
        self.vel.y += GRAVIDADE * 0.35
        self.rect.x += int(self.vel.x)
        self.rect.y += int(self.vel.y)
        if (self.rect.right < -120 or self.rect.left > 4500
                or self.rect.top > ALTURA_TELA + 250):
            self.kill()

    def desenhar(self, superficie, deslocamento_x):
        superficie.blit(self.image, (self.rect.x - deslocamento_x, self.rect.y))


class Plataforma(pygame.sprite.Sprite):
    def __init__(self, x, y, largura, altura, nivel):
        super().__init__()
        self.rect = pygame.Rect(x, y, largura, altura)
        self.image = pygame.Surface((largura, altura))
        if nivel == 0:
            self.image.fill(COR_PLATAFORMA)
        else:
            self.image.fill(COR_PLATAFORMA_ESCURA)


class Inimigo(pygame.sprite.Sprite):
    def __init__(self, x, y, largura=30, altura=40):
        super().__init__()
        self.image = pygame.Surface((largura, altura), pygame.SRCALPHA)
        pygame.draw.rect(self.image, COR_INIMIGO, (0, 8, largura, altura - 10))
        pygame.draw.rect(self.image, (40, 0, 0), (0, altura - 8, largura, 8))
        pygame.draw.rect(self.image, (0, 0, 0), (6, 0, largura - 12, 10))
        self.rect = self.image.get_rect(bottomleft=(x, y))

        self.pos = pygame.Vector2(self.rect.x, self.rect.y)
        self.vel = pygame.Vector2(0, 0)
        self.direcao = random.choice([-1, 1])
        self.vel_patrol = 1.6 + random.random()
        self.raio_patrol = 120 + random.randint(-30, 40)
        self.x_base = x
        self.vida = 40

    def update(self, dt, plataformas):
        deslocamento = self.pos.x - self.x_base
        if abs(deslocamento) > self.raio_patrol:
            self.direcao *= -1

        self.vel.x = self.vel_patrol * self.direcao
        self.vel.y += GRAVIDADE
        if self.vel.y > LIMITE_QUEDA:
            self.vel.y = LIMITE_QUEDA

        self.pos.x += self.vel.x
        self.rect.x = int(self.pos.x)
        self._resolver_colisoes(plataformas, "x")

        self.pos.y += self.vel.y
        self.rect.y = int(self.pos.y)
        self._resolver_colisoes(plataformas, "y")

    def _resolver_colisoes(self, plataformas, eixo):
        for p in plataformas:
            if self.rect.colliderect(p.rect):
                if eixo == "x":
                    if self.vel.x > 0:
                        self.rect.right = p.rect.left
                    elif self.vel.x < 0:
                        self.rect.left = p.rect.right
                    self.pos.x = self.rect.x
                    self.vel.x = 0
                    self.direcao *= -1
                else:
                    if self.vel.y > 0:
                        self.rect.bottom = p.rect.top
                        self.pos.y = self.rect.y
                        self.vel.y = 0
                    elif self.vel.y < 0:
                        self.rect.top = p.rect.bottom
                        self.pos.y = self.rect.y
                        self.vel.y = 0

    def levar_dano(self, quantidade):
        self.vida -= quantidade
        if self.vida <= 0:
            self.kill()


class Coletavel(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.base_y = y
        self.fase = random.random() * math.pi * 2
        self.image = pygame.Surface((18, 18), pygame.SRCALPHA)
        pygame.draw.polygon(
            self.image,
            COR_COLETAVEL,
            [(9, 0), (18, 9), (9, 18), (0, 9)],
        )
        pygame.draw.circle(self.image, (255, 255, 255), (9, 9), 3)
        self.rect = self.image.get_rect(center=(x, y))

    def update(self, dt):
        self.fase += dt * 3
        self.rect.centery = int(self.base_y + math.sin(self.fase) * 4)


# ------------------------------------------------------------
# Fase / Mundo
# ------------------------------------------------------------

class FaseCidadeDeserto:
    def __init__(self):
        self.plataformas = pygame.sprite.Group()
        self.inimigos = pygame.sprite.Group()
        self.coletaveis = pygame.sprite.Group()

        self.largura_total = 4000

        # solo principal
        self.plataformas.add(
            Plataforma(0, ALTURA_TELA - 40, self.largura_total, 40, 0)
        )

        # telhados e varandas
        blocos = [
            (200, ALTURA_TELA - 120, 230, 20),
            (520, ALTURA_TELA - 180, 190, 20),
            (840, ALTURA_TELA - 150, 210, 20),
            (1180, ALTURA_TELA - 210, 220, 20),
            (1520, ALTURA_TELA - 160, 260, 20),
            (1920, ALTURA_TELA - 120, 200, 20),
            (2260, ALTURA_TELA - 180, 230, 20),
            (2620, ALTURA_TELA - 140, 260, 20),
            (3020, ALTURA_TELA - 190, 240, 20),
            (3420, ALTURA_TELA - 150, 260, 20),
        ]
        for (x, y, w, h) in blocos:
            self.plataformas.add(Plataforma(x, y, w, h, 1))

        # caixas baixas
        obstaculos = [
            (430, ALTURA_TELA - 60, 40, 20),
            (900, ALTURA_TELA - 60, 50, 20),
            (1350, ALTURA_TELA - 60, 45, 20),
            (2100, ALTURA_TELA - 60, 55, 20),
        ]
        for (x, y, w, h) in obstaculos:
            self.plataformas.add(Plataforma(x, y, w, h, 1))

        # inimigos
        inimigos_info = [
            (350, ALTURA_TELA - 40),
            (650, ALTURA_TELA - 180),
            (980, ALTURA_TELA - 40),
            (1400, ALTURA_TELA - 210),
            (1760, ALTURA_TELA - 40),
            (2360, ALTURA_TELA - 180),
            (2820, ALTURA_TELA - 40),
            (3220, ALTURA_TELA - 190),
            (3660, ALTURA_TELA - 40),
        ]
        for (x, y) in inimigos_info:
            self.inimigos.add(Inimigo(x, y))

        # joias
        for (x, y, w, h) in blocos:
            if random.random() < 0.7:
                cx = x + w // 2
                cy = y - 26
                self.coletaveis.add(Coletavel(cx, cy))

    def atualizar(self, dt):
        self.coletaveis.update(dt)
        self.inimigos.update(dt, self.plataformas)

    def desenhar_fundo(self, superficie, deslocamento_x):
        superficie.fill(COR_CEU)

        fator_longe = 0.22
        desloc_longe = int(deslocamento_x * fator_longe)
        for i in range(-2, 12):
            base_x = i * 420 - desloc_longe
            pygame.draw.polygon(
                superficie,
                COR_FUNDO_LONGE,
                [
                    (base_x, ALTURA_TELA - 130),
                    (base_x + 210, ALTURA_TELA - 230),
                    (base_x + 420, ALTURA_TELA - 130),
                ],
            )

        fator_perto = 0.5
        desloc_perto = int(deslocamento_x * fator_perto)
        for i in range(-2, 14):
            base_x = i * 320 - desloc_perto
            pygame.draw.rect(
                superficie,
                COR_FUNDO_PERTO,
                (base_x, ALTURA_TELA - 210, 260, 210),
            )
            pygame.draw.ellipse(
                superficie,
                (230, 200, 120),
                (base_x + 60, ALTURA_TELA - 250, 140, 80),
            )

    def desenhar_frente(self, superficie, deslocamento_x):
        for p in self.plataformas:
            superficie.blit(p.image, (p.rect.x - deslocamento_x, p.rect.y))

        for col in self.coletaveis:
            superficie.blit(col.image, (col.rect.x - deslocamento_x, col.rect.y))

        for inim in self.inimigos:
            superficie.blit(inim.image, (inim.rect.x - deslocamento_x, inim.rect.y))


# ------------------------------------------------------------
# HUD e combate
# ------------------------------------------------------------

def desenhar_hud(superficie, jogador, pontos, tempo_seg):
    largura_barra = 220
    altura_barra = 20
    x = 20
    y = 20

    pygame.draw.rect(
        superficie,
        COR_VIDA_FUNDO,
        (x - 2, y - 2, largura_barra + 4, altura_barra + 4),
        border_radius=6,
    )
    proporcao = max(0.0, jogador.vida / jogador.vida_maxima)
    largura_preenchida = int(largura_barra * proporcao)
    pygame.draw.rect(
        superficie,
        COR_VIDA,
        (x, y, largura_preenchida, altura_barra),
        border_radius=4,
    )

    fonte = pygame.font.SysFont("arial", 18, bold=True)
    texto_vida = fonte.render(f"Vida: {jogador.vida:3d}", True, COR_TEXTO)
    superficie.blit(texto_vida, (x, y - 22))

    texto_pontos = fonte.render(f"Joias: {pontos:03d}", True, COR_TEXTO)
    superficie.blit(texto_pontos, (x, y + altura_barra + 10))

    minutos = int(tempo_seg // 60)
    segundos = int(tempo_seg % 60)
    texto_tempo = fonte.render(
        f"Tempo: {minutos:02d}:{segundos:02d}", True, COR_TEXTO
    )
    superficie.blit(texto_tempo, (LARGURA_TELA - 170, 20))


def combater_e_coletar(jogador, fase, pontos):
    # facas x inimigos
    for faca in list(jogador.facas):
        inimigo = pygame.sprite.spritecollideany(faca, fase.inimigos)
        if inimigo:
            inimigo.levar_dano(20)
            faca.kill()
            pontos += 5

    # jogador x inimigos (contato)
    inimigo_colisao = pygame.sprite.spritecollideany(
        jogador, fase.inimigos, collided=lambda pl, inim: pl.rect.colliderect(inim.rect)
    )
    if inimigo_colisao:
        jogador.levar_dano(12)

    # jogador x joias
    coletados = [
        col
        for col in fase.coletaveis
        if jogador.rect.colliderect(col.rect)
    ]
    for col in coletados:
        col.kill()
    if coletados:
        pontos += 10 * len(coletados)

    return pontos


# ------------------------------------------------------------
# Loop principal do jogo
# ------------------------------------------------------------

def executar_jogo():
    pygame.init()
    pygame.display.set_caption("Herói do Deserto — Fase 1 (Pygame)")
    tela = pygame.display.set_mode((LARGURA_TELA, ALTURA_TELA))
    relogio = pygame.time.Clock()

    fonte_grande = pygame.font.SysFont("arial", 40, bold=True)
    fonte_media = pygame.font.SysFont("arial", 26, bold=True)

    fase = FaseCidadeDeserto()
    jogador = Jogador(60, ALTURA_TELA - 120)

    pontos = 0
    tempo_total = 0.0
    deslocamento_x = 0.0
    rodando = True
    jogo_terminado = False
    vitoria = False

    while rodando:
        dt_ms = relogio.tick(FPS)
        dt = dt_ms / 1000.0
        if not jogo_terminado:
            tempo_total += dt

        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                rodando = False
            elif evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_ESCAPE:
                    rodando = False
                if jogo_terminado and evento.key in (pygame.K_RETURN, pygame.K_SPACE):
                    # reiniciar fase
                    fase = FaseCidadeDeserto()
                    jogador = Jogador(60, ALTURA_TELA - 120)
                    pontos = 0
                    tempo_total = 0.0
                    deslocamento_x = 0.0
                    jogo_terminado = False
                    vitoria = False

        teclas = pygame.key.get_pressed()
        if not jogo_terminado:
            jogador.processar_entrada(teclas)

            fase.atualizar(dt)
            jogador.aplicar_fisica(dt, fase.plataformas)
            jogador.atualizar_estado(dt)

            pontos = combater_e_coletar(jogador, fase, pontos)

            if jogador.vida <= 0 or jogador.rect.top > ALTURA_TELA + 200:
                jogo_terminado = True
                vitoria = False

            if jogador.rect.x > fase.largura_total - 160:
                jogo_terminado = True
                vitoria = True

            # câmera
            alvo = jogador.rect.centerx - LARGURA_TELA * 0.35
            deslocamento_x += (alvo - deslocamento_x) * min(10 * dt, 1.0)
            deslocamento_x = max(0, min(deslocamento_x, fase.largura_total - LARGURA_TELA))

        # desenho
        fase.desenhar_fundo(tela, deslocamento_x)
        fase.desenhar_frente(tela, deslocamento_x)
        jogador.desenhar(tela, deslocamento_x)
        desenhar_hud(tela, jogador, pontos, tempo_total)

        if jogo_terminado:
            caixa = pygame.Surface((LARGURA_TELA, 130), pygame.SRCALPHA)
            caixa.fill((0, 0, 0, 180))
            tela.blit(caixa, (0, ALTURA_TELA // 2 - 80))

            if vitoria:
                texto1 = fonte_grande.render("Vitória!", True, (255, 240, 130))
                texto2 = fonte_media.render(
                    "Você atravessou a cidade do deserto.",
                    True,
                    (235, 235, 235),
                )
            else:
                texto1 = fonte_grande.render("Game Over", True, (255, 190, 190))
                texto2 = fonte_media.render(
                    "Pressione ENTER para recomeçar.",
                    True,
                    (235, 235, 235),
                )

            texto3 = fonte_media.render(
                f"Joias: {pontos:03d}   Tempo: {tempo_total:05.1f}s",
                True,
                (235, 235, 235),
            )

            tela.blit(
                texto1,
                (
                    LARGURA_TELA // 2 - texto1.get_width() // 2,
                    ALTURA_TELA // 2 - 60,
                ),
            )
            tela.blit(
                texto2,
                (
                    LARGURA_TELA // 2 - texto2.get_width() // 2,
                    ALTURA_TELA // 2 - 20,
                ),
            )
            tela.blit(
                texto3,
                (
                    LARGURA_TELA // 2 - texto3.get_width() // 2,
                    ALTURA_TELA // 2 + 20,
                ),
            )

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    executar_jogo()
