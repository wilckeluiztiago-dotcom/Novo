# ============================================================
# MODELO GEOFÍSICO 1D DO POÇO DE PETRÓLEO + VISUALIZAÇÃO PYGAME
# Autor: Luiz Tiago Wilcke 
# ============================================================

import math
import sys
from dataclasses import dataclass

import numpy as np
import pygame


# ------------------------------------------------------------
# 1. Parâmetros físicos e numéricos do poço
# ------------------------------------------------------------

@dataclass
class ParametrosPoco:
    profundidade_total: float = 3000.0       # m
    numero_nos: int = 80                     # discretização vertical
    pressao_reservatorio: float = 320e5      # Pa (fundo do poço)
    pressao_boca_base: float = 80e5          # Pa (superfície, sem produção)
    temperatura_superficie: float = 300.0    # K
    temperatura_reservatorio: float = 380.0  # K
    difusividade_pressao: float = 0.015      # m^2/s (difusão de pressão)
    difusividade_temperatura: float = 0.008  # m^2/s (condução térmica)
    velocidade_media_subida: float = 0.15    # m/s (escoamento médio do fluido)
    ganho_vazao_pressao: float = 2.0e6       # Pa por unidade de vazão (estrangulador)
    capacidade_termica_rocha: float = 2.0e6  # J/(m^3 K) — rocha + fluido
    fonte_calor_atrito: float = 2.0e3        # W/m^3 (atrito do escoamento)
    passo_tempo_base: float = 1.0            # s (será corrigido por estabilidade)


# ------------------------------------------------------------
# 2. Núcleo numérico: EDPs 1D para pressão e temperatura
#    ∂p/∂t = Dp ∂²p/∂z² − v ∂p/∂z + q_p(z,t)
#    ∂T/∂t = Dt ∂²T/∂z² − v ∂T/∂z + q_T(z,t)/(ρ c)
# ------------------------------------------------------------

class ModeloGeofisicoPoco:
    def __init__(self, parametros: ParametrosPoco):
        self.parametros = parametros
        self.profundidade_total = parametros.profundidade_total
        self.numero_nos = parametros.numero_nos
        self.delta_z = self.profundidade_total / (self.numero_nos - 1)

        # Malha 1D: z = 0 (superfície) até z = profundidade_total (fundo)
        self.profundidades = np.linspace(0.0, self.profundidade_total, self.numero_nos)

        # Campos dinâmicos
        self.pressao = np.zeros(self.numero_nos, dtype=float)
        self.temperatura = np.zeros(self.numero_nos, dtype=float)

        # Parâmetros de controle operacional
        self.vazao_superficie = 0.5   # 0 (fechado) a 1 (máxima produção)
        self.tempo_simulacao = 0.0

        self._inicializar_campos()
        self.delta_tempo_estavel = self._calcular_delta_tempo_estavel()

    def _inicializar_campos(self):
        """Perfil inicial: gradiente quase hidrostático + gradiente térmico."""
        p_topo = self.parametros.pressao_boca_base
        p_fundo = self.parametros.pressao_reservatorio
        t_topo = self.parametros.temperatura_superficie
        t_fundo = self.parametros.temperatura_reservatorio

        # Interpolação linear inicial
        self.pressao = np.linspace(p_topo, p_fundo, self.numero_nos)
        self.temperatura = np.linspace(t_topo, t_fundo, self.numero_nos)

    def _calcular_delta_tempo_estavel(self) -> float:
        """Critério de estabilidade simples (CFL) para difusão + advecção."""
        dz = self.delta_z
        D_max = max(self.parametros.difusividade_pressao,
                    self.parametros.difusividade_temperatura)
        v = abs(self.parametros.velocidade_media_subida)

        # Difusão: dt <= 0.45 * dz^2 / D
        termo_difusao = 0.45 * dz * dz / max(D_max, 1e-8)

        # Advecção: dt <= 0.5 * dz / v
        if v > 1e-8:
            termo_adveccao = 0.5 * dz / v
        else:
            termo_adveccao = 1e9

        dt_estavel = min(termo_difusao, termo_adveccao, self.parametros.passo_tempo_base)
        return float(dt_estavel)

    def _aplicar_condicoes_contorno(self):
        """Condiciona pressão e temperatura em topo e fundo do poço."""
        # Topo: pressão depende da vazão (mais produção => menor pressão na boca)
        pressao_boca = (
            self.parametros.pressao_boca_base
            - self.parametros.ganho_vazao_pressao * self.vazao_superficie
        )
        # Garante limite físico mínimo (tipo pressão próxima à atmosférica)
        pressao_boca = max(2.0e5, pressao_boca)

        self.pressao[0] = pressao_boca
        self.temperatura[0] = self.parametros.temperatura_superficie

        # Fundo: condições quase fixas (reservatório grande)
        self.pressao[-1] = self.parametros.pressao_reservatorio
        self.temperatura[-1] = self.parametros.temperatura_reservatorio

    def passo_tempo(self, fator_escala_tempo: float = 1.0):
        """Avança solução em um passo de tempo, mantendo estabilidade."""
        delta_tempo = self.delta_tempo_estavel * fator_escala_tempo
        dz = self.delta_z

        # Copia para evitar interferência durante o cálculo
        pressao_atual = self.pressao.copy()
        temperatura_atual = self.temperatura.copy()

        Dp = self.parametros.difusividade_pressao
        Dt = self.parametros.difusividade_temperatura
        v = self.parametros.velocidade_media_subida
        fonte_calor = self.parametros.fonte_calor_atrito
        capacidade_termica = self.parametros.capacidade_termica_rocha

        # Derivadas espaciais interiores (esquema de diferenças finitas)
        # Índices 1..N-2 (os extremos são dados por contorno)
        for i in range(1, self.numero_nos - 1):
            # Difusão (segunda derivada)
            d2p_dz2 = (pressao_atual[i + 1] - 2.0 * pressao_atual[i] + pressao_atual[i - 1]) / (dz * dz)
            d2T_dz2 = (temperatura_atual[i + 1] - 2.0 * temperatura_atual[i] + temperatura_atual[i - 1]) / (dz * dz)

            # Advecção (primeira derivada, central)
            dp_dz = (pressao_atual[i + 1] - pressao_atual[i - 1]) / (2.0 * dz)
            dT_dz = (temperatura_atual[i + 1] - temperatura_atual[i - 1]) / (2.0 * dz)

            # Termo-fonte simplificado para pressão: acoplamento fraco com gradiente térmico
            # q_p ~ pequeno ganho quando fluido está mais quente (menor viscosidade)
            termo_fonte_pressao = 0.02 * (temperatura_atual[i] - self.parametros.temperatura_superficie)

            # Termo-fonte térmico: atrito do escoamento ao longo do poço
            termo_fonte_temperatura = fonte_calor / capacidade_termica

            dp_dt = Dp * d2p_dz2 - v * dp_dz + termo_fonte_pressao
            dT_dt = Dt * d2T_dz2 - v * dT_dz + termo_fonte_temperatura

            self.pressao[i] = pressao_atual[i] + delta_tempo * dp_dt
            self.temperatura[i] = temperatura_atual[i] + delta_tempo * dT_dt

        # Condições de contorno em topo e fundo
        self._aplicar_condicoes_contorno()

        # Avança o relógio interno
        self.tempo_simulacao += delta_tempo

    def obter_metricas(self):
        """Alguns indicadores numéricos para análise rápida."""
        pressao_topo = float(self.pressao[0])
        pressao_fundo = float(self.pressao[-1])
        gradiente_medio_pressao = (pressao_fundo - pressao_topo) / self.profundidade_total

        temperatura_topo = float(self.temperatura[0])
        temperatura_fundo = float(self.temperatura[-1])
        gradiente_medio_temperatura = (temperatura_fundo - temperatura_topo) / self.profundidade_total

        return {
            "tempo_s": self.tempo_simulacao,
            "pressao_topo_Pa": pressao_topo,
            "pressao_fundo_Pa": pressao_fundo,
            "gradiente_pressao_Pa_m": gradiente_medio_pressao,
            "temperatura_topo_K": temperatura_topo,
            "temperatura_fundo_K": temperatura_fundo,
            "gradiente_temperatura_K_m": gradiente_medio_temperatura,
            "vazao_superficie": self.vazao_superficie,
        }


# ------------------------------------------------------------
# 3. Visualização interativa com Pygame
# ------------------------------------------------------------

class SimuladorPygamePoco:
    def __init__(self, modelo: ModeloGeofisicoPoco):
        pygame.init()
        self.modelo = modelo

        self.largura_janela = 1000
        self.altura_janela = 700
        self.janela = pygame.display.set_mode((self.largura_janela, self.altura_janela))
        pygame.display.setcaption = pygame.display.set_caption(
            "Simulação Geofísica do Poço de Petróleo — LT"
        )

        self.relogio = pygame.time.Clock()
        self.fonte_pequena = pygame.font.SysFont("consolas", 16)
        self.fonte_media = pygame.font.SysFont("consolas", 22, bold=True)

        # Escalas de visualização
        self.margem_superior = 50
        self.margem_inferior = 60
        self.coluna_altura = self.altura_janela - self.margem_superior - self.margem_inferior
        self.coluna_largura = 60

        # Fator de aceleração do tempo (simulação numérica)
        self.fator_escala_tempo = 5.0

    def _cor_por_valor(self, valor, minimo, maximo):
        """Mapa de cores simples: azul (baixo) -> verde -> vermelho (alto)."""
        if maximo <= minimo:
            return (255, 255, 255)
        x = (valor - minimo) / (maximo - minimo)
        x = max(0.0, min(1.0, x))
        # Interpolação manual: azul -> verde -> vermelho
        if x < 0.5:
            # Azul (0,0,255) para Verde (0,255,0)
            t = x / 0.5
            r = 0
            g = int(255 * t)
            b = int(255 * (1 - t))
        else:
            # Verde (0,255,0) para Vermelho (255,0,0)
            t = (x - 0.5) / 0.5
            r = int(255 * t)
            g = int(255 * (1 - t))
            b = 0
        return (r, g, b)

    def _desenhar_coluna(self, valores, xmin, xmax, x_centro):
        """Desenha uma coluna vertical discretizada ao longo da profundidade."""
        numero_nos = len(valores)
        altura_celula = self.coluna_altura / numero_nos

        valor_min = float(np.min(valores))
        valor_max = float(np.max(valores))

        for i, valor in enumerate(valores):
            y_topo = self.margem_superior + i * altura_celula
            retangulo = pygame.Rect(
                x_centro - self.coluna_largura // 2,
                int(y_topo),
                self.coluna_largura,
                int(altura_celula) + 1,
            )
            cor = self._cor_por_valor(
                valor,
                xmin if xmin is not None else valor_min,
                xmax if xmax is not None else valor_max,
            )
            pygame.draw.rect(self.janela, cor, retangulo)

        # Moldura
        cor_borda = (220, 220, 220)
        moldura = pygame.Rect(
            x_centro - self.coluna_largura // 2,
            self.margem_superior,
            self.coluna_largura,
            self.coluna_altura,
        )
        pygame.draw.rect(self.janela, cor_borda, moldura, 1)

    def _desenhar_interface(self):
        self.janela.fill((7, 10, 25))

        # Atualiza o modelo algumas vezes por quadro para acelerar a dinâmica
        self.modelo.passo_tempo(self.fator_escala_tempo)

        # Coluna de pressão (esquerda)
        x_pressao = self.largura_janela // 3
        self._desenhar_coluna(
            self.modelo.pressao,
            xmin=40e5,
            xmax=340e5,
            x_centro=x_pressao,
        )

        # Coluna de temperatura (direita)
        x_temperatura = 2 * self.largura_janela // 3
        self._desenhar_coluna(
            self.modelo.temperatura,
            xmin=290.0,
            xmax=390.0,
            x_centro=x_temperatura,
        )

        # Títulos
        texto_pressao = self.fonte_media.render(
            "Pressão ao longo do poço (Pa)", True, (240, 240, 240)
        )
        texto_temperatura = self.fonte_media.render(
            "Temperatura ao longo do poço (K)", True, (240, 240, 240)
        )
        self.janela.blit(
            texto_pressao,
            (x_pressao - texto_pressao.get_width() // 2, 10),
        )
        self.janela.blit(
            texto_temperatura,
            (x_temperatura - texto_temperatura.get_width() // 2, 10),
        )

        # Escala de profundidade
        cor_texto = (200, 200, 200)
        profundidade_total = self.modelo.profundidade_total
        for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
            y = self.margem_superior + frac * self.coluna_altura
            profundidade = int(frac * profundidade_total)
            pygame.draw.line(
                self.janela,
                (80, 80, 80),
                (x_pressao - 80, int(y)),
                (x_temperatura + 80, int(y)),
                1,
            )
            texto = self.fonte_pequena.render(f"{profundidade:4d} m", True, cor_texto)
            self.janela.blit(texto, (20, int(y) - 8))

        # Métricas numéricas
        metricas = self.modelo.obter_metricas()
        linha = 0
        deslocamento_x = 40
        deslocamento_y = self.altura_janela - self.margem_inferior + 5

        def escrever(linha_texto: str):
            nonlocal linha
            texto = self.fonte_pequena.render(linha_texto, True, cor_texto)
            self.janela.blit(texto, (deslocamento_x, deslocamento_y + 18 * linha))
            linha += 1

        escrever(f"Tempo simulado: {metricas['tempo_s']/3600.0:6.2f} h")
        escrever(f"Pressão topo:   {metricas['pressao_topo_Pa']/1e5:6.2f} bar")
        escrever(f"Pressão fundo:  {metricas['pressao_fundo_Pa']/1e5:6.2f} bar")
        escrever(
            f"Gradiente P:    {metricas['gradiente_pressao_Pa_m']/1e4:6.2f} x10^4 Pa/m"
        )
        escrever(f"Temp. topo:     {metricas['temperatura_topo_K']:6.1f} K")
        escrever(f"Temp. fundo:    {metricas['temperatura_fundo_K']:6.1f} K")
        escrever(
            f"Gradiente T:    {metricas['gradiente_temperatura_K_m']*1000:6.3f} K/km"
        )
        escrever(
            f"Vazão (controle): {metricas['vazao_superficie']:.2f} "
            "(0 = fechado, 1 = máximo)"
        )

        # Instruções
        instrucoes = [
            "Controles: ↑ aumenta produção (menor pressão na boca do poço)",
            "           ↓ reduz produção",
            "           → acelera simulação   ← desacelera simulação",
            "           ESC ou Q para sair",
        ]
        for idx, txt in enumerate(instrucoes):
            texto = self.fonte_pequena.render(txt, True, (180, 180, 220))
            self.janela.blit(
                texto,
                (
                    self.largura_janela - texto.get_width() - 30,
                    self.altura_janela - self.margem_inferior + 5 + 18 * idx,
                ),
            )

    def _processar_eventos(self):
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                return False
            if evento.type == pygame.KEYDOWN:
                if evento.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False
                if evento.key == pygame.K_UP:
                    self.modelo.vazao_superficie = min(
                        1.0, self.modelo.vazao_superficie + 0.05
                    )
                if evento.key == pygame.K_DOWN:
                    self.modelo.vazao_superficie = max(
                        0.0, self.modelo.vazao_superficie - 0.05
                    )
                if evento.key == pygame.K_RIGHT:
                    self.fator_escala_tempo = min(
                        20.0, self.fator_escala_tempo * 1.2
                    )
                if evento.key == pygame.K_LEFT:
                    self.fator_escala_tempo = max(
                        0.25, self.fator_escala_tempo / 1.2
                    )
        return True

    def rodar(self):
        """Loop principal da simulação gráfica."""
        while True:
            if not self._processar_eventos():
                break
            self._desenhar_interface()
            pygame.display.flip()
            self.relogio.tick(30)  # FPS alvo


# ------------------------------------------------------------
# 4. Execução direta
# ------------------------------------------------------------

def main():
    parametros = ParametrosPoco()
    modelo = ModeloGeofisicoPoco(parametros)
    simulador = SimuladorPygamePoco(modelo)
    simulador.rodar()
    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
