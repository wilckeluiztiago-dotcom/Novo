# ============================================================
# Modelo de Congestionamento em Rede com Tráfego de Lévy
# Autor: (pode colocar: Luiz Tiago Wilcke - LT)
# ============================================================
# Ideia geral:
#   - Um link de rede com capacidade C (pacotes/unidade de tempo).
#   - Chegadas de tráfego em rajadas pesadas, modeladas por uma
#     distribuição estável de Lévy (α = 1/2, totalmente assimétrica).
#   - Evolução da fila:
#         F_{t+1} = min( B,
#                        max( 0, F_t + A_t - C ) )
#     onde:
#       F_t = tamanho da fila no instante t,
#       A_t = tráfego que chega no intervalo [t, t+1),
#       C   = capacidade de serviço do link,
#       B   = tamanho máximo do buffer.
#   - Se F_t + A_t - C > B, o excedente é considerado "perda" por
#     congestionamento (pacotes descartados).
#
#   - O processo A_t é gerado a partir de uma distribuição de Lévy
#     (estável com α = 1/2, β = 1), produzindo caudas pesadas e
#     congestionamentos intermitentes mas extremos.
#
#   - O código inclui:
#       * Gerador de amostras de Lévy (via algoritmo de Chambers–Mallows–Stuck).
#       * Simulação de um link com tráfego puramente de Lévy.
#       * Simulação de um link com mistura de tráfego Lévy + Gaussiano.
#       * Simulação de uma cadeia multi-hop de links.
#       * Cálculo de métricas de congestionamento.
# ============================================================

from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 1. Gerador de distribuição de Lévy (estável α = 1/2, β = 1)
# ------------------------------------------------------------

@dataclass
class ParametrosLevy:
    """Parâmetros do ruído de Lévy (forma estável α=1/2, β=1).

    escala: controla a intensidade média das rajadas.
    deslocamento: desloca a distribuição (não usado no modelo da fila,
                  mas deixado para generalidade).
    """
    escala: float = 1.0
    deslocamento: float = 0.0


def gerar_levy(n_amostras: int,
               parametros: ParametrosLevy,
               seed: Optional[int] = None) -> np.ndarray:
    """Gera amostras de uma distribuição de Lévy (estável) positiva.

    Implementação baseada no algoritmo de Chambers–Mallows–Stuck
    para distribuições α-estáveis. Aqui fixamos:
        α = 1/2  (índice de estabilidade)
        β = 1    (assimetria totalmente à direita)

    O resultado é uma variável aleatória de cauda extremamente pesada,
    suportada em (0, +∞), típica dos "Lévy flights".

    Retorna
    -------
    np.ndarray
        Vetor de tamanho n_amostras com valores > 0.
    """
    rng = np.random.default_rng(seed)

    alpha = 0.5
    beta = 1.0

    # Passos do algoritmo CMS:
    U = rng.uniform(-np.pi / 2.0, np.pi / 2.0, size=n_amostras)
    W = rng.exponential(1.0, size=n_amostras)

    # Constantes B_{α,β} e S_{α,β} (parâmetros de Nolan)
    B = (1.0 / alpha) * np.arctan(beta * np.tan(np.pi * alpha / 2.0))
    S = (1.0 + (beta ** 2) * (np.tan(np.pi * alpha / 2.0) ** 2)) ** (1.0 / (2.0 * alpha))

    # Amostra estável geral:
    X = S * np.sin(alpha * (U + B)) / (np.cos(U) ** (1.0 / alpha)) * \
        (np.cos(U - alpha * (U + B)) / W) ** ((1.0 - alpha) / alpha)

    # Escala e deslocamento (mantendo apenas valores positivos)
    X = parametros.deslocamento + parametros.escala * X
    # Como α=1/2, β=1, a distribuição é naturalmente suportada em (0,∞)
    # (numericamente pode haver ruído muito pequeno; forçamos positividade mínima)
    X = np.clip(X, 1e-9, None)

    return X


# ------------------------------------------------------------
# 2. Configuração do modelo de rede
# ------------------------------------------------------------

@dataclass
class ConfigRede:
    horizonte_tempo: int = 10_000      # número de passos de simulação
    capacidade_link: float = 100.0     # capacidade de serviço por passo (C)
    tamanho_buffer: float = 5_000.0    # tamanho máximo do buffer (B)
    intensidade_levy: float = 80.0     # escala das rajadas de Lévy
    intensidade_gauss: float = 80.0    # média do tráfego gaussiano (tráfego de fundo)
    desvio_gauss: float = 20.0         # desvio padrão do ruído gaussiano
    seed: Optional[int] = 42           # semente para reprodutibilidade
    cortar_extremos: float = 10_000.0  # truncagem de rajadas absurdamente grandes


# ------------------------------------------------------------
# 3. Núcleo do modelo de fila com congestionamento
# ------------------------------------------------------------

def simular_link_levy(config: ConfigRede) -> Dict[str, np.ndarray]:
    """Simula um único link com tráfego puramente de Lévy.

    Equação da fila:
        F_{t+1} = min(B, max(0, F_t + A_t - C))

    Perda por congestionamento:
        L_t = max(0, F_t + A_t - C - B)

    Retorna um dicionário com séries temporais:
        - trafego_entrada
        - fila
        - perdas
    """
    rng = np.random.default_rng(config.seed)

    # Geração do tráfego de Lévy (rajadas de pacotes)
    params = ParametrosLevy(escala=config.intensidade_levy)
    trafego_levy = gerar_levy(config.horizonte_tempo, params, seed=config.seed)

    # Truncagem opcional de extremos para evitar números gigantescos
    if config.cortar_extremos is not None:
        trafego_levy = np.clip(trafego_levy, 0.0, config.cortar_extremos)

    fila = np.zeros(config.horizonte_tempo + 1)
    perdas = np.zeros(config.horizonte_tempo)

    for t in range(config.horizonte_tempo):
        entrada = trafego_levy[t]

        # tamanho provisório da fila após chegada e serviço
        fila_prox = fila[t] + entrada - config.capacidade_link

        # perdas ocorrem se acima do buffer
        if fila_prox > config.tamanho_buffer:
            perdas[t] = fila_prox - config.tamanho_buffer
            fila_prox = config.tamanho_buffer
        else:
            perdas[t] = 0.0

        # fila não pode ser negativa
        fila[t + 1] = max(0.0, fila_prox)

    return {
        "trafego_entrada": trafego_levy,
        "fila": fila,
        "perdas": perdas,
    }


def simular_link_misturado(config: ConfigRede) -> Dict[str, np.ndarray]:
    """Simula um link com tráfego misto:
        A_t = A_t^(Lévy) + A_t^(Gaussiano)

    Útil para modelar situações em que há tráfego "normal"
    (gaussiano) com rajadas extremas (Lévy).
    """
    rng = np.random.default_rng(config.seed + 1 if config.seed is not None else None)

    params = ParametrosLevy(escala=config.intensidade_levy)
    trafego_levy = gerar_levy(config.horizonte_tempo, params, seed=config.seed)

    # Tráfego de fundo aproximadamente gaussiano (não-negativo)
    trafego_gauss = rng.normal(loc=config.intensidade_gauss,
                               scale=config.desvio_gauss,
                               size=config.horizonte_tempo)
    trafego_gauss = np.clip(trafego_gauss, 0.0, None)

    trafego_total = trafego_levy + trafego_gauss

    if config.cortar_extremos is not None:
        trafego_total = np.clip(trafego_total, 0.0, config.cortar_extremos)

    fila = np.zeros(config.horizonte_tempo + 1)
    perdas = np.zeros(config.horizonte_tempo)

    for t in range(config.horizonte_tempo):
        entrada = trafego_total[t]

        fila_prox = fila[t] + entrada - config.capacidade_link

        if fila_prox > config.tamanho_buffer:
            perdas[t] = fila_prox - config.tamanho_buffer
            fila_prox = config.tamanho_buffer
        else:
            perdas[t] = 0.0

        fila[t + 1] = max(0.0, fila_prox)

    return {
        "trafego_levy": trafego_levy,
        "trafego_gauss": trafego_gauss,
        "trafego_total": trafego_total,
        "fila": fila,
        "perdas": perdas,
    }


def simular_multihop(config: ConfigRede,
                     num_links: int = 3) -> Dict[str, np.ndarray]:
    """Simula uma pequena cadeia de links (multi-hop) em série.

    - O tráfego entra no primeiro link como um processo de Lévy.
    - A saída de cada link (limitada pela capacidade) se torna a
      entrada do próximo link.

    Isso ilustra como uma única fonte de tráfego pesado pode propagar
    congestionamento ao longo de uma rota inteira.
    """
    rng = np.random.default_rng(config.seed + 2 if config.seed is not None else None)

    params = ParametrosLevy(escala=config.intensidade_levy)
    trafego_entrada = gerar_levy(config.horizonte_tempo, params, seed=config.seed)

    if config.cortar_extremos is not None:
        trafego_entrada = np.clip(trafego_entrada, 0.0, config.cortar_extremos)

    filas = np.zeros((num_links, config.horizonte_tempo + 1))
    perdas = np.zeros((num_links, config.horizonte_tempo))
    saidas = np.zeros((num_links, config.horizonte_tempo))

    # Entrada no primeiro link
    entrada_atual = trafego_entrada.copy()

    for idx_link in range(num_links):
        for t in range(config.horizonte_tempo):
            entrada = entrada_atual[t]

            # fila provisória após chegada e serviço
            fila_prox = filas[idx_link, t] + entrada - config.capacidade_link

            # quantidade que efetivamente sai do link (até a capacidade + o que havia na fila)
            servico = min(config.capacidade_link, filas[idx_link, t] + entrada)
            saidas[idx_link, t] = max(0.0, servico)

            if fila_prox > config.tamanho_buffer:
                perdas[idx_link, t] = fila_prox - config.tamanho_buffer
                fila_prox = config.tamanho_buffer
            else:
                perdas[idx_link, t] = 0.0

            filas[idx_link, t + 1] = max(0.0, fila_prox)

        # A saída deste link é a entrada do próximo
        if idx_link < num_links - 1:
            entrada_atual = saidas[idx_link].copy()

    return {
        "trafego_entrada": trafego_entrada,
        "filas": filas,
        "perdas": perdas,
        "saidas": saidas,
    }


# ------------------------------------------------------------
# 4. Métricas de congestionamento e utilidade do link
# ------------------------------------------------------------

def calcular_metricas_basicas(fila: np.ndarray,
                              perdas: np.ndarray,
                              capacidade_link: float) -> Dict[str, float]:
    """Calcula métricas básicas da fila e do congestionamento."""
    horizonte_tempo = len(perdas)

    ocupacao_media = float(np.mean(fila))
    prob_congestionamento = float(np.mean(fila[:-1] >= capacidade_link))
    perda_total = float(np.sum(perdas))
    perda_media_por_passo = float(np.mean(perdas))
    utilizacao_media = float(
        min(1.0, ocupacao_media / (capacidade_link + 1e-9))
    )

    return {
        "ocupacao_media": ocupacao_media,
        "prob_congestionamento": prob_congestionamento,
        "perda_total": perda_total,
        "perda_media_por_passo": perda_media_por_passo,
        "utilizacao_media": utilizacao_media,
    }


# ------------------------------------------------------------
# 5. Rotinas de visualização (opcionais)
# ------------------------------------------------------------

def plotar_resultados_link_levy(resultados: Dict[str, np.ndarray],
                                config: ConfigRede,
                                titulo_prefixo: str = "Link com tráfego de Lévy"):
    """Plota algumas séries: tráfego de entrada, fila e perdas."""
    trafego = resultados["trafego_entrada"]
    fila = resultados["fila"]
    perdas = resultados["perdas"]

    tempo = np.arange(len(trafego))

    plt.figure(figsize=(12, 7))

    plt.subplot(3, 1, 1)
    plt.plot(tempo, trafego, linewidth=0.8)
    plt.ylabel("Tráfego (pacotes)")
    plt.title(f"{titulo_prefixo} — Tráfego de entrada (Lévy)")

    plt.subplot(3, 1, 2)
    plt.plot(tempo, fila[:-1], linewidth=0.8)
    plt.axhline(config.tamanho_buffer, linestyle="--", linewidth=0.8, label="Buffer")
    plt.ylabel("Tamanho da fila")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(tempo, perdas, linewidth=0.8)
    plt.ylabel("Perdas")
    plt.xlabel("Tempo (passos)")
    plt.tight_layout()
    plt.show()


def plotar_histograma_levy(amostras: np.ndarray,
                           bins: int = 200,
                           titulo: str = "Histograma de amostras de Lévy"):
    """Mostra o histograma em escala log para destacar a cauda pesada."""
    plt.figure(figsize=(10, 5))
    plt.hist(amostras, bins=bins, density=True, alpha=0.7)
    plt.yscale("log")
    plt.xlabel("x")
    plt.ylabel("densidade (log)")
    plt.title(titulo)
    plt.tight_layout()
    plt.show()




def exemplo():
    """Executa alguns cenários de exemplo para o modelo."""
    config = ConfigRede(
        horizonte_tempo=5000,
        capacidade_link=120.0,
        tamanho_buffer=4000.0,
        intensidade_levy=80.0,
        intensidade_gauss=60.0,
        desvio_gauss=15.0,
        seed=123,
        cortar_extremos=5000.0,
    )

    # 1) Link com tráfego puramente de Lévy
    res_levy = simular_link_levy(config)
    met_levy = calcular_metricas_basicas(res_levy["fila"], res_levy["perdas"], config.capacidade_link)

    print("=== MÉTRICAS — LINK COM TRÁFEGO DE LÉVY ===")
    for k, v in met_levy.items():
        print(f"{k:30s}: {v:.4f}")
    plotar_resultados_link_levy(res_levy, config, titulo_prefixo="Lévy puro")

    # 2) Link com mistura Lévy + Gaussiano (tráfego de fundo)
    res_mix = simular_link_misturado(config)
    met_mix = calcular_metricas_basicas(res_mix["fila"], res_mix["perdas"], config.capacidade_link)

    print("\n=== MÉTRICAS — LINK COM TRÁFEGO MISTO (Lévy + Gaussiano) ===")
    for k, v in met_mix.items():
        print(f"{k:30s}: {v:.4f}")

    # Um gráfico simples da fila misturada:
    plt.figure(figsize=(12, 5))
    plt.plot(res_mix["fila"][:-1], linewidth=0.8)
    plt.axhline(config.tamanho_buffer, linestyle="--", linewidth=0.8, label="Buffer")
    plt.title("Fila ao longo do tempo — Tráfego misto (Lévy + Gaussiano)")
    plt.xlabel("Tempo")
    plt.ylabel("Tamanho da fila")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 3) Cadeia multi-hop (3 links em série)
    res_multi = simular_multihop(config, num_links=3)

    print("\n=== MULTI-HOP — RESUMO DAS PERDAS POR LINK ===")
    perdas_link = np.sum(res_multi["perdas"], axis=1)
    for i, perda in enumerate(perdas_link):
        print(f"Link {i+1}: perda total = {perda:.2f}")

    # Visualizar fila do primeiro e último link
    tempo = np.arange(config.horizonte_tempo)
    plt.figure(figsize=(12, 6))
    plt.plot(tempo, res_multi["filas"][0, :-1], label="Link 1", linewidth=0.8)
    plt.plot(tempo, res_multi["filas"][-1, :-1], label=f"Link {res_multi['filas'].shape[0]}", linewidth=0.8)
    plt.axhline(config.tamanho_buffer, linestyle="--", linewidth=0.8, label="Buffer")
    plt.xlabel("Tempo")
    plt.ylabel("Tamanho da fila")
    plt.title("Fila em uma cadeia multi-hop com tráfego de Lévy")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 4) Histograma da distribuição de Lévy usada
    amostras_levy = gerar_levy(10000, ParametrosLevy(escala=config.intensidade_levy), seed=321)
    plotar_histograma_levy(amostras_levy, titulo="Histograma — Tráfego de Lévy (rajadas)")


if __name__ == "__main__":
    exemplo()
