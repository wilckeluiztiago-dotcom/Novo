# ============================================================
# REDES DE RISCO SISTÊMICO — GRAFO + CONTÁGIO + ANIMAÇÃO PYGAME
# Autor: Luiz Tiago Wilcke (LT)
# ============================================================
# Conceito:
#   - Cada nó = instituição financeira
#   - Arestas direcionadas = exposições (quem perde se quem quebrar)
#   - Dinâmica de contágio em tempo discreto:
#
#       h_i(0) = choque_inicial_i
#
#       Δh_j(t)      = max( h_j(t) - h_j(t-1), 0 )
#       impacto_i(t) = Σ_j Ŵ_ij * Δh_j(t)
#
#       h_i(t+1) = min{ 1,
#                      h_i(t) + (1 - h_i(t)) * β * impacto_i(t)
#                    }
#
#     onde Ŵ_ij = exposição_ij / capital_inicial_i
#
#   - Capital atualizado:
#
#       perda_i(t+1)   = Σ_j exposição_ij * Δh_j(t+1)
#       capital_i(t+1) = max( capital_i(t) - perda_i(t+1), 0 )
#
#       se capital_i(t+1) <= 0  ⇒  h_i(t+1) = 1 (default)
#
#   - Índice de risco sistêmico:
#
#       SR(t) = (1 / Σ_i v_i) * Σ_i v_i * h_i(t)
#
#     onde v_i = valor econômico do nó i.
#
#   - Animação Pygame:
#       - cor do nó vai de verde (h≈0) a vermelho (h≈1)
#       - espessura da aresta ~ exposição
#       - mostra t e SR(t) na tela
# ============================================================

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import networkx as nx

import pygame


# ------------------------------------------------------------
# 1) Estruturas de dados
# ------------------------------------------------------------

@dataclass
class ParametrosRede:
    beta: float = 0.9           # intensidade da propagação
    passos_maximos: int = 40    # número máximo de passos na simulação
    tolerancia: float = 1e-4    # critério de convergência |h(t+1) - h(t)|_∞
    fator_choque_padrao: float = 0.4  # choque inicial se não especificado


# ------------------------------------------------------------
# 2) Criação da rede de risco sistêmico (exemplo)
# ------------------------------------------------------------

def criar_rede_exemplo() -> nx.DiGraph:
    """
    Cria um grafo direcionado de instituições financeiras com:
      - exposições (quem está exposto a quem)
      - capital inicial
      - valor econômico
      - choque inicial em alguns nós

    Convenção:
      exposicao_ij = quanto a instituição i perde se a instituição j quebrar.
    """
    grafo = nx.DiGraph()

    # Nós = instituições
    # valor_economico: tamanho relativo (para SR e visualização)
    # capital_inicial: buffer de absorção de perdas
    # choque_inicial: distress inicial h_i(0)
    dados_nos = {
        "BancoA": {"valor_economico": 100.0, "capital_inicial": 20.0, "choque_inicial": 0.6},
        "BancoB": {"valor_economico": 80.0,  "capital_inicial": 16.0, "choque_inicial": 0.0},
        "BancoC": {"valor_economico": 60.0,  "capital_inicial": 12.0, "choque_inicial": 0.0},
        "BancoD": {"valor_economico": 50.0,  "capital_inicial": 10.0, "choque_inicial": 0.0},
        "BancoE": {"valor_economico": 40.0,  "capital_inicial": 8.0,  "choque_inicial": 0.0},
        "BancoF": {"valor_economico": 30.0,  "capital_inicial": 6.0,  "choque_inicial": 0.2},
    }

    for nome, atributos in dados_nos.items():
        grafo.add_node(nome, **atributos)

    # Exposições direcionadas (i exposto a j)
    # (i, j, exposicao_ij)
    # pense como: i comprou títulos de j; se j sofre, i perde.
    lista_arestas = [
        ("BancoA", "BancoB", 4.0),
        ("BancoA", "BancoC", 3.0),
        ("BancoB", "BancoC", 2.0),
        ("BancoB", "BancoD", 4.0),
        ("BancoC", "BancoD", 3.0),
        ("BancoC", "BancoE", 2.0),
        ("BancoD", "BancoE", 2.5),
        ("BancoD", "BancoF", 1.5),
        ("BancoE", "BancoF", 2.0),
        ("BancoF", "BancoA", 1.0),  # feedback
    ]

    for origem, destino, exposicao in lista_arestas:
        grafo.add_edge(origem, destino, exposicao=exposicao)

    return grafo


# ------------------------------------------------------------
# 3) Preparação das matrizes e vetores
# ------------------------------------------------------------

def construir_matrizes(grafo: nx.DiGraph):
    """
    A partir do grafo, constrói:
      - lista_nos
      - dicionario_indices
      - matriz_exposicao W  (W[i,j] = exposicao_ij)
      - matriz_exposicao_normalizada W_til (Ŵ_ij)
      - vetor_capital_inicial
      - vetor_valor_economico
      - vetor_choque_inicial h(0)
    """
    lista_nos = list(grafo.nodes())
    N = len(lista_nos)
    dicionario_indices: Dict[str, int] = {no: i for i, no in enumerate(lista_nos)}

    W = np.zeros((N, N), dtype=float)
    capital_inicial = np.zeros(N, dtype=float)
    valor_economico = np.zeros(N, dtype=float)
    choque_inicial = np.zeros(N, dtype=float)

    for no in lista_nos:
        i = dicionario_indices[no]
        capital_inicial[i] = grafo.nodes[no].get("capital_inicial", 1.0)
        valor_economico[i] = grafo.nodes[no].get("valor_economico", 1.0)
        choque_inicial[i] = grafo.nodes[no].get("choque_inicial", 0.0)

    # Preenche matriz de exposicoes W
    for origem, destino, dados in grafo.edges(data=True):
        i = dicionario_indices[origem]
        j = dicionario_indices[destino]
        W[i, j] = dados.get("exposicao", 0.0)

    # Normalização por capital_inicial (Ŵ_ij)
    W_til = np.zeros_like(W)
    for i in range(N):
        if capital_inicial[i] > 0:
            W_til[i, :] = W[i, :] / capital_inicial[i]
        else:
            W_til[i, :] = 0.0

    return (
        lista_nos,
        dicionario_indices,
        W,
        W_til,
        capital_inicial,
        valor_economico,
        choque_inicial,
    )


# ------------------------------------------------------------
# 4) Simulação da dinâmica de contágio
# ------------------------------------------------------------

def simular_contagio(
    parametros: ParametrosRede,
    W: np.ndarray,
    W_til: np.ndarray,
    capital_inicial: np.ndarray,
    valor_economico: np.ndarray,
    choque_inicial: np.ndarray,
):
    """
    Executa a simulação da dinâmica de contágio:
      - h(t)  vetor de distress
      - SR(t) índice de risco sistêmico

    Retorna:
      - lista_h:  lista de vetores h(t) (cada elemento é um np.array)
      - lista_capital: lista de vetores capital(t)
      - lista_SR: lista com SR(t)
    """
    N = len(capital_inicial)
    beta = parametros.beta

    # Vetores de estado
    h_anterior = np.zeros(N, dtype=float)
    h_atual = choque_inicial.copy()
    capital_atual = capital_inicial.copy()

    # Históricos
    lista_h: List[np.ndarray] = [h_atual.copy()]
    lista_capital: List[np.ndarray] = [capital_atual.copy()]
    lista_SR: List[float] = []

    # Função SR(t)
    soma_valor = valor_economico.sum()

    # SR(0)
    SR_0 = float(np.dot(valor_economico, h_atual) / soma_valor)
    lista_SR.append(SR_0)

    for t in range(1, parametros.passos_maximos + 1):
        # Incremento de distress Δh_j(t) = max(h_j(t) - h_j(t-1), 0)
        delta_h = np.maximum(h_atual - h_anterior, 0.0)

        # impacto_i(t) = Σ_j Ŵ_ij * Δh_j(t)
        impacto = W_til.dot(delta_h)

        # h_i(t+1)
        h_novo = h_atual + (1.0 - h_atual) * beta * impacto
        h_novo = np.clip(h_novo, 0.0, 1.0)

        # Atualização de capital
        # perda_i(t+1) = Σ_j W_ij * Δh_j(t+1)
        delta_h_novo = np.maximum(h_novo - h_atual, 0.0)
        perda = W.dot(delta_h_novo)
        capital_novo = capital_atual - perda
        capital_novo = np.maximum(capital_novo, 0.0)

        # Se capital foi a zero, força h=1 (default total)
        for i in range(N):
            if capital_novo[i] <= 0.0 and h_novo[i] < 1.0:
                h_novo[i] = 1.0

        # Índice sistêmico SR(t)
        SR_t = float(np.dot(valor_economico, h_novo) / soma_valor)
        lista_SR.append(SR_t)

        lista_h.append(h_novo.copy())
        lista_capital.append(capital_novo.copy())

        # Critério de parada
        max_dif = float(np.max(np.abs(h_novo - h_atual)))
        if max_dif < parametros.tolerancia:
            # completa listas com estado final para facilitar animação
            # (repete o último estado até o tamanho passado se necessário na animação)
            break

        h_anterior = h_atual
        h_atual = h_novo
        capital_atual = capital_novo

    return lista_h, lista_capital, lista_SR


# ------------------------------------------------------------
# 5) Medidas de centralidade e raio espectral
# ------------------------------------------------------------

def calcular_centralidades_e_raio_espectral(W: np.ndarray):
    """
    Calcula:
      - centralidade de grau de saída e entrada (ponderado)
      - vetor de eigenvetor (W^T x = λ_max x)
      - raio espectral de uma matriz de propagação P = W_til^T (exemplo)
    """
    # Graus ponderados (saída e entrada)
    grau_saida = W.sum(axis=1)
    grau_entrada = W.sum(axis=0)

    # Autovalores de W^T (para centralidade de eigenvetor)
    autovalores, autovetores = np.linalg.eig(W.T)
    idx_max = int(np.argmax(np.real(autovalores)))
    lambda_max = float(np.real(autovalores[idx_max]))
    vetor_eigen = np.real(autovetores[:, idx_max])
    # normaliza para valores positivos somando módulo
    if np.all(vetor_eigen == 0):
        vetor_eigen_normalizado = vetor_eigen
    else:
        vetor_eigen_normalizado = np.abs(vetor_eigen) / np.sum(np.abs(vetor_eigen))

    return grau_saida, grau_entrada, lambda_max, vetor_eigen_normalizado


def calcular_raio_espectral_matriz_prop(W_til: np.ndarray, beta: float):
    """
    Considera matriz de propagação P = beta * W_til^T
    e calcula o raio espectral ρ(P).
    """
    P = beta * W_til.T
    autovalores = np.linalg.eigvals(P)
    raio = float(np.max(np.abs(autovalores)))
    return raio


# ------------------------------------------------------------
# 6) Animação Pygame
# ------------------------------------------------------------

def inicializar_pygame(largura: int = 1000, altura: int = 600):
    pygame.init()
    tela = pygame.display.set_mode((largura, altura))
    pygame.display.set_caption("Rede de Risco Sistêmico — Animação de Contágio")
    fonte = pygame.font.SysFont("arial", 18)
    return tela, fonte


def gerar_posicoes_circulares(lista_nos: List[str], largura: int, altura: int) -> Dict[str, Tuple[int, int]]:
    """
    Distribui os nós em um círculo para facilitar visualização.
    """
    N = len(lista_nos)
    centro_x = largura // 2
    centro_y = altura // 2
    raio = min(largura, altura) // 3

    posicoes: Dict[str, Tuple[int, int]] = {}
    for k, no in enumerate(lista_nos):
        angulo = 2 * math.pi * k / N
        x = int(centro_x + raio * math.cos(angulo))
        y = int(centro_y + raio * math.sin(angulo))
        posicoes[no] = (x, y)
    return posicoes


def cor_por_distress(h: float) -> Tuple[int, int, int]:
    """
    Mapeia h ∈ [0,1] em cor RGB:
      h = 0 → verde
      h = 1 → vermelho
      interpolando linearmente.
    """
    h_clamp = max(0.0, min(1.0, h))
    r = int(255 * h_clamp)
    g = int(255 * (1.0 - h_clamp))
    b = 60
    return (r, g, b)


def desenhar_quadro(
    tela,
    fonte,
    grafo: nx.DiGraph,
    lista_nos: List[str],
    posicoes: Dict[str, Tuple[int, int]],
    dicionario_indices: Dict[str, int],
    W: np.ndarray,
    h_t: np.ndarray,
    SR_t: float,
    t: int,
):
    tela.fill((15, 23, 42))  # fundo escuro

    # Escala de espessura das arestas
    exposicoes = [dados.get("exposicao", 0.0) for _, _, dados in grafo.edges(data=True)]
    expo_max = max(exposicoes) if exposicoes else 1.0

    # Desenhar arestas
    for origem, destino, dados in grafo.edges(data=True):
        exposicao = dados.get("exposicao", 0.0)
        peso_rel = exposicao / expo_max if expo_max > 0 else 0.0
        largura = 1 + int(4 * peso_rel)

        x1, y1 = posicoes[origem]
        x2, y2 = posicoes[destino]

        # linha
        pygame.draw.line(tela, (148, 163, 184), (x1, y1), (x2, y2), largura)

        # pequena seta (triângulo simples) na ponta aproximada
        dx = x2 - x1
        dy = y2 - y1
        angulo = math.atan2(dy, dx)
        seta_tamanho = 10
        ponta_x = x2
        ponta_y = y2
        esquerda_x = ponta_x - seta_tamanho * math.cos(angulo - math.pi / 6)
        esquerda_y = ponta_y - seta_tamanho * math.sin(angulo - math.pi / 6)
        direita_x = ponta_x - seta_tamanho * math.cos(angulo + math.pi / 6)
        direita_y = ponta_y - seta_tamanho * math.sin(angulo + math.pi / 6)
        pygame.draw.polygon(
            tela,
            (148, 163, 184),
            [(ponta_x, ponta_y), (esquerda_x, esquerda_y), (direita_x, direita_y)],
        )

    # Desenhar nós
    for no in lista_nos:
        i = dicionario_indices[no]
        h_i = float(h_t[i])
        cor_no = cor_por_distress(h_i)

        x, y = posicoes[no]
        raio_no = 25
        pygame.draw.circle(tela, cor_no, (x, y), raio_no)
        pygame.draw.circle(tela, (15, 23, 42), (x, y), raio_no, 2)

        # rótulo com nome + h_i
        texto_no = fonte.render(f"{no}", True, (248, 250, 252))
        rect_texto = texto_no.get_rect(center=(x, y - 20))
        tela.blit(texto_no, rect_texto)

        texto_h = fonte.render(f"h={h_i:.2f}", True, (226, 232, 240))
        rect_h = texto_h.get_rect(center=(x, y + 15))
        tela.blit(texto_h, rect_h)

    # Informações do tempo e SR(t)
    texto_t = fonte.render(f"Passo t = {t}", True, (248, 250, 252))
    tela.blit(texto_t, (20, 20))

    texto_SR = fonte.render(f"SR(t) = {SR_t:.3f}", True, (248, 250, 252))
    tela.blit(texto_SR, (20, 45))

    pygame.display.flip()


def animar_processo(
    grafo: nx.DiGraph,
    lista_nos: List[str],
    dicionario_indices: Dict[str, int],
    W: np.ndarray,
    lista_h: List[np.ndarray],
    lista_SR: List[float],
):
    largura, altura = 1000, 600
    tela, fonte = inicializar_pygame(largura, altura)
    clock = pygame.time.Clock()

    posicoes = gerar_posicoes_circulares(lista_nos, largura, altura)

    # Animação: percorre os estados da simulação
    indice_t = 0
    numero_passos = len(lista_h)
    frames_por_passo = 20  # quantos frames cada estado fica visível
    contador_frames = 0

    rodando = True
    while rodando:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                rodando = False
            elif evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_ESCAPE:
                    rodando = False

        # Estado atual
        h_t = lista_h[indice_t]
        SR_t = lista_SR[indice_t]

        desenhar_quadro(
            tela,
            fonte,
            grafo,
            lista_nos,
            posicoes,
            dicionario_indices,
            W,
            h_t,
            SR_t,
            t=indice_t,
        )

        # Controle de passo
        contador_frames += 1
        if contador_frames >= frames_por_passo:
            contador_frames = 0
            if indice_t < numero_passos - 1:
                indice_t += 1
            # se chegou ao fim, apenas mantém no último estado

        clock.tick(30)

    pygame.quit()


# ------------------------------------------------------------
# 7) Função principal
# ------------------------------------------------------------

def main():
    # 1) Criar rede
    grafo = criar_rede_exemplo()
    (
        lista_nos,
        dicionario_indices,
        W,
        W_til,
        capital_inicial,
        valor_economico,
        choque_inicial,
    ) = construir_matrizes(grafo)

    parametros = ParametrosRede()

    # 2) Simular dinâmica de contágio
    lista_h, lista_capital, lista_SR = simular_contagio(
        parametros,
        W,
        W_til,
        capital_inicial,
        valor_economico,
        choque_inicial,
    )

    # 3) Centralidades e raio espectral
    grau_saida, grau_entrada, lambda_max_W, vetor_eigen = calcular_centralidades_e_raio_espectral(W)
    raio_espectral_P = calcular_raio_espectral_matriz_prop(W_til, parametros.beta)

    print("===== RESUMO ESTATÍSTICO DA REDE =====")
    print("Nós:", lista_nos)
    print("\nGrau de saída ponderado (exposição total fornecida):")
    for no, g in zip(lista_nos, grau_saida):
        print(f"  {no}: {g:.2f}")

    print("\nGrau de entrada ponderado (exposição total recebida):")
    for no, g in zip(lista_nos, grau_entrada):
        print(f"  {no}: {g:.2f}")

    print(f"\nMaior autovalor de W^T (lambda_max): {lambda_max_W:.4f}")
    print("Centralidade de eigenvetor normalizada (aprox.):")
    for no, c in zip(lista_nos, vetor_eigen):
        print(f"  {no}: {c:.4f}")

    print(f"\nRaio espectral da matriz de propagação P = beta * W_til^T: ρ(P) = {raio_espectral_P:.4f}")
    if raio_espectral_P < 1:
        print("  → regime absorvente (tendência a absorver choques ao longo do tempo).")
    else:
        print("  → regime potencialmente instável (choques podem se amplificar).")

    print("\nEvolução do índice de risco sistêmico SR(t):")
    for t, SR_t in enumerate(lista_SR):
        print(f"  t={t:2d}  SR={SR_t:.4f}")

    # 4) Animação Pygame
    animar_processo(
        grafo,
        lista_nos,
        dicionario_indices,
        W,
        lista_h,
        lista_SR,
    )


if __name__ == "__main__":
    main()
