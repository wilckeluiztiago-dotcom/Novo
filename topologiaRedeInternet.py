# ============================================================
# TDA em Tráfego de Rede — Detecção Topológica de Anomalias
# Autor: Luiz Tiago Wilcke (LT)
# ============================================================

import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# TDA — homologia persistente (Vietoris–Rips)
# ------------------------------------------------------------
try:
    from ripser import ripser
except ImportError as e:
    raise ImportError(
        "Pacote 'ripser' não encontrado. "
        "Instale com: pip install ripser"
    ) from e


# ============================================================
# 1. Configurações do modelo topológico
# ============================================================

@dataclass
class ConfiguracaoTopologica:
    """
    Configurações da análise topológica do tráfego.
    """
    janela_segundos: float = 10.0      # largura da janela temporal
    dimensao_maxima: int = 1           # H0 e H1 (componentes conexas e ciclos)
    limiar_barra_min: float = 0.01     # mínimo de persistência para contar uma barra
    limiar_anomalia_z: float = 3.0     # limiar do score (norma de z-score)
    max_pacotes_por_janela: int = 200  # limite para evitar explosão de complexidade


# ============================================================
# 2. Espaço métrico do tráfego
#    - Cada pacote -> ponto em R^4:
#      [tempo_normalizado, tamanho_normalizado, ip_origem_hash, ip_destino_hash]
# ============================================================

def hash_ip_para_real(ip: str) -> float:
    """
    Converte um IP v4 (ex: '192.168.0.10') em um número real em [0,1].
    Serve apenas como imersão no espaço métrico.
    """
    partes = ip.split(".")
    valor = 0
    for p in partes:
        try:
            valor = valor * 257 + int(p)  # 257 é primo, mistura os bytes
        except ValueError:
            # fallback em caso de IP estranho
            valor = valor * 257
    # normalizar para [0,1]
    return valor / float(257 ** 4)


def construir_nuvem_pontos(df_janela: pd.DataFrame,
                           cfg: ConfiguracaoTopologica,
                           tempo_inicio: float,
                           tempo_fim: float) -> np.ndarray:
    """
    Constrói nuvem de pontos em R^4 para uma janela de tráfego.
    """
    if df_janela.empty:
        return np.empty((0, 4), dtype=float)

    df = df_janela.copy()

    # Normalização do tempo dentro da janela [0,1]
    intervalo = max(tempo_fim - tempo_inicio, 1e-6)
    df["tempo_norm"] = (df["tempo_seg"] - tempo_inicio) / intervalo

    # Normalização do tamanho (assumindo MTU ~ 1500 bytes)
    df["tam_norm"] = df["tamanho_bytes"] / 1500.0
    df["tam_norm"] = df["tam_norm"].clip(0, 5)  # evita outliers absurdos

    # Imersão dos IPs em [0,1]
    df["ip_origem_hash"] = df["ip_origem"].map(hash_ip_para_real)
    df["ip_destino_hash"] = df["ip_destino"].map(hash_ip_para_real)

    # Amostragem para limitar número de pontos (controle de complexidade)
    if len(df) > cfg.max_pacotes_por_janela:
        df = df.sample(cfg.max_pacotes_por_janela, random_state=42)

    nuvem = df[["tempo_norm", "tam_norm", "ip_origem_hash", "ip_destino_hash"]].values
    return nuvem.astype(float)


# ============================================================
# 3. Homologia persistente (Vietoris–Rips via ripser)
# ============================================================

def calcular_diagramas_persistencia(nuvem: np.ndarray,
                                    cfg: ConfiguracaoTopologica) -> Optional[List[np.ndarray]]:
    """
    Calcula os diagramas de persistência (H0, H1, ...) da nuvem de pontos.
    """
    if nuvem.shape[0] < 5:
        # Poucos pontos => pouca informação topológica
        return None

    resultado = ripser(nuvem, maxdim=cfg.dimensao_maxima)
    diagramas = resultado["dgms"]  # lista: [H0, H1, ...]
    return diagramas


def extrair_caracteristicas_topologicas(diagramas: List[np.ndarray],
                                        cfg: ConfiguracaoTopologica) -> np.ndarray:
    """
    Extrai um vetor de características topológicas a partir dos diagramas.
    Para cada dimensão k:
      - quantidade de barras "longas"
      - comprimento médio das barras "longas"
      - entropia de persistência

    Vetor final (se maxdim=1):
      [qtd_H0, media_H0, entropia_H0,  qtd_H1, media_H1, entropia_H1]
    """
    caracteristicas: List[float] = []

    for dim, diag in enumerate(diagramas):
        if diag.size == 0:
            # Sem barras nesta dimensão
            caracteristicas.extend([0.0, 0.0, 0.0])
            continue

        comprimentos = []
        for birth, death in diag:
            if math.isinf(death):
                # barras infinitas (componentes que nunca colapsam) — opcional
                continue
            comprimento = float(death - birth)
            if comprimento >= cfg.limiar_barra_min:
                comprimentos.append(comprimento)

        if len(comprimentos) == 0:
            caracteristicas.extend([0.0, 0.0, 0.0])
            continue

        comprimentos = np.array(comprimentos, dtype=float)

        # Quantidade de barras "significativas"
        qtd = float(len(comprimentos))

        # Comprimento médio
        media = float(comprimentos.mean())

        # Entropia de persistência (medida global da "complexidade" do diagrama)
        soma = float(comprimentos.sum())
        p = comprimentos / (soma + 1e-12)
        entropia = float(-np.sum(p * np.log(p + 1e-12)))

        caracteristicas.extend([qtd, media, entropia])

    return np.array(caracteristicas, dtype=float)


# ============================================================
# 4. Janelamento temporal do tráfego
# ============================================================

def iterar_janelas_trafego(df_trafego: pd.DataFrame,
                           cfg: ConfiguracaoTopologica):
    """
    Gera janelas temporais sucessivas do tráfego.
    """
    if df_trafego.empty:
        return

    t0 = float(df_trafego["tempo_seg"].min())
    tmax = float(df_trafego["tempo_seg"].max())

    inicio = t0
    while inicio < tmax:
        fim = inicio + cfg.janela_segundos
        mascara = (df_trafego["tempo_seg"] >= inicio) & (df_trafego["tempo_seg"] < fim)
        df_janela = df_trafego.loc[mascara].copy()
        yield inicio, fim, df_janela
        inicio = fim


# ============================================================
# 5. Geração de tráfego sintético (normal + DDoS + botnet)
#    - Apenas para demonstração do modelo.
# ============================================================

def gerar_trafego_sintetico(
    duracao_total: float = 300.0,
    taxa_normal: float = 50.0,   # pacotes/s "normais"
    taxa_ddos: float = 400.0,    # pacotes/s na janela de DDoS
    inicio_ddos: float = 120.0,
    fim_ddos: float = 180.0,
    taxa_botnet: float = 80.0,
    inicio_botnet: float = 220.0,
    fim_botnet: float = 260.0,
) -> pd.DataFrame:
    """
    Cria um DataFrame com tráfego sintético contendo:
        - regime normal
        - ataque DDoS
        - atividade de botnet
    """

    registros: List[Dict] = []

    rng = np.random.default_rng(42)

    # Conjunto de IPs "normais"
    ips_clientes = [f"10.0.0.{i}" for i in range(2, 50)]
    ips_servidores = [f"192.168.0.{i}" for i in range(2, 10)]

    ip_alvo_ddos = "192.168.0.100"
    ip_c2_botnet = "172.16.0.66"

    # 5.1 Tráfego normal (espalhado na linha do tempo toda)
    n_normal = int(duracao_total * taxa_normal)
    tempos_normal = rng.uniform(0, duracao_total, size=n_normal)

    for t in tempos_normal:
        # gerar tamanho e "clipar" via max/min (corrigindo o erro do clip)
        tam = int(rng.normal(600, 200))
        tam = max(100, min(1500, tam))
        registros.append({
            "tempo_seg": float(t),
            "ip_origem": rng.choice(ips_clientes),
            "ip_destino": rng.choice(ips_servidores),
            "tamanho_bytes": tam,
            "porta_destino": int(rng.choice([80, 443, 53, 22, 8080])),
            "tipo": "normal"
        })

    # 5.2 Ataque DDoS: muitos pacotes em pequena janela, muitos IPs origem para um alvo
    n_ddos = int((fim_ddos - inicio_ddos) * taxa_ddos)
    tempos_ddos = rng.uniform(inicio_ddos, fim_ddos, size=n_ddos)

    for t in tempos_ddos:
        registros.append({
            "tempo_seg": float(t),
            "ip_origem": f"203.0.113.{rng.integers(1, 254)}",  # muitos bots espalhados
            "ip_destino": ip_alvo_ddos,
            "tamanho_bytes": int(rng.integers(200, 1200)),
            "porta_destino": 80,
            "tipo": "ddos"
        })

    # 5.3 Botnet: padrão mais sutil, fluxos periódicos entre bots e C2
    n_botnet = int((fim_botnet - inicio_botnet) * taxa_botnet)
    tempos_botnet = rng.uniform(inicio_botnet, fim_botnet, size=n_botnet)

    for t in tempos_botnet:
        if rng.random() < 0.5:
            ip_bot = f"198.51.100.{rng.integers(1, 254)}"
            ip_dest = ip_c2_botnet
        else:
            ip_bot = ip_c2_botnet
            ip_dest = f"198.51.100.{rng.integers(1, 254)}"

        registros.append({
            "tempo_seg": float(t),
            "ip_origem": ip_bot,
            "ip_destino": ip_dest,
            "tamanho_bytes": int(rng.integers(80, 800)),
            "porta_destino": int(rng.choice([443, 8080, 9001])),
            "tipo": "botnet"
        })

    df = pd.DataFrame(registros)

    # Sanitizar tamanhos (caso o normal gere algo estranho)
    df["tamanho_bytes"] = df["tamanho_bytes"].clip(64, 1500)

    # Ordenar por tempo
    df = df.sort_values("tempo_seg").reset_index(drop=True)
    return df


# Pequeno helper genérico
def _clip(x, a, b):
    return max(a, min(b, x))


# Corrigir o uso de clip na geração normal (substituindo inline)
def _corrigir_tamanho(row):
    v = row["tamanho_bytes"]
    # inclui numpy.int64 / numpy.float64 etc.
    if isinstance(v, (int, float, np.integer, np.floating)):
        return _clip(float(v), 100, 1500)
    try:
        return _clip(float(v), 100, 1500)
    except Exception:
        return 600


# ============================================================
# 6. Detecção de anomalias baseada em TDA
# ============================================================

def calcular_scores_anomalia(matriz_caracteristicas: np.ndarray,
                             cfg: ConfiguracaoTopologica) -> np.ndarray:
    """
    Calcula um score de anomalia por janela com base no z-score
    multidimensional das características topológicas.

    Score = norma_2(z) = || (x - média) / desvio ||
    """
    if matriz_caracteristicas.shape[0] < 5:
        # Poucos dados para estatística robusta
        return np.zeros(matriz_caracteristicas.shape[0])

    media = matriz_caracteristicas.mean(axis=0)
    desvio = matriz_caracteristicas.std(axis=0)
    desvio[desvio < 1e-6] = 1e-6  # evita divisão por zero

    z = (matriz_caracteristicas - media) / desvio
    scores = np.linalg.norm(z, axis=1)
    return scores


# ============================================================
# 7. Demonstração completa
# ============================================================

def demonstracao_topologica():
    cfg = ConfiguracaoTopologica(
        janela_segundos=10.0,
        dimensao_maxima=1,
        limiar_barra_min=0.01,
        limiar_anomalia_z=3.0,
        max_pacotes_por_janela=250
    )

    print("Gerando tráfego sintético (normal + DDoS + botnet)...")
    df_trafego = gerar_trafego_sintetico()
    df_trafego["tamanho_bytes"] = df_trafego.apply(_corrigir_tamanho, axis=1)

    caracteristicas_todas: List[np.ndarray] = []
    tempos_centro: List[float] = []
    rotulos_reais: List[str] = []

    print("Calculando diagramas de persistência por janela...")
    for inicio, fim, df_janela in iterar_janelas_trafego(df_trafego, cfg):
        nuvem = construir_nuvem_pontos(df_janela, cfg, inicio, fim)
        diagramas = calcular_diagramas_persistencia(nuvem, cfg)

        if diagramas is None:
            # janela vazia / muito pobre em pontos
            vetor = np.zeros(6, dtype=float)  # para maxdim=1
        else:
            vetor = extrair_caracteristicas_topologicas(diagramas, cfg)

        caracteristicas_todas.append(vetor)
        tempos_centro.append(0.5 * (inicio + fim))

        # rótulo "verdadeiro" para fins de validação da demo
        if df_janela.empty:
            rotulo = "vazio"
        else:
            # rótulo mais frequente na janela
            rotulo = df_janela["tipo"].value_counts().idxmax()
        rotulos_reais.append(rotulo)

    matriz_car = np.vstack(caracteristicas_todas)
    scores = calcular_scores_anomalia(matriz_car, cfg)

    # Marcar janelas acima do limiar
    anomalias = scores > cfg.limiar_anomalia_z

    # --------------------------------------------------------
    # Impressão textual
    # --------------------------------------------------------
    print("\n========== RESUMO TOPOLOGICO POR JANELA ==========")
    for t, sc, rot, anom in zip(tempos_centro, scores, rotulos_reais, anomalias):
        marca = " <-- ANOMALIA TOPOLOGICA" if anom else ""
        print(f"t ~ {t:6.1f}s | score_topologico = {sc:6.2f} | rotulo_simulado = {rot}{marca}")

    # --------------------------------------------------------
    # Visualização
    # --------------------------------------------------------
    plt.figure(figsize=(10, 4))
    plt.plot(tempos_centro, scores, marker="o", linewidth=1)
    plt.axhline(cfg.limiar_anomalia_z, linestyle="--", label="limiar anomalia")

    # pintar janelas sinteticamente rotuladas
    for t, sc, rot in zip(tempos_centro, scores, rotulos_reais):
        if rot == "ddos":
            plt.scatter(t, sc, marker="o", s=60, edgecolors="k", label="DDoS")
        elif rot == "botnet":
            plt.scatter(t, sc, marker="s", s=60, edgecolors="k", label="Botnet")

    plt.xlabel("Tempo (s) — centro da janela")
    plt.ylabel("Score topológico (norma do z-score)")
    plt.title("Anomalias em tráfego como mudanças bruscas na topologia do espaço métrico")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    demonstracao_topologica()
