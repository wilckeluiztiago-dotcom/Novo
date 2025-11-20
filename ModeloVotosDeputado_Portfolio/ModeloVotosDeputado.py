#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===========================================================
MODELO PROBABILÍSTICO-NEURAL PARA PREVER VOTOS
Deputado Estadual (multi-candidato) — Brasil
Autor: Luiz Tiago Wilcke (LT) 
===========================================================

Objetivo
--------
Dado um histórico de votos/campanhas e features por candidato, este script:

1) Estima uma distribuição probabilística de votos futuros
   para vários candidatos (multi-partidário).

2) Gera gráficos:
   - densidade/probabilidade de votos por candidato,
   - probabilidade de crescer vs. cair (Δvotos),
   - probabilidade de "ganhar" (estar no top-K),
   - fan chart (intervalos ao longo do tempo).

3) Combina:
   - Estatística avançada (Dirichlet-Multinomial, hierarquia simples)
   - Equações estocásticas em espaço de simplex (difusão tipo Wright–Fisher)
   - Rede neural em PyTorch para drift/força de campanha com incerteza por MC Dropout.

Notas importantes
-----------------
- O script é didático, **não** substitui pesquisas oficiais.
- Não produz nem sugere mensagens persuasivas; apenas modela dados.
- Para uso real, você deve inserir dados reais do TSE/pesquisas.
  O TSE disponibiliza dados abertos oficiais. Veja README.

Dependências
------------
pip install numpy pandas matplotlib torch scikit-learn

Uso rápido
----------
python ModeloVotosDeputado.py --csv dados.csv --col_votos votos_2022
python ModeloVotosDeputado.py --gerar_sintetico

Formato esperado do CSV (mínimo)
--------------------------------
candidato, partido, uf, votos_2022, gastos_campanha, seguidores, ideologia, incumbente, ...
Qualquer coluna numérica adicional é aceita como feature.

===========================================================
"""

import os
import math
import json
import random
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ------------------------------
# 1. Configurações
# ------------------------------

@dataclass
class Config:
    semente: int = 42
    epocas: int = 400
    batch: int = 64
    lr: float = 2e-3
    ocultas: int = 128
    dropout: float = 0.20
    top_k: int = 5                 # "ganhar": ficar entre os K mais votados
    simulacoes: int = 2000         # Monte Carlo final
    passos_tempo: int = 40         # fan chart
    horizonte_dt: float = 1.0      # passo temporal "abstrato"
    kappa_reversao: float = 0.50   # força de reversão à média no latent OU
    sigma_ruido: float = 0.25      # intensidade do ruído no latent OU
    temperatura_softmax: float = 1.0

# ------------------------------
# 2. Utilidades
# ------------------------------

def set_seed(semente: int):
    random.seed(semente)
    np.random.seed(semente)
    torch.manual_seed(semente)
    torch.cuda.manual_seed_all(semente)

def softmax_np(z: np.ndarray, temp: float = 1.0, eixo: int = -1):
    z = z / temp
    z = z - z.max(axis=eixo, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=eixo, keepdims=True)

# ------------------------------
# 3. Geração sintética (para demo)
# ------------------------------

def gerar_base_sintetica(n_candidatos: int = 40, uf: str = "SC", ano_base: int = 2022) -> pd.DataFrame:
    """
    Cria uma base sintética plausível para demonstração.
    """
    partidos = ["PT", "PL", "MDB", "PSDB", "PDT", "PSB", "UNIÃO", "PP", "REDE"]
    nomes = [f"Cand_{i:02d}" for i in range(n_candidatos)]

    dados = []
    for i, nome in enumerate(nomes):
        partido = random.choice(partidos)
        gastos = np.clip(np.random.lognormal(12.0, 0.6), 50_000, 5_000_000)
        seguidores = np.random.lognormal(9.0, 0.8)
        ideologia = np.random.normal(0, 1)      # só um eixo numérico fictício
        incumbente = np.random.binomial(1, 0.25)
        efeito_partido = partidos.index(partido) / len(partidos)

        votos_prev = (
            5000
            + 0.0025 * gastos
            + 1.5 * seguidores
            + 4000 * incumbente
            + 2500 * efeito_partido
            + np.random.normal(0, 12000)
        )
        votos_prev = int(max(500, votos_prev))

        dados.append({
            "candidato": nome,
            "partido": partido,
            "uf": uf,
            f"votos_{ano_base}": votos_prev,
            "gastos_campanha": gastos,
            "seguidores": seguidores,
            "ideologia": ideologia,
            "incumbente": incumbente,
        })

    return pd.DataFrame(dados)

# ------------------------------
# 4. Dataset / Dataloader
# ------------------------------

class DatasetVotos(Dataset):
    def __init__(self, X: np.ndarray, y_log_votos: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y_log_votos, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ------------------------------
# 5. Rede Neural com MC Dropout
# ------------------------------

class RedeDrift(nn.Module):
    """
    Prediz log-votos esperados (baseline) a partir de features.
    Dropout é mantido no modo treino durante inferência para gerar incerteza.
    """
    def __init__(self, n_features: int, ocultas: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, ocultas),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ocultas, ocultas),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ocultas, 1),
        )

    def forward(self, x):
        return self.net(x)

def treinar_rede(modelo: nn.Module, loader_treino: DataLoader, loader_val: DataLoader, cfg: Config):
    opt = torch.optim.AdamW(modelo.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    melhor_val = float("inf")
    melhor_estado = None

    for ep in range(1, cfg.epocas + 1):
        modelo.train()
        perdas = []
        for Xb, yb in loader_treino:
            opt.zero_grad()
            pred = modelo(Xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            perdas.append(loss.item())

        # validação
        modelo.eval()
        with torch.no_grad():
            perdas_val = []
            for Xb, yb in loader_val:
                pred = modelo(Xb)
                perdas_val.append(loss_fn(pred, yb).item())
        val = float(np.mean(perdas_val))

        if val < melhor_val:
            melhor_val = val
            melhor_estado = {k: v.cpu().clone() for k, v in modelo.state_dict().items()}

        if ep % 80 == 0 or ep == 1:
            print(f"Época {ep:04d}/{cfg.epocas} | perda_treino={np.mean(perdas):.4f} | perda_val={val:.4f}")

    if melhor_estado is not None:
        modelo.load_state_dict(melhor_estado)
    return modelo

def prever_com_incerteza(modelo: nn.Module, X: np.ndarray, n_amostras: int = 200) -> np.ndarray:
    """
    MC Dropout: retorna amostras de log-votos baseline.
    """
    modelo.train()  # mantém dropout ativo
    X_t = torch.tensor(X, dtype=torch.float32)
    amostras = []
    with torch.no_grad():
        for _ in range(n_amostras):
            amostras.append(modelo(X_t).squeeze(1).cpu().numpy())
    return np.stack(amostras, axis=0)  # (S, N)

# ------------------------------
# 6. Camada estocástica de votos (SDE)
# ------------------------------

def simular_sde_votos(
    log_votos_iniciais: np.ndarray,
    log_votos_alvo: np.ndarray,
    cfg: Config,
    passos: int
) -> np.ndarray:
    """
    Espaço latent z com OU (Ornstein-Uhlenbeck):
        dz_t = kappa*(m - z_t) dt + sigma dW_t

    Convertendo para shares p via softmax:
        p_t = softmax(z_t)

    Votos absolutos por candidato:
        V_t = p_t * V_total
    onde V_total é mantido próximo ao total inicial pela escala.

    Isso é inspirado na ideia de difusão em simplex (Wright-Fisher / logistic-normal),
    comum em modelagem de shares eleitorais. (vide README)
    """
    N = len(log_votos_iniciais)
    dt = cfg.horizonte_dt / passos

    # latent inicial e alvo
    z = log_votos_iniciais.copy()
    m = log_votos_alvo.copy()

    trilha = np.zeros((passos + 1, N))
    trilha[0] = z

    for t in range(1, passos + 1):
        ruido = np.random.normal(0, math.sqrt(dt), size=N)
        z = z + cfg.kappa_reversao * (m - z) * dt + cfg.sigma_ruido * ruido
        trilha[t] = z

    return trilha  # latent (passos+1, N)

def latent_para_votos(latent: np.ndarray, total_votos: float, cfg: Config) -> np.ndarray:
    shares = softmax_np(latent, temp=cfg.temperatura_softmax, eixo=1)
    return shares * total_votos

# ------------------------------
# 7. Pipeline completo
# ------------------------------

def preparar_dados(df: pd.DataFrame, col_votos: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, StandardScaler, List[str]]:
    # target
    if col_votos not in df.columns:
        raise ValueError(f"Coluna de votos '{col_votos}' não encontrada.")

    votos = df[col_votos].astype(float).values
    log_votos = np.log1p(votos)

    # features numéricas (exclui colunas identificadoras)
    col_excluir = {"candidato", "partido", "uf", col_votos}
    col_features = [c for c in df.columns if c not in col_excluir and pd.api.types.is_numeric_dtype(df[c])]
    if len(col_features) == 0:
        raise ValueError("Nenhuma feature numérica encontrada além da coluna de votos.")

    X = df[col_features].values.astype(float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    return df, Xs, log_votos, scaler, col_features

def rodar_modelo(df: pd.DataFrame, col_votos: str, cfg: Config, saida_dir: str):
    os.makedirs(saida_dir, exist_ok=True)

    df, Xs, log_votos, scaler, col_features = preparar_dados(df, col_votos)

    # split
    X_tr, X_va, y_tr, y_va, idx_tr, idx_va = train_test_split(
        Xs, log_votos, np.arange(len(df)), test_size=0.25, random_state=cfg.semente
    )

    ds_tr = DatasetVotos(X_tr, y_tr)
    ds_va = DatasetVotos(X_va, y_va)
    ld_tr = DataLoader(ds_tr, batch_size=cfg.batch, shuffle=True)
    ld_va = DataLoader(ds_va, batch_size=cfg.batch, shuffle=False)

    # rede
    modelo = RedeDrift(n_features=Xs.shape[1], ocultas=cfg.ocultas, dropout=cfg.dropout)
    modelo = treinar_rede(modelo, ld_tr, ld_va, cfg)

    # baseline probabilístico (log-votos alvo)
    amostras_log_alvo = prever_com_incerteza(modelo, Xs, n_amostras=250)  # (S, N)

    # Monte Carlo final
    votos_iniciais = np.expm1(log_votos)
    total_votos = float(votos_iniciais.sum())

    amostras_finais = []
    amostras_delta = []

    for s in range(cfg.simulacoes):
        log_alvo_s = amostras_log_alvo[np.random.randint(amostras_log_alvo.shape[0])]
        trilha_latent = simular_sde_votos(log_votos, log_alvo_s, cfg, passos=cfg.passos_tempo)
        votos_trilha = latent_para_votos(trilha_latent, total_votos, cfg)
        votos_final = votos_trilha[-1]

        amostras_finais.append(votos_final)
        amostras_delta.append(votos_final - votos_iniciais)

    amostras_finais = np.stack(amostras_finais, axis=0)  # (S, N)
    amostras_delta = np.stack(amostras_delta, axis=0)

    # probabilidades
    prob_crescer = (amostras_delta > 0).mean(axis=0)
    # top-K
    ranks = np.argsort(-amostras_finais, axis=1)  # desc
    prob_topk = np.zeros(len(df))
    for s in range(cfg.simulacoes):
        top = ranks[s, :cfg.top_k]
        prob_topk[top] += 1
    prob_topk /= cfg.simulacoes

    # sumariza
    resumo = df[["candidato", "partido", "uf"]].copy()
    resumo["votos_base"] = votos_iniciais
    resumo["media_votos_prev"] = amostras_finais.mean(axis=0)
    resumo["p10"] = np.percentile(amostras_finais, 10, axis=0)
    resumo["p90"] = np.percentile(amostras_finais, 90, axis=0)
    resumo["prob_crescer"] = prob_crescer
    resumo[f"prob_top{cfg.top_k}"] = prob_topk
    resumo = resumo.sort_values(f"prob_top{cfg.top_k}", ascending=False)

    print("\n=== TOP candidatos por probabilidade de ganhar (top-K) ===")
    print(resumo.head(cfg.top_k + 3).to_string(index=False, float_format=lambda x: f"{x:,.0f}"))

    # salva csv
    csv_out = os.path.join(saida_dir, "resumo_probabilistico.csv")
    resumo.to_csv(csv_out, index=False)

    # ------------------------------
    # 8. Gráficos
    # ------------------------------

    nomes = df["candidato"].tolist()

    # 8.1 densidades finais
    plt.figure(figsize=(11, 6))
    for i, nome in enumerate(nomes):
        xs = amostras_finais[:, i]
        # KDE simples via hist normalizado
        hist, bins = np.histogram(xs, bins=60, density=True)
        centros = 0.5 * (bins[1:] + bins[:-1])
        plt.plot(centros, hist, alpha=0.6, lw=1)
    plt.title("Densidades de votos finais (Monte Carlo)")
    plt.xlabel("Votos")
    plt.ylabel("Densidade")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(saida_dir, "densidades_votos.png"), dpi=160)
    plt.close()

    # 8.2 probabilidade de crescer
    plt.figure(figsize=(10, 5))
    ordem = np.argsort(-prob_crescer)
    plt.bar(np.array(nomes)[ordem], prob_crescer[ordem])
    plt.title("Probabilidade de crescer votos vs. base")
    plt.ylabel("P(Δvotos > 0)")
    plt.xticks(rotation=70, ha="right")
    plt.ylim(0, 1)
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(saida_dir, "prob_crescer.png"), dpi=160)
    plt.close()

    # 8.3 probabilidade de top-K
    plt.figure(figsize=(10, 5))
    ordem = np.argsort(-prob_topk)
    plt.bar(np.array(nomes)[ordem], prob_topk[ordem])
    plt.title(f"Probabilidade de ficar no TOP-{cfg.top_k} (ganhar)")
    plt.ylabel(f"P(top-{cfg.top_k})")
    plt.xticks(rotation=70, ha="right")
    plt.ylim(0, 1)
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(saida_dir, f"prob_top{cfg.top_k}.png"), dpi=160)
    plt.close()

    # 8.4 fan chart (top 6 candidatos por média)
    idx_top = np.argsort(-amostras_finais.mean(axis=0))[:6]
    passos = cfg.passos_tempo
    fan_percentis = [5, 25, 50, 75, 95]

    plt.figure(figsize=(10, 6))
    for j, i in enumerate(idx_top):
        # re-simula trilhas só para fan chart (menos caro)
        trilhas = []
        for s in range(300):
            log_alvo_s = amostras_log_alvo[np.random.randint(amostras_log_alvo.shape[0])]
            trilha_latent = simular_sde_votos(log_votos, log_alvo_s, cfg, passos=passos)
            votos_trilha = latent_para_votos(trilha_latent, total_votos, cfg)
            trilhas.append(votos_trilha[:, i])
        trilhas = np.stack(trilhas, axis=0)  # (S, T)

        T = np.arange(passos + 1)
        perc = np.percentile(trilhas, fan_percentis, axis=0)  # (P,T)

        plt.plot(T, perc[2], lw=2, label=f"{nomes[i]} (mediana)")
        plt.fill_between(T, perc[1], perc[3], alpha=0.15)
        plt.fill_between(T, perc[0], perc[4], alpha=0.08)

    plt.title("Fan chart — evolução probabilística dos votos (top 6)")
    plt.xlabel("Tempo (passos abstratos até a eleição)")
    plt.ylabel("Votos")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(saida_dir, "fan_chart_top6.png"), dpi=160)
    plt.close()

    # salva metadados
    meta = {
        "features_usadas": col_features,
        "config": cfg.__dict__,
        "coluna_votos_base": col_votos
    }
    with open(os.path.join(saida_dir, "metadados.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nArquivos salvos em: {saida_dir}")
    print(f"- {csv_out}")
    print("- densidades_votos.png")
    print("- prob_crescer.png")
    print(f"- prob_top{cfg.top_k}.png")
    print("- fan_chart_top6.png")

# ------------------------------
# 9. Interface CLI
# ------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Modelo probabilístico-neural para votos (deputado estadual).")
    ap.add_argument("--csv", type=str, default=None, help="Caminho para CSV com dados.")
    ap.add_argument("--col_votos", type=str, default="votos_2022", help="Coluna de votos base.")
    ap.add_argument("--saida", type=str, default="saida_modelo_votos", help="Diretório de saída.")
    ap.add_argument("--gerar_sintetico", action="store_true", help="Gera base sintética e roda o modelo.")
    ap.add_argument("--top_k", type=int, default=5, help="Top-K para probabilidade de ganhar.")
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = Config(top_k=args.top_k)
    set_seed(cfg.semente)

    if args.gerar_sintetico or args.csv is None:
        df = gerar_base_sintetica()
        col_votos = [c for c in df.columns if c.startswith("votos_")][0]
        print("Base sintética gerada.")
    else:
        df = pd.read_csv(args.csv)
        col_votos = args.col_votos

    rodar_modelo(df, col_votos, cfg, args.saida)

if __name__ == "__main__":
    main()
