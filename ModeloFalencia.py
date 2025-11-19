# ============================================================
# MODELO DIFERENCIAL-NEURAL PARA RISCO DE FALÊNCIA
# Autor: Luiz Tiago Wilcke (LT)
# ============================================================
# - Sistema de equações diferenciais avançadas para "campo de estresse" financeiro
# - Cada empresa é representada por um campo psi(x,t) em 1D (setores/linhas do balanço)
# - A dinâmica do campo é acoplada a parâmetros contábeis/macroeconômicos
# - Um classificador neural em PyTorch aprende P(falência | parâmetros + dinâmica)
# ============================================================

import math
import random
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ------------------------------------------------------------
# 1. Configurações gerais
# ------------------------------------------------------------

@dataclass
class ConfiguracoesSimulacao:
    num_pontos_espaco: int = 50      # discretização em "setores" de risco
    tempo_total: float = 2.0         # horizonte temporal de simulação
    passo_tempo: float = 0.01        # passo de integração temporal
    limiar_falencia: float = 3.0     # ainda usado como referência de estresse
    semente_aleatoria: int = 42


@dataclass
class ConfiguracoesTreinamento:
    tamanho_lote: int = 64
    epocas: int = 20
    taxa_aprendizado: float = 1e-3
    proporcao_treino: float = 0.8


@dataclass
class ParametrosCenario:
    liquidez_relativa: float        # caixa / passivo de curto prazo
    alavancagem: float              # dívida / patrimônio
    volatilidade_setorial: float    # proxy de volatilidade do setor
    taxa_juros: float               # taxa livre de risco normalizada
    tamanho_empresa: float          # log(ativo total) normalizado
    margem_lucro: float             # margem líquida
    indice_governanca: float        # 0-1 (qualidade de governança)


# ------------------------------------------------------------
# 2. Operador diferencial (Laplaciano discreto)
# ------------------------------------------------------------

def construir_laplaciano_1d(num_pontos: int) -> np.ndarray:
    L = np.zeros((num_pontos, num_pontos), dtype=np.float32)
    for i in range(num_pontos):
        L[i, i] = -2.0
        if i > 0:
            L[i, i - 1] = 1.0
        if i < num_pontos - 1:
            L[i, i + 1] = 1.0
    # Contorno refletivo
    L[0, 0] = -1.0
    L[-1, -1] = -1.0
    return L


# ------------------------------------------------------------
# 3. Mapeamento dos parâmetros financeiros para a dinâmica
# ------------------------------------------------------------

def mapear_parametros_para_dinamica(cenario: ParametrosCenario) -> Dict[str, float]:
    # Velocidade de propagação de estresse
    velocidade = 0.5 + 1.5 * (
        0.6 * cenario.volatilidade_setorial + 0.4 * max(0.0, cenario.alavancagem - 1.0)
    )

    # Amortecimento aumenta com liquidez e governança
    amortecimento_base = 0.3 + 0.7 * (cenario.liquidez_relativa + cenario.indice_governanca) / 2.0

    # Potencial confina mais com juros altos e margem baixa
    potencial_base = 0.3 + 0.7 * (
        0.6 * cenario.taxa_juros + 0.4 * max(0.0, 0.3 - cenario.margem_lucro)
    )

    # Fonte de estresse ligada a alavancagem, volatilidade e tamanho
    intensidade_fonte = 0.5 + 2.0 * (
        0.5 * max(0.0, cenario.alavancagem - 1.0)
        + 0.3 * cenario.volatilidade_setorial
        + 0.2 * cenario.tamanho_empresa
    )

    return {
        "velocidade": float(velocidade),
        "amortecimento": float(amortecimento_base),
        "potencial_base": float(potencial_base),
        "intensidade_fonte": float(intensidade_fonte),
    }


# ------------------------------------------------------------
# 4. Índice de risco puramente financeiro (analítico)
# ------------------------------------------------------------

def calcular_indice_risco_financeiro(c: ParametrosCenario) -> float:
    # Normalizações aproximadas usando os ranges usados na base sintética
    # liquidez_relativa ~ [0.3, 2.0] -> liquidez_baixa alto quando liquidez é baixa
    liquidez_norm = (c.liquidez_relativa - 0.3) / (2.0 - 0.3)
    liquidez_norm = float(np.clip(liquidez_norm, 0.0, 1.0))
    liquidez_baixa = 1.0 - liquidez_norm

    # alavancagem ~ [0.5, 5.0]
    alav_norm = (c.alavancagem - 0.5) / (5.0 - 0.5)
    alav_norm = float(np.clip(alav_norm, 0.0, 1.0))

    # volatilidade ~ [0.05, 0.6]
    vol_norm = (c.volatilidade_setorial - 0.05) / (0.6 - 0.05)
    vol_norm = float(np.clip(vol_norm, 0.0, 1.0))

    # taxa_juros ~ [0.02, 0.25]
    juros_norm = (c.taxa_juros - 0.02) / (0.25 - 0.02)
    juros_norm = float(np.clip(juros_norm, 0.0, 1.0))

    # margem_lucro ~ [-0.1, 0.4] -> margem_baixa alto quando margem é baixa
    margem_norm = (c.margem_lucro + 0.1) / (0.4 + 0.1)
    margem_norm = float(np.clip(margem_norm, 0.0, 1.0))
    margem_baixa = 1.0 - margem_norm

    # governança boa reduz risco
    gov_norm = float(np.clip(c.indice_governanca, 0.0, 1.0))
    gov_baixa = 1.0 - gov_norm

    # Combinação linear simples (pode virar algo de paper)
    indice = (
        0.25 * alav_norm +
        0.20 * liquidez_baixa +
        0.15 * vol_norm +
        0.15 * juros_norm +
        0.15 * margem_baixa +
        0.10 * gov_baixa
    )

    return float(np.clip(indice, 0.0, 1.0))


# ------------------------------------------------------------
# 5. Simulação do campo de estresse psi(x,t)
# ------------------------------------------------------------

def simular_campo_estresse(
    cfg: ConfiguracoesSimulacao,
    cenario: ParametrosCenario,
    laplaciano: np.ndarray,
) -> Dict[str, np.ndarray]:
    parametros = mapear_parametros_para_dinamica(cenario)
    v = parametros["velocidade"]
    gamma = parametros["amortecimento"]
    potencial_base = parametros["potencial_base"]
    intensidade_fonte = parametros["intensidade_fonte"]

    num_pontos = cfg.num_pontos_espaco
    dt = cfg.passo_tempo
    num_passos = int(cfg.tempo_total / dt)

    psi = np.zeros(num_pontos, dtype=np.float32)
    momento = np.zeros(num_pontos, dtype=np.float32)

    x = np.linspace(-1.0, 1.0, num_pontos, dtype=np.float32)
    potencial = potencial_base * (1.0 + 0.5 * x ** 2)

    centro_fonte = int(0.7 * num_pontos)
    largura_fonte = int(0.1 * num_pontos)
    mascara_fonte = np.zeros(num_pontos, dtype=np.float32)
    mascara_fonte[
        max(0, centro_fonte - largura_fonte) : min(num_pontos, centro_fonte + largura_fonte)
    ] = 1.0

    serie_max_abs_psi = []
    serie_energia = []

    for passo in range(num_passos):
        t = passo * dt

        pulso_temporal = math.exp(-((t - 0.5 * cfg.tempo_total) ** 2) / (0.1 * cfg.tempo_total) ** 2)
        fonte = intensidade_fonte * mascara_fonte * pulso_temporal

        lap_psi = laplaciano @ psi

        dpsi = momento
        dmomento = (v ** 2) * lap_psi - gamma * momento - potencial * psi + fonte

        psi = psi + dt * dpsi
        momento = momento + dt * dmomento

        max_abs = float(np.max(np.abs(psi)))
        energia = float(np.sum(psi ** 2 + momento ** 2))

        serie_max_abs_psi.append(max_abs)
        serie_energia.append(energia)

    return {
        "serie_max_abs_psi": np.array(serie_max_abs_psi, dtype=np.float32),
        "serie_energia": np.array(serie_energia, dtype=np.float32),
    }


# ------------------------------------------------------------
# 6. Features dinâmicas + rótulo de falência
# ------------------------------------------------------------

def extrair_caracteristicas_dinamicas(
    cfg: ConfiguracoesSimulacao,
    resultados: Dict[str, np.ndarray],
    cenario: ParametrosCenario,
) -> Tuple[np.ndarray, int]:
    max_abs_psi = resultados["serie_max_abs_psi"]
    energia = resultados["serie_energia"]

    max_estresse = float(np.max(max_abs_psi))
    energia_final = float(energia[-1])
    energia_media = float(np.mean(energia))
    estresse_final = float(max_abs_psi[-1])

    indices_acima = np.where(max_abs_psi >= cfg.limiar_falencia)[0]
    if len(indices_acima) > 0:
        tempo_cruzamento = float(indices_acima[0] / len(max_abs_psi))
    else:
        tempo_cruzamento = 1.0

    # Normalizações simples
    max_estresse_norm = max_estresse / (max_estresse + 1.0)
    energia_norm = energia_media / (energia_media + 1.0)

    # Índice de risco puramente financeiro
    indice_fin = calcular_indice_risco_financeiro(cenario)

    # Índice combinado (dinâmica + finanças)
    indice_risco = 0.5 * indice_fin + 0.3 * max_estresse_norm + 0.2 * energia_norm
    indice_risco = float(np.clip(indice_risco, 0.0, 1.0))

    # Regra de falência didática
    falencia = 1 if indice_risco > 0.5 else 0

    # Coloco o índice de risco como mais uma feature
    features = np.array(
        [
            max_estresse,
            energia_final,
            energia_media,
            estresse_final,
            tempo_cruzamento,
            indice_risco,
        ],
        dtype=np.float32,
    )

    return features, falencia


# ------------------------------------------------------------
# 7. Geração de base sintética
# ------------------------------------------------------------

def gerar_cenario_aleatorio() -> ParametrosCenario:
    return ParametrosCenario(
        liquidez_relativa=random.uniform(0.3, 2.0),
        alavancagem=random.uniform(0.5, 5.0),
        volatilidade_setorial=random.uniform(0.05, 0.6),
        taxa_juros=random.uniform(0.02, 0.25),
        tamanho_empresa=random.uniform(0.2, 1.0),
        margem_lucro=random.uniform(-0.1, 0.4),
        indice_governanca=random.uniform(0.0, 1.0),
    )


def construir_base_sintetica(
    cfg_sim: ConfiguracoesSimulacao,
    num_amostras: int = 2000,
) -> Tuple[np.ndarray, np.ndarray]:
    random.seed(cfg_sim.semente_aleatoria)
    np.random.seed(cfg_sim.semente_aleatoria)

    laplaciano = construir_laplaciano_1d(cfg_sim.num_pontos_espaco)

    lista_features_globais = []
    lista_features_dinamica = []
    lista_rotulos = []

    for _ in range(num_amostras):
        cenario = gerar_cenario_aleatorio()

        features_globais = np.array(
            [
                cenario.liquidez_relativa,
                cenario.alavancagem,
                cenario.volatilidade_setorial,
                cenario.taxa_juros,
                cenario.tamanho_empresa,
                cenario.margem_lucro,
                cenario.indice_governanca,
            ],
            dtype=np.float32,
        )

        resultados = simular_campo_estresse(cfg_sim, cenario, laplaciano)
        features_dinam, rotulo = extrair_caracteristicas_dinamicas(cfg_sim, resultados, cenario)

        lista_features_globais.append(features_globais)
        lista_features_dinamica.append(features_dinam)
        lista_rotulos.append(rotulo)

    X_globais = np.stack(lista_features_globais, axis=0)
    X_dinam = np.stack(lista_features_dinamica, axis=0)
    y = np.array(lista_rotulos, dtype=np.int64)

    X = np.concatenate([X_globais, X_dinam], axis=1)

    proporcao_falencia = float(y.mean())
    print(f"Proporção de 'falência' na base sintética: {proporcao_falencia:.3f}")

    return X, y


# ------------------------------------------------------------
# 8. Dataset e modelo neural
# ------------------------------------------------------------

class DatasetRiscoFalencia(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class ClassificadorFalencia(nn.Module):
    def __init__(self, dimensao_entrada: int):
        super().__init__()
        self.rede = nn.Sequential(
            nn.Linear(dimensao_entrada, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rede(x)


# ------------------------------------------------------------
# 9. Treinamento
# ------------------------------------------------------------

def treinar_modelo(
    X: np.ndarray,
    y: np.ndarray,
    cfg_treino: ConfiguracoesTreinamento,
) -> Tuple[ClassificadorFalencia, Dict[str, float]]:
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_amostras = X.shape[0]
    idx = np.arange(num_amostras)
    np.random.shuffle(idx)

    limite_treino = int(cfg_treino.proporcao_treino * num_amostras)
    idx_treino = idx[:limite_treino]
    idx_teste = idx[limite_treino:]

    X_treino, y_treino = X[idx_treino], y[idx_treino]
    X_teste, y_teste = X[idx_teste], y[idx_teste]

    ds_treino = DatasetRiscoFalencia(X_treino, y_treino)
    ds_teste = DatasetRiscoFalencia(X_teste, y_teste)

    dl_treino = DataLoader(ds_treino, batch_size=cfg_treino.tamanho_lote, shuffle=True)
    dl_teste = DataLoader(ds_teste, batch_size=cfg_treino.tamanho_lote, shuffle=False)

    modelo = ClassificadorFalencia(dimensao_entrada=X.shape[1]).to(dispositivo)
    criterio = nn.CrossEntropyLoss()
    otimizador = torch.optim.Adam(modelo.parameters(), lr=cfg_treino.taxa_aprendizado)

    for epoca in range(cfg_treino.epocas):
        modelo.train()
        perda_acumulada = 0.0
        corretos = 0
        total = 0

        for lotes_X, lotes_y in dl_treino:
            lotes_X = lotes_X.to(dispositivo)
            lotes_y = lotes_y.to(dispositivo)

            otimizador.zero_grad()
            logits = modelo(lotes_X)
            perda = criterio(logits, lotes_y)
            perda.backward()
            otimizador.step()

            perda_acumulada += float(perda.item()) * lotes_X.size(0)
            pred = torch.argmax(logits, dim=1)
            corretos += int((pred == lotes_y).sum().item())
            total += int(lotes_X.size(0))

        perda_media = perda_acumulada / total
        acuracia_treino = corretos / total

        modelo.eval()
        corretos_teste = 0
        total_teste = 0
        with torch.no_grad():
            for lotes_X, lotes_y in dl_teste:
                lotes_X = lotes_X.to(dispositivo)
                lotes_y = lotes_y.to(dispositivo)
                logits = modelo(lotes_X)
                pred = torch.argmax(logits, dim=1)
                corretos_teste += int((pred == lotes_y).sum().item())
                total_teste += int(lotes_X.size(0))
        acuracia_teste = corretos_teste / total_teste

        print(
            f"Época {epoca+1:02d}/{cfg_treino.epocas} | "
            f"Perda treino: {perda_media:.4f} | "
            f"Acurácia treino: {acuracia_treino:.3f} | "
            f"Acurácia teste: {acuracia_teste:.3f}"
        )

    metricas = {
        "acuracia_treino": acuracia_treino,
        "acuracia_teste": acuracia_teste,
    }

    return modelo, metricas


# ------------------------------------------------------------
# 10. Previsão de probabilidade de falência
# ------------------------------------------------------------

def prever_probabilidade_falencia(
    modelo: ClassificadorFalencia,
    cfg_sim: ConfiguracoesSimulacao,
    parametros: ParametrosCenario,
) -> float:
    laplaciano = construir_laplaciano_1d(cfg_sim.num_pontos_espaco)
    resultados = simular_campo_estresse(cfg_sim, parametros, laplaciano)

    # Features dinâmicas (sem rótulo aqui)
    max_abs_psi = resultados["serie_max_abs_psi"]
    energia = resultados["serie_energia"]

    max_estresse = float(np.max(max_abs_psi))
    energia_media = float(np.mean(energia))

    max_estresse_norm = max_estresse / (max_estresse + 1.0)
    energia_norm = energia_media / (energia_media + 1.0)

    indice_fin = calcular_indice_risco_financeiro(parametros)
    indice_risco_analitico = 0.5 * indice_fin + 0.3 * max_estresse_norm + 0.2 * energia_norm
    indice_risco_analitico = float(np.clip(indice_risco_analitico, 0.0, 1.0))

    features_globais = np.array(
        [
            parametros.liquidez_relativa,
            parametros.alavancagem,
            parametros.volatilidade_setorial,
            parametros.taxa_juros,
            parametros.tamanho_empresa,
            parametros.margem_lucro,
            parametros.indice_governanca,
        ],
        dtype=np.float32,
    )

    # Reaproveito a mesma construção de features dinâmicas da base,
    # mas aqui sem recalcular o rótulo.
    cfg_tmp = cfg_sim
    # Só para reaproveitar a função, gero um cenario idêntico:
    resultados2 = resultados
    features_dinam, _ = extrair_caracteristicas_dinamicas(cfg_tmp, resultados2, parametros)

    X = np.concatenate([features_globais, features_dinam], axis=0)[None, :]
    tensor_X = torch.from_numpy(X.astype(np.float32))

    modelo.eval()
    with torch.no_grad():
        logits = modelo(tensor_X)
        probabilidades = torch.softmax(logits, dim=1).cpu().numpy()[0]

    prob_rede = float(probabilidades[1])

    # Mistura: mais peso no índice analítico para garantir sensibilidade
    prob_final = 0.3 * prob_rede + 0.7 * indice_risco_analitico
    prob_final = float(np.clip(prob_final, 0.0, 1.0))

    return prob_final


# ------------------------------------------------------------
# 11. Exemplo de uso
# ------------------------------------------------------------

def exemplo_uso():
    cfg_sim = ConfiguracoesSimulacao()
    cfg_treino = ConfiguracoesTreinamento()

    print("Gerando base sintética de cenários...")
    X, y = construir_base_sintetica(cfg_sim, num_amostras=1000)

    print("Treinando classificador neural...")
    modelo, metricas = treinar_modelo(X, y, cfg_treino)

    print("Métricas finais (didáticas):", metricas)

    empresa_solida = ParametrosCenario(
        liquidez_relativa=1.5,
        alavancagem=1.0,
        volatilidade_setorial=0.1,
        taxa_juros=0.05,
        tamanho_empresa=0.8,
        margem_lucro=0.25,
        indice_governanca=0.9,
    )

    empresa_fragil = ParametrosCenario(
        liquidez_relativa=0.5,
        alavancagem=4.0,
        volatilidade_setorial=0.5,
        taxa_juros=0.18,
        tamanho_empresa=0.4,
        margem_lucro=-0.05,
        indice_governanca=0.2,
    )

    prob_solida = prever_probabilidade_falencia(modelo, cfg_sim, empresa_solida)
    prob_fragil = prever_probabilidade_falencia(modelo, cfg_sim, empresa_fragil)

    print(f"Probabilidade estimada de falência (empresa sólida): {prob_solida:.3f}")
    print(f"Probabilidade estimada de falência (empresa frágil): {prob_fragil:.3f}")


if __name__ == "__main__":
    exemplo_uso()
