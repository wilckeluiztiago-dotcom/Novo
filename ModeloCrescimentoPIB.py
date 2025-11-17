# ============================================================
# Modelo Dinâmico de Crescimento do PIB do Brasil
# Capital físico + capital humano + tecnologia + demografia + dívida
# Unidades:
#   - PIB e capital: trilhões de R$
#   - População: milhões de pessoas
#   - Dívida: razão dívida/PIB (adimensional)
#   - Autor : Luiz Tiago Wilcke
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# ------------------------------------------------------------
# 1) Parâmetros do modelo
# ------------------------------------------------------------
@dataclass
class ParametrosModelo:
    # Produção Cobb-Douglas estendida: Y = A K^α H^ψ L^(1-α-ψ)
    alpha: float = 0.33          # elasticidade do capital físico
    psi: float = 0.15            # elasticidade do capital humano

    # Acumulação de capital físico (Solow)
    taxa_poupanca_base: float = 0.18   # fração do PIB investida
    taxa_depreciacao: float = 0.05     # depreciação anual do capital

    # Tecnologia / inovação (Romer simplificado + ruído)
    taxa_inovacao: float = 0.03        # intensidade de inovação
    obsolescencia: float = 0.01        # obsolescência tecnológica
    fracao_inovacao: float = 0.02      # fração do PIB em P&D
    sigma_tecnologia: float = 0.03     # intensidade do choque estocástico em A

    # Capital humano (Lucas)
    taxa_educacao: float = 0.06        # taxa de acumulação de capital humano
    u_produtivo: float = 0.8           # fração do tempo em produção (1-u em educação)

    # Demografia (logística)
    taxa_crescimento_pop: float = 0.007   # ~0,7% ao ano
    capacidade_populacional: float = 260.0  # milhões (capacidade de longo prazo)

    # Dívida pública (razão dívida/PIB com reversão à média)
    kappa_divida: float = 0.1           # velocidade de ajuste da dívida
    divida_alvo: float = 0.8            # dívida/PIB de longo prazo (80%)
    gamma_divida: float = 0.02          # quanto dívida alta reduz a poupança

    # Ruído estocástico
    usar_ruido: bool = True
    seed: int = 42


# ------------------------------------------------------------
# 2) Dinâmica do sistema (EDOs + ruído em A)
# ------------------------------------------------------------
def derivadas(estado, t_atual, params: ParametrosModelo):
    """
    estado = [capital_fisico, capital_humano, tecnologia, populacao, divida]
    """
    capital_fisico, capital_humano, tecnologia, populacao, divida = estado

    # Função de produção Cobb-Douglas estendida
    pib = (
        tecnologia
        * (capital_fisico ** params.alpha)
        * (capital_humano ** params.psi)
        * (populacao ** (1.0 - params.alpha - params.psi))
    )

    # Poupança efetiva com feedback da dívida
    # Se a dívida > alvo, a taxa de poupança cai (maior carga de juros / incerteza)
    poupanca_efetiva = params.taxa_poupanca_base - params.gamma_divida * max(
        0.0, divida - params.divida_alvo
    )
    # Mantém em um intervalo razoável
    poupanca_efetiva = max(0.05, min(poupanca_efetiva, 0.30))

    investimento = poupanca_efetiva * pib
    investimento_PeD = params.fracao_inovacao * pib

    # 2.1) Equação de Solow para capital físico
    d_capital_fisico = investimento - params.taxa_depreciacao * capital_fisico

    # 2.2) Equação de Lucas para capital humano
    # dh/dt = taxa_educacao * h * (1 - u_produtivo)
    d_capital_humano = params.taxa_educacao * capital_humano * (1.0 - params.u_produtivo)

    # 2.3) Equação de Romer simplificada para tecnologia
    # dA/dt = taxa_inovacao * (I_PeD / PIB) * A - obsolescencia * A
    # Aqui (I_PeD / PIB) = fracao_inovacao, constante; fica proporcional a A.
    d_tecnologia = params.taxa_inovacao * params.fracao_inovacao * tecnologia \
                   - params.obsolescencia * tecnologia

    # 2.4) Demografia logística
    # dL/dt = r L (1 - L / L_max)
    d_populacao = (
        params.taxa_crescimento_pop
        * populacao
        * (1.0 - populacao / params.capacidade_populacional)
    )

    # 2.5) Dinâmica simplificada da dívida pública (razão dívida/PIB)
    # dD/dt = kappa (D_alvo - D)
    d_divida = params.kappa_divida * (params.divida_alvo - divida)

    derivadas_vetor = np.array([
        d_capital_fisico,
        d_capital_humano,
        d_tecnologia,
        d_populacao,
        d_divida,
    ])

    return derivadas_vetor, pib


# ------------------------------------------------------------
# 3) Integração numérica (RK4 + ruído em A)
# ------------------------------------------------------------
def simular_pib_brasil(anos=40.0, dt=0.25, params: ParametrosModelo | None = None):
    """
    Simula o modelo ao longo de 'anos' com passo 'dt' (em anos).
    Retorna:
        tempos, matriz_estado, serie_pib, serie_crescimento_anual
    """
    if params is None:
        params = ParametrosModelo()

    passos = int(anos / dt) + 1
    tempos = np.linspace(0.0, anos, passos)

    # estado = [K, H, A, L, D]
    matriz_estado = np.zeros((passos, 5))
    pib = np.zeros(passos)

    # Condições iniciais aproximadas para o Brasil (valores ilustrativos)
    capital_fisico_inicial = 20.0   # ~20 trilhões em "capital efetivo" (estoque)
    capital_humano_inicial = 1.0    # índice normalizado
    tecnologia_inicial = 0.25       # calibrado para PIB ~10 trilhões
    populacao_inicial = 203.0       # milhões de pessoas
    divida_inicial = 0.90           # dívida/PIB inicial ~90%

    matriz_estado[0, :] = [
        capital_fisico_inicial,
        capital_humano_inicial,
        tecnologia_inicial,
        populacao_inicial,
        divida_inicial,
    ]

    # PIB inicial
    _, pib[0] = derivadas(matriz_estado[0, :], tempos[0], params)

    # Gerador de números aleatórios para os choques em A(t)
    rng = np.random.default_rng(params.seed)

    for i in range(1, passos):
        estado_atual = matriz_estado[i - 1, :].copy()
        t = tempos[i - 1]

        # Passo de Runge–Kutta 4
        k1, pib_k1 = derivadas(estado_atual, t, params)
        k2, _      = derivadas(estado_atual + 0.5 * dt * k1, t + 0.5 * dt, params)
        k3, _      = derivadas(estado_atual + 0.5 * dt * k2, t + 0.5 * dt, params)
        k4, pib_k4 = derivadas(estado_atual + dt * k3,       t + dt,       params)

        estado_prox = estado_atual + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Ruído estocástico na tecnologia A(t) (SDE: dA = ... + sigma*A dW)
        if params.usar_ruido:
            dW = np.sqrt(dt) * rng.normal()
            estado_prox[2] += params.sigma_tecnologia * estado_prox[2] * dW

        # Garante que variáveis críticas não fiquem negativas
        estado_prox[0] = max(1e-6, estado_prox[0])   # capital_fisico
        estado_prox[1] = max(1e-6, estado_prox[1])   # capital_humano
        estado_prox[2] = max(1e-6, estado_prox[2])   # tecnologia
        estado_prox[3] = max(1e-6, estado_prox[3])   # populacao

        matriz_estado[i, :] = estado_prox

        # Recalcula o PIB no novo estado
        _, pib[i] = derivadas(estado_prox, tempos[i], params)

    # Taxa de crescimento anualizada aproximada:
    # g_t ≈ (PIB_t+Δt - PIB_t) / (PIB_t * Δt)
    crescimento = (pib[1:] - pib[:-1]) / (pib[:-1] * dt)

    return tempos, matriz_estado, pib, crescimento


# ------------------------------------------------------------
# 4) Execução e gráficos
# ------------------------------------------------------------
if __name__ == "__main__":
    params = ParametrosModelo()

    anos_simulacao = 40.0   # simular 40 anos para frente
    dt = 0.25               # passo trimestral (0,25 ano)

    tempos, estado, pib, crescimento = simular_pib_brasil(
        anos=anos_simulacao, dt=dt, params=params
    )

    capital_fisico = estado[:, 0]
    capital_humano = estado[:, 1]
    tecnologia = estado[:, 2]
    populacao = estado[:, 3]
    divida = estado[:, 4]

    crescimento_medio_anual = np.mean(crescimento) * 100.0

    print("===============================================")
    print("PIB inicial (trilhões de R$): ", pib[0])
    print("PIB final   (trilhões de R$): ", pib[-1])
    print("Crescimento médio anual simulado: "
          f"{crescimento_medio_anual:.2f}% ao ano")
    print("Dívida/PIB inicial: ", divida[0])
    print("Dívida/PIB final:   ", divida[-1])
    print("===============================================")

    # ----- Gráfico do PIB -----
    plt.figure(figsize=(10, 6))
    plt.plot(tempos, pib, label="PIB (trilhões de R$)")
    plt.xlabel("Tempo (anos)")
    plt.ylabel("PIB (trilhões de R$)")
    plt.title("Trajetória simulada do PIB real do Brasil")
    plt.grid(True)
    plt.legend()

    # ----- Gráfico das taxas de crescimento -----
    plt.figure(figsize=(10, 6))
    plt.plot(tempos[1:], crescimento * 100.0, label="Crescimento do PIB (% a.a.)")
    plt.axhline(np.mean(crescimento) * 100.0, linestyle="--",
                label="Média do período")
    plt.xlabel("Tempo (anos)")
    plt.ylabel("Taxa de crescimento do PIB (% ao ano)")
    plt.title("Taxa de crescimento do PIB (simulação)")
    plt.grid(True)
    plt.legend()

    # ----- Gráfico das principais variáveis de estado -----
    plt.figure(figsize=(10, 6))
    plt.plot(tempos, capital_fisico, label="Capital físico (K)")
    plt.plot(tempos, capital_humano, label="Capital humano (H)")
    plt.plot(tempos, tecnologia, label="Tecnologia (A)")
    plt.plot(tempos, populacao, label="População (milhões)")
    plt.xlabel("Tempo (anos)")
    plt.ylabel("Nível (unidades do modelo)")
    plt.title("Trajetória das variáveis de estado")
    plt.grid(True)
    plt.legend()

    # ----- Dívida/PIB -----
    plt.figure(figsize=(10, 6))
    plt.plot(tempos, divida, label="Dívida/PIB")
    plt.axhline(params.divida_alvo, linestyle="--", label="Dívida alvo")
    plt.xlabel("Tempo (anos)")
    plt.ylabel("Razão dívida/PIB")
    plt.title("Dinâmica da Dívida Pública (razão dívida/PIB)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
