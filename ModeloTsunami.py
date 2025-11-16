# ============================================================
# Modelo 1D de Tsunami em Águas Rasas (Equações Diferenciais)
#   - Equação: d²η/dt² = ∂/∂x [ c(x)² ∂η/∂x ]
#   - Discretização em x -> grande sistema de EDOs
#   - Integração explícita no tempo
#
# Autor: Luiz Tiago Wilcke 
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# ------------------------------------------------------------
# 1) Parâmetros do modelo
# ------------------------------------------------------------

@dataclass
class ParametrosTsunami:
    comprimento_dominio_km: float = 400.0    # extensão 1D (km)
    n_pontos: int = 401                      # número de pontos da malha
    profundidade_offshore_m: float = 4000.0  # profundidade no mar profundo (m)
    profundidade_costa_m: float = 50.0       # profundidade perto da costa (m)
    g: float = 9.81                          # gravidade (m/s²)

    # Condição inicial (pulso gaussiano)
    amplitude_inicial_m: float = 1.0         # elevação inicial máxima (m)
    posicao_pulso_km: float = 300.0          # onde ocorre o "terremoto" (km)
    largura_pulso_km: float = 20.0           # largura do pulso (km)

    # Integração temporal
    tempo_simulacao_s: float = 7200.0        # 2 horas de simulação (s)
    dt_s: float = 2.0                        # passo de tempo (s)

    # Saída
    salvar_cada_n_passos: int = 50           # salvar estado a cada N passos


# ------------------------------------------------------------
# 2) Construção da malha e batimetria
# ------------------------------------------------------------

def construir_malha_e_batimetria(p: ParametrosTsunami):
    """
    Constrói o eixo espacial (em km) e a profundidade h(x) em metros.
    Fazemos uma transição linear entre mar profundo e costa rasa.
    """
    # Eixo em km
    x_km = np.linspace(0.0, p.comprimento_dominio_km, p.n_pontos)
    # Converter para metros para cálculos internos
    x_m = x_km * 1000.0

    # Batimetria simples: profundidade varia linearmente da costa ao offshore
    h_m = np.linspace(p.profundidade_costa_m,
                      p.profundidade_offshore_m,
                      p.n_pontos)

    # Velocidade de onda longa: c(x) = sqrt(g * h(x))
    c_m_s = np.sqrt(p.g * h_m)

    # c² em cada ponto
    c2 = c_m_s ** 2

    # c² nas faces i+1/2 (média entre pontos vizinhos)
    c2_faces = 0.5 * (c2[:-1] + c2[1:])

    return x_km, x_m, h_m, c2, c2_faces


# ------------------------------------------------------------
# 3) Condição inicial do tsunami (pulso gaussiano)
# ------------------------------------------------------------

def condicao_inicial_eta(x_km: np.ndarray, p: ParametrosTsunami):
    """
    Pulso gaussiano representando a deformação inicial da superfície.
    """
    x0 = p.posicao_pulso_km
    sigma = p.largura_pulso_km

    eta0 = p.amplitude_inicial_m * np.exp(
        -0.5 * ((x_km - x0) / sigma) ** 2
    )
    return eta0


# ------------------------------------------------------------
# 4) Operador espacial: calcula dv/dt dado η(t)
# ------------------------------------------------------------

def calcular_dvdt(eta: np.ndarray, c2_faces: np.ndarray, dx_m: float):
    """
    Calcula dv_i/dt usando a discretização:
      dv_i/dt = (1/dx) [ c_{i+1/2}² (eta_{i+1}-eta_i)/dx
                        - c_{i-1/2}² (eta_i-eta_{i-1})/dx ]
    Assumimos condições de contorno simples (derivada ~ 0) nas bordas.
    """
    n = eta.size
    dvdt = np.zeros_like(eta)

    # Derivada espacial nas faces internas
    # d_eta_dx_face[i] ≈ (eta[i+1] - eta[i]) / dx
    d_eta_dx_faces = (eta[1:] - eta[:-1]) / dx_m  # tamanho n-1

    # Fluxos em cada face: F = c² * d_eta/dx
    fluxos = c2_faces * d_eta_dx_faces  # tamanho n-1

    # dvdt interior: (F[i] - F[i-1]) / dx
    dvdt[1:-1] = (fluxos[1:] - fluxos[:-1]) / dx_m

    # Bordas (condição de contorno reflexiva aproximada: fluxo ~ 0)
    dvdt[0] = (fluxos[0] - 0.0) / dx_m
    dvdt[-1] = (0.0 - fluxos[-1]) / dx_m

    return dvdt


# ------------------------------------------------------------
# 5) Integração temporal
# ------------------------------------------------------------

def simular_tsunami(p: ParametrosTsunami):
    # Malha e batimetria
    x_km, x_m, h_m, c2, c2_faces = construir_malha_e_batimetria(p)
    dx_m = x_m[1] - x_m[0]

    # Condições iniciais
    eta = condicao_inicial_eta(x_km, p)  # elevação da superfície
    v = np.zeros_like(eta)               # derivada temporal inicial (dη/dt = 0)

    # Número de passos no tempo
    n_passos = int(p.tempo_simulacao_s / p.dt_s)

    # Listas para salvar estados (para plotar depois)
    tempos_salvos = []
    etas_salvas = []

    tempos_salvos.append(0.0)
    etas_salvas.append(eta.copy())

    # Loop temporal (esquema de Euler explícito simples)
    for n in range(1, n_passos + 1):
        # Calcular dv/dt a partir de eta
        dvdt = calcular_dvdt(eta, c2_faces, dx_m)

        # Atualizar v e eta
        v = v + p.dt_s * dvdt
        eta = eta + p.dt_s * v

        # Guardar de tempos em tempos
        if n % p.salvar_cada_n_passos == 0:
            t_atual = n * p.dt_s
            tempos_salvos.append(t_atual)
            etas_salvas.append(eta.copy())
            print(f"Salvando t = {t_atual/60:.1f} min")

    return x_km, h_m, np.array(tempos_salvos), np.array(etas_salvas)


# ------------------------------------------------------------
# 6) Visualização dos resultados
# ------------------------------------------------------------

def plotar_resultados(x_km, h_m, tempos, etas):
    """
    Plota a batimetria e alguns snapshots da superfície da água.
    """
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # --- Batimetria ---
    axs[0].fill_between(x_km, -h_m, 0, alpha=0.7)
    axs[0].set_ylabel("Profundidade (m)")
    axs[0].set_title("Batimetria (perfil 1D)")
    axs[0].invert_yaxis()
    axs[0].grid(True, linestyle="--", alpha=0.3)

    # --- Elevação da superfície em diferentes tempos ---
    axs[1].set_title("Propagação da onda de tsunami")
    axs[1].set_xlabel("Distância ao longo da costa (km)")
    axs[1].set_ylabel("Elevação da superfície (m)")
    axs[1].grid(True, linestyle="--", alpha=0.3)

    n_snapshots = min(5, etas.shape[0])  # plota até 5 snapshots
    indices = np.linspace(0, etas.shape[0] - 1, n_snapshots, dtype=int)

    for idx in indices:
        t = tempos[idx]
        axs[1].plot(x_km, etas[idx], label=f"t = {t/60:.1f} min")

    axs[1].legend()
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 7) Execução principal
# ------------------------------------------------------------

if __name__ == "__main__":
    p = ParametrosTsunami(
        comprimento_dominio_km=400.0,
        n_pontos=401,
        profundidade_offshore_m=4000.0,
        profundidade_costa_m=50.0,
        amplitude_inicial_m=1.0,
        posicao_pulso_km=300.0,
        largura_pulso_km=20.0,
        tempo_simulacao_s=7200.0,  # 2h
        dt_s=2.0,
        salvar_cada_n_passos=50
    )

    x_km, h_m, tempos, etas = simular_tsunami(p)
    plotar_resultados(x_km, h_m, tempos, etas)
