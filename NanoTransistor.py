# ============================================================
# NANOTRANSISTOR 2D — Poisson + Density-Gradient + Corrente
# Autor: Luiz Tiago Wilcke (LT)
#
# - Domínio 2D: x -> fonte → canal → dreno
#               y -> vertical (canal de Si + óxido + gate)
# - Resolve Poisson 2D acoplado à densidade de elétrons
# - Inclui correção quântica via modelo density-gradient
# - Calcula mapa de potencial, densidade e corrente de elétrons
#
# OBS:
#   • Modelo físico-realista didático (não um TCAD comercial).
#   • Discretização por diferenças finitas em malha retangular.
# ============================================================

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from dataclasses import dataclass

# ------------------------------------------------------------
# 1. Constantes físicas
# ------------------------------------------------------------
q = 1.602e-19          # carga do elétron [C]
k_B = 1.381e-23        # constante de Boltzmann [J/K]
T = 300.0              # temperatura [K]
V_T = k_B * T / q      # tensão térmica [V]
hbar = 1.055e-34       # constante de Planck reduzida [J·s]
m0 = 9.109e-31         # massa do elétron no vácuo [kg]

epsilon0 = 8.854e-12   # permissividade do vácuo [F/m]


# ------------------------------------------------------------
# 2. Configurações do nanotransistor 2D
# ------------------------------------------------------------
@dataclass
class ConfiguracoesNanotransistor2D:
    # Geometria (nm)
    comprimento_canal_nm: float = 30.0    # comprimento total fonte–dreno
    altura_silicio_nm: float = 10.0       # espessura da região de Si
    altura_oxido_nm: float = 5.0          # espessura do óxido (gate oxide)

    # Malha numérica
    num_pontos_x: int = 81
    num_pontos_y: int = 51

    # Tensões [V]
    tensao_fonte: float = 0.0
    tensao_dreno: float = 0.6
    tensao_gate: float = 0.8
    tensao_substrato: float = 0.0

    # Propriedades do material
    epsilon_rel_silicio: float = 11.7
    epsilon_rel_oxido: float = 3.9
    massa_efetiva_relativa: float = 0.26   # Si
    mobilidade_eletrons: float = 0.05      # [m^2/(V·s)] valor didático
    densidade_intrinseca: float = 1e16     # [m^-3] valor didático

    # Dopagem (apenas em Si)
    dopagem_doadores: float = 1e24         # [m^-3] n+ fonte/dreno/canal (simplif.)
    dopagem_aceptores: float = 0.0         # [m^-3]

    # Iterações de acoplamento
    max_iteracoes: int = 80
    tolerancia: float = 1e-3               # tolerância relativa para φ e n
    fator_relaxamento: float = 0.5         # underrelaxation para estabilidade


# ------------------------------------------------------------
# 3. Construção da malha 2D
# ------------------------------------------------------------
def criar_malha_2d(cfg: ConfiguracoesNanotransistor2D):
    Lx = cfg.comprimento_canal_nm * 1e-9
    Ly = (cfg.altura_silicio_nm + cfg.altura_oxido_nm) * 1e-9

    x = np.linspace(0.0, Lx, cfg.num_pontos_x)
    y = np.linspace(0.0, Ly, cfg.num_pontos_y)  # y=0 (substrato) → y=Ly (gate)

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    X, Y = np.meshgrid(x, y)  # formato (Ny, Nx)
    return x, y, X, Y, dx, dy


# ------------------------------------------------------------
# 4. Laplaciano 2D (diferenças finitas)
# ------------------------------------------------------------
def montar_laplaciano_2d(num_x: int, num_y: int, dx: float, dy: float):
    """
    Monta o operador Laplaciano 2D em malha retangular:
        ∇²f ≈ d²f/dx² + d²f/dy²
    com esquema de 5 pontos.
    """
    Nx, Ny = num_x, num_y
    N = Nx * Ny

    L = sp.lil_matrix((N, N))
    coef_x = 1.0 / dx**2
    coef_y = 1.0 / dy**2

    for j in range(Ny):
        for i in range(Nx):
            k = j * Nx + i

            # centro
            L[k, k] = -2.0 * (coef_x + coef_y)

            # vizinho x-1
            if i > 0:
                L[k, k - 1] = coef_x
            # vizinho x+1
            if i < Nx - 1:
                L[k, k + 1] = coef_x
            # vizinho y-1
            if j > 0:
                L[k, k - Nx] = coef_y
            # vizinho y+1
            if j < Ny - 1:
                L[k, k + Nx] = coef_y

    return L.tocsc()


def aplicar_condicoes_contorno_dirichlet(A: sp.csc_matrix,
                                         b: np.ndarray,
                                         mascara_bc: np.ndarray,
                                         valores_bc: np.ndarray):
    """
    Impõe condições de contorno de Dirichlet:
      φ(k) = valores_bc(k) para k onde mascara_bc(k) = True
    """
    A_mod = A.tolil()
    for k in np.where(mascara_bc)[0]:
        A_mod[k, :] = 0.0
        A_mod[k, k] = 1.0
        b[k] = valores_bc[k]
    return A_mod.tocsc(), b


# ------------------------------------------------------------
# 5. Potencial quântico — density-gradient 2D
# ------------------------------------------------------------
def calcular_potencial_quantico_density_gradient_2d(
    densidade_flat: np.ndarray,
    laplaciano: sp.csc_matrix,
    cfg: ConfiguracoesNanotransistor2D
) -> np.ndarray:
    """
    Q_n(x,y) = - (ħ² / (12 m* )) ( ∇² √n / √n )
    Implementação discreta em 2D usando o mesmo Laplaciano do potencial.
    """
    massa_efetiva = cfg.massa_efetiva_relativa * m0

    densidade_clipada = np.clip(densidade_flat, 1e10, None)
    raiz_n = np.sqrt(densidade_clipada)

    laplaciano_raiz_n = laplaciano @ raiz_n

    Q_n = - (hbar**2) / (12.0 * massa_efetiva) * (laplaciano_raiz_n / raiz_n)
    return Q_n  # [J]


# ------------------------------------------------------------
# 6. Construção dos mapas de material e dopagem
# ------------------------------------------------------------
def construir_mapas_materiais_e_dopagem(
    cfg: ConfiguracoesNanotransistor2D,
    X: np.ndarray,
    Y: np.ndarray
):
    """
    Define:
      • máscara de silício
      • máscara de óxido
      • mapas de ε_relativo e dopagem doadora
    """
    Ny, Nx = Y.shape

    altura_silicio_m = cfg.altura_silicio_nm * 1e-9

    mascara_silicio = (Y <= altura_silicio_m)
    mascara_oxido = (Y > altura_silicio_m)

    epsilon_rel = np.zeros_like(X)
    epsilon_rel[mascara_silicio] = cfg.epsilon_rel_silicio
    epsilon_rel[mascara_oxido] = cfg.epsilon_rel_oxido

    dopagem_doadores = np.zeros_like(X)
    dopagem_doadores[mascara_silicio] = cfg.dopagem_doadores

    return mascara_silicio, mascara_oxido, epsilon_rel, dopagem_doadores


# ------------------------------------------------------------
# 7. Máscaras de contato (fonte, dreno, gate, substrato)
# ------------------------------------------------------------
def construir_mascaras_contatos(cfg: ConfiguracoesNanotransistor2D,
                                X: np.ndarray,
                                Y: np.ndarray,
                                mascara_silicio: np.ndarray):
    Ny, Nx = X.shape

    mascara_fonte = np.zeros_like(X, dtype=bool)
    mascara_dreno = np.zeros_like(X, dtype=bool)
    mascara_gate = np.zeros_like(X, dtype=bool)
    mascara_substrato = np.zeros_like(X, dtype=bool)

    # Fonte e dreno: laterais no silício
    mascara_fonte[:, 0] = mascara_silicio[:, 0]
    mascara_dreno[:, -1] = mascara_silicio[:, -1]

    # Gate: topo (j = Ny-1) sobre o óxido
    mascara_gate[-1, :] = True

    # Substrato: base do silício (y=0)
    mascara_substrato[0, :] = mascara_silicio[0, :]

    return mascara_fonte, mascara_dreno, mascara_gate, mascara_substrato


# ------------------------------------------------------------
# 8. Simulação acoplada Poisson + density-gradient
# ------------------------------------------------------------
def simular_nanotransistor_2d(cfg: ConfiguracoesNanotransistor2D):
    # --- malha ---
    x, y, X, Y, dx, dy = criar_malha_2d(cfg)
    Ny, Nx = Y.shape
    N = Nx * Ny

    # --- materiais e dopagem ---
    mascara_silicio, mascara_oxido, epsilon_rel, dopagem_doadores = \
        construir_mapas_materiais_e_dopagem(cfg, X, Y)

    # --- contatos ---
    mascara_fonte, mascara_dreno, mascara_gate, mascara_substrato = \
        construir_mascaras_contatos(cfg, X, Y, mascara_silicio)

    # --- campos auxiliares achatados ---
    epsilon_rel_flat = epsilon_rel.ravel()
    dopagem_flat = dopagem_doadores.ravel()
    mascara_silicio_flat = mascara_silicio.ravel()
    mascara_fonte_flat = mascara_fonte.ravel()
    mascara_dreno_flat = mascara_dreno.ravel()
    mascara_gate_flat = mascara_gate.ravel()
    mascara_substrato_flat = mascara_substrato.ravel()

    # --- operador Laplaciano 2D ---
    L = montar_laplaciano_2d(Nx, Ny, dx, dy)

    # Vamos tratar ε aproximadamente constante por região (didático):
    # usamos ε do silício para toda a equação de Poisson
    epsilon_silicio = cfg.epsilon_rel_silicio * epsilon0

    # --- inicializações ---
    # Potencial inicial: interpolação linear fonte→dreno
    phi_2d = np.zeros((Ny, Nx))
    for i in range(Nx):
        alpha = i / (Nx - 1)
        phi_2d[:, i] = (1 - alpha) * cfg.tensao_fonte + alpha * cfg.tensao_dreno
    # teto (gate) vai ser puxado depois como Dirichlet

    phi_flat = phi_2d.ravel()

    # Densidade inicial: quase-neutralidade no silício
    n_flat = np.zeros(N)
    n_flat[mascara_silicio_flat] = dopagem_flat[mascara_silicio_flat]
    n_flat[~mascara_silicio_flat] = cfg.densidade_intrinseca

    # Difusão de elétrons (relação de Einstein)
    D_n = cfg.mobilidade_eletrons * V_T

    # Índices de fonte para referência de potencial
    indices_fonte = np.where(mascara_fonte_flat & mascara_silicio_flat)[0]
    if len(indices_fonte) == 0:
        raise RuntimeError("Nenhum nó de fonte encontrado no silício.")

    # --- laço de acoplamento não-linear ---
    for it in range(cfg.max_iteracoes):
        # 1) Potencial quântico Q_n (J)
        Q_n_J = calcular_potencial_quantico_density_gradient_2d(
            n_flat, L, cfg
        )
        Q_n_V = Q_n_J / q  # em Volts (energia/q)

        # 2) Equação de Poisson: -ε ∇² φ = ρ
        rho = np.zeros(N)
        # apenas no silício (no óxido consideramos ρ ≅ 0)
        rho[mascara_silicio_flat] = q * (
            - n_flat[mascara_silicio_flat]
            + dopagem_flat[mascara_silicio_flat]
            - cfg.dopagem_aceptores
        )

        A = -epsilon_silicio * L.copy()
        b = rho.copy()

        # 3) Condições de contorno (Dirichlet)
        mascara_bc = np.zeros(N, dtype=bool)
        valores_bc = np.zeros(N)

        # Fonte
        mascara_bc[mascara_fonte_flat] = True
        valores_bc[mascara_fonte_flat] = cfg.tensao_fonte

        # Dreno
        mascara_bc[mascara_dreno_flat] = True
        valores_bc[mascara_dreno_flat] = cfg.tensao_dreno

        # Gate (topo)
        mascara_bc[mascara_gate_flat] = True
        valores_bc[mascara_gate_flat] = cfg.tensao_gate

        # Substrato (base)
        mascara_bc[mascara_substrato_flat] = True
        valores_bc[mascara_substrato_flat] = cfg.tensao_substrato

        # Aplica BCs
        A_bc, b_bc = aplicar_condicoes_contorno_dirichlet(A, b, mascara_bc, valores_bc)

        # Resolve Poisson
        phi_novo_flat = spla.spsolve(A_bc, b_bc)

        # Relaxação
        phi_flat = (1 - cfg.fator_relaxamento) * phi_flat \
                   + cfg.fator_relaxamento * phi_novo_flat

        # 4) Atualiza densidade de elétrons (aprox. Boltzmann + termo quântico)
        # Referência de potencial: média na fonte
        phi_fonte_media = np.mean(phi_flat[indices_fonte])

        expo = (phi_flat + Q_n_V - phi_fonte_media) / V_T
        expo = np.clip(expo, -40.0, 40.0)

        n_novo_flat = np.zeros_like(n_flat)
        # no silício, dominado por dopagem
        n_novo_flat[mascara_silicio_flat] = \
            dopagem_flat[mascara_silicio_flat] * np.exp(expo[mascara_silicio_flat])
        # no óxido, densidade intrínseca muito menor
        n_novo_flat[~mascara_silicio_flat] = \
            cfg.densidade_intrinseca * np.exp(expo[~mascara_silicio_flat])

        n_novo_flat = np.clip(n_novo_flat, 1e8, None)

        # Relaxação
        n_flat = (1 - cfg.fator_relaxamento) * n_flat \
                 + cfg.fator_relaxamento * n_novo_flat

        # 5) Critério de convergência
        erro_phi = np.linalg.norm(phi_novo_flat - phi_flat) / (np.linalg.norm(phi_flat) + 1e-20)
        erro_n = np.linalg.norm(n_novo_flat - n_flat) / (np.linalg.norm(n_flat) + 1e-20)

        print(f"Iteração {it+1:02d} | erro_phi = {erro_phi:.3e} | erro_n = {erro_n:.3e}")

        if max(erro_phi, erro_n) < cfg.tolerancia:
            print("Convergência atingida.")
            break

    # Reshape para 2D
    phi_2d = phi_flat.reshape(Ny, Nx)
    n_2d = n_flat.reshape(Ny, Nx)

    # --------------------------------------------------------
    # 9. Cálculo da corrente de elétrons J_n (2D)
    # --------------------------------------------------------
    # Gradientes numéricos (diferenças centrais internas)
    grad_phi_x = np.zeros_like(phi_2d)
    grad_phi_y = np.zeros_like(phi_2d)

    grad_n_x = np.zeros_like(n_2d)
    grad_n_y = np.zeros_like(n_2d)

    # Interior
    grad_phi_x[:, 1:-1] = (phi_2d[:, 2:] - phi_2d[:, :-2]) / (2 * dx)
    grad_phi_y[1:-1, :] = (phi_2d[2:, :] - phi_2d[:-2, :]) / (2 * dy)

    grad_n_x[:, 1:-1] = (n_2d[:, 2:] - n_2d[:, :-2]) / (2 * dx)
    grad_n_y[1:-1, :] = (n_2d[2:, :] - n_2d[:-2, :]) / (2 * dy)

    # Bordas (primeira derivada unidirecional simples)
    grad_phi_x[:, 0] = (phi_2d[:, 1] - phi_2d[:, 0]) / dx
    grad_phi_x[:, -1] = (phi_2d[:, -1] - phi_2d[:, -2]) / dx
    grad_phi_y[0, :] = (phi_2d[1, :] - phi_2d[0, :]) / dy
    grad_phi_y[-1, :] = (phi_2d[-1, :] - phi_2d[-2, :]) / dy

    grad_n_x[:, 0] = (n_2d[:, 1] - n_2d[:, 0]) / dx
    grad_n_x[:, -1] = (n_2d[:, -1] - n_2d[:, -2]) / dx
    grad_n_y[0, :] = (n_2d[1, :] - n_2d[0, :]) / dy
    grad_n_y[-1, :] = (n_2d[-1, :] - n_2d[-2, :]) / dy

    # Campo elétrico: E = -∇φ
    E_x = -grad_phi_x
    E_y = -grad_phi_y

    # Corrente de elétrons (drift + difusão):
    # J_n = q μ_n n E + q D_n ∇n
    mu_n = cfg.mobilidade_eletrons
    J_n_x = q * mu_n * n_2d * E_x + q * D_n * grad_n_x
    J_n_y = q * mu_n * n_2d * E_y + q * D_n * grad_n_y

    # Corrente total drenando pelo lado do dreno (integração ao longo de y)
    indice_coluna_dreno = Nx - 2  # um pouco antes do contato para evitar borda
    corrente_dreno = np.trapz(J_n_x[:, indice_coluna_dreno], y)
    print(f"Corrente aproximada de dreno (|Id|) ≈ {abs(corrente_dreno):.3e} A por largura [m].")

    resultados = {
        "x": x,
        "y": y,
        "X": X,
        "Y": Y,
        "phi_2d": phi_2d,
        "n_2d": n_2d,
        "J_n_x": J_n_x,
        "J_n_y": J_n_y,
        "corrente_dreno": corrente_dreno,
        "mascara_silicio": mascara_silicio,
        "mascara_oxido": mascara_oxido
    }
    return resultados


# ------------------------------------------------------------
# 10. Rotina de visualização básica
# ------------------------------------------------------------
def plotar_resultados_nanotransistor(resultados):
    X = resultados["X"]
    Y = resultados["Y"]
    phi = resultados["phi_2d"]
    n = resultados["n_2d"]
    Jx = resultados["J_n_x"]
    mascara_silicio = resultados["mascara_silicio"]

    fig, axs = plt.subplots(1, 3, figsize=(16, 4))

    im0 = axs[0].pcolormesh(X * 1e9, Y * 1e9, phi, shading="auto")
    axs[0].set_title("Potencial elétrico φ (V)")
    axs[0].set_xlabel("x [nm]")
    axs[0].set_ylabel("y [nm]")
    fig.colorbar(im0, ax=axs[0])

    im1 = axs[1].pcolormesh(X * 1e9, Y * 1e9, np.log10(n + 1.0), shading="auto")
    axs[1].set_title("log10 densidade de elétrons n(x,y) [m⁻³]")
    axs[1].set_xlabel("x [nm]")
    axs[1].set_ylabel("y [nm]")
    fig.colorbar(im1, ax=axs[1])

    # Visualização da corrente Jx apenas no silício
    Jx_plot = np.where(mascara_silicio, Jx, np.nan)
    im2 = axs[2].pcolormesh(X * 1e9, Y * 1e9, Jx_plot, shading="auto")
    axs[2].set_title("Componente J_n,x (A/m²)")
    axs[2].set_xlabel("x [nm]")
    axs[2].set_ylabel("y [nm]")
    fig.colorbar(im2, ax=axs[2])

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 11. Execução direta
# ------------------------------------------------------------
if __name__ == "__main__":
    cfg = ConfiguracoesNanotransistor2D(
        comprimento_canal_nm=30.0,
        altura_silicio_nm=10.0,
        altura_oxido_nm=5.0,
        num_pontos_x=81,
        num_pontos_y=51,
        tensao_fonte=0.0,
        tensao_dreno=0.6,
        tensao_gate=0.8,
        tensao_substrato=0.0,
        max_iteracoes=60,
        tolerancia=5e-3,
        fator_relaxamento=0.5
    )

    resultados = simular_nanotransistor_2d(cfg)
    plotar_resultados_nanotransistor(resultados)
