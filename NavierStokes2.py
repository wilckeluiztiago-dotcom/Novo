"""
============================================================
SIMULAÇÕES NUMÉRICAS DE EQUAÇÕES DE FLUIDOS
  - Burgers 1D (diferença finita explícita, esquema RK4)
  - Navier–Stokes 2D (forma vorticidade, método pseudo-espectral)

Autor: Luiz Tiago Wilcke (LT)
-------------------------------------------------------------
1) Equação de Burgers viscosa 1D
-------------------------------------------------------------
Equação (em domínio periódico x ∈ [0, L]):

    ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²,

onde:
    - u(x,t) é a velocidade escalar,
    - ν > 0 é a viscosidade cinemática.

Implementação:
    - Derivadas espaciais via diferenças finitas centrais (ordem 2),
    - Integração no tempo via método de Runge–Kutta de 4ª ordem (RK4),
    - Condições de contorno periódicas (via np.roll),
    - Cálculo da energia cinética:
        E(t) = 1/2 ∫ u(x,t)² dx.

-------------------------------------------------------------
2) Equações de Navier–Stokes incompressíveis 2D
-------------------------------------------------------------
Modelo na forma de vorticidade ω(x,y,t) e função corrente ψ(x,y,t),
em domínio periódico (x,y) ∈ [0, L] × [0, L]:

    ∂ω/∂t + u ∂ω/∂x + v ∂ω/∂y = ν ∇²ω,

com:
    ω = ∂v/∂x - ∂u/∂y,
    ∇²ψ = -ω,
    u =  ∂ψ/∂y,
    v = -∂ψ/∂x.

Implementação:
    - Método pseudo-espectral:
        · transformadas de Fourier 2D (FFT) para derivadas espaciais,
        · inversão de Poisson em espaço de Fourier para ψ,
    - Integração no tempo via RK4,
    - Dealiasing (regra 2/3) aplicado ao termo não-linear,
    - Cálculo de:
        · energia cinética total:
              E(t) = 1/2 ∬ (u² + v²) dA,
        · enstrofia:
              Z(t) = 1/2 ∬ ω² dA,
        · espectro de energia E(k) (média em cascas de número de onda).

Esta estrutura é típica de códigos de pesquisa em turbulência 2D,
funciona como base conceitual para estudos em cascata de energia,
regimes dissipativos, etc.
============================================================
"""

# ============================================================
# IMPORTAÇÕES
# ============================================================
import numpy as np
import matplotlib.pyplot as plt

# Deixar gráficos mais agradáveis (opcional)
plt.rcParams["figure.figsize"] = (7, 5)
plt.rcParams["font.size"] = 11


# ============================================================
# PARTE 1 — BURGERS 1D (DIFERENÇA FINITA, RK4)
# ============================================================

def inicializar_campo_burgers(nx: int, comprimento: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Inicializa a malha 1D e o campo de velocidade u(x,0) para a
    equação de Burgers viscosa.

    Aqui escolhemos como condição inicial uma combinação de modos
    senoidais, o que gera não linearidade forte e formação de frentes.

    Parâmetros
    ----------
    nx : int
        Número de pontos na malha 1D.
    comprimento : float
        Comprimento do domínio [0, L].

    Retorna
    -------
    x : ndarray
        Vetor de coordenadas espaciais.
    u0 : ndarray
        Campo inicial de velocidade.
    """
    x = np.linspace(0.0, comprimento, nx, endpoint=False)
    # Combinação de modos: seno fundamental + harmônicos
    u0 = np.sin(x) + 0.5 * np.sin(2.0 * x) - 0.25 * np.sin(3.0 * x)
    return x, u0


def derivada_central_periodica(campo: np.ndarray, dx: float) -> np.ndarray:
    """
    Aplica derivada primeira via diferenças finitas centrais
    com condições periódicas em 1D.
    """
    return (np.roll(campo, -1) - np.roll(campo, 1)) / (2.0 * dx)


def derivada_segunda_periodica(campo: np.ndarray, dx: float) -> np.ndarray:
    """
    Aplica derivada segunda via diferenças finitas centrais
    com condições periódicas em 1D.
    """
    return (np.roll(campo, -1) - 2.0 * campo + np.roll(campo, 1)) / (dx**2)


def rhs_burgers(u: np.ndarray, viscosidade: float, dx: float) -> np.ndarray:
    """
    Calcula o lado direito da equação de Burgers viscosa 1D:

        ∂u/∂t = -u ∂u/∂x + ν ∂²u/∂x²
    """
    du_dx = derivada_central_periodica(u, dx)
    d2u_dx2 = derivada_segunda_periodica(u, dx)
    return -u * du_dx + viscosidade * d2u_dx2


def passo_rk4_burgers(u: np.ndarray, dt: float, viscosidade: float, dx: float) -> np.ndarray:
    """
    Executa um passo de tempo usando método de Runge–Kutta de 4ª ordem
    para a equação de Burgers.
    """
    k1 = rhs_burgers(u, viscosidade, dx)
    k2 = rhs_burgers(u + 0.5 * dt * k1, viscosidade, dx)
    k3 = rhs_burgers(u + 0.5 * dt * k2, viscosidade, dx)
    k4 = rhs_burgers(u + dt * k3, viscosidade, dx)
    return u + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def energia_burgers(u: np.ndarray, dx: float) -> float:
    """
    Energia cinética da solução de Burgers:

        E = 1/2 ∫ u(x)² dx
    """
    return 0.5 * np.sum(u**2) * dx


def simular_burgers_1d():
    """
    Roda a simulação de Burgers 1D e gera gráficos:

        - Campo inicial e final u(x),
        - Evolução da energia E(t).
    """
    # Parâmetros
    nx = 256
    comprimento = 2.0 * np.pi
    viscosidade = 5e-3
    tempo_final = 3.0
    dt = 2e-4
    passos = int(tempo_final / dt)

    x, u = inicializar_campo_burgers(nx, comprimento)
    dx = comprimento / nx

    vetor_tempo = np.zeros(passos + 1)
    vetor_energia = np.zeros(passos + 1)
    vetor_energia[0] = energia_burgers(u, dx)

    u_inicial = u.copy()

    for n in range(1, passos + 1):
        u = passo_rk4_burgers(u, dt, viscosidade, dx)
        vetor_tempo[n] = n * dt
        vetor_energia[n] = energia_burgers(u, dx)

    # Gráficos
    plt.figure()
    plt.plot(x, u_inicial, label="u(x,0)")
    plt.plot(x, u, label=f"u(x,{tempo_final:.2f})")
    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    plt.title("Equação de Burgers 1D — Campo inicial e final")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.figure()
    plt.plot(vetor_tempo, vetor_energia, "-")
    plt.xlabel("tempo t")
    plt.ylabel("Energia E(t)")
    plt.title("Equação de Burgers 1D — Decaimento da energia")
    plt.grid(True)
    plt.tight_layout()


# ============================================================
# PARTE 2 — NAVIER–STOKES 2D (PSEUDO-ESPECTRAL, VORTICIDADE)
# ============================================================

class NavierStokes2D:
    """
    Classe que encapsula uma simulação pseudo-espectral 2D da
    equação de Navier–Stokes incompressível na forma de vorticidade.

    Domínio periódico em [0, L] x [0, L].

    Equação de vorticidade:

        ∂ω/∂t + u ∂ω/∂x + v ∂ω/∂y = ν ∇²ω,

    com:
        ∇²ψ = -ω,
        u =  ∂ψ/∂y,
        v = -∂ψ/∂x.

    Implementação:
        - FFT 2D para derivadas espaciais,
        - inversão do Laplaciano em espaço de Fourier,
        - termo não-linear calculado em espaço físico (pseudo-espectral),
        - regra 2/3 para dealiasing dos modos de alta frequência,
        - integração em tempo via RK4.
    """

    def __init__(self,
                 nx: int = 128,
                 ny: int = 128,
                 comprimento: float = 2.0 * np.pi,
                 viscosidade: float = 1e-3):
        self.nx = nx
        self.ny = ny
        self.comprimento = comprimento
        self.viscosidade = viscosidade

        # Espaço físico
        self.x = np.linspace(0.0, comprimento, nx, endpoint=False)
        self.y = np.linspace(0.0, comprimento, ny, endpoint=False)
        self.xx, self.yy = np.meshgrid(self.x, self.y, indexing="ij")

        # Espaço de Fourier
        dx = comprimento / nx
        dy = comprimento / ny
        kx = np.fft.fftfreq(nx, d=dx) * 2.0 * np.pi
        ky = np.fft.fftfreq(ny, d=dy) * 2.0 * np.pi
        self.kx, self.ky = np.meshgrid(kx, ky, indexing="ij")
        self.k2 = self.kx**2 + self.ky**2
        self.k2[0, 0] = 1.0  # evita divisão por zero

        # Máscara de dealiasing (regra 2/3 simplificada)
        k_mod = np.sqrt(self.kx**2 + self.ky**2)
        k_max = np.max(k_mod)
        limite = (2.0 / 3.0) * k_max
        self.mascara_dealias = (k_mod <= limite).astype(float)

        # Área de cada célula (para integrais)
        self.dx = dx
        self.dy = dy
        self.area_celula = dx * dy

        # Campo de vorticidade em Fourier
        self.omega_chapeu = None

    # --------------------------
    # Inicialização
    # --------------------------

    def inicializar_vorticidade_turbulenta(self, amplitud: float = 1.0, semente: int = 42):
        """
        Cria um campo inicial de vorticidade com estrutura tipo turbulência 2D,
        a partir de combinação de alguns modos aleatórios de baixa frequência.
        """
        rng = np.random.default_rng(semente)
        omega_hat = np.zeros((self.nx, self.ny), dtype=complex)

        max_modo = 4
        for i in range(-max_modo, max_modo + 1):
            for j in range(-max_modo, max_modo + 1):
                if i == 0 and j == 0:
                    continue
                idx = i % self.nx
                idy = j % self.ny
                fase = rng.uniform(0, 2 * np.pi)
                amp = amplitud * rng.normal()
                omega_hat[idx, idy] = amp * np.exp(1j * fase)

        omega = np.fft.ifft2(omega_hat).real
        omega -= np.mean(omega)
        self.omega_chapeu = np.fft.fft2(omega)

    # --------------------------
    # Campos derivados
    # --------------------------

    def calcular_vorticidade(self) -> np.ndarray:
        """
        Retorna a vorticidade ω no espaço físico.
        """
        return np.fft.ifft2(self.omega_chapeu).real

    def calcular_streamfunction(self) -> np.ndarray:
        """
        Resolve ∇²ψ = -ω em espaço de Fourier:

            k² ψ̂ = -ω̂  => ψ̂ = -ω̂ / k².

        O modo (0,0) é fixado em 0 (não afeta gradientes).
        """
        psi_hat = -self.omega_chapeu / self.k2
        psi_hat[0, 0] = 0.0
        psi = np.fft.ifft2(psi_hat).real
        return psi

    def calcular_velocidade(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Calcula o campo de velocidade (u, v) a partir da função corrente:

            u =  ∂ψ/∂y,
            v = -∂ψ/∂x,
        via derivadas em espaço de Fourier.
        """
        psi_hat = -self.omega_chapeu / self.k2
        psi_hat[0, 0] = 0.0

        dpsi_dx_hat = 1j * self.kx * psi_hat
        dpsi_dy_hat = 1j * self.ky * psi_hat

        dpsi_dx = np.fft.ifft2(dpsi_dx_hat).real
        dpsi_dy = np.fft.ifft2(dpsi_dy_hat).real

        u = dpsi_dy
        v = -dpsi_dx
        return u, v

    # --------------------------
    # RHS e integrais globais
    # --------------------------

    def rhs_vorticidade(self) -> np.ndarray:
        """
        Calcula o lado direito da equação de vorticidade em espaço de Fourier:

            ∂ω̂/∂t = - FFT[ u ∂ω/∂x + v ∂ω/∂y ] + ν ∇² ω̂,

        com dealiasing aplicado ao termo não-linear.
        """
        # Campos no espaço físico
        omega = np.fft.ifft2(self.omega_chapeu).real
        u, v = self.calcular_velocidade()

        # Derivadas de ω em Fourier
        domega_dx_hat = 1j * self.kx * self.omega_chapeu
        domega_dy_hat = 1j * self.ky * self.omega_chapeu

        domega_dx = np.fft.ifft2(domega_dx_hat).real
        domega_dy = np.fft.ifft2(domega_dy_hat).real

        # Termo não-linear em espaço físico
        termo_nao_linear = u * domega_dx + v * domega_dy

        # FFT do termo não-linear e dealiasing
        termo_nao_linear_hat = np.fft.fft2(termo_nao_linear)
        termo_nao_linear_hat *= self.mascara_dealias

        # Laplaciano de ω em Fourier
        laplaciano_omega_hat = -self.k2 * self.omega_chapeu

        rhs_hat = -termo_nao_linear_hat + self.viscosidade * laplaciano_omega_hat
        return rhs_hat

    def energia_e_enstrofia(self) -> tuple[float, float]:
        """
        Calcula energia cinética E e enstrofia Z:

            E = 1/2 ∬ (u² + v²) dA,
            Z = 1/2 ∬ ω² dA.
        """
        omega = np.fft.ifft2(self.omega_chapeu).real
        u, v = self.calcular_velocidade()

        energia = 0.5 * np.sum(u**2 + v**2) * self.area_celula
        enstrofia = 0.5 * np.sum(omega**2) * self.area_celula
        return energia, enstrofia

    def espectro_energia(self, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Calcula o espectro de energia E(k) médio em cascas de número de onda.

        Procedimento:
            1) Calcula FFT de u e v,
            2) Define energia espectral ponto-a-ponto:
                   e(kx,ky) = 0.5 (|û|² + |v̂|²),
            3) Agrupa em cascas de raio k = sqrt(kx² + ky²),
               produzindo E(k).
        """
        u_hat = np.fft.fft2(u)
        v_hat = np.fft.fft2(v)

        energia_espectral = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2)

        k_mod = np.sqrt(self.kx**2 + self.ky**2)
        k_fund = 2.0 * np.pi / self.comprimento
        indices_casca = np.floor(k_mod / k_fund).astype(int)
        k_max_indice = indices_casca.max()

        espectro = np.zeros(k_max_indice + 1)
        contagem = np.zeros(k_max_indice + 1, dtype=int)

        for i in range(self.nx):
            for j in range(self.ny):
                ind = indices_casca[i, j]
                espectro[ind] += energia_espectral[i, j]
                contagem[ind] += 1

        mascara = contagem > 0
        espectro[mascara] /= contagem[mascara]

        k_valores = np.arange(k_max_indice + 1) * k_fund
        return k_valores, espectro

    # --------------------------
    # Integração no tempo (RK4)
    # --------------------------

    def passo_rk4(self, dt: float):
        """
        Executa um passo de tempo (Δt) via RK4 para o campo de vorticidade ω̂.
        """
        k1 = self.rhs_vorticidade()
        omega_tmp = self.omega_chapeu + 0.5 * dt * k1

        k2 = self.rhs_vorticidade_aux(omega_tmp)
        omega_tmp = self.omega_chapeu + 0.5 * dt * k2

        k3 = self.rhs_vorticidade_aux(omega_tmp)
        omega_tmp = self.omega_chapeu + dt * k3

        k4 = self.rhs_vorticidade_aux(omega_tmp)

        self.omega_chapeu = self.omega_chapeu + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def rhs_vorticidade_aux(self, omega_hat: np.ndarray) -> np.ndarray:
        """
        Versão auxiliar de rhs_vorticidade para RK4,
        permitindo avaliar o RHS em estados intermediários ω̂.
        """
        # Salva o estado original
        omega_original = self.omega_chapeu
        self.omega_chapeu = omega_hat

        rhs_hat = self.rhs_vorticidade()

        # Restaura estado original
        self.omega_chapeu = omega_original
        return rhs_hat

    # --------------------------
    # Rotina de simulação principal
    # --------------------------

    def simular(self, tempo_final: float, dt: float, salvar_cada: int = 50):
        """
        Simula a dinâmica de Navier–Stokes 2D até tempo_final com passo dt.

        Parâmetros
        ----------
        tempo_final : float
            Tempo final da simulação.
        dt : float
            Passo de tempo.
        salvar_cada : int
            Frequência (em passos) de coleta de diagnósticos.
        """
        passos = int(tempo_final / dt)

        tempos = []
        energias = []
        enstrofias = []

        for n in range(passos + 1):
            t = n * dt

            if n % salvar_cada == 0:
                E, Z = self.energia_e_enstrofia()
                tempos.append(t)
                energias.append(E)
                enstrofias.append(Z)

            if n < passos:
                self.passo_rk4(dt)

        return np.array(tempos), np.array(energias), np.array(enstrofias)


def rodar_simulacao_navier_stokes_2d():
    """
    Roda uma simulação 2D de Navier–Stokes incompressível em forma de vorticidade,
    partindo de uma condição inicial turbulenta, e gera:

        - mapa de vorticidade final,
        - campo de velocidade (setas) em grade reduzida,
        - evolução de energia e enstrofia,
        - espectro de energia E(k) no tempo final.
    """
    # Parâmetros "pesados" mas ainda razoáveis
    nx = 128
    ny = 128
    comprimento = 2.0 * np.pi
    viscosidade = 2e-3

    modelo = NavierStokes2D(nx=nx, ny=ny, comprimento=comprimento, viscosidade=viscosidade)

    # Campo inicial de vorticidade "turbulento"
    modelo.inicializar_vorticidade_turbulenta(amplitud=1.0, semente=123)

    # Parâmetros de tempo
    tempo_final = 3.0
    dt = 2e-3
    salvar_cada = 10

    tempos, energias, enstrofias = modelo.simular(tempo_final=tempo_final, dt=dt, salvar_cada=salvar_cada)

    # Campos finais
    omega_final = modelo.calcular_vorticidade()
    u_final, v_final = modelo.calcular_velocidade()

    # Espectro de energia no tempo final
    k_vals, E_k = modelo.espectro_energia(u_final, v_final)

    # --------------------------------------------------------
    # Gráfico 1 — Vorticidade final
    # --------------------------------------------------------
    plt.figure(figsize=(6, 5))
    im = plt.pcolormesh(modelo.xx, modelo.yy, omega_final, shading="auto")
    plt.colorbar(im, label="ω(x,y)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Navier–Stokes 2D — Vorticidade no tempo final")
    plt.tight_layout()

    # --------------------------------------------------------
    # Gráfico 2 — Campo de velocidade (amostrado)
    # --------------------------------------------------------
    passo_plot = 4
    plt.figure(figsize=(6, 5))
    plt.quiver(modelo.xx[::passo_plot, ::passo_plot],
               modelo.yy[::passo_plot, ::passo_plot],
               u_final[::passo_plot, ::passo_plot],
               v_final[::passo_plot, ::passo_plot])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Navier–Stokes 2D — Campo de velocidade (tempo final)")
    plt.tight_layout()

    # --------------------------------------------------------
    # Gráfico 3 — Energia e enstrofia vs tempo
    # --------------------------------------------------------
    plt.figure()
    plt.plot(tempos, energias, label="Energia E(t)")
    plt.plot(tempos, enstrofias, label="Enstrofia Z(t)")
    plt.xlabel("tempo t")
    plt.ylabel("Medida global")
    plt.title("Navier–Stokes 2D — Energia e enstrofia")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # --------------------------------------------------------
    # Gráfico 4 — Espectro de energia E(k)
    # --------------------------------------------------------
    plt.figure()
    # Evita k=0 no log
    mascara = k_vals > 0
    plt.loglog(k_vals[mascara], E_k[mascara], "-o", markersize=3)
    plt.xlabel("número de onda k")
    plt.ylabel("E(k)")
    plt.title("Navier–Stokes 2D — Espectro de energia (tempo final)")
    plt.grid(True, which="both")
    plt.tight_layout()




if __name__ == "__main__":
    # 1) Burgers 1D
    simular_burgers_1d()

    # 2) Navier–Stokes 2D
    rodar_simulacao_navier_stokes_2d()

    plt.show()
