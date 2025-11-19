# ============================================================
# MODELO NOVO DE PRECIFICAÇÃO DE AÇÕES PARA A B3
# ------------------------------------------------------------
# - Processo em LOG-PREÇO com drift e volatilidade NÃO lineares
# - Equação de Fokker–Planck 1D em x = ln(S)
# - Esquema numérico: Lax–Friedrichs (parte convectiva)
#   + Difusão explícita (segunda derivada central)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. Parâmetros do modelo estocástico em log-preço
# ------------------------------------------------------------

class ParametrosModelo:
    def __init__(self):
        # Referência para escala de preço (por ex. média histórica)
        self.preco_referencia = 30.0      # R$ (ex.: ação média na B3)
        self.log_preco_ref = np.log(self.preco_referencia)

        # Drift "inteligente" em log-preço:
        # a(x) = mu0 + mu1 * x - kappa * (x - x_ref)
        #        - lambda_ordem * sign(x - x_ref) * |x - x_ref|^gamma
        self.mu0 = 0.02       # componente estrutural anual (~2%)
        self.mu1 = 0.05       # sensibilidade a ciclos macro (depende de x)
        self.kappa = 0.8      # força de reversão à média em log-preço
        self.lambda_ordem = 0.6  # intensidade de pressão de fluxo de ordens
        self.gamma = 1.3      # não-linearidade da pressão de ordens

        # Volatilidade local em log-preço:
        # b(x) = sigma0 * (1 + beta * |x - x_ref|)^eta
        self.sigma0 = 0.35    # vol de base (35% ao ano, típica de ações)
        self.beta = 0.4       # sensibilidade a desvios de preço
        self.eta = 0.8        # grau de não-linearidade da vol

        # Horizonte temporal (em anos) e malha numérica
        self.horizonte_anos = 1.0   # 1 ano
        self.n_pontos_espaco = 401  # número de pontos em x
        self.n_passos_tempo = 5000  # passos de tempo (para ~6 dígitos de precisão)

        # Intervalo em log-preço (x_min, x_max) cobrindo várias sigmas
        self.margem_sigmas = 4.0

        # Parâmetros numéricos para estabilidade
        self.cfl_seguro = 0.4  # fator de segurança do CFL


param = ParametrosModelo()

# ------------------------------------------------------------
# 2. Funções de drift e volatilidade em log-preço
# ------------------------------------------------------------

def drift_logpreco(x: np.ndarray, p: ParametrosModelo) -> np.ndarray:
    """Drift a(x) em log-preço com reversão à média + pressão de ordens."""
    dx = x - p.log_preco_ref
    componente_basica = p.mu0 + p.mu1 * x - p.kappa * dx
    pressao_ordens = - p.lambda_ordem * np.sign(dx) * np.abs(dx) ** p.gamma
    return componente_basica + pressao_ordens

def volatilidade_logpreco(x: np.ndarray, p: ParametrosModelo) -> np.ndarray:
    """Volatilidade local b(x) em log-preço."""
    dx = np.abs(x - p.log_preco_ref)
    return p.sigma0 * (1.0 + p.beta * dx) ** p.eta

# ------------------------------------------------------------
# 3. Construção da malha em x e t
# ------------------------------------------------------------

# Estimativa de desvio-padrão efetivo em log-preço para definir domínio
sigma_efetiva = param.sigma0 * np.sqrt(param.horizonte_anos) * (1 + param.beta * 0.5) ** param.eta
x0 = param.log_preco_ref
x_min = x0 - param.margem_sigmas * sigma_efetiva
x_max = x0 + param.margem_sigmas * sigma_efetiva

nx = param.n_pontos_espaco
dx = (x_max - x_min) / (nx - 1)
grade_x = np.linspace(x_min, x_max, nx)

nt = param.n_passos_tempo
dt_bruto = param.horizonte_anos / nt

# Ajuste de dt com base em condições de estabilidade (CFL)
a_vals = drift_logpreco(grade_x, param)
b_vals = volatilidade_logpreco(grade_x, param)
vel_max = np.max(np.abs(a_vals)) + 1e-8
dif_max = 0.5 * np.max(b_vals ** 2)

dt_cfl_conv = param.cfl_seguro * dx / vel_max
dt_cfl_dif = param.cfl_seguro * dx**2 / (2.0 * dif_max + 1e-12)
dt = min(dt_bruto, dt_cfl_conv, dt_cfl_dif)

# Corrige número de passos com o dt estável
nt = int(np.ceil(param.horizonte_anos / dt))
dt = param.horizonte_anos / nt

print(f"dx   = {dx:.6e}")
print(f"dt   = {dt:.6e}  (nt = {nt})")
print(f"CFL conv = {vel_max * dt / dx:.3f}")
print(f"CFL dif  = {dif_max * dt / dx**2:.3f}")

# ------------------------------------------------------------
# 4. Condição inicial p(x,0): distribuição concentrada no preço inicial
# ------------------------------------------------------------

preco_inicial = 28.0  # exemplo: ação começa em R$ 28
x_inicial = np.log(preco_inicial)

# Gaussiana em x: p(x,0) ~ Normal(x_inicial, sigma_ini^2)
sigma_ini = 0.10  # desvio inicial em log-preço
p0 = np.exp(-0.5 * ((grade_x - x_inicial) / sigma_ini) ** 2)
p0 /= np.trapz(p0, grade_x)  # normalizar para integral = 1

densidade = p0.copy()

# ------------------------------------------------------------
# 5. Esquema numérico: Lax–Friedrichs + difusão explícita
# ------------------------------------------------------------

def passo_tempo_lax_friedrichs(densidade_atual: np.ndarray,
                               grade_x: np.ndarray,
                               dt: float,
                               dx: float,
                               param: ParametrosModelo) -> np.ndarray:
    """
    Avança a densidade p(x,t) -> p(x,t+dt) para a equação de Fokker–Planck:
        ∂p/∂t = -∂(a(x)p)/∂x + 1/2 ∂²( b(x)^2 p )/∂x²
    Parte convectiva com Lax–Friedrichs, difusiva com esquema central explícito.
    """
    p = densidade_atual
    nx = len(p)

    # Drift e volatilidade na malha
    a = drift_logpreco(grade_x, param)
    b = volatilidade_logpreco(grade_x, param)
    D = 0.5 * b**2  # coeficiente de difusão em log-preço

    # Fluxo convectivo F = a(x) * p
    F = a * p

    # Vetor para próxima etapa
    p_meio = np.zeros_like(p)
    p_novo = np.zeros_like(p)

    # -----------------------------------------
    # 5.1. Parte convectiva: Lax–Friedrichs
    # p^{n+1/2}_i = 0.5 (p_{i+1}^n + p_{i-1}^n)
    #               - dt/(2dx) (F_{i+1}^n - F_{i-1}^n)
    # -----------------------------------------
    for i in range(1, nx - 1):
        p_meio[i] = 0.5 * (p[i+1] + p[i-1]) - 0.5 * dt / dx * (F[i+1] - F[i-1])

    # Condições de fronteira aproximadas (derivada nula em p)
    p_meio[0] = p_meio[1]
    p_meio[-1] = p_meio[-2]

    # -----------------------------------------
    # 5.2. Parte difusiva: esquema central explícito
    # p^{n+1}_i = p^{n+1/2}_i + dt * [ D_i * (p_{i+1}-2p_i+p_{i-1}) / dx^2 ]
    # -----------------------------------------
    for i in range(1, nx - 1):
        laplace_p = (p_meio[i+1] - 2.0 * p_meio[i] + p_meio[i-1]) / dx**2
        p_novo[i] = p_meio[i] + dt * D[i] * laplace_p

    # Fronteiras com derivada nula (Neumann ~ refletivo)
    p_novo[0] = p_novo[1]
    p_novo[-1] = p_novo[-2]

    # Forçar não-negatividade numérica
    p_novo = np.maximum(p_novo, 0.0)

    # Renormalizar para manter integral ~ 1
    integral = np.trapz(p_novo, grade_x)
    if integral > 0:
        p_novo /= integral

    return p_novo

# ------------------------------------------------------------
# 6. Evolução temporal e cálculo de preço esperado
# ------------------------------------------------------------

tempos = np.linspace(0.0, param.horizonte_anos, nt + 1)
trajetoria_preco_medio = np.zeros_like(tempos)

# Armazenar algumas densidades intermediárias para visualização
indices_snapshot = [0, int(0.25 * nt), int(0.5 * nt), int(0.75 * nt), nt]
snapshots = {}

densidade = p0.copy()
for k, t in enumerate(tempos):
    # Preço esperado E[S_t] = ∫ exp(x) p(x,t) dx
    preco_esperado = np.trapz(np.exp(grade_x) * densidade, grade_x)
    trajetoria_preco_medio[k] = preco_esperado

    if k in indices_snapshot:
        snapshots[k] = densidade.copy()

    if k < nt:
        densidade = passo_tempo_lax_friedrichs(densidade, grade_x, dt, dx, param)

# Verificação da conservação de probabilidade
erro_massa = 1.0 - np.trapz(densidade, grade_x)
print(f"Erro de massa no tempo final: {erro_massa:.6e}")

# ------------------------------------------------------------
# 7. Conversão de x -> preço S para visualização
# ------------------------------------------------------------

grade_preco = np.exp(grade_x)

def converter_densidade_x_para_preco(densidade_x: np.ndarray,
                                     grade_x: np.ndarray):
    """
    Converte densidade em x=ln(S) para densidade em S.
    Relação: p_S(S) = p_X(ln S) * (1/S)
    """
    S = np.exp(grade_x)
    dens_S = densidade_x / S
    # Normalizar em S para evitar erros numéricos
    integral_S = np.trapz(dens_S, S)
    if integral_S > 0:
        dens_S /= integral_S
    return S, dens_S

# ------------------------------------------------------------
# 8. Gráficos
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))
for idx in indices_snapshot:
    dens_x = snapshots[idx]
    S, dens_S = converter_densidade_x_para_preco(dens_x, grade_x)
    t_rel = tempos[idx]
    plt.plot(S, dens_S, label=f"t = {t_rel:.2f} anos")
plt.xlabel("Preço da ação S (R$)")
plt.ylabel("Densidade em S")
plt.title("Evolução da distribuição de preços – Modelo não linear em log-preço")
plt.legend()
plt.grid(True, alpha=0.3)

plt.figure(figsize=(10, 5))
plt.plot(tempos, trajetoria_preco_medio)
plt.xlabel("Tempo (anos)")
plt.ylabel("Preço médio E[S_t] (R$)")
plt.title("Trajetória do preço médio esperado sob o modelo proposto")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
