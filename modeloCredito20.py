# ============================================================
# Modelo Híbrido Complexo — Juros + Risco de Crédito + Bond
# Autor: Luiz Tiago Wilcke (LT)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1) Configurações da simulação
# ------------------------------------------------------------
horizonte_anos = 5.0
dt = 1/252            # passo diário
n_passos = int(horizonte_anos / dt) + 1
tempo = np.linspace(0, horizonte_anos, n_passos)

n_caminhos = 5000     # número de cenários de Monte Carlo

# ------------------------------------------------------------
# 2) Parâmetros do processo de juros r(t) (Vasicek)
#    dr_t = kappa_r (theta_r - r_t) dt + sigma_r dW_r
# ------------------------------------------------------------
kappa_r = 0.6         # velocidade de reversão
theta_r = 0.05        # nível médio de juros
sigma_r = 0.02        # volatilidade dos juros
r_inicial = 0.04      # taxa de juros inicial

# ------------------------------------------------------------
# 3) Parâmetros do processo de crédito λ(t) (CIR)
#    dλ_t = kappa_l (theta_l - λ_t) dt + sigma_l √λ_t dW_l
# ------------------------------------------------------------
kappa_l = 0.7         # velocidade de reversão
theta_l = 0.08        # nível médio da intensidade de default
sigma_l = 0.25        # volatilidade da intensidade
lambda_inicial = 0.05 # intensidade de default inicial

# ------------------------------------------------------------
# 4) Parâmetros do título (bond) com risco de crédito
#    dp/dt = r(t) p(t) + λ(t) (R - p(t))   (Duffie–Singleton)
# ------------------------------------------------------------
recovery = 0.40       # taxa de recuperação R
valor_face = 1.0      # payoff no vencimento

# ------------------------------------------------------------
# 5) Correlação entre os Brownianos de juros e crédito
# ------------------------------------------------------------
rho = 0.45  # correlação entre dW_r e dW_lambda (crédito)

# ------------------------------------------------------------
# 6) Preparação das matrizes (tempo x caminhos)
# ------------------------------------------------------------
taxa_juros = np.zeros((n_passos, n_caminhos))
intensidade_default = np.zeros((n_passos, n_caminhos))
preco_bond = np.zeros((n_passos, n_caminhos))

taxa_juros[0, :] = r_inicial
intensidade_default[0, :] = lambda_inicial

# ------------------------------------------------------------
# 7) Geração dos incrementos Brownianos correlacionados
# ------------------------------------------------------------
# z1, z2 ~ N(0,1) independentes
z1 = np.random.normal(size=(n_passos - 1, n_caminhos))
z2 = np.random.normal(size=(n_passos - 1, n_caminhos))

dW_r = np.sqrt(dt) * z1
dW_indep = np.sqrt(dt) * z2
dW_l = rho * dW_r + np.sqrt(1 - rho**2) * dW_indep  # correlacionado

# ------------------------------------------------------------
# 8) Simulação das SDEs (Euler–Maruyama)
# ------------------------------------------------------------
for t in range(1, n_passos):
    # Juros r(t) — Vasicek
    r_anterior = taxa_juros[t-1, :]
    dr = kappa_r * (theta_r - r_anterior) * dt + sigma_r * dW_r[t-1, :]
    taxa_juros[t, :] = r_anterior + dr

    # Intensidade λ(t) — CIR (garantir positividade)
    lambda_anterior = intensidade_default[t-1, :]
    lambda_raiz = np.sqrt(np.maximum(lambda_anterior, 0.0))
    d_lambda = (
        kappa_l * (theta_l - lambda_anterior) * dt +
        sigma_l * lambda_raiz * dW_l[t-1, :]
    )
    intensidade_default[t, :] = np.maximum(lambda_anterior + d_lambda, 0.0)

# ------------------------------------------------------------
# 9) Precificação do bond via Duffie–Singleton (backward)
#    dp/dt = r p + λ (R - p)
#    Integramos de T para 0 para garantir p(T) = valor_face
# ------------------------------------------------------------
preco_bond[-1, :] = valor_face

for t in range(n_passos - 2, -1, -1):
    r_fut = taxa_juros[t+1, :]
    lambda_fut = intensidade_default[t+1, :]
    p_fut = preco_bond[t+1, :]

    dp = (r_fut * p_fut + lambda_fut * (recovery - p_fut)) * dt
    preco_bond[t, :] = p_fut - dp

# ------------------------------------------------------------
# 10) Estatísticas agregadas: média e bandas (5% e 95%)
# ------------------------------------------------------------
def estatisticas_por_tempo(matriz):
    media = np.mean(matriz, axis=1)
    p5 = np.percentile(matriz, 5, axis=1)
    p95 = np.percentile(matriz, 95, axis=1)
    return media, p5, p95

media_r, r_p5, r_p95 = estatisticas_por_tempo(taxa_juros)
media_lambda, l_p5, l_p95 = estatisticas_por_tempo(intensidade_default)
media_preco, p_p5, p_p95 = estatisticas_por_tempo(preco_bond)

# ------------------------------------------------------------
# 11) Impressão de resultados resumidos (6 casas decimais)
# ------------------------------------------------------------
print("=============== RESULTADOS MÉDIOS ===============")
print(f"r(0) médio         = {media_r[0]:.6f}")
print(f"r(T) médio         = {media_r[-1]:.6f}")
print(f"lambda(0) médio    = {media_lambda[0]:.6f}")
print(f"lambda(T) médio    = {media_lambda[-1]:.6f}")
print(f"Preço bond p(0)    = {media_preco[0]:.6f}")
print(f"Preço bond p(T)    = {media_preco[-1]:.6f} (esperado ~ {valor_face:.6f})")
print("=================================================")

# Correlação empírica entre r(T) e λ(T) (último ponto)
corr_empirica = np.corrcoef(taxa_juros[-1, :], intensidade_default[-1, :])[0, 1]
print(f"Correlação empírica r(T) x lambda(T): {corr_empirica:.6f}")

# ------------------------------------------------------------
# 12) Gráficos com bandas (média, 5%, 95%)
# ------------------------------------------------------------
plt.figure(figsize=(12, 12))

# Juros
plt.subplot(3, 1, 1)
plt.fill_between(tempo, r_p5, r_p95, alpha=0.2, label="Banda 5%–95%")
plt.plot(tempo, media_r, lw=2, label="Média")
plt.title("Taxa de Juros r(t) — Processo de Vasicek (múltiplos cenários)")
plt.xlabel("Tempo (anos)")
plt.ylabel("r(t)")
plt.grid(True, alpha=0.3)
plt.legend()

# Intensidade de crédito
plt.subplot(3, 1, 2)
plt.fill_between(tempo, l_p5, l_p95, alpha=0.2, color="orange", label="Banda 5%–95%")
plt.plot(tempo, media_lambda, lw=2, color="red", label="Média")
plt.title("Intensidade de Default λ(t) — Processo CIR (múltiplos cenários)")
plt.xlabel("Tempo (anos)")
plt.ylabel("λ(t)")
plt.grid(True, alpha=0.3)
plt.legend()

# Preço do bond
plt.subplot(3, 1, 3)
plt.fill_between(tempo, p_p5, p_p95, alpha=0.2, color="green", label="Banda 5%–95%")
plt.plot(tempo, media_preco, lw=2, color="darkgreen", label="Média")
plt.title("Preço do Título com Risco de Crédito p(t) — Duffie–Singleton")
plt.xlabel("Tempo (anos)")
plt.ylabel("Preço do bond")
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()
