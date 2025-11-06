# ============================================
# Previsão de Bitcoin via Heston + Saltos (Bates)
# Filtro de Partículas (pseudo-MLE) + Previsão
# Autor: Luiz Tiago Wilcke (LT)
# ============================================
import numpy as np
import math
import matplotlib.pyplot as plt
from dataclasses import dataclass

np.random.seed(42)


precos_btc = np.array([
    61000, 61480, 60550, 59800, 60220, 61210, 62300, 61850, 62540, 64010,
    63700, 64580, 65120, 64890, 66200, 65950, 67110, 66800, 67520, 68230,
    69010, 70150, 69500, 69990, 70700, 71420, 70950, 72030, 72810, 73500,
    74220, 73810, 74900, 76050, 75550, 77020, 78500, 78010, 79230, 80500,
    80040, 81220, 82560, 83900, 83200, 84440, 85890, 85200, 86510, 87800,
    87050, 88900, 90250, 89700, 91020, 92400, 91800, 93210, 94600, 94020,
    95410, 96850, 96000, 97500, 98950, 98200, 99680, 101300, 100200, 101900,
    103500, 102400, 104200, 105900, 105100, 106800, 108400, 107600, 109300, 110900
], dtype=float)

# Intervalo de tempo (1 dia em fração de ano ~ 1/252)
dt = 1.0/252.0
retornos = np.diff(np.log(precos_btc))

# ------------------------------
# 2) Utilidades estatísticas
# ------------------------------
def log_normal_pdf(x, mu, var):
    """Log densidade Normal(x | mu, var)."""
    return -0.5*(np.log(2*np.pi*var) + ((x-mu)**2)/var)

def mistura_normal_logpdf(x, pesos, mus, vars_):
    """Log densidade de mistura de Normais por soma estável."""
    # soma pes_i * N(x|mu_i,var_i) em log-space (log-sum-exp)
    comps = [np.log(w) + log_normal_pdf(x, m, v) for w, m, v in zip(pesos, mus, vars_)]
    mmax = np.max(comps, axis=0)
    return mmax + np.log(np.sum(np.exp(comps - mmax), axis=0))

# ------------------------------
# 3) Estruturas de parâmetros
# ------------------------------
@dataclass
class Parametros:
    mu: float
    kappa: float
    theta: float
    sigma_v: float
    rho: float
    lambda_j: float
    m_j: float
    s_j: float

def limites_parametros():
    # Faixas plausíveis para cripto
    return {
        "mu":       (-0.5, 0.5),
        "kappa":    (0.10, 8.0),
        "theta":    (1e-5, 0.6),
        "sigma_v":  (1e-4,  3.0),
        "rho":      (-0.95, 0.95),
        "lambda_j": (0.0,   1.5),   # salto médio até ~1.5/dia é forte
        "m_j":      (-0.20, 0.20),  # média do log-salto
        "s_j":      (1e-3,  0.50)   # desvio do log-salto
    }

def amostra_parametros_aleatorios():
    lim = limites_parametros()
    return Parametros(
        mu=np.random.uniform(*lim["mu"]),
        kappa=np.random.uniform(*lim["kappa"]),
        theta=np.random.uniform(*lim["theta"]),
        sigma_v=np.random.uniform(*lim["sigma_v"]),
        rho=np.random.uniform(*lim["rho"]),
        lambda_j=np.random.uniform(*lim["lambda_j"]),
        m_j=np.random.uniform(*lim["m_j"]),
        s_j=np.random.uniform(*lim["s_j"]),
    )

# ------------------------------
# 4) Filtro de partículas (Bates)
# Estado oculto: v_t (variância instantânea)
# Observável: retorno r_t
# ------------------------------
def filtro_particulas_loglike(ret, pars: Parametros, Npart=1000):
    """
    Filtro bootstrap simples para v_t.
    - Transição de v_t: Euler-Maruyama com "full truncation" p/ positividade.
    - Likelihood de r_t | v_t: mistura (sem salto / com salto lognormal).
    Retorna: log-verossimilhança, trajetória suavizada (v_hat_t).
    """
    T = len(ret)
    if T == 0:
        return -np.inf, np.array([])

    mu, kappa, theta, sigma_v = pars.mu, pars.kappa, pars.theta, pars.sigma_v
    rho, lambda_j, m_j, s_j = pars.rho, pars.lambda_j, pars.m_j, pars.s_j

    # Inicialização dos partículas de v (volatilidade instantânea)
    v = np.abs(np.random.normal(loc=theta, scale=0.5*theta, size=Npart))
    v = np.clip(v, 1e-8, None)

    loglike = 0.0
    v_suav = np.zeros(T)

    # Para correlação entre choques (opcional na observação, mas aqui só no predict)
    # No peso observacional usamos mistura marginal em r_t.
    for t in range(T):
        # 4.1) Previsão de v_{t+1}
        # dW2 ~ N(0, dt). Usamos choque Normal(0,1)*sqrt(dt).
        z2 = np.random.normal(size=Npart)
        v = v + kappa*(theta - v)*dt + sigma_v*np.sqrt(np.maximum(v, 0.0))*np.sqrt(dt)*z2
        v = np.clip(v, 1e-10, None)

        # 4.2) Likelihood r_t | v. Mistura:
        # Peso sem salto: prob = exp(-lambda*dt), salto: 1 - prob
        p_nojump = np.exp(-lambda_j*dt)
        p_jump = 1.0 - p_nojump

        # Componente "sem salto": r ~ Normal(mu*dt - 0.5*v*dt, v*dt)
        media_sem = mu*dt - 0.5*v*dt
        var_sem = v*dt

        # Componente "com salto": r ~ Normal(mu*dt - 0.5*v*dt + m_j, v*dt + s_j^2)
        media_com = media_sem + m_j
        var_com = var_sem + s_j**2

        # Densidade de mistura por partícula
        # (usamos soma sobre partículas via pesos)
        # Para estabilidade numérica, calculamos em log e depois normalizamos.
        # w_i ∝ p_nojump*N(r|media_sem_i, var_sem_i) + p_jump*N(r|media_com_i, var_com_i)
        # Como "r" é escalar por tempo t, calculamos vetor de log-densidades mistura.
        r_obs = ret[t]
        # Log-sum-exp por partícula (2 componentes)
        logw_no = np.log(p_nojump + 1e-300) + (-0.5*(np.log(2*np.pi*var_sem) + (r_obs - media_sem)**2/var_sem))
        logw_jp = np.log(p_jump   + 1e-300) + (-0.5*(np.log(2*np.pi*var_com) + (r_obs - media_com)**2/var_com))

        # peso log total = log( exp(logw_no) + exp(logw_jp) )
        mmax = np.maximum(logw_no, logw_jp)
        logw = mmax + np.log(np.exp(logw_no - mmax) + np.exp(logw_jp - mmax))

        # Atualiza log-verossimilhança total pelo log-média dos pesos
        lw_max = np.max(logw)
        w = np.exp(logw - lw_max)
        soma_w = np.sum(w)
        loglike += lw_max + np.log(soma_w) - np.log(Npart + 0.0)

        # 4.3) Reamostragem (multinomial simples)
        w_norm = w / (soma_w + 1e-300)
        idx = np.random.choice(Npart, size=Npart, replace=True, p=w_norm)
        v = v[idx]

        # Guardar um estimador suavizado simples (média das partículas)
        v_suav[t] = np.mean(v)

    return float(loglike), v_suav

# ------------------------------
# 5) Busca aleatória guiada (pseudo-MLE)
# ------------------------------
def avalia_parametros(ret, pars, Npart):
    # Penalizações leves para garantir estacionariedade do CIR (2*kappa*theta > 0)
    if pars.kappa <= 0 or pars.theta <= 0 or pars.sigma_v <= 0 or pars.s_j <= 0:
        return -np.inf, None
    # estabilidade do CIR (momentos finitos): força heurística
    if (2*pars.kappa*pars.theta) <= (pars.sigma_v**2)*0.5:
        return -np.inf, None
    ll, v_suav = filtro_particulas_loglike(ret, pars, Npart=Npart)
    return ll, v_suav

def otimiza_parametros(ret, n_iniciais=40, n_refinos=30, Npart=800):
    melhor_ll = -np.inf
    melhor_pars = None
    melhor_v = None

    # Amostras iniciais amplas
    candidatos = [amostra_parametros_aleatorios() for _ in range(n_iniciais)]
    for c in candidatos:
        ll, vsv = avalia_parametros(ret, c, Npart)
        if ll > melhor_ll:
            melhor_ll, melhor_pars, melhor_v = ll, c, vsv

    # Refino local: perturbações decrescentes
    escala = 0.25
    lim = limites_parametros()
    for _ in range(n_refinos):
        base = melhor_pars
        def clamp(val, a, b): return max(a, min(b, val))
        prop = Parametros(
            mu       = clamp(base.mu       + np.random.normal(0, escala*0.05), *lim["mu"]),
            kappa    = clamp(base.kappa    + np.random.normal(0, escala*0.8),  *lim["kappa"]),
            theta    = clamp(base.theta    + np.random.normal(0, escala*0.08), *lim["theta"]),
            sigma_v  = clamp(base.sigma_v  + np.random.normal(0, escala*0.4),  *lim["sigma_v"]),
            rho      = clamp(base.rho      + np.random.normal(0, escala*0.2),  *lim["rho"]),
            lambda_j = clamp(base.lambda_j + np.random.normal(0, escala*0.2),  *lim["lambda_j"]),
            m_j      = clamp(base.m_j      + np.random.normal(0, escala*0.04), *lim["m_j"]),
            s_j      = clamp(base.s_j      + np.random.normal(0, escala*0.05), *lim["s_j"]),
        )
        ll, vsv = avalia_parametros(ret, prop, Npart)
        if ll > melhor_ll:
            melhor_ll, melhor_pars, melhor_v = ll, prop, vsv
        escala *= 0.95

    return melhor_pars, melhor_ll, melhor_v

# ------------------------------
# 6) Previsão amostral (simulação preditiva)
# ------------------------------
def simula_caminhos_futuros(S_ultimo, v_ultimo, pars: Parametros, passos=30, n_caminhos=500):
    dt = 1.0/252.0
    S = np.full((n_caminhos, passos+1), S_ultimo, dtype=float)
    v = np.full((n_caminhos, passos+1), max(1e-8, float(v_ultimo)), dtype=float)

    mu, kappa, theta, sigma_v = pars.mu, pars.kappa, pars.theta, pars.sigma_v
    rho, lambda_j, m_j, s_j = pars.rho, pars.lambda_j, pars.m_j, pars.s_j

    for t in range(passos):
        z1 = np.random.normal(size=n_caminhos)
        z2 = np.random.normal(size=n_caminhos)
        # correlacionar z1 e z2
        z2_corr = rho*z1 + np.sqrt(max(1e-12, 1-rho**2))*z2

        # Evolução da variância (CIR discretizado)
        v[:, t+1] = v[:, t] + kappa*(theta - v[:, t])*dt + sigma_v*np.sqrt(np.maximum(v[:, t],0))*np.sqrt(dt)*z2_corr
        v[:, t+1] = np.clip(v[:, t+1], 1e-10, None)

        # Saltos (Merton): indicador de salto ~ Bernoulli(1 - exp(-lambda*dt))
        houve_salto = np.random.uniform(size=n_caminhos) < (1 - np.exp(-lambda_j*dt))
        logJ = np.where(houve_salto, np.random.normal(m_j, s_j, size=n_caminhos), 0.0)

        # Evolução do preço (retorno log)
        r = (mu - 0.5*v[:, t])*dt + np.sqrt(v[:, t]*dt)*z1 + logJ
        S[:, t+1] = S[:, t]*np.exp(r)

    return S, v

# ===========================================================
# Execução: estimação, suavização, previsão e gráficos
# ===========================================================
print("Ajustando parâmetros (pseudo-MLE com filtro de partículas)...")
pars_hat, ll_hat, v_suav = otimiza_parametros(retornos, n_iniciais=36, n_refinos=28, Npart=600)
print("Parâmetros estimados:")
print(pars_hat)
print(f"log-verossimilhança ≈ {ll_hat:.2f}")

# Previsão para 30 dias
passos_futuros = 30
S_ult = float(precos_btc[-1])
v_ult = float(v_suav[-1] if len(v_suav)>0 else pars_hat.theta)
caminhos_S, caminhos_v = simula_caminhos_futuros(S_ult, v_ult, pars_hat, passos=passos_futuros, n_caminhos=800)

# Estatísticas preditivas
mediana = np.median(caminhos_S, axis=0)
p10 = np.percentile(caminhos_S, 10, axis=0)
p90 = np.percentile(caminhos_S, 90, axis=0)
p2 = np.percentile(caminhos_S, 2.5, axis=0)
p97 = np.percentile(caminhos_S, 97.5, axis=0)

# -----------------------------------------------------------
# Gráficos
# -----------------------------------------------------------
plt.figure(figsize=(11,5))
plt.title("Preço BTC: histórico e previsão (faixas 80% e 95%)")
plt.plot(precos_btc, lw=1.8, label="Histórico")
base = len(precos_btc)-1
eixo_prev = np.arange(base, base+passos_futuros+1)

plt.plot(eixo_prev, mediana, lw=2.2, label="Mediana preditiva")
plt.fill_between(eixo_prev, p10, p90, alpha=0.25, label="80%")
plt.fill_between(eixo_prev, p2, p97, alpha=0.15, label="95%")
plt.scatter([base], [precos_btc[-1]], zorder=5, label="Último preço")
plt.legend()
plt.xlabel("Tempo (dias)")
plt.ylabel("Preço (USD)")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()

plt.figure(figsize=(11,4))
plt.title("Volatilidade instantânea suavizada (E[v_t])")
plt.plot(np.sqrt(v_suav)*np.sqrt(252.0), lw=1.8, label="Vol anualizada (suavizada)")
plt.xlabel("Tempo (dias)")
plt.ylabel("Vol anualizada")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.show()
