# ============================================================
# Estimativa Bayesiana de N_estrela (Via Láctea) com Seleção Realista
# Autor: Luiz Tiago Wilcke (LT)
# Descrição:
#   - Processo de Contagem: n_i ~ Poisson( lambda_i ),  lambda_i = N_total * p_i
#   - p_i = fração detectável no campo i (estimada via Monte Carlo físico)
#   - Física simplificada mas plausível:
#       * Disco exponencial: rho(R,z) ∝ exp(-R/R_d) * exp(-|z|/h_z)
#       * Sol no plano: R0=8.2 kpc, z0=0 pc
#       * Função de "luminosidade" (M_abs): mistura Gaussiana (proxy da LF)
#       * Extinção: A(d,b) = a0 * d_kpc * exp(-|sin b|/b0)
#       * Seleção: m = M + 5 log10(d/10pc) + A(d,b) <= m_lim_i e dentro do cone do campo
#   - Inferência: MCMC (Metropolis-Hastings) em N_total com prior lognormal fraca
#   - Saídas:
#       * Tabela com campos, limites de magnitude, contagens observadas/simuladas
#       * Gráficos: diagnóstico da simulação, posterior de N_total
#       * Sumário numérico (MAP, média, mediana, IC 95%)
# Dependências: numpy, scipy, pandas, matplotlib
# ============================================================

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict

np.random.seed(123)

# ---------------------------
# 1) Constantes e conversões
# ---------------------------
PC_PARA_KPC = 1e-3
KPC_PARA_PC = 1e3
RAD_PARA_GRAU = 180.0 / math.pi
GRAU_PARA_RAD = math.pi / 180.0

# ---------------------------
# 2) Modelo Galáctico
# ---------------------------
@dataclass
class ParametrosDisco:
    R_d_kpc: float = 2.6     # escala radial (kpc)
    h_z_pc:  float = 300.0   # escala vertical (pc)
    R0_kpc:  float = 8.2     # posição do Sol (kpc)
    z0_pc:   float = 0.0     # altura do Sol (pc)

@dataclass
class ParametrosExtincao:
    a0_mag_por_kpc: float = 1.0    # mag/kpc base
    b0: float = 0.3                 # escala (adimensional) para dependência com latitude

@dataclass
class MisturaMagnitudes:
    # Mistura de normais para M_abs (magnitudes absolutas bolométricas proxy)
    pesos: List[float] = None
    medias: List[float] = None
    sigmas: List[float] = None

    def __post_init__(self):
        if self.pesos is None:
            # Três "tipos" (muito grosseiro): anãs K/M, G-solar, A/F mais brilhantes
            self.pesos  = [0.65, 0.30, 0.05]
            self.medias = [7.5,  5.0,  1.5]
            self.sigmas = [1.2,  0.9,  0.8]
        # normalizar pesos
        s = sum(self.pesos)
        self.pesos = [w/s for w in self.pesos]

    def amostrar_M(self, n: int) -> np.ndarray:
        comp = np.random.choice(len(self.pesos), size=n, p=self.pesos)
        M = np.random.normal(loc=np.take(self.medias, comp),
                             scale=np.take(self.sigmas, comp))
        return M

# -------------------------------------
# 3) Geometria de campos observacionais
# -------------------------------------
@dataclass
class CampoCeleste:
    nome: str
    long_graus: float
    lat_graus: float
    omega_sr: float      # área sólida do campo (steradian)
    m_lim: float         # limite de magnitude aparente

def raio_cone_rad(omega_sr: float) -> float:
    # Para cones pequenos, omega ≈ pi * theta^2 -> theta ≈ sqrt(omega/pi)
    return math.sqrt(omega_sr / math.pi)

# -------------------------------------
# 4) Utilidades geométricas / celestes
# -------------------------------------
def angulo_grande_circulo(l1, b1, l2, b2):
    # todos em radianos; retorna separação angular (rad)
    sb1, sb2 = math.sin(b1), math.sin(b2)
    cb1, cb2 = math.cos(b1), math.cos(b2)
    dl = abs(l1 - l2)
    c = sb1*sb2 + cb1*cb2*math.cos(dl)
    c = max(min(c, 1.0), -1.0)
    return math.acos(c)

def cartesiano_dispara_para_lb(x, y, z, x_sol, y_sol, z_sol):
    # vetor do Sol ao astro
    rx, ry, rz = x - x_sol, y - y_sol, z - z_sol
    d = math.sqrt(rx*rx + ry*ry + rz*rz)
    b = math.asin(rz / d)
    l = math.atan2(ry, rx)
    if l < 0: l += 2*math.pi
    return l, b, d

# -------------------------------------
# 5) Densidade e amostragem do disco
# -------------------------------------
def densidade_log(R_kpc: float, z_pc: float, prm: ParametrosDisco) -> float:
    # log da densidade (até constante), para aceitação-rejeição/IS se preciso
    return -R_kpc/prm.R_d_kpc - abs(z_pc)/prm.h_z_pc

def amostrar_estrelas_disco(n: int, prm: ParametrosDisco,
                            R_max_kpc: float = 20.0,
                            z_max_pc: float = 3000.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Amostra aproximada de (x,y,z) em kpc/pc com distribuição ~ disco exponencial.
    Método: amostra por inversão + thinning simples para R e z independentes aproximados.
    """
    # R ~ exponencial com escala R_d, truncado em R_max
    u = np.random.uniform(size=n)
    R = -prm.R_d_kpc * np.log(1 - u*(1 - math.exp(-R_max_kpc/prm.R_d_kpc)))
    # phi ~ U[0,2pi)
    phi = np.random.uniform(0, 2*math.pi, size=n)
    # z ~ Laplace(0, h_z) truncado
    u2 = np.random.uniform(-0.5, 0.5, size=n)
    z = -prm.h_z_pc * np.sign(u2) * np.log(1 - 2*np.abs(u2))
    z = np.clip(z, -z_max_pc, z_max_pc)
    # Coordenadas cartesianas galactocêntricas (kpc/pc coerentes)
    x = R * np.cos(phi)           # kpc
    y = R * np.sin(phi)           # kpc
    z_kpc = z * PC_PARA_KPC       # converter z para kpc
    return x, y, z_kpc

# -------------------------
# 6) Extinção e magnitude
# -------------------------
def extincao_mag(d_pc: float, b_rad: float, prm_ext: ParametrosExtincao) -> float:
    d_kpc = d_pc * PC_PARA_KPC
    return prm_ext.a0_mag_por_kpc * d_kpc * math.exp(-abs(math.sin(b_rad))/prm_ext.b0)

def magnitude_aparente(M_abs: float, d_pc: float, A_mag: float) -> float:
    return M_abs + 5.0*math.log10(max(d_pc, 1.0)/10.0) + A_mag

# -------------------------------------------------------
# 7) Construir campos (ex.: latitudes variadas, m_lim)
# -------------------------------------------------------
def construir_campos_exemplo() -> List[CampoCeleste]:
    # 8 campos com diferentes latitudes e profundidades
    campos = [
        CampoCeleste("Plano_1",   30.0,   0.0,   5e-4, 17.5),
        CampoCeleste("Plano_2",  200.0,   2.0,   5e-4, 18.0),
        CampoCeleste("Inter_1",  120.0,  20.0,   5e-4, 19.0),
        CampoCeleste("Inter_2",  300.0, -15.0,   5e-4, 19.5),
        CampoCeleste("Alto_1",   250.0,  40.0,   5e-4, 20.0),
        CampoCeleste("Alto_2",    60.0, -50.0,   5e-4, 20.0),
        CampoCeleste("Prof_1",   180.0,  30.0,   5e-4, 21.0),
        CampoCeleste("Prof_2",   350.0, -30.0,   5e-4, 21.5),
    ]
    return campos

# ---------------------------------------------------------
# 8) Fração detectável p_i por Monte Carlo (grande N_mc)
# ---------------------------------------------------------
def fracao_detectavel_por_campo(
    campos: List[CampoCeleste],
    prm_disco: ParametrosDisco,
    prm_ext: ParametrosExtincao,
    mistura: MisturaMagnitudes,
    N_mc: int = 120_000
) -> Dict[str, float]:
    """
    Retorna p_i para cada campo: fração de todas as estrelas galácticas que
    cairiam (direção + magnitude) no campo e seriam detectadas.
    """
    # Amostrar posições galácticas
    x, y, z_kpc = amostrar_estrelas_disco(N_mc, prm_disco)
    # Sol em (R0,0, z0)
    x_sol, y_sol, z_sol = prm_disco.R0_kpc, 0.0, prm_disco.z0_pc * PC_PARA_KPC
    # Converter para l,b,d
    lbds = np.array([cartesiano_dispara_para_lb(x[i], y[i], z_kpc[i], x_sol, y_sol, z_sol)
                     for i in range(N_mc)])
    l_rad = lbds[:,0]
    b_rad = lbds[:,1]
    d_pc  = lbds[:,2] * KPC_PARA_PC
    # Amostrar magnitudes absolutas
    M_abs = mistura.amostrar_M(N_mc)
    # Precomputar extinção e mags aparentes para cada estrela (dependem de b)
    A_mag = np.array([extincao_mag(d_pc[i], b_rad[i], prm_ext) for i in range(N_mc)])
    m_ap = M_abs + 5.0*np.log10(np.maximum(d_pc, 1.0)/10.0) + A_mag

    # Para cada campo: dentro do cone & m <= m_lim
    p_dict = {}
    for c in campos:
        l0 = c.long_graus * GRAU_PARA_RAD
        b0 = c.lat_graus  * GRAU_PARA_RAD
        theta = raio_cone_rad(c.omega_sr)
        # separação angular
        sep = np.array([angulo_grande_circulo(l_rad[i], b_rad[i], l0, b0) for i in range(N_mc)])
        dentro = sep <= theta
        detect = np.logical_and(dentro, m_ap <= c.m_lim)
        p_i = detect.mean()  # fração em relação a TODAS as estrelas do disco (amostragem proporcional à densidade)
        p_dict[c.nome] = float(p_i)
    return p_dict

# ----------------------------------------------------------
# 9) Gerar contagens observadas (ou carregar, se tiver)
# ----------------------------------------------------------
def simular_contagens(campos: List[CampoCeleste],
                      p_dict: Dict[str,float],
                      N_total_verdadeiro: float,
                      usar_poisson: bool = True) -> Dict[str,int]:
    n_dict = {}
    for c in campos:
        lam = N_total_verdadeiro * p_dict[c.nome]
        if usar_poisson:
            n = np.random.poisson(lam=lam)
        else:
            n = int(round(lam))
        n_dict[c.nome] = int(n)
    return n_dict

# ----------------------------------------------------------
# 10) Inferência Bayesiana de N_total (p fixos)
# ----------------------------------------------------------
@dataclass
class PriorLognormal:
    mu_log: float = math.log(1e11)   # centro fraco ~ 1e11
    sigma_log: float = 2.0           # bem largo

    def logpdf(self, N: float) -> float:
        if N <= 0:
            return -np.inf
        x = (math.log(N) - self.mu_log)/self.sigma_log
        return -0.5*x*x - math.log(N) - math.log(self.sigma_log*math.sqrt(2*math.pi))

def logverossimilhanca_poisson(N: float,
                               n_dict: Dict[str,int],
                               p_dict: Dict[str,float]) -> float:
    if N <= 0:
        return -np.inf
    lv = 0.0
    for k, n in n_dict.items():
        lam = N * max(p_dict[k], 1e-30)
        # log Poisson: n*log(lam) - lam - log(n!)
        lv += n * math.log(lam) - lam - math.lgamma(n+1)
    return lv

def mcmc_metropolis_N(
    n_dict: Dict[str,int],
    p_dict: Dict[str,float],
    prior: PriorLognormal,
    N_inicial: float = 1e11,
    passos: int = 50_000,
    saltolog: float = 0.2,
    burn: int = 10_000,
    thin: int = 10
):
    """
    MCMC no espaço log N, proposta lognormal simétrica (~ Normal no log).
    'saltolog' é o desvio-padrão da proposta no log.
    """
    logN_atual = math.log(N_inicial)
    logp_atual = logverossimilhanca_poisson(math.exp(logN_atual), n_dict, p_dict) + prior.logpdf(math.exp(logN_atual))
    amostras = []
    aceitacoes = 0

    for s in range(passos):
        prop = np.random.normal(logN_atual, saltolog)
        N_prop = math.exp(prop)
        logp_prop = logverossimilhanca_poisson(N_prop, n_dict, p_dict) + prior.logpdf(N_prop)
        if np.log(np.random.rand()) < (logp_prop - logp_atual):
            logN_atual = prop
            logp_atual = logp_prop
            aceitacoes += 1
        if s >= burn and ((s - burn) % thin == 0):
            amostras.append(math.exp(logN_atual))

        # ajuste adaptativo leve de saltolog para taxa ~ 0.25-0.4
        if (s+1) % 1000 == 0:
            taxa = aceitacoes / (s+1)
            if taxa < 0.2:
                saltolog *= 0.9
            elif taxa > 0.5:
                saltolog *= 1.1

    return np.array(amostras), aceitacoes / passos

# ----------------------------------------------------------
# 11) Pipeline completo
# ----------------------------------------------------------
def rodar_pipeline(
    N_total_verdadeiro: float = 1.2e11,
    N_mc: int = 120_000
):
    prm_disco = ParametrosDisco(R_d_kpc=2.6, h_z_pc=300.0, R0_kpc=8.2, z0_pc=0.0)
    prm_ext   = ParametrosExtincao(a0_mag_por_kpc=1.0, b0=0.3)
    mistura   = MisturaMagnitudes()
    campos    = construir_campos_exemplo()

    print(">> Estimando frações detectáveis p_i por Monte Carlo...")
    p_dict = fracao_detectavel_por_campo(campos, prm_disco, prm_ext, mistura, N_mc=N_mc)

    print(">> Simulando (ou carregando) contagens observadas n_i (Poisson)...")
    n_dict = simular_contagens(campos, p_dict, N_total_verdadeiro, usar_poisson=True)

    # Tabela resumo por campo
    linhas = []
    for c in campos:
        linhas.append({
            "campo": c.nome,
            "l(grau)": c.long_graus,
            "b(grau)": c.lat_graus,
            "omega(sr)": c.omega_sr,
            "m_lim": c.m_lim,
            "p_i (fração)": p_dict[c.nome],
            "n_i (observado)": n_dict[c.nome],
            "lambda_i esper. (com N_true)": N_total_verdadeiro * p_dict[c.nome]
        })
    df_campos = pd.DataFrame(linhas).sort_values("b(grau)").reset_index(drop=True)
    print("\n=== Resumo dos Campos ===")
    print(df_campos.to_string(index=False, float_format=lambda v: f"{v:,.6g}"))

    # Inferência Bayesiana para N_total
    prior = PriorLognormal(mu_log=math.log(1e11), sigma_log=2.0)
    print("\n>> Rodando MCMC (Metropolis–Hastings) em N_total ...")
    amostras, taxa = mcmc_metropolis_N(n_dict, p_dict, prior,
                                       N_inicial=1e11,
                                       passos=60_000, burn=15_000, thin=10,
                                       saltolog=0.25)
    print(f"Taxa de aceitação ~ {100*taxa:.1f}%   |  Amostras efetivas: {len(amostras)}")

    # Sumário posterior
    def resumo(v):
        return {
            "media": float(np.mean(v)),
            "mediana": float(np.median(v)),
            "dp": float(np.std(v, ddof=1)),
            "q2.5%": float(np.quantile(v, 0.025)),
            "q97.5%": float(np.quantile(v, 0.975))
        }
    sumN = resumo(amostras)
    sumN["MAP (aprox)"] = float(amostras[np.argmax(np.histogram(amostras, bins=80)[0])])
    print("\n=== Sumário Posterior de N_total (número de estrelas) ===")
    for k,v in sumN.items():
        print(f"{k:>10s}: {v:,.6g}")
    print(f"N_total verdadeiro (simulado): {N_total_verdadeiro:,.6g}")

    # ------------- Gráficos -------------
    plt.figure(figsize=(7.2,4.5))
    plt.hist(amostras/1e11, bins=60, density=True, alpha=0.8)
    plt.axvline(N_total_verdadeiro/1e11, linestyle="--", linewidth=2)
    plt.title("Posterior de N_total (em unidades de 1e11 estrelas)")
    plt.xlabel("N_total / 1e11")
    plt.ylabel("Densidade")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7.2,4.5))
    plt.bar(df_campos["campo"], df_campos["n_i (observado)"], label="observado")
    plt.plot(df_campos["campo"], df_campos["lambda_i esper. (com N_true)"], "o-", label="esperado (N_true)")
    plt.xticks(rotation=30)
    plt.title("Contagens por campo: observado vs. esperado (N_true)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {
        "campos": df_campos,
        "amostras_N": amostras,
        "sumario": sumN,
        "p_dict": p_dict,
        "n_dict": n_dict
    }

# --------------------------
# 12) Executar (exemplo)
# --------------------------
if __name__ == "__main__":
    resultados = rodar_pipeline(
        N_total_verdadeiro = 1.2e11,   # você pode alterar
        N_mc = 120_000                 # ↑ aumenta precisão; 60–150k é Ok em notebook
    )

