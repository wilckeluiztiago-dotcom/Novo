"""
Modelo Gauss–Markov aplicado a um modelo climático com luminosidade estelar
-----------------------------------------------------------------------------
Autor: Luiz Tiago Wilcke (LT)
Linguagem: Python 3.x
Dependências sugeridas: numpy, pandas, matplotlib, statsmodels, scipy

Resumo da ideia (para TCC):
- Construímos um modelo físico‑estatístico simples para temperatura média de um
  planeta que orbita uma estrela. A temperatura média de equilíbrio depende da
  energia recebida da estrela (luminosidade L) e é modulada por distância
  orbital (d), albedo planetário (α) e um índice de efeito estufa (G).
- Do balanço de energia de primeira ordem, a potência média absorvida por m² é:
      Q_in = (1 - α) * S / 4,   onde S = L / (4π d²) é o "solar constant" na órbita.
  No equilíbrio radiativo simples (sem efeito estufa), σ T_e^4 = Q_in.
- Para aproximar o efeito estufa, modelamos que a emissão efetiva é:
      σ T^4 = Q_in * (1 + γ G)   (γ > 0)  
  onde G é um índice (e.g., CO₂ equivalente padronizado).
- Para estimar parâmetros por MQO (Gauss–Markov), definimos:
      y   = σ T^4 (resposta)
      x1  = (1 - α) * S          (forçante radiativa limpa)
      x2  = G                    (índice GEE)
      x12 = x1 * x2              (interação efeito estufa × forçante)
      y = β0 + β1 x1 + β2 x2 + β3 x12 + ε
  É linear nos parâmetros (condição de linearidade do Teorema de Gauss–Markov).

O script inclui:
1) Geração de dados sintéticos fisicamente plausíveis (com ruído controlado).
2) Estimação por MQO implementada "na mão" (β̂ = (X'X)⁻¹ X'y), com erros‑padrão
   clássicos, White (robustos a heteroscedasticidade) e HC3.
3) Diagnósticos: VIF (multicolinearidade), Durbin–Watson (autocorrelação),
   Breusch–Pagan e White (heteroscedasticidade), Jarque–Bera (normalidade).
4) Checagem operacional das hipóteses de Gauss–Markov e discussão nos comentários.
5) Visualizações (opcionais) para resíduos e ajuste.
6) Comparação com statsmodels (para conferência de resultados e p‑values).

Notas didáticas:
- O teorema de Gauss–Markov garante que, sob: (i) linearidade em β, (ii) posto
  completo (sem multicolinearidade perfeita), (iii) exogeneidade E[ε|X]=0 e
  (iv) homoscedasticidade Var(ε|X)=σ² I, o estimador MQO é BLUE (Best Linear
  Unbiased Estimator). A normalidade de ε não é necessária para insesgamento,
  mas ajuda na inferência exata finita; com N grande, CLT ajuda.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass

# Opcional: se disponíveis, usaremos estes pacotes para testes e comparação.
try:
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import het_breuschpagan, het_white, acorr_breusch_godfrey
    from statsmodels.stats.stattools import jarque_bera
    from statsmodels.stats.outliers_influence import variance_inflation_factor
except Exception:
    sm = None

try:
    from scipy import stats
except Exception:
    stats = None

# ----------------------------- Constantes físicas -----------------------------
SIGMA_SB = 5.670374419e-8   # Constante de Stefan-Boltzmann [W m^-2 K^-4]
PI = np.pi

# ----------------------------- Geração de dados -------------------------------
@dataclass
class ParametrosGeracao:
    n_observacoes: int = 600               # número de observações (e.g., horas ou dias)
    media_L: float = 3.828e26              # Luminosidade solar [W] (Sol)
    cv_L: float = 0.001                    # variação relativa (p.ex. atividade estelar)
    distancia_media: float = 1.496e11      # 1 UA [m]
    excentricidade: float = 0.0167         # similar à da Terra
    albedo_medio: float = 0.30             # albedo planetário médio
    ruido_albedo: float = 0.03             # amplitude de variação de α
    G_medio: float = 1.0                   # índice GEE médio (adimensional)
    ruido_G: float = 0.20                  # variação do índice GEE
    gamma_true: float = 0.35               # intensidade verdadeira do efeito estufa
    beta0_true: float = 0.0                # intercepto verdadeiro em σ T^4
    desvio_padrao_eps: float = 7.5         # σ do erro (em W/m²) para σT^4
    hetero: bool = True                    # se True, gera heteroscedasticidade leve
    seed: int = 42


def gerar_dados(param: ParametrosGeracao) -> pd.DataFrame:
    """Gera dados sintéticos fisicamente coerentes para o modelo.

    Produz colunas: tempo, L, d, albedo, G, S, x1, x2, x12, y (σT^4), T
    """
    rng = np.random.default_rng(param.seed)
    t = np.arange(param.n_observacoes)

    # Variação estocástica suave da luminosidade estelar
    L = rng.normal(loc=param.media_L, scale=param.cv_L*param.media_L, size=param.n_observacoes)

    # Distância orbital variando com excentricidade (aprox. elipse) + ruído pequeno
    fase = 2*PI*t/param.n_observacoes
    d = param.distancia_media * (1 - param.excentricidade*np.cos(fase))
    d = d * (1 + rng.normal(0, 0.002, size=param.n_observacoes))

    # Albedo (α) variando por nuvens/gelo
    albedo = np.clip(rng.normal(param.albedo_medio, param.ruido_albedo, size=param.n_observacoes), 0.05, 0.9)

    # Índice de gases de efeito estufa (CO2eq padronizado)
    G = np.clip(rng.normal(param.G_medio, param.ruido_G, size=param.n_observacoes), 0.1, None)

    # Solar constant na órbita: S = L /(4π d²)
    S = L / (4*PI*d**2)

    # Forçante limpa x1 = (1 - α) * S  (NOTA: já incorpora o /4 do balanço mais adiante.)
    x1 = (1 - albedo) * S

    # x2 = G  (índice GEE)
    x2 = G

    # Interação física: maior estufa amplifica energia efetiva
    x12 = x1 * x2

    # Verdade do gerador: σ T^4 = β0 + b1*x1/4 + b3*(x1*x2)/4   com b1≈1, b3≈gamma_true
    b0 = param.beta0_true
    b1 = 1.0
    b2 = 0.0  # termo linear em G puro (fica para ser testado/estimado)
    b3 = param.gamma_true

    mu = b0 + (b1 * x1 + b3 * x12)/4.0 + b2 * x2

    # Heteroscedasticidade opcional: var(ε|X) proporcional a (1 + x1)*σ²
    if param.hetero:
        sd = param.desvio_padrao_eps * np.sqrt(1 + (x1/np.mean(x1)))
    else:
        sd = np.full_like(mu, param.desvio_padrao_eps)

    eps = rng.normal(0.0, sd)
    y = mu + eps                   # y = σ T^4 (W/m²)

    # Converter para temperatura de superfície (aprox.)
    T = np.maximum( (y / SIGMA_SB), 1e-9 ) ** 0.25   # K

    df = pd.DataFrame({
        'tempo': t,
        'L': L,
        'distancia': d,
        'albedo': albedo,
        'G': G,
        'S': S,
        'x1': x1,
        'x2': x2,
        'x12': x12,
        'y_sigma_T4': y,
        'T_K': T
    })
    return df

# ------------------------ MQO (Gauss–Markov) "na mão" ------------------------
@dataclass
class ResultadoMQO:
    beta_hat: np.ndarray
    se_classico: np.ndarray
    se_white: np.ndarray
    se_hc3: np.ndarray
    residuos: np.ndarray
    y_hat: np.ndarray
    sigma2_hat: float
    R2: float
    R2_adj: float
    X: np.ndarray
    y: np.ndarray
    nomes: list


def ols_manual(X: np.ndarray, y: np.ndarray, nomes: list[str]) -> ResultadoMQO:
    """Estimador MQO com erros‑padrão clássico, White e HC3.

    - X deve incluir a coluna de 1s (intercepto).
    - y é vetor coluna (n,).
    """
    n, k = X.shape
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    beta_hat = XtX_inv @ (X.T @ y)
    y_hat = X @ beta_hat
    residuos = y - y_hat
    # Estimador clássico de σ² (com gl = n - k)
    sigma2_hat = (residuos @ residuos) / (n - k)

    # Erros‑padrão clássicos
    Var_beta_classico = sigma2_hat * XtX_inv
    se_classico = np.sqrt(np.diag(Var_beta_classico))

    # White (HC0): Var = (X'X)⁻¹ [X' diag(e²) X] (X'X)⁻¹
    meat = np.zeros((k, k))
    for i in range(n):
        xi = X[i:i+1, :]  # (1,k)
        ei = residuos[i]
        meat += (xi.T @ xi) * (ei**2)
    Var_beta_white = XtX_inv @ meat @ XtX_inv
    se_white = np.sqrt(np.diag(Var_beta_white))

    # HC3 (mais conservador): diag(e_i²/(1-h_ii)²)
    H = X @ XtX_inv @ X.T
    h = np.clip(np.diag(H), 1e-9, 1 - 1e-9)
    meat_hc3 = np.zeros((k, k))
    for i in range(n):
        xi = X[i:i+1, :]
        ei = residuos[i] / (1 - h[i])
        meat_hc3 += (xi.T @ xi) * (ei**2)
    Var_beta_hc3 = XtX_inv @ meat_hc3 @ XtX_inv
    se_hc3 = np.sqrt(np.diag(Var_beta_hc3))

    # R² e R² ajustado
    SST = ((y - y.mean())**2).sum()
    SSR = ((y_hat - y.mean())**2).sum()
    R2 = SSR / SST
    R2_adj = 1 - (1-R2)*(n-1)/(n-k)

    return ResultadoMQO(
        beta_hat=beta_hat, se_classico=se_classico, se_white=se_white, se_hc3=se_hc3,
        residuos=residuos, y_hat=y_hat, sigma2_hat=float(sigma2_hat), R2=R2, R2_adj=R2_adj,
        X=X, y=y, nomes=nomes
    )

# -------------------------- Métricas e diagnósticos --------------------------

def vif(X: np.ndarray) -> np.ndarray:
    """Variance Inflation Factor para cada coluna de X (assume intercepto na col 0)."""
    n_cols = X.shape[1]
    vifs = np.zeros(n_cols)
    for j in range(n_cols):
        if j == 0:  # intercepto
            vifs[j] = np.nan
            continue
        mask = np.ones(n_cols, dtype=bool)
        mask[j] = False
        X_j = X[:, j]
        X_others = X[:, mask]
        # MQO para X_j ~ X_others
        beta = np.linalg.lstsq(X_others, X_j, rcond=None)[0]
        y_hat = X_others @ beta
        resid = X_j - y_hat
        R2_j = 1 - (resid @ resid)/(((X_j - X_j.mean())**2).sum())
        vifs[j] = 1 / (1 - R2_j + 1e-12)
    return vifs


def breusch_pagan(residuos: np.ndarray, X: np.ndarray) -> tuple[float,float]:
    """Teste de Breusch–Pagan para heteroscedasticidade (aprox. LM ~ χ²)."""
    n = X.shape[0]
    e2 = residuos**2
    # Regressão auxiliar: e² ~ X
    beta = np.linalg.lstsq(X, e2, rcond=None)[0]
    y_hat = X @ beta
    SSR = ((y_hat - e2.mean())**2).sum()
    R2 = SSR / ((e2 - e2.mean())**2).sum()
    LM = n * R2
    # p‑valor (qui‑quadrado com k-1 gl; sem intercepto efetivo)
    k = X.shape[1]
    try:
        from scipy.stats import chi2
        p = 1 - chi2.cdf(LM, k-1)
    except Exception:
        p = np.nan
    return LM, p


def durbin_watson(residuos: np.ndarray) -> float:
    num = ((np.diff(residuos))**2).sum()
    den = (residuos**2).sum()
    return float(num/den)


def jarque_bera_test(residuos: np.ndarray) -> tuple[float,float]:
    r = residuos
    n = r.size
    m2 = np.mean((r - r.mean())**2)
    m3 = np.mean((r - r.mean())**3)
    m4 = np.mean((r - r.mean())**4)
    skew = m3 / (m2**1.5 + 1e-12)
    kurt = m4 / (m2**2 + 1e-12)
    JB = n/6 * (skew**2 + (1/4)*(kurt - 3)**2)
    try:
        from scipy.stats import chi2
        p = 1 - chi2.cdf(JB, 2)
    except Exception:
        p = np.nan
    return float(JB), float(p)

# ------------------------------- Pipeline principal --------------------------

def preparar_e_ajustar(df: pd.DataFrame) -> ResultadoMQO:
    """Monta a matriz de desenho e ajusta MQO para y = β0 + β1 x1/4 + β2 x2 + β3 x1x2/4 + ε.
    Trabalhamos diretamente em unidades de y = σ T^4 (W/m²)."""
    y = df['y_sigma_T4'].to_numpy()
    x1 = df['x1'].to_numpy() / 4.0
    x2 = df['x2'].to_numpy()
    x12 = (df['x12'].to_numpy()) / 4.0

    # Matriz X com intercepto
    X = np.column_stack([np.ones_like(y), x1, x2, x12])
    nomes = ['Intercepto', 'x1=(1-α)S/4', 'x2=GEE', 'x1x2/4']

    resultado = ols_manual(X, y, nomes)
    return resultado


def tabela_resultados(res: ResultadoMQO) -> pd.DataFrame:
    t_classico = res.beta_hat / res.se_classico
    t_white = res.beta_hat / res.se_white
    t_hc3 = res.beta_hat / res.se_hc3
    df = pd.DataFrame({
        'Regressor': res.nomes,
        'Beta_hat': res.beta_hat,
        'SE_classico': res.se_classico,
        't_classico': t_classico,
        'SE_White': res.se_white,
        't_White': t_white,
        'SE_HC3': res.se_hc3,
        't_HC3': t_hc3,
    })
    return df

# ----------------------------- Visualizações (opcional) ----------------------

def plot_residuos(res: ResultadoMQO, df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    # Resíduos vs Ajustados
    axes[0].scatter(res.y_hat, res.residuos, s=12)
    axes[0].axhline(0, ls='--')
    axes[0].set_xlabel('Ajustado (σT⁴)')
    axes[0].set_ylabel('Resíduo')
    axes[0].set_title('Resíduos vs Ajustado')
    # QQ‑plot (se scipy disponível)
    if stats is not None:
        stats.probplot(res.residuos, dist="norm", plot=axes[1])
        axes[1].set_title('QQ‑plot Resíduos')
    else:
        axes[1].text(0.5, 0.5, 'scipy ausente', ha='center')
    # Resíduo vs x1
    axes[2].scatter(df['x1']/4.0, res.residuos, s=12)
    axes[2].axhline(0, ls='--')
    axes[2].set_xlabel('x1=(1-α)S/4')
    axes[2].set_title('Resíduo vs x1')
    plt.tight_layout()
    plt.show()

# ----------------------------- Execução de exemplo ---------------------------
if __name__ == '__main__':
    # 1) Gerar dados
    param = ParametrosGeracao(
        n_observacoes=720,      # por exemplo, 720 horas (~30 dias)
        hetero=True,            # criamos heteroscedasticidade para demonstrar robustez
        gamma_true=0.35,
        desvio_padrao_eps=6.0,
        seed=123
    )
    df = gerar_dados(param)

    # 2) Ajustar MQO
    resultado = preparar_e_ajustar(df)

    # 3) Tabela de coeficientes
    print('\n=== Estimação MQO (manual) — Modelo σT^4 ~ 1 + x1/4 + x2 + (x1x2)/4 ===')
    print(tabela_resultados(resultado).to_string(index=False))

    print(f"\nR² = {resultado.R2:.4f} | R² ajustado = {resultado.R2_adj:.4f}")
    print(f"σ²̂ (clássico) = {resultado.sigma2_hat:.3f} (W/m²)²")

    # 4) Diagnósticos Gauss–Markov
    VIFs = vif(resultado.X)
    print('\nVIF (inflacao de variancia):')
    for nome, v in zip(resultado.nomes, VIFs):
        print(f"  {nome:>12s}: {v:.3f}")

    DW = durbin_watson(resultado.residuos)
    print(f"\nDurbin–Watson = {DW:.3f} (≈2 sugere pouca autocorrelação)")

    LM, p_bp = breusch_pagan(resultado.residuos, resultado.X)
    print(f"Breusch–Pagan: LM={LM:.3f}, p≈{p_bp:.4f}  (p pequeno ⇒ heteroscedasticidade)")

    JB, p_jb = jarque_bera_test(resultado.residuos)
    print(f"Jarque–Bera: JB={JB:.3f}, p≈{p_jb:.4f}  (p grande ⇒ proximidade de normalidade)")

    # 5) (Opcional) Comparar com statsmodels
    if sm is not None:
        sm_res = sm.OLS(df['y_sigma_T4'], resultado.X).fit()
        print('\n=== statsmodels resumo ===')
        print(sm_res.summary())

        # Testes adicionais (se disponíveis)
        try:
            bp = het_breuschpagan(sm_res.resid, sm_res.model.exog)
            print(f"\n[statsmodels] Breusch–Pagan: LM={bp[0]:.3f}, p={bp[1]:.4f}")
            wh = het_white(sm_res.resid, sm_res.model.exog)
            print(f"[statsmodels] White: LM={wh[0]:.3f}, p={wh[1]:.4f}")
            jb = jarque_bera(sm_res.resid)
            print(f"[statsmodels] Jarque–Bera: JB={jb[0]:.3f}, p={jb[1]:.4f}")
        except Exception:
            pass

    # 6) Visualizações (descomente para ver gráficos)
    # plot_residuos(resultado, df)

    # 7) Interpretação sucinta (como guia para o TCC):
    print("\nInterpretação didática:")
    print("- β1 perto de 1 indica que σT⁴ cresce aproximadamente 1‑para‑1 com a forçante limpa (x1/4).")
    print("- β3 > 0 quantifica o aumento no aquecimento efetivo conforme o índice GEE (interação).")
    print("- Se BP indica heteroscedasticidade, prefira SE White/HC3 para inferência.")
    print("- Se VIF for alto (>10), há multicolinearidade; considere reespecificar variáveis.")
    print("- Se DW muito <2, há autocorrelação; MQO segue insesgado, mas SEs e testes podem falhar.")
