# ============================================================
# Modelo de Previsão do Crescimento Econômico com EDOs
# (oferta: Solow estendido K-H-A; demanda: hiato; nominal: Phillips + Taylor)
# Autor: Luiz Tiago Wilcke (LT)
# ============================================================

import numpy as np
import pandas as pd
import math
from dataclasses import dataclass
from typing import Dict, Tuple
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ==========================
# 1) Parâmetros do modelo
# ==========================
@dataclass
class Parametros:
    # Produção (Cobb-Douglas) e depreciação
    alpha: float = 0.33          # participação do capital físico
    delta_k: float = 0.05        # depreciação anual do capital físico
    delta_h: float = 0.02        # depreciação/obsolescência do capital humano
    delta_a: float = 0.01        # obsolescência tecnológica

    # Demografia
    n_pop: float = 0.008         # crescimento intrínseco anual da população
    L_max: float = 300.0         # limite logístico (milhões)

    # Investimentos endógenos
    s0: float = 0.22             # taxa de investimento privado de referência
    s_i: float = 0.08            # sensibilidade de investimento a juros
    tau_edu: float = 0.045       # fração do PIB em educação
    tau_pd: float  = 0.020       # fração do PIB em P&D
    gamma_h: float = 0.30        # eficiência gasto-educação → H
    gamma_a: float = 0.40        # eficiência P&D → A

    # Bloco fiscal
    g_share: float = 0.18        # G/PIB
    t_share: float = 0.19        # T/PIB

    # Bloco nominal (inflação e juros)
    pi_estrela: float = 0.035    # meta de inflação
    r_neutro: float = 0.03       # juros reais neutros
    a_pi: float = 1.5            # peso da inflação na regra de Taylor
    a_y: float  = 0.5            # peso do hiato na regra de Taylor
    kappa: float = 0.60          # inclinação da Phillips (π responde ao hiato)
    theta: float = 0.80          # ancoragem à meta (reversão)

    # Hiato do produto (x = ln(Y/Y_pot))
    lambda_x: float = 0.6        # auto-reversão do hiato
    psi_i: float = 0.8           # efeito de juros no hiato

    # Choques (nível)
    choque_demanda: float = 0.0
    choque_inflacao: float = 0.0

    # Limites/segurança numérica
    inv_min: float = 0.02
    inv_max: float = 0.35


# ==============================
# 2) Funções auxiliares
# ==============================
def producao_potencial(K: float, H: float, A: float, L: float, alpha: float) -> float:
    """Y_pot = K^α * (A*H*L)^(1-α)."""
    return (K**alpha) * ((A * H * L)**(1.0 - alpha))

def limites_investimento(s: float, inv_min: float, inv_max: float) -> float:
    return max(inv_min, min(inv_max, s))

def regras_politica(pi: float, x: float, p: Parametros) -> float:
    """
    Regra de Taylor (juros reais i):
        i = r_neutro + a_pi*(pi - pi_estrela) + a_y*x
    Piso em 0 (ZLB real) por robustez.
    """
    i = p.r_neutro + p.a_pi*(pi - p.pi_estrela) + p.a_y * x
    return max(0.0, i)

def fluxo_fiscal(Y: float, p: Parametros) -> Tuple[float, float, float, float]:
    """
    G, I_pub (25% de G), T e resultado primário S_prim = T - (G - I_pub).
    """
    G = p.g_share * Y
    I_pub = 0.25 * G
    T = p.t_share * Y
    S_prim = T - (G - I_pub)
    return G, I_pub, T, S_prim


# ==============================
# 3) Sistema dinâmico (EDOs)
# ==============================
def dinamica(t: float, estado: np.ndarray, p: Parametros) -> np.ndarray:
    """
    Estado: [K, H, A, L, Drel, pi, x]
      K: capital físico
      H: capital humano
      A: tecnologia (TFP)
      L: população (milhões)
      Drel: dívida/PIB (razão)
      pi: inflação (a.a., decimal)
      x: hiato do produto (log)
    """
    K, H, A, L, Drel, pi, x = estado

    # Produção
    Y_pot = producao_potencial(K, H, A, L, p.alpha)
    Y = Y_pot * math.exp(x)

    # Política monetária
    i_real = regras_politica(pi, x, p)

    # Investimento privado dependente de juros
    s_priv = limites_investimento(p.s0 - p.s_i * (i_real - p.r_neutro), p.inv_min, p.inv_max)
    I_priv = s_priv * Y

    # Fiscal
    G, I_pub, T, S_prim = fluxo_fiscal(Y, p)
    I_total = I_priv + I_pub

    # Dinâmica de estoques
    dK = I_total - p.delta_k * K
    dH = p.gamma_h * p.tau_edu * Y - p.delta_h * H
    dA = p.gamma_a * p.tau_pd * Y - p.delta_a * A

    # População (logística)
    dL = p.n_pop * L * (1.0 - L/p.L_max)

    # Inflação (Phillips + âncora)
    dpi = p.kappa * x - p.theta * (pi - p.pi_estrela) + p.choque_inflacao

    # Hiato (reversão e juros)
    dx = -p.lambda_x * x - p.psi_i * (i_real - p.r_neutro) + p.choque_demanda

    # Dinâmica da dívida/PIB (Drel = D/Y):
    # d(D/Y) = (i_real + pi)*D/Y + (G - T)/Y - g_nominal*D/Y
    eps = 1e-12
    dlnY_pot = p.alpha * (dK/max(K,eps)) + (1.0 - p.alpha) * (
        (dA/max(A,eps)) + (dH/max(H,eps)) + (dL/max(L,eps))
    )
    g_real = dlnY_pot + dx
    g_nominal = g_real + pi
    deficit_primario_share = (G - T)/Y
    dDrel = (i_real + pi) * Drel + deficit_primario_share - g_nominal * Drel

    return np.array([dK, dH, dA, dL, dDrel, dpi, dx], dtype=float)


# ==================================
# 4) Simulação e utilitários
# ==================================
def simular(horizonte_anos: float = 25.0, p: Parametros = Parametros(),
            estado_inicial: Dict[str, float] = None, passos: int = 2001):
    """
    Resolve as EDOs em [0, horizonte_anos] e retorna (DataFrame, parâmetros).
    """
    if estado_inicial is None:
        estado_inicial = {
            "K": 100.0,
            "H": 50.0,
            "A": 1.0,
            "L": 200.0,
            "Drel": 0.75,
            "pi": 0.05,
            "x": -0.02
        }

    y0 = np.array([estado_inicial[k] for k in ["K","H","A","L","Drel","pi","x"]], dtype=float)
    t_span = (0.0, horizonte_anos)
    t_eval = np.linspace(t_span[0], t_span[1], passos)

    sol = solve_ivp(lambda t, y: dinamica(t, y, p),
                    t_span=t_span, y0=y0, t_eval=t_eval,
                    max_step=0.1, rtol=1e-7, atol=1e-9)

    # Reconstruções
    registros = []
    for ti, vec in zip(sol.t, sol.y.T):
        K, H, A, L, Drel, pi, x = vec
        Y_pot = producao_potencial(K, H, A, L, p.alpha)
        Y = Y_pot * math.exp(x)
        i_real = regras_politica(pi, x, p)
        registros.append({
            "t": ti,
            "K": K, "H": H, "A": A, "L": L,
            "PIB_pot": Y_pot, "PIB": Y,
            "divida_PIB": Drel,
            "inflacao": pi,
            "juros_reais": i_real,
            "hiato_log": x
        })

    df = pd.DataFrame(registros)

    # Crescimentos (diferença de log * 100)
    for col in ["PIB", "PIB_pot", "K", "H", "A", "L"]:
        df[f"g_{col}"] = 100.0 * np.gradient(
            np.log(np.maximum(df[col].values, 1e-12)), df["t"].values
        )
    df["g_real_aprox_%"] = df["g_PIB"]
    df["inflacao_%"] = 100.0 * df["inflacao"]
    df["juros_reais_%"] = 100.0 * df["juros_reais"]
    return df, p


# =========================
# 5) Rodar e visualizar
# =========================
if __name__ == "__main__":
    df, pars = simular(horizonte_anos=25.0, passos=2001)

    # Gráfico 1: Crescimento real aproximado (% a.a.)
    plt.figure()
    plt.plot(df["t"], df["g_real_aprox_%"])
    plt.xlabel("Tempo (anos)")
    plt.ylabel("Crescimento do PIB real (% a.a.)")
    plt.title("Trajetória do Crescimento do PIB (modelo EDO)")
    plt.grid(True)
    plt.show()

    # Gráfico 2: Inflação (% a.a.)
    plt.figure()
    plt.plot(df["t"], df["inflacao_%"])
    plt.xlabel("Tempo (anos)")
    plt.ylabel("Inflação (% a.a.)")
    plt.title("Inflação (Phillips + meta)")
    plt.grid(True)
    plt.show()

    # Gráfico 3: Juros reais (% a.a.)
    plt.figure()
    plt.plot(df["t"], df["juros_reais_%"])
    plt.xlabel("Tempo (anos)")
    plt.ylabel("Juros reais (% a.a.)")
    plt.title("Juros Reais (Regra de Taylor)")
    plt.grid(True)
    plt.show()

    # Gráfico 4: Dívida/PIB
    plt.figure()
    plt.plot(df["t"], df["divida_PIB"])
    plt.xlabel("Tempo (anos)")
    plt.ylabel("Dívida / PIB (razão)")
    plt.title("Dinâmica da Dívida Pública (D/PIB)")
    plt.grid(True)
    plt.show()

    # Gráfico 5: Níveis de PIB e PIB potencial
    plt.figure()
    plt.plot(df["t"], df["PIB"])
    plt.xlabel("Tempo (anos)")
    plt.ylabel("PIB (nível relativo)")
    plt.title("PIB Efetivo")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(df["t"], df["PIB_pot"])
    plt.xlabel("Tempo (anos)")
    plt.ylabel("PIB Potencial (nível relativo)")
    plt.title("PIB Potencial (Cobb-Douglas)")
    plt.grid(True)
    plt.show()

    # Gráfico 6: Estoques e demografia
    plt.figure(); plt.plot(df["t"], df["K"]); plt.xlabel("Tempo (anos)"); plt.ylabel("Capital Físico (nível)"); plt.title("Evolução do Capital Físico (K)"); plt.grid(True); plt.show()
    plt.figure(); plt.plot(df["t"], df["H"]); plt.xlabel("Tempo (anos)"); plt.ylabel("Capital Humano (nível)"); plt.title("Evolução do Capital Humano (H)"); plt.grid(True); plt.show()
    plt.figure(); plt.plot(df["t"], df["A"]); plt.xlabel("Tempo (anos)"); plt.ylabel("Tecnologia (índice)"); plt.title("Evolução da Tecnologia (A)"); plt.grid(True); plt.show()
    plt.figure(); plt.plot(df["t"], df["L"]); plt.xlabel("Tempo (anos)"); plt.ylabel("População (milhões)"); plt.title("Dinâmica Populacional (L)"); plt.grid(True); plt.show()

    # Salvar CSV com a trajetória completa
    caminho_csv = "trajetoria_modelo_EDO_crescimento.csv"
    df.to_csv(caminho_csv, index=False)
    print(f"CSV salvo em: {caminho_csv}")
