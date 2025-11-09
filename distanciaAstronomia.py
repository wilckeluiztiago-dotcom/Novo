# ============================================================
# Estimativa de Raio e Distância Estelar via ODE de Fluxo
# Autor: Luiz Tiago Wilcke (LT)
# Descrição:
#   - Modelo físico: dF/dr = -2F/r - k_ext*F  (espalhamento esférico + extinção)
#   - Condição de contorno: F(R_*) = sigma_SB * T_eff^4 (aprox. corpo negro)
#   - Se o usuário possuir F_obs (fluxo bolométrico medido), o método estima R_* e d.
#   - Quando só houver T_eff e não houver F_obs, o exemplo abaixo SIMULA F_obs
#     a partir de parâmetros conhecidos de uma estrela próxima (para teste).
#   - Inclui integração numérica da ODE, tabela de resultados e gráfico comparativo.
# Dependências: numpy, scipy, pandas, matplotlib
# ============================================================

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Tuple
from scipy.integrate import solve_ivp

# -------------------------
# Constantes físicas (SI)
# -------------------------
sigma_SB = 5.670374419e-8     # Constante de Stefan-Boltzmann (W m^-2 K^-4)
R_solar  = 6.957e8            # Raio solar (m)
L_solar  = 3.828e26           # Luminosidade solar (W)
pc       = 3.085677581491367e16  # Parsec (m)
ly       = 9.4607304725808e15     # Ano-luz (m)

# --------------------------------------------------------
# Aproximação de sequência principal: R_* ~ (T/5772)^alpha
# --------------------------------------------------------
def raio_por_temperatura(T_eff: float, alpha: float = 1.8) -> float:
    """Retorna R_* em metros a partir de T_eff usando uma lei de potência simples."""
    return R_solar * (T_eff / 5772.0)**alpha

# --------------------------------------------------------
# ODE: dF/dr = -2F/r - k_ext*F
# --------------------------------------------------------
def ode_fluxo(r: float, F: np.ndarray, k_ext: float) -> np.ndarray:
    # F é escalar (colocado em array para solve_ivp)
    return np.array([ -2.0 * F[0] / r - k_ext * F[0] ])

def integrar_fluxo(R_estrela: float, T_eff: float, k_ext: float,
                   r_alvo: float, n_pontos: int = 400) -> Tuple[np.ndarray, np.ndarray]:
    """Integra a ODE do fluxo de r=R_* até r=r_alvo."""
    F_R = sigma_SB * (T_eff**4)  # condição de contorno na fotosfera
    def f(r, y): return ode_fluxo(r, y, k_ext)
    sol = solve_ivp(f, (R_estrela, r_alvo), y0=[F_R], dense_output=True, rtol=1e-9, atol=1e-12)
    r_grid = np.linspace(R_estrela, r_alvo, n_pontos)
    F_grid = sol.sol(r_grid)[0]
    return r_grid, F_grid

# --------------------------------------------------------
# Estimadores
# --------------------------------------------------------
def estimar_raio_distancia_por_fluxo(T_eff: float,
                                     fluxo_bolometrico_observado: float,
                                     k_ext: float = 0.0,
                                     alpha_TR: float = 1.8) -> Tuple[float, float, float]:
    """
    Estima R_* e d usando:
      1) R_*/d via F_obs = sigma*T^4*(R_*/d)^2 * exp[-k_ext (d - R_*)]
      2) R_* via relação R ~ (T/5772)^alpha
    Retorna (R_estrela [m], d [m], razao_R_sobre_d).
    Observação: se k_ext não for zero, há um termo em d dentro do expoente.
    Aqui, como aproxima inicial robusta, consideramos k_ext pequeno para (R/d) e
    depois checamos com a ODE (integração) a consistência do F(d).
    """
    # Passo 1: razão (R/d) assumindo extinção pequena (ou absorvida no F_obs calibrado)
    razao_R_sobre_d = math.sqrt( max(1e-300, fluxo_bolometrico_observado / (sigma_SB * T_eff**4)) )

    # Passo 2: R_* pela aproximação de sequência principal
    R_estrela = raio_por_temperatura(T_eff, alpha_TR)

    # Passo 3: distância d pela razão estimada
    d = R_estrela / razao_R_sobre_d

    return R_estrela, d, razao_R_sobre_d

# --------------------------------------------------------
# Modo de demonstração: simulando F_obs de uma estrela próxima
# --------------------------------------------------------
@dataclass
class ConfigEstrela:
    nome: str = "Proxima Centauri (exemplo)"
    T_eff: float = 3040.0            # K (aprox M5.5V)
    # Para simulação de F_obs (opcional): use valores “literatura” aproximados
    # para testar a recuperação do método. Isso NÃO é necessário se você tiver F_obs medido.
    L_em_Lsol: float = 0.0017        # luminosidade ~0.17% do Sol (aprox)
    distancia_pc_real: float = 1.301 # parsecs (aprox)
    k_ext: float = 0.0               # extinção ~0 no entorno solar (teste)
    alpha_TR: float = 1.8            # expoente da relação R~(T/5772)^alpha

def simular_fluxo_bolometrico(L_em_Lsol: float, distancia_pc: float) -> float:
    """Gera um F_obs 'verdadeiro' a partir de L e d (para teste do método)."""
    L = L_em_Lsol * L_solar
    d_m = distancia_pc * pc
    return L / (4.0 * math.pi * d_m**2)

def checar_fluxo_por_ode(R_estrela: float, T_eff: float, k_ext: float, d: float) -> float:
    """Integra a ODE até r=d e retorna F_ode(d) para conferência."""
    _, F_grid = integrar_fluxo(R_estrela, T_eff, k_ext, r_alvo=d, n_pontos=400)
    return float(F_grid[-1])

# --------------------------------------------------------
# Execução principal (exemplo)
# --------------------------------------------------------
if __name__ == "__main__":
    cfg = ConfigEstrela()  # pode trocar por Alpha Centauri A, Sirius, etc., ajustando T_eff, L, d, k_ext

    # 1) (Opcional) Simular F_obs a partir de L e d "reais" (apenas para demonstrar)
    fluxo_bolometrico_observado = simular_fluxo_bolometrico(cfg.L_em_Lsol, cfg.distancia_pc_real)

    # 2) Estimar R_* e d usando F_obs e T_eff (com a hipótese R~(T/5772)^alpha)
    R_est, d_est, R_sobre_d = estimar_raio_distancia_por_fluxo(
        T_eff=cfg.T_eff,
        fluxo_bolometrico_observado=fluxo_bolometrico_observado,
        k_ext=cfg.k_ext,
        alpha_TR=cfg.alpha_TR
    )

    # 3) Conferir por ODE qual F(d_est) o modelo produz
    F_ode_em_dest = checar_fluxo_por_ode(R_est, cfg.T_eff, cfg.k_ext, d_est)

    # 4) Comparar com solução analítica (quando k_ext constante)
    F_R = sigma_SB * (cfg.T_eff**4)
    F_analitica_em_dest = F_R * (R_est/d_est)**2 * math.exp(-cfg.k_ext * (d_est - R_est))

    # 5) Juntar resultados em tabela
    dados = {
        "estrela": [cfg.nome],
        "T_eff_K": [cfg.T_eff],
        "k_ext_menos1": [cfg.k_ext],
        "alpha_TR": [cfg.alpha_TR],
        "fluxo_obs_W_m2": [fluxo_bolometrico_observado],
        "raio_estimado_m": [R_est],
        "raio_estimado_em_Rsol": [R_est / R_solar],
        "distancia_estimado_m": [d_est],
        "distancia_estimado_pc": [d_est / pc],
        "distancia_estimado_ly": [d_est / ly],
        "R_sobre_d": [R_sobre_d],
        "F_ode_em_d_W_m2": [F_ode_em_dest],
        "F_analitica_em_d_W_m2": [F_analitica_em_dest],
        "erro_relativo_ode_vs_obs_%": [100.0 * abs(F_ode_em_dest - fluxo_bolometrico_observado) / fluxo_bolometrico_observado],
    }
    df = pd.DataFrame(dados)
    pd.set_option("display.precision", 6)
    print("\n=== Resultados (Tabela) ===")
    print(df.to_string(index=False))

    # 6) Curvas F(r): ODE vs solução analítica vs 1/r^2 sem extinção (para inspeção)
    r_plot = np.linspace(R_est, d_est, 400)
    # ODE
    r_grid, F_grid = integrar_fluxo(R_est, cfg.T_eff, cfg.k_ext, r_alvo=d_est, n_pontos=400)
    # Analítica
    F_analitica = (sigma_SB * cfg.T_eff**4) * (R_est / r_plot)**2 * np.exp(-cfg.k_ext * (r_plot - R_est))
    # 1/r^2 puro (k_ext=0)
    F_invquad = (sigma_SB * cfg.T_eff**4) * (R_est / r_plot)**2

    plt.figure(figsize=(7.2, 4.4))
    plt.loglog(r_grid/R_est, F_grid, label="ODE (numérico)")
    plt.loglog(r_plot/R_est, F_analitica, linestyle="--", label="Analítica")
    plt.loglog(r_plot/R_est, F_invquad, linestyle=":", label="1/r^2 (k_ext=0)")
    plt.xlabel("r / R_*")
    plt.ylabel("Fluxo F(r)  [W/m^2]")
    plt.title(f"Perfil de Fluxo — {cfg.nome}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 7) Dica: Se você tiver F_obs real (bolométrico) da sua fotometria:
    #    - Substitua 'fluxo_bolometrico_observado' pelo seu valor medido.
    #    - Ajuste T_eff (p.ex., via cor B-V ou ajuste de SED).
    #    - Ajuste k_ext (p.ex., a partir de E(B-V) -> A_V -> k_ext ~ A_V / d).
