# ============================================================
# Emissão de Neutrinos Solares 
# Autor: Luiz Tiago Wilcke (LT)
# ============================================================

import math
from dataclasses import dataclass

# -------------------------
# 1) Constantes físicas
# -------------------------
AU = 1.495978707e11        # metro
NA = 6.02214076e23         # mol^-1
L_solar = 3.828e26         # Watt
MeV = 1.602176634e-13      # Joule
Q_He = 26.73 * MeV         # J por 4p->He (inclui energia dos neutrinos)
cm2 = 1e-4                 # m^2 -> cm^2

# -------------------------
# 2) Parâmetros do núcleo
# -------------------------
@dataclass
class NucleoSolar:
    T_c: float = 1.57e7           # K (temperatura central típica)
    rho_c: float = 1.5e5          # kg/m^3 (densidade central ~150 g/cm^3)
    X: float = 0.34               # fração em massa de H no núcleo atual
    Y: float = 0.64               # fração em massa de He no núcleo atual
    Z: float = 0.02               # metais
    Z_CNO: float = 0.01           # fração C+N+O (parte de Z)
    f_CNO_energia: float = 0.01   # fração da L_solar via CNO (1% didático)

p = NucleoSolar()

# -------------------------
# 3) Expoentes de sensibilidade (didáticos)
# -------------------------
expo = {
    "pp":   4.0,
    "pep": -2.4,
    "Be7": 10.0,
    "B8":  24.0,
    "hep": 24.0,
    "CNO": 20.0
}


phi_ref = {
    "pp":  6.0e10,
    "pep": 1.4e8,
    "Be7": 4.8e9,
    "B8":  5.0e6,
    "hep": 8.0e3,
    "N13": 3.0e8,
    "O15": 2.2e8
}
T_ref = 1.57e7
rho_ref = 1.5e5
X_ref, Y_ref, ZCNO_ref = 0.34, 0.64, 0.01

def escala_txy(valor_ref, T, rho, X, Y, Zcno, canal):
    """Escala um fluxo referência por leis de potência aproximadas."""
    # Dependências em composição (muito simplificadas)
    if canal in ("pp","pep"):
        comp = (X/X_ref)**2 * (rho/rho_ref)
    elif canal in ("Be7","B8"):
        comp = (X/X_ref)*(Y/Y_ref) * (rho/rho_ref)
    elif canal in ("N13","O15"):
        comp = (X/X_ref)*(Zcno/ZCNO_ref) * (rho/rho_ref)
    elif canal == "hep":
        comp = (X/X_ref)*(rho/rho_ref)
    else:
        comp = (rho/rho_ref)

    # Dependência em temperatura
    if canal == "N13" or canal == "O15":
        nu = expo["CNO"]
    elif canal == "Be7":
        nu = expo["Be7"]
    elif canal == "B8":
        nu = expo["B8"]
    elif canal == "pep":
        nu = expo["pep"]
    elif canal == "hep":
        nu = expo["hep"]
    else:
        nu = expo["pp"]

    return valor_ref * (T/T_ref)**nu * comp


def estimar_fluxos_brutos(nucleo: NucleoSolar):
    T, rho, X, Y, Zcno = nucleo.T_c, nucleo.rho_c, nucleo.X, nucleo.Y, nucleo.Z_CNO

    phi = {}
    for k in ["pp","pep","Be7","B8","hep","N13","O15"]:
        phi[k] = escala_txy(phi_ref[k], T, rho, X, Y, Zcno, k)  # cm^-2 s^-1

    
    phi_cno = phi["N13"] + phi["O15"]
    phi_ppfam = phi["pp"] + phi["pep"] + phi["Be7"] + phi["B8"] + phi["hep"]

    # Estimar energia relativa
    Erel_cno = phi_cno / 2.0
    Erel_pp  = phi_ppfam / 2.2
    frac_cno_calc = Erel_cno / (Erel_cno + Erel_pp + 1e-30)
    alvo = nucleo.f_CNO_energia
    if frac_cno_calc > 0:
        fator_cno = (alvo / frac_cno_calc)**0.5
        phi["N13"] *= fator_cno
        phi["O15"] *= fator_cno

    return phi

# -------------------------
# 6) Oscilações: P_ee(E) suave (vacuum -> MSW)
# -------------------------
def Pee(E_MeV, P_low=0.62, P_high=0.31, E0=2.0, dE=0.5):
    # Forma logística: alta energia -> P_high; baixa energia -> P_low
    x = -(E_MeV - E0)/dE
    return P_low - (P_low - P_high)/(1.0 + math.exp(x))

# Energias representativas (MeV) para aplicar P_ee
E_repr = {
    "pp": 0.265,     # média aproximada do contínuo pp
    "pep": 1.44,     # linha pep
    "Be7": 0.862,    # dominante
    "B8": 7.0,       # típico no contínuo B8
    "hep": 10.0,     # alto
    "N13": 0.70,     # típico
    "O15": 1.00      # típico
}

# -------------------------
# 7) Empacotar tudo
# -------------------------
def calcular_fluxos(nucleo: NucleoSolar):
    # Fluxos produzidos no Sol chegando à Terra (sem oscilação), cm^-2 s^-1
    phi = estimar_fluxos_brutos(nucleo)

    # Aplicar probabilidade de sobrevivência ν_e
    phi_surv = {}
    for k, val in phi.items():
        p = Pee(E_repr[k])
        phi_surv[k] = val * p

    # Calcular número de reações 4p->He por segundo a partir da luminosidade
    Ndot_He = L_solar / Q_He  # s^-1

    # Consistência: estimar "neutrinos por He" efetivo do conjunto (diagnóstico)
    # Fluxo total (todos canais) -> taxa total passando pela esfera de 1 UA:
    esfera = 4*math.pi*(AU**2) / cm2  # em cm^2
    Ndot_nu_total = sum(phi.values()) * esfera  # s^-1
    nu_por_He_efetivo = Ndot_nu_total / (Ndot_He + 1e-30)

    return {
        "produzido_cm2s": phi,
        "sobrevivente_cm2s": phi_surv,
        "nu_por_He_efetivo": nu_por_He_efetivo,
        "Ndot_He_s": Ndot_He
    }

# -------------------------
# 8) Rodar e imprimir
# -------------------------
res = calcular_fluxos(p)

def fmt(x): 
    # formato compacto
    if x == 0: return "0"
    expo = int(math.floor(math.log10(abs(x))))
    mant = x / (10**expo)
    return f"{mant:.3f}e{expo:+d}"

print("=== Parâmetros do núcleo (1-zona) ===")
print(f"T_c = {p.T_c:.3e} K, rho_c = {p.rho_c:.3e} kg/m^3, X={p.X:.3f}, Y={p.Y:.3f}, Z_CNO={p.Z_CNO:.3f}")
print(f"Fração de energia via CNO (alvo) = {p.f_CNO_energia:.3%}")
print()
print("=== Fluxos de neutrinos na Terra (produzidos) [cm^-2 s^-1] ===")
for k in ["pp","pep","Be7","B8","hep","N13","O15"]:
    print(f"{k:>4}: {fmt(res['produzido_cm2s'][k])}")

print("\n=== Após oscilação (ν_e sobreviventes) [cm^-2 s^-1] ===")
for k in ["pp","pep","Be7","B8","hep","N13","O15"]:
    print(f"{k:>4}: {fmt(res['sobrevivente_cm2s'][k])}")

print("\n=== Diagnósticos ===")
print(f"Reações 4p->He por segundo (do balanço L_solar): {res['Ndot_He_s']:.3e} s^-1")
print(f"Neutrinos por He (efetivo, a partir dos fluxos):  {res['nu_por_He_efetivo']:.3f}")
print("\nObs.: Os valores são didáticos e devem cair nas ordens de grandeza do SSM.")
