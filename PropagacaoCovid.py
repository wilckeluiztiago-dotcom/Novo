# ============================================================
# MODELO NEURAL–ESTOCÁSTICO–EM-REDE (SIR COVID)
# Autor: Luiz Tiago Wilcke
# ============================================================
# - SIR estocástico (SDE) com ruído multiplicativo
# - SIR discreto em grafo (contato em rede)
# - Rede neural aprende beta(t) de dados ruidosos
# - Modelo híbrido: SIR(SDE) + beta(t) neural
# ============================================================

import math, random, warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import torch
import torch.nn as nn

# -------------------------------
# 0. Utilidades gerais
# -------------------------------
def set_semente(semente=42):
    random.seed(semente)
    np.random.seed(semente)
    torch.manual_seed(semente)

def imprimir_resumo(t, s, i, r, titulo="RESUMO"):
    print("\n" + "="*70)
    print(titulo)
    print("="*70)
    print(f"t_final = {t[-1]:.8f} dias")
    print(f"S_final = {s[-1]:.8f}")
    print(f"I_final = {i[-1]:.8f}")
    print(f"R_final = {r[-1]:.8f}")
    pico_idx = np.argmax(i)
    print(f"I_pico  = {i[pico_idx]:.8f} em t = {t[pico_idx]:.8f} dias")
    print("="*70 + "\n")

# -------------------------------
# 1. SIR Estocástico (SDE)
# -------------------------------
# Equações (forma pesada):
#
# dS = - β(t) S I dt  +  σ_S * S I dW1
# dI = (β(t) S I - γ I) dt + σ_I * S I dW1 - σ_R * I dW2
# dR = γ I dt + σ_R * I dW2
#
# onde W1 e W2 são Wiener independentes.
#
# Discretização Euler–Maruyama:
# X_{t+dt} = X_t + f(X_t,t) dt + G(X_t,t) sqrt(dt) ξ
#
def simular_sir_sde(
    N=1.0,
    beta_const=0.35,
    gamma=0.10,
    sigma_S=0.12,
    sigma_I=0.12,
    sigma_R=0.05,
    t_max=160,
    dt=0.2,
    I0=0.001,
    beta_t_func=None
):
    passos = int(t_max/dt) + 1
    t = np.linspace(0, t_max, passos)

    S = np.zeros(passos); I = np.zeros(passos); R = np.zeros(passos)
    S[0] = N - I0
    I[0] = I0
    R[0] = 0.0

    for k in range(passos-1):
        tk = t[k]
        Sk, Ik, Rk = S[k], I[k], R[k]

        beta_t = beta_t_func(tk) if beta_t_func is not None else beta_const

        # ruídos Gaussianos independentes
        xi1 = np.random.normal()
        xi2 = np.random.normal()

        # drift
        dS_det = -beta_t * Sk * Ik
        dI_det =  beta_t * Sk * Ik - gamma * Ik
        dR_det =  gamma * Ik

        # difusão (multiplicativa)
        dS_sto =  sigma_S * Sk * Ik * math.sqrt(dt) * xi1
        dI_sto =  sigma_I * Sk * Ik * math.sqrt(dt) * xi1 - sigma_R * Ik * math.sqrt(dt) * xi2
        dR_sto =  sigma_R * Ik * math.sqrt(dt) * xi2

        S_next = Sk + dS_det*dt + dS_sto
        I_next = Ik + dI_det*dt + dI_sto
        R_next = Rk + dR_det*dt + dR_sto

        # projeção simples para manter limites
        S_next = max(0.0, min(N, S_next))
        I_next = max(0.0, min(N, I_next))
        R_next = max(0.0, min(N, R_next))
        soma = S_next + I_next + R_next
        if soma != 0:
            S_next, I_next, R_next = (S_next/soma)*N, (I_next/soma)*N, (R_next/soma)*N

        S[k+1], I[k+1], R[k+1] = S_next, I_next, R_next

    return t, S, I, R

def monte_carlo_sir_sde(n_traj=200, **kwargs):
    todas_S, todas_I, todas_R = [], [], []
    for _ in range(n_traj):
        t, S, I, R = simular_sir_sde(**kwargs)
        todas_S.append(S); todas_I.append(I); todas_R.append(R)
    return t, np.array(todas_S), np.array(todas_I), np.array(todas_R)

# -------------------------------
# 2. SIR Estocástico em Grafo
# -------------------------------
# Cada nó: estado {S, I, R}
# Evento de infecção: S -> I com taxa beta * (#vizinhos infectados)
# Evento de recuperação: I -> R com taxa gamma
#
def simular_sir_grafo(
    n=400,
    p_conexao=0.02,
    beta=0.35,
    gamma=0.10,
    t_max=160,
    I0_frac=0.01
):
    G = nx.erdos_renyi_graph(n, p_conexao)
    estados = np.zeros(n, dtype=int)  # 0=S, 1=I, 2=R

    infectados_iniciais = np.random.choice(n, int(I0_frac*n), replace=False)
    estados[infectados_iniciais] = 1

    t = [0.0]
    S_hist = [np.mean(estados == 0)]
    I_hist = [np.mean(estados == 1)]
    R_hist = [np.mean(estados == 2)]

    tempo = 0.0

    while tempo < t_max and np.any(estados == 1):
        taxas = []
        eventos = []

        # construir lista de taxas evento-a-evento
        for u in range(n):
            if estados[u] == 0:
                # taxa de infecção proporcional aos vizinhos infectados
                viz_inf = sum(estados[v] == 1 for v in G.neighbors(u))
                if viz_inf > 0:
                    taxa_inf = beta * viz_inf
                    taxas.append(taxa_inf)
                    eventos.append(("inf", u))
            elif estados[u] == 1:
                taxa_rec = gamma
                taxas.append(taxa_rec)
                eventos.append(("rec", u))

        taxa_total = sum(taxas)
        if taxa_total == 0:
            break

        # tempo próximo evento (Gillespie)
        dt = np.random.exponential(1.0/taxa_total)
        tempo += dt

        # escolher evento
        r = np.random.uniform(0, taxa_total)
        acumulado = 0.0
        escolhido = None
        for taxa, ev in zip(taxas, eventos):
            acumulado += taxa
            if r <= acumulado:
                escolhido = ev
                break

        if escolhido is None:
            continue

        tipo, no = escolhido
        if tipo == "inf":
            estados[no] = 1
        else:
            estados[no] = 2

        t.append(tempo)
        S_hist.append(np.mean(estados == 0))
        I_hist.append(np.mean(estados == 1))
        R_hist.append(np.mean(estados == 2))

    return G, np.array(t), np.array(S_hist), np.array(I_hist), np.array(R_hist)

# -------------------------------
# 3. Dados sintéticos com beta(t)
# -------------------------------
def beta_real(t):
    # "ondas" de transmissão (lockdown/relaxamento)
    return 0.25 + 0.18*np.exp(-(t-35)**2/300) + 0.12*np.exp(-(t-95)**2/500)

def gerar_dados_sinteticos():
    t, S, I, R = simular_sir_sde(
        beta_const=0.0, gamma=0.10,
        sigma_S=0.09, sigma_I=0.09, sigma_R=0.04,
        t_max=160, dt=0.2, I0=0.002,
        beta_t_func=beta_real
    )
    # observação ruidosa de I(t)
    ruido = np.random.normal(0, 0.01, size=len(I))
    I_obs = np.clip(I + ruido, 0, 1)
    return t, S, I, R, I_obs

# -------------------------------
# 4. Rede Neural: aprender beta(t)
# -------------------------------
class RedeBeta(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1), nn.Softplus()  # força beta>=0
        )

    def forward(self, t):
        return self.net(t)

def treinar_rede_beta(t, I_obs, gamma=0.10, epocas=2000, lr=2e-3):
    device = "cpu"
    t_torch = torch.tensor(t, dtype=torch.float32, device=device).view(-1,1)
    I_torch = torch.tensor(I_obs, dtype=torch.float32, device=device).view(-1,1)

    rede = RedeBeta().to(device)
    opt = torch.optim.Adam(rede.parameters(), lr=lr)

    # Treino "fisicamente informado" (PINN simples):
    # Usamos relação aproximada:
    # dI/dt + gamma I = beta(t) S I
    # => beta(t) ≈ (dI/dt + gamma I) / (S I)
    #
    # Mas S não observado; aproximamos S ≈ 1 - I_obs (assume R pequeno no começo)
    # Isso coloca a rede para aprender um beta coerente com a dinâmica.
    #
    I_np = I_obs
    dIdt = np.gradient(I_np, t)
    S_aprox = np.clip(1.0 - I_np, 1e-6, 1.0)
    beta_alvo = (dIdt + gamma*I_np) / (S_aprox * np.clip(I_np, 1e-6, 1.0))
    beta_alvo = np.clip(beta_alvo, 0, 1.5)

    beta_alvo_torch = torch.tensor(beta_alvo, dtype=torch.float32, device=device).view(-1,1)

    for ep in range(1, epocas+1):
        opt.zero_grad()
        beta_pred = rede(t_torch)

        loss = torch.mean((beta_pred - beta_alvo_torch)**2)
        loss.backward()
        opt.step()

        if ep % 400 == 0:
            print(f"Época {ep:4d}/{epocas} | perda_beta = {loss.item():.8f}")

    return rede

# -------------------------------
# 5. SIR híbrido com beta(t) neural
# -------------------------------
def simular_hibrido(rede_beta, gamma=0.10, **kwargs):
    def beta_neural(t):
        with torch.no_grad():
            tt = torch.tensor([[t]], dtype=torch.float32)
            return float(rede_beta(tt).item())
    return simular_sir_sde(beta_const=0.0, gamma=gamma, beta_t_func=beta_neural, **kwargs)

# -------------------------------
# 6. Pipeline principal
# -------------------------------
def main():
    set_semente(7)

    # (A) Dados sintéticos / "verdade"
    t, S_true, I_true, R_true, I_obs = gerar_dados_sinteticos()

    # (B) Treino rede beta(t)
    rede_beta = treinar_rede_beta(t, I_obs, gamma=0.10, epocas=2000, lr=2e-3)

    # (C) Monte Carlo SDE com beta neural
    t_mc, S_mc, I_mc, R_mc = monte_carlo_sir_sde(
        n_traj=300,
        beta_const=0.0, gamma=0.10,
        sigma_S=0.10, sigma_I=0.10, sigma_R=0.05,
        t_max=160, dt=0.2, I0=0.002,
        beta_t_func=lambda x: float(rede_beta(torch.tensor([[x]], dtype=torch.float32)).item())
    )

    # médias e bandas
    S_med = S_mc.mean(axis=0); I_med = I_mc.mean(axis=0); R_med = R_mc.mean(axis=0)
    I_q05 = np.quantile(I_mc, 0.05, axis=0)
    I_q95 = np.quantile(I_mc, 0.95, axis=0)

    imprimir_resumo(t, S_true, I_true, R_true, "SIR(SDE) VERDADE SINTÉTICA")
    imprimir_resumo(t_mc, S_med, I_med, R_med, "SIR HÍBRIDO (MÉDIA MONTE CARLO)")

    # (D) SIR em grafo
    G, t_g, S_g, I_g, R_g = simular_sir_grafo(
        n=350, p_conexao=0.025, beta=0.33, gamma=0.10, t_max=160, I0_frac=0.01
    )

    # -------------------------------
    # 7. Gráficos
    # -------------------------------
    plt.figure(figsize=(10,6))
    plt.plot(t, S_true, label="S verdadeiro")
    plt.plot(t, I_true, label="I verdadeiro")
    plt.plot(t, R_true, label="R verdadeiro")
    plt.scatter(t, I_obs, s=10, alpha=0.4, label="I observado (ruído)")
    plt.title("SIR Estocástico (SDE) — Trajetória verdadeira + observação")
    plt.xlabel("Tempo (dias)"); plt.ylabel("Proporção")
    plt.legend(); plt.grid(True)

    plt.figure(figsize=(10,6))
    plt.plot(t_mc, I_med, label="I média (híbrido)")
    plt.fill_between(t_mc, I_q05, I_q95, alpha=0.3, label="Banda 90% (MC)")
    plt.plot(t, I_true, "--", label="I verdadeiro")
    plt.title("Infectados — Modelo híbrido com incerteza")
    plt.xlabel("Tempo (dias)"); plt.ylabel("Proporção infectada")
    plt.legend(); plt.grid(True)

    # Beta real vs beta neural
    beta_real_vec = np.array([beta_real(tt) for tt in t])
    with torch.no_grad():
        beta_neural_vec = rede_beta(torch.tensor(t, dtype=torch.float32).view(-1,1)).numpy().flatten()

    plt.figure(figsize=(10,6))
    plt.plot(t, beta_real_vec, label="β(t) real")
    plt.plot(t, beta_neural_vec, "--", label="β(t) neural")
    plt.title("Taxa de transmissão aprendida")
    plt.xlabel("Tempo (dias)"); plt.ylabel("β(t)")
    plt.legend(); plt.grid(True)

    # Plano de fase I vs S
    plt.figure(figsize=(7,6))
    plt.plot(S_true, I_true, label="Verdade")
    plt.plot(S_med, I_med, "--", label="Híbrido média")
    plt.title("Plano de fase: I(t) vs S(t)")
    plt.xlabel("S"); plt.ylabel("I")
    plt.legend(); plt.grid(True)

    # Grafo: layout e snapshots simples
    pos = nx.spring_layout(G, seed=3)
    plt.figure(figsize=(7,7))
    nx.draw_networkx_nodes(G, pos, node_size=20, node_color="#999999")
    nx.draw_networkx_edges(G, pos, alpha=0.15)
    plt.title("Rede de contatos (grafo Erdős–Rényi)")
    plt.axis("off")

    # Comparar SDE (média) vs grafo
    plt.figure(figsize=(10,6))
    plt.plot(t_mc, I_med, label="I híbrido (SDE média)")
    plt.plot(t_g, I_g, label="I em grafo (discreto)")
    plt.title("Comparação: dinâmica média vs dinâmica em rede")
    plt.xlabel("Tempo (dias)"); plt.ylabel("Proporção infectada")
    plt.legend(); plt.grid(True)

    # Previsão curta para frente
    # prolonga beta neural e roda mais 40 dias
    t_fut, S_fut, I_fut, R_fut = simular_hibrido(
        rede_beta,
        gamma=0.10,
        sigma_S=0.10, sigma_I=0.10, sigma_R=0.05,
        t_max=200, dt=0.2, I0=I_true[-1]
    )

    plt.figure(figsize=(10,6))
    plt.plot(t, I_true, label="I verdadeiro (até 160d)")
    plt.plot(t_fut, I_fut, "--", label="I previsto híbrido (até 200d)")
    plt.axvline(160, color="k", linestyle=":")
    plt.title("Previsão híbrida (SDE + β neural)")
    plt.xlabel("Tempo (dias)"); plt.ylabel("Proporção infectada")
    plt.legend(); plt.grid(True)

    plt.show()

if __name__ == "__main__":
    main()
