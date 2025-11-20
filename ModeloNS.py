# ============================================================
# NEURAL NAVIER–STOKES (PINN 2D) — Taylor-Green Vortex
# Autor: Luiz Tiago Wilcke (LT)
# ------------------------------------------------------------
# - Resolve Navier–Stokes incompressível 2D via PINN (PyTorch)
# - Benchmark turbulento: Taylor-Green Vortex
# - Treina rede para u,v,p obedecerem PDE + IC + BC periódicas
# - Gera gráficos: campos, vorticidade, pressão, perdas
# ============================================================

import os, time, math, random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ---------------------------
# 0. SEMENTES / DISPOSITIVO
# ---------------------------
def fixar_semente(semente=1234):
    random.seed(semente)
    np.random.seed(semente)
    torch.manual_seed(semente)
    torch.cuda.manual_seed_all(semente)

fixar_semente(7)
dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo:", dispositivo)

# ---------------------------
# 1. PARAMETROS FISICOS
# ---------------------------
L = 2.0 * math.pi        # domínio [0, 2pi]
nu = 0.01                # viscosidade cinemática (Re ~ 1/nu)
t_max = 1.0              # tempo final
print(f"nu = {nu} (Re efetivo ~ {1/nu:.1f})")

# ---------------------------
# 2. SOLUÇÃO ANALÍTICA (TGV 2D)
# ---------------------------
def solucao_taylor_green(x, y, t, nu):
    """
    x,y,t -> arrays numpy
    retorna u, v, p
    """
    fator_u = np.exp(-2.0 * nu * t)
    fator_p = np.exp(-4.0 * nu * t)

    u = -np.cos(x) * np.sin(y) * fator_u
    v =  np.sin(x) * np.cos(y) * fator_u
    p = -0.25 * (np.cos(2*x) + np.cos(2*y)) * fator_p
    return u, v, p

# ---------------------------
# 3. REDE NEURAL MLP
# ---------------------------
class MLPNavierStokes(nn.Module):
    def __init__(self, camadas):
        super().__init__()
        lista = []
        for i in range(len(camadas) - 2):
            lista.append(nn.Linear(camadas[i], camadas[i+1]))
            lista.append(nn.Tanh())
        lista.append(nn.Linear(camadas[-2], camadas[-1]))
        self.rede = nn.Sequential(*lista)

        # inicialização Xavier
        for m in self.rede:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, xyt):
        return self.rede(xyt)

# camadas: entrada (x,y,t)=3 -> saída (u,v,p)=3
modelo = MLPNavierStokes([3, 128, 128, 128, 128, 3]).to(dispositivo)

# ---------------------------
# 4. AMOSTRAGEM DE PONTOS
# ---------------------------
def amostrar_dominio(n_pontos):
    """
    Amostra pontos (x,y,t) no domínio
    """
    x = np.random.uniform(0, L, (n_pontos, 1))
    y = np.random.uniform(0, L, (n_pontos, 1))
    t = np.random.uniform(0, t_max, (n_pontos, 1))
    return x, y, t

def amostrar_inicial(n_pontos):
    """
    t=0, IC analítica
    """
    x = np.random.uniform(0, L, (n_pontos, 1))
    y = np.random.uniform(0, L, (n_pontos, 1))
    t = np.zeros((n_pontos, 1))
    u, v, p = solucao_taylor_green(x, y, t, nu)
    return x, y, t, u, v, p

def amostrar_periodico(n_pontos):
    """
    Pega pares periódicos:
      (x=0,y,t) <-> (x=L,y,t)
      (x,y=0,t) <-> (x,y=L,t)
    """
    t = np.random.uniform(0, t_max, (n_pontos, 1))

    # bordas x
    y1 = np.random.uniform(0, L, (n_pontos, 1))
    x_esq = np.zeros((n_pontos, 1))
    x_dir = L * np.ones((n_pontos, 1))

    # bordas y
    x2 = np.random.uniform(0, L, (n_pontos, 1))
    y_baixo = np.zeros((n_pontos, 1))
    y_cima = L * np.ones((n_pontos, 1))

    return (x_esq, y1, t, x_dir, y1, t,
            x2, y_baixo, t, x2, y_cima, t)

# hiperparâmetros de amostragem
N_colocacao = 50000   # pontos internos (resíduo PDE)
N_inicial   = 8000    # pontos IC
N_periodico = 6000    # pares periódicos

# ---------------------------
# 5. DERIVADAS AUTOMÁTICAS
# ---------------------------
def gradiente(y, x):
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y),
        create_graph=True, retain_graph=True
    )[0]

# ---------------------------
# 6. RESÍDUOS NAVIER-STOKES
# ---------------------------
def residuos_pde(modelo, x, y, t, nu):
    """
    Calcula resíduos:
        r_u, r_v, r_cont
    """
    x.requires_grad_(True)
    y.requires_grad_(True)
    t.requires_grad_(True)

    entrada = torch.cat([x, y, t], dim=1)
    saida = modelo(entrada)
    u = saida[:, 0:1]
    v = saida[:, 1:2]
    p = saida[:, 2:3]

    # derivadas
    u_t = gradiente(u, t)
    u_x = gradiente(u, x)
    u_y = gradiente(u, y)
    u_xx = gradiente(u_x, x)
    u_yy = gradiente(u_y, y)

    v_t = gradiente(v, t)
    v_x = gradiente(v, x)
    v_y = gradiente(v, y)
    v_xx = gradiente(v_x, x)
    v_yy = gradiente(v_y, y)

    p_x = gradiente(p, x)
    p_y = gradiente(p, y)

    # Navier–Stokes
    r_u = u_t + (u * u_x + v * u_y) + p_x - nu * (u_xx + u_yy)
    r_v = v_t + (u * v_x + v * v_y) + p_y - nu * (v_xx + v_yy)

    # incompressibilidade
    r_cont = u_x + v_y

    return r_u, r_v, r_cont

# ---------------------------
# 7. FUNÇÃO DE PERDA
# ---------------------------
def perda_total(modelo):
    # --- 7.1 Resíduo PDE interno ---
    x_c, y_c, t_c = amostrar_dominio(N_colocacao)
    x_c = torch.tensor(x_c, dtype=torch.float32, device=dispositivo)
    y_c = torch.tensor(y_c, dtype=torch.float32, device=dispositivo)
    t_c = torch.tensor(t_c, dtype=torch.float32, device=dispositivo)

    r_u, r_v, r_cont = residuos_pde(modelo, x_c, y_c, t_c, nu)
    perda_pde = (r_u**2).mean() + (r_v**2).mean() + (r_cont**2).mean()

    # --- 7.2 Condição Inicial ---
    x_i, y_i, t_i, u_i, v_i, p_i = amostrar_inicial(N_inicial)
    x_i = torch.tensor(x_i, dtype=torch.float32, device=dispositivo)
    y_i = torch.tensor(y_i, dtype=torch.float32, device=dispositivo)
    t_i = torch.tensor(t_i, dtype=torch.float32, device=dispositivo)
    u_i = torch.tensor(u_i, dtype=torch.float32, device=dispositivo)
    v_i = torch.tensor(v_i, dtype=torch.float32, device=dispositivo)
    p_i = torch.tensor(p_i, dtype=torch.float32, device=dispositivo)

    entrada_i = torch.cat([x_i, y_i, t_i], dim=1)
    saida_i = modelo(entrada_i)
    perda_ic = ((saida_i[:,0:1]-u_i)**2).mean() \
             + ((saida_i[:,1:2]-v_i)**2).mean() \
             + ((saida_i[:,2:3]-p_i)**2).mean()

    # --- 7.3 Periodicidade (BC) ---
    (x_esq, y1, t1, x_dir, y2, t2,
     x3, y_baixo, t3, x4, y_cima, t4) = amostrar_periodico(N_periodico)

    x_esq = torch.tensor(x_esq, dtype=torch.float32, device=dispositivo)
    y1    = torch.tensor(y1, dtype=torch.float32, device=dispositivo)
    t1    = torch.tensor(t1, dtype=torch.float32, device=dispositivo)
    x_dir = torch.tensor(x_dir, dtype=torch.float32, device=dispositivo)
    y2    = torch.tensor(y2, dtype=torch.float32, device=dispositivo)
    t2    = torch.tensor(t2, dtype=torch.float32, device=dispositivo)

    x3 = torch.tensor(x3, dtype=torch.float32, device=dispositivo)
    y_baixo = torch.tensor(y_baixo, dtype=torch.float32, device=dispositivo)
    t3 = torch.tensor(t3, dtype=torch.float32, device=dispositivo)
    x4 = torch.tensor(x4, dtype=torch.float32, device=dispositivo)
    y_cima = torch.tensor(y_cima, dtype=torch.float32, device=dispositivo)
    t4 = torch.tensor(t4, dtype=torch.float32, device=dispositivo)

    saida_esq = modelo(torch.cat([x_esq, y1, t1], dim=1))
    saida_dir = modelo(torch.cat([x_dir, y2, t2], dim=1))
    saida_baixo = modelo(torch.cat([x3, y_baixo, t3], dim=1))
    saida_cima  = modelo(torch.cat([x4, y_cima,  t4], dim=1))

    perda_bc = ((saida_esq - saida_dir)**2).mean() \
             + ((saida_baixo - saida_cima)**2).mean()

    # pesos (ajustáveis)
    w_pde = 1.0
    w_ic  = 5.0
    w_bc  = 2.0

    perda = w_pde*perda_pde + w_ic*perda_ic + w_bc*perda_bc
    return perda, perda_pde.detach(), perda_ic.detach(), perda_bc.detach()

# ---------------------------
# 8. TREINAMENTO
# ---------------------------
otimizador = torch.optim.Adam(modelo.parameters(), lr=1e-3)
epocas = 6000

historico_total = []
historico_pde = []
historico_ic  = []
historico_bc  = []

inicio = time.time()
for ep in range(1, epocas+1):
    otimizador.zero_grad()
    perda, p_pde, p_ic, p_bc = perda_total(modelo)
    perda.backward()
    otimizador.step()

    historico_total.append(perda.item())
    historico_pde.append(p_pde.item())
    historico_ic.append(p_ic.item())
    historico_bc.append(p_bc.item())

    if ep % 200 == 0 or ep == 1:
        print(f"Época {ep:05d}/{epocas} | "
              f"Perda total={perda.item():.3e} | "
              f"PDE={p_pde.item():.3e} | IC={p_ic.item():.3e} | BC={p_bc.item():.3e}")

fim = time.time()
print(f"Treino finalizado em {(fim-inicio)/60:.2f} min")

# ---------------------------
# 9. AVALIAÇÃO EM GRADE
# ---------------------------
def avaliar_grade(modelo, n=128, tempo=0.5):
    x_lin = np.linspace(0, L, n)
    y_lin = np.linspace(0, L, n)
    X, Y = np.meshgrid(x_lin, y_lin)
    T = tempo * np.ones_like(X)

    x_t = torch.tensor(X.reshape(-1,1), dtype=torch.float32, device=dispositivo)
    y_t = torch.tensor(Y.reshape(-1,1), dtype=torch.float32, device=dispositivo)
    t_t = torch.tensor(T.reshape(-1,1), dtype=torch.float32, device=dispositivo)

    with torch.no_grad():
        saida = modelo(torch.cat([x_t, y_t, t_t], dim=1)).cpu().numpy()

    u_pred = saida[:,0].reshape(n,n)
    v_pred = saida[:,1].reshape(n,n)
    p_pred = saida[:,2].reshape(n,n)

    u_ex, v_ex, p_ex = solucao_taylor_green(X, Y, T, nu)
    return X, Y, u_pred, v_pred, p_pred, u_ex, v_ex, p_ex

# tempo de avaliação
tempo_avaliar = 0.6
X, Y, u_pred, v_pred, p_pred, u_ex, v_ex, p_ex = avaliar_grade(modelo, n=160, tempo=tempo_avaliar)

# erro L2 relativo
def erro_relativo(a, b):
    return np.linalg.norm(a-b) / (np.linalg.norm(b) + 1e-12)

erro_u = erro_relativo(u_pred, u_ex)
erro_v = erro_relativo(v_pred, v_ex)
erro_p = erro_relativo(p_pred, p_ex)
print(f"Erros relativos L2 em t={tempo_avaliar}: u={erro_u:.3e}, v={erro_v:.3e}, p={erro_p:.3e}")

# ---------------------------
# 10. PÓS-PROCESSAMENTO: VORTICIDADE
# ---------------------------
def vorticidade(u, v, dx):
    du_dy = np.gradient(u, dx, axis=0)
    dv_dx = np.gradient(v, dx, axis=1)
    return dv_dx - du_dy

dx = L/(u_pred.shape[0]-1)
omega_pred = vorticidade(u_pred, v_pred, dx)
omega_ex   = vorticidade(u_ex, v_ex, dx)

# ---------------------------
# 11. GRÁFICOS
# ---------------------------
os.makedirs("figuras", exist_ok=True)

# 11.1 perdas
plt.figure()
plt.semilogy(historico_total, label="total")
plt.semilogy(historico_pde,   label="PDE")
plt.semilogy(historico_ic,    label="IC")
plt.semilogy(historico_bc,    label="BC")
plt.xlabel("época")
plt.ylabel("perda (log)")
plt.legend()
plt.title("Histórico de Treinamento PINN")
plt.tight_layout()
plt.savefig("figuras/perdas.png", dpi=200)

# 11.2 campos u e v
def plotar_campo(Z, titulo, nome):
    plt.figure()
    plt.contourf(X, Y, Z, 60)
    plt.colorbar()
    plt.title(titulo)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(f"figuras/{nome}.png", dpi=200)

plotar_campo(u_pred, f"u_pred (t={tempo_avaliar})", "u_pred")
plotar_campo(u_ex,   f"u_exato (t={tempo_avaliar})", "u_exato")
plotar_campo(u_pred-u_ex, "erro u", "erro_u")

plotar_campo(v_pred, f"v_pred (t={tempo_avaliar})", "v_pred")
plotar_campo(v_ex,   f"v_exato (t={tempo_avaliar})", "v_exato")
plotar_campo(v_pred-v_ex, "erro v", "erro_v")

plotar_campo(p_pred, f"p_pred (t={tempo_avaliar})", "p_pred")
plotar_campo(p_ex,   f"p_exato (t={tempo_avaliar})", "p_exato")
plotar_campo(p_pred-p_ex, "erro p", "erro_p")

# 11.3 vorticidade
plotar_campo(omega_pred, f"vorticidade ω_pred (t={tempo_avaliar})", "omega_pred")
plotar_campo(omega_ex,   f"vorticidade ω_exata (t={tempo_avaliar})", "omega_exata")
plotar_campo(omega_pred-omega_ex, "erro ω", "erro_omega")

# 11.4 quiver (vetores velocidade)
plt.figure()
passo = 6
plt.quiver(X[::passo,::passo], Y[::passo,::passo],
           u_pred[::passo,::passo], v_pred[::passo,::passo])
plt.title(f"Campo Vetorial Velocidade (PINN) t={tempo_avaliar}")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.savefig("figuras/velocidade_quiver.png", dpi=200)

plt.show()

print("Figuras salvas em ./figuras/")
