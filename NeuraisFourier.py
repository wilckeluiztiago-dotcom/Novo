# ============================================================
# FOURIER NEURAL OPERATOR (FNO) para Navier–Stokes 2D
# Autor: Luiz Tiago Wilcke (LT)
# ============================================================
# Resolve Navier–Stokes 2D incompressível via:
#   - Solver espectral (FFT) para gerar dados
#   - Fourier Neural Operator para aprender o operador temporal
#
# Forma em vorticidade:
#   ∂ω/∂t + u·∇ω = ν Δω + f
#   u = ∇⊥ ψ,  Δψ = ω
#
# Onde:
#   ω = vorticidade
#   ψ = função corrente
#   ν = viscosidade cinemática
#   f = forçante externa (opcional; aqui usamos pequena)
# ============================================================

import os, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.fft as fft
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ----------------------------
# 1. Configurações gerais
# ----------------------------
class Configuracoes:
    # grade espacial
    n = 64                # resolução NxN
    L = 2*np.pi           # tamanho do domínio [0, L]^2 (periódico)
    # simulação física
    viscosidade = 1e-3
    dt = 1e-3
    passos_tempo = 50     # passos por amostra
    # dataset
    num_amostras = 200    # total de trajetórias
    num_treino = 160
    num_teste = 40
    # FNO
    modos_fourier = 12
    largura = 64
    camadas = 4
    # treino
    epocas = 20
    batch_size = 8
    lr = 1e-3
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Configuracoes()


# ----------------------------
# 2. Utilitários espectrais
# ----------------------------
def criar_grade_fourier(n, L):
    """Cria malhas de números de onda kx, ky para FFT periódica."""
    k = np.fft.fftfreq(n, d=L/n) * 2*np.pi  # escala correta no domínio [0,L]
    kx, ky = np.meshgrid(k, k, indexing="ij")
    return kx, ky

def laplaciano_espectral(campo_hat, kx, ky):
    """Δ em Fourier: -(kx^2 + ky^2) * campo_hat"""
    return -(kx**2 + ky**2) * campo_hat

def derivadas_espectrais(campo_hat, kx, ky):
    """Retorna derivadas espaciais no espaço físico via FFT."""
    d_dx_hat = 1j*kx*campo_hat
    d_dy_hat = 1j*ky*campo_hat
    d_dx = np.fft.ifft2(d_dx_hat).real
    d_dy = np.fft.ifft2(d_dy_hat).real
    return d_dx, d_dy


# ----------------------------
# 3. Solver Navier–Stokes 2D (vorticidade)
# ----------------------------
def passo_navier_stokes(omega, kx, ky, viscosidade, dt, forca=None):
    """
    Um passo de Euler semi-implícito:
      - termo não linear explícito
      - difusão implícita no espectro
    """
    n = omega.shape[0]

    # FFT da vorticidade
    omega_hat = np.fft.fft2(omega)

    # Resolver Poisson: Δψ = ω => ψ_hat = -omega_hat/(kx^2+ky^2)
    k2 = kx**2 + ky**2
    k2[0,0] = 1.0  # evita divisão por zero no modo 0
    psi_hat = -omega_hat / k2

    # velocidades: u = (∂ψ/∂y, -∂ψ/∂x)
    dpsi_dx, dpsi_dy = derivadas_espectrais(psi_hat, kx, ky)
    u = dpsi_dy
    v = -dpsi_dx

    # gradiente de ω
    domega_dx, domega_dy = derivadas_espectrais(omega_hat, kx, ky)

    # termo convectivo: u·∇ω
    nao_linear = u*domega_dx + v*domega_dy

    # forçante
    if forca is None:
        forca = 0.0

    # passo explícito no espaço físico
    rhs = omega - dt*nao_linear + dt*forca

    # difusão implícita em Fourier:
    rhs_hat = np.fft.fft2(rhs)
    denom = (1.0 + dt*viscosidade*(kx**2 + ky**2))
    omega_hat_novo = rhs_hat / denom

    omega_novo = np.fft.ifft2(omega_hat_novo).real
    return omega_novo


def simular_trajetoria(cfg, semente=None):
    """Gera uma trajetória omega(t) usando NS 2D periódica."""
    if semente is not None:
        np.random.seed(semente)

    n, L = cfg.n, cfg.L
    kx, ky = criar_grade_fourier(n, L)

    # condição inicial: campo aleatório suave (filtrado no espectro)
    ruido = np.random.randn(n, n)
    ruido_hat = np.fft.fft2(ruido)
    k2 = kx**2 + ky**2
    filtro = np.exp(-0.1 * k2)  # suaviza altas frequências
    omega0 = np.fft.ifft2(ruido_hat * filtro).real

    # pequena forçante estacionária (opcional)
    forca = 0.1*np.sin(np.linspace(0, L, n))[:, None]

    omegas = [omega0]
    omega = omega0.copy()
    for _ in range(cfg.passos_tempo):
        omega = passo_navier_stokes(
            omega, kx, ky, cfg.viscosidade, cfg.dt, forca=forca
        )
        omegas.append(omega)

    # shape: (T+1, n, n)
    return np.stack(omegas, axis=0)


# ----------------------------
# 4. Dataset: pares (omega_t -> omega_{t+1})
# ----------------------------
class DatasetNavierStokes(Dataset):
    def __init__(self, cfg, modo="treino"):
        super().__init__()
        self.cfg = cfg

        # gera trajetórias
        trajs = []
        for i in range(cfg.num_amostras):
            trajs.append(simular_trajetoria(cfg, semente=i))

        trajs = np.stack(trajs, axis=0)  # (A, T+1, n, n)

        # separa treino/teste
        if modo == "treino":
            trajs = trajs[:cfg.num_treino]
        else:
            trajs = trajs[cfg.num_treino:cfg.num_treino+cfg.num_teste]

        # cria pares t -> t+1
        entradas, saidas = [], []
        for traj in trajs:
            for t in range(cfg.passos_tempo):
                entradas.append(traj[t])
                saidas.append(traj[t+1])

        self.entradas = torch.tensor(np.array(entradas), dtype=torch.float32)
        self.saidas = torch.tensor(np.array(saidas), dtype=torch.float32)

        # normalização simples global
        self.media = self.entradas.mean()
        self.desvio = self.entradas.std() + 1e-8
        self.entradas = (self.entradas - self.media)/self.desvio
        self.saidas = (self.saidas - self.media)/self.desvio

    def __len__(self):
        return self.entradas.shape[0]

    def __getitem__(self, idx):
        # retorna (n,n) -> adiciona canal em frente no loader
        return self.entradas[idx], self.saidas[idx]


# ----------------------------
# 5. Camada espectral do FNO
# ----------------------------
class CamadaFourier2D(nn.Module):
    """
    Aplica convolução no espaço de Fourier:
      y_hat(k) = W(k) * x_hat(k)  (apenas modos baixos)
    """
    def __init__(self, canais_in, canais_out, modos):
        super().__init__()
        self.canais_in = canais_in
        self.canais_out = canais_out
        self.modos = modos

        escala = 1/(canais_in*canais_out)
        # pesos complexos separados em real/imag
        self.pesos_real = nn.Parameter(
            escala*torch.randn(canais_in, canais_out, modos, modos)
        )
        self.pesos_imag = nn.Parameter(
            escala*torch.randn(canais_in, canais_out, modos, modos)
        )

    def forward(self, x):
        """
        x: (batch, canais, n, n)
        """
        b, c, n, _ = x.shape

        x_hat = fft.rfft2(x)  # (b, c, n, n//2+1)

        # saida no espectro
        y_hat = torch.zeros(
            b, self.canais_out, n, n//2 + 1,
            device=x.device, dtype=torch.cfloat
        )

        # aplica pesos nos modos baixos
        modos = self.modos
        pesos = torch.complex(self.pesos_real, self.pesos_imag)

        y_hat[:, :, :modos, :modos] = torch.einsum(
            "b i x y, i o x y -> b o x y",
            x_hat[:, :, :modos, :modos],
            pesos
        )

        # volta ao espaço físico
        y = fft.irfft2(y_hat, s=(n, n))
        return y


# ----------------------------
# 6. Fourier Neural Operator 2D
# ----------------------------
class FNO2D(nn.Module):
    def __init__(self, modos, largura, camadas):
        super().__init__()
        self.modos = modos
        self.largura = largura
        self.camadas = camadas

        # lifting: 1 canal (omega) + 2 coords -> largura
        self.lift = nn.Conv2d(3, largura, 1)

        self.fourier_camadas = nn.ModuleList([
            CamadaFourier2D(largura, largura, modos) for _ in range(camadas)
        ])
        self.w_camadas = nn.ModuleList([
            nn.Conv2d(largura, largura, 1) for _ in range(camadas)
        ])

        # projeção final
        self.proj1 = nn.Conv2d(largura, 128, 1)
        self.proj2 = nn.Conv2d(128, 1, 1)
        self.ativacao = nn.GELU()

    def forward(self, omega):
        """
        omega: (batch, n, n) normalizado
        retorna: (batch, n, n)
        """
        b, n, _ = omega.shape

        # adiciona canal
        omega = omega.unsqueeze(1)  # (b,1,n,n)

        # coords (para dar noção espacial ao operador)
        x = torch.linspace(0, 1, n, device=omega.device)
        y = torch.linspace(0, 1, n, device=omega.device)
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        xx = xx.unsqueeze(0).unsqueeze(0).repeat(b, 1, 1, 1)
        yy = yy.unsqueeze(0).unsqueeze(0).repeat(b, 1, 1, 1)

        entrada = torch.cat([omega, xx, yy], dim=1)  # (b,3,n,n)

        z = self.lift(entrada)

        for k in range(self.camadas):
            z1 = self.fourier_camadas[k](z)
            z2 = self.w_camadas[k](z)
            z = self.ativacao(z1 + z2)

        z = self.ativacao(self.proj1(z))
        z = self.proj2(z)           # (b,1,n,n)

        return z.squeeze(1)


# ----------------------------
# 7. Treinamento
# ----------------------------
def treinar_fno(cfg):
    ds_treino = DatasetNavierStokes(cfg, modo="treino")
    ds_teste  = DatasetNavierStokes(cfg, modo="teste")

    loader_treino = DataLoader(ds_treino, batch_size=cfg.batch_size, shuffle=True)
    loader_teste  = DataLoader(ds_teste, batch_size=cfg.batch_size, shuffle=False)

    modelo = FNO2D(cfg.modos_fourier, cfg.largura, cfg.camadas).to(cfg.dispositivo)

    otimizador = torch.optim.Adam(modelo.parameters(), lr=cfg.lr, weight_decay=1e-6)
    perda_fn = nn.MSELoss()

    historico = {"perda_treino": [], "perda_teste": []}

    for epoca in range(1, cfg.epocas+1):
        modelo.train()
        perdas = []

        for omega_t, omega_tp1 in loader_treino:
            omega_t = omega_t.to(cfg.dispositivo)
            omega_tp1 = omega_tp1.to(cfg.dispositivo)

            pred = modelo(omega_t)
            perda = perda_fn(pred, omega_tp1)

            otimizador.zero_grad()
            perda.backward()
            otimizador.step()

            perdas.append(perda.item())

        perda_media_treino = float(np.mean(perdas))

        # avalia em teste
        modelo.eval()
        perdas_teste = []
        with torch.no_grad():
            for omega_t, omega_tp1 in loader_teste:
                omega_t = omega_t.to(cfg.dispositivo)
                omega_tp1 = omega_tp1.to(cfg.dispositivo)
                pred = modelo(omega_t)
                perdas_teste.append(perda_fn(pred, omega_tp1).item())

        perda_media_teste = float(np.mean(perdas_teste))

        historico["perda_treino"].append(perda_media_treino)
        historico["perda_teste"].append(perda_media_teste)

        print(f"Época {epoca:03d} | perda treino={perda_media_treino:.6e} | perda teste={perda_media_teste:.6e}")

    return modelo, ds_teste, historico


# ----------------------------
# 8. Avaliação visual
# ----------------------------
def avaliar_e_plotar(modelo, ds_teste, cfg, amostra=0):
    modelo.eval()

    omega_t, omega_tp1 = ds_teste[amostra]
    omega_t = omega_t.unsqueeze(0).to(cfg.dispositivo)
    omega_tp1 = omega_tp1.numpy()

    with torch.no_grad():
        pred = modelo(omega_t).cpu().numpy()[0]

    # desnormaliza (usa estatísticas do dataset)
    media = float(ds_teste.media)
    desvio = float(ds_teste.desvio)
    omega_tp1_real = omega_tp1*desvio + media
    pred_real = pred*desvio + media

    erro = np.abs(pred_real - omega_tp1_real)

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.title("Verdadeiro ω(t+dt)")
    plt.imshow(omega_tp1_real, origin="lower")
    plt.colorbar()

    plt.subplot(1,3,2)
    plt.title("Predito FNO")
    plt.imshow(pred_real, origin="lower")
    plt.colorbar()

    plt.subplot(1,3,3)
    plt.title("|Erro|")
    plt.imshow(erro, origin="lower")
    plt.colorbar()

    plt.tight_layout()
    plt.show()


# ----------------------------
# 9. Rodar tudo
# ----------------------------
if __name__ == "__main__":
    print("Dispositivo:", cfg.dispositivo)
    print("Gerando dados + treinando FNO...\n")

    modelo, ds_teste, hist = treinar_fno(cfg)

    # curvas de perda
    plt.figure()
    plt.plot(hist["perda_treino"], label="treino")
    plt.plot(hist["perda_teste"], label="teste")
    plt.yscale("log")
    plt.xlabel("Época")
    plt.ylabel("MSE")
    plt.legend()
    plt.title("Histórico de treino do FNO")
    plt.show()

    # avaliação visual
    avaliar_e_plotar(modelo, ds_teste, cfg, amostra=3)
