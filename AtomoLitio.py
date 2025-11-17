import numpy as np
import pygame
import math
import random

# ============================================================
#  CONFIGURAÇÕES DO MODELO QUÂNTICO (Átomo de Lítio)
# ============================================================
Z = 3
N_orbitais = 3           # 1s (duplo) + 2s (simples)
l_orb = [0, 0, 0]        # Todos orbitais s (l = 0)
ocupacao = [2, 1, 0]     # 2 elétrons no 1s, 1 no 2s

# Grade radial
r_max = 30
N = 5000
r = np.linspace(1e-6, r_max, N)
dr = r[1] - r[0]

# ============================================================
#  Funções auxiliares: potenciais Hartree e Troca
# ============================================================
def calcular_potencial_hartree(rho):
    # V_H(r_j) = integral_0^r  rho(r') * r'^2 dr' + (1/r) integral_r^inf rho(r') r'^2 dr'
    integ1 = np.cumsum(rho * r**2) * dr
    integ2 = (np.sum(rho * r**2) - integ1) * dr
    return integ1 / r + integ2 / r

def potencial_troca_local(rho):
    return -(3/np.pi * rho)**(1/3)

def hamiltoniano_radial(Veff, E):
    """Construção da matriz tridiagonal para método de Numerov."""
    diag = 2 + (10*dr**2) * (Veff - E)
    off = -1 * np.ones(N-1)
    return diag, off

def resolver_orbital(Veff, tentativa_E):
    """Solve radial Schrödinger using Numerov."""
    E = tentativa_E
    u = np.zeros(N)
    u[0] = 0
    u[1] = 1e-6

    for i in range(1, N-1):
        k_i = 2*(Veff[i] - E)
        k_im1 = 2*(Veff[i-1] - E)
        k_ip1 = 2*(Veff[i+1] - E)

        u[i+1] = ((2*u[i]*(1 - (5*dr**2/12)*k_i) -
                   u[i-1]*(1 + (dr**2/12)*k_im1)) /
                   (1 + (dr**2/12)*k_ip1))

    # Normalizar
    norm = np.sqrt(np.trapz(u**2, r))
    return u / norm

# ============================================================
#  LOOP DE HARTREE-FOCK AUTO-CONSISTENTE
# ============================================================
def hartree_fock(max_iter=30):
    # Orbitais iniciais
    orbitais = [
        np.exp(-Z*r),         # 1s aproximado
        np.exp(-Z*r),
        (1 - Z*r/2)*np.exp(-Z*r/2)  # 2s aproximado
    ]

    orbitais = [u/np.sqrt(np.trapz(u**2, r)) for u in orbitais]

    for it in range(max_iter):
        # Densidade total
        rho = np.zeros(N)
        for i in range(N_orbitais):
            rho += ocupacao[i] * orbitais[i]**2

        # Potenciais
        V_H = calcular_potencial_hartree(rho)
        V_X = potencial_troca_local(rho)
        Veff = -Z/r + V_H + V_X

        # Resolver novos orbitais
        novos = []
        for i in range(N_orbitais):
            E_guess = - (Z - 0.3*i)**2 / 2
            u_new = resolver_orbital(Veff, E_guess)
            novos.append(u_new)

        # Critério de convergência
        erro = sum(np.max(np.abs(novos[i] - orbitais[i])) for i in range(N_orbitais))
        orbitais = novos
        print(f"Iteração {it+1}, erro = {erro:.4e}")
        if erro < 1e-6:
            break

    return orbitais, rho, Veff

orbitais, densidade_total, Veff = hartree_fock(max_iter=25)

# ============================================================
#  CONVERSÃO PARA PROBABILIDADE RADIAL 3D
# ============================================================
def amostrar_posicao(u):
    psi2 = u**2
    pdf = psi2 * r**2
    pdf /= np.sum(pdf)
    cdf = np.cumsum(pdf)

    x = random.random()
    idx = np.searchsorted(cdf, x)
    R = r[idx]

    # Distribuição angular uniforme
    theta = random.uniform(0, 2*np.pi)
    phi = math.acos(2*random.random() - 1)

    return R*np.sin(phi)*np.cos(theta), R*np.sin(phi)*np.sin(theta)

# ============================================================
#  PYGAME — VISUALIZAÇÃO DINÂMICA DOS 3 ELÉTRONS
# ============================================================
pygame.init()
L = 900
tela = pygame.display.set_mode((L, L))
clock = pygame.time.Clock()
ESCALA = 10

class Eletron:
    def __init__(self, u):
        x,y = amostrar_posicao(u)
        self.x = x*ESCALA
        self.y = y*ESCALA

    def atualizar(self, u):
        # Movimentação difusiva + recolocação por densidade radial
        if random.random() < 0.01:
            x,y = amostrar_posicao(u)
            self.x = x*ESCALA
            self.y = y*ESCALA
        else:
            self.x += random.uniform(-0.3,0.3)
            self.y += random.uniform(-0.3,0.3)

    def desenhar(self, cor):
        pygame.draw.circle(
            tela,
            cor,
            (int(self.x + L/2), int(self.y + L/2)),
            3
        )

# criar elétrons
eletrons = [
    Eletron(orbitais[0]),
    Eletron(orbitais[0]),
    Eletron(orbitais[2])
]

rodando = True
while rodando:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            rodando = False

    tela.fill((0,0,20))

    # núcleo
    pygame.draw.circle(tela, (255,50,50), (L//2, L//2), 12)

    # atualizar e desenhar elétrons
    for i,e in enumerate(eletrons):
        e.atualizar(orbitais[i])
        cor = (150,150,255) if i < 2 else (255,255,150)
        e.desenhar(cor)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
