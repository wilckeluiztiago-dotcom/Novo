# ============================================================
# Relatividade no Sistema Solar — Tkinter + Pygame 
# Autor: Luiz Tiago Wilcke (LT)
# Descrição:
#   • UI em Tkinter para configurar parâmetros (massas, distância, velocidade,
#     correção relativística 1-PN e amortecimento).
#   • Ao clicar "Calcular e Animar", abre Pygame e anima a Terra
#     orbitando/caindo no Sol, com "funil" de espaço-tempo no fundo.
#   • Integração via RK4. Tempo de simulação (t_sim) separado de tempo real (t_real),
#     evitando que a janela abra e feche imediatamente.
#   • Teclas: P (pausa) e ESC (sair).
#   • Renderiza a equação: G_{μν} + Λ g_{μν} = (8πG/c^4) T_{μν}
# ============================================================

import math
import tkinter as tk
from tkinter import ttk, messagebox

# Tente importar pygame
try:
    import pygame
except ImportError:
    pygame = None

# ---------------------------
# Constantes físicas (SI)
# ---------------------------
G = 6.67430e-11         # m^3 kg^-1 s^-2
C = 299_792_458.0       # m/s
M_SOL = 1.98847e30      # kg
M_TERRA = 5.9722e24     # kg
UA = 1.495978707e11     # m
R_SOL = 6.9634e8        # m
R_TERRA = 6.371e6       # m

# ---------------------------
# Dinâmica (Newton + 1-PN simples) + amortecimento
# ---------------------------
def aceleracao_relativistica(pos, vel, massa_central, usar_1pn, gamma_amort):
    """Aceleração newtoniana com correção radial 1-PN simples e amortecimento."""
    x, y = pos
    vx, vy = vel
    r2 = x*x + y*y
    r = math.sqrt(r2) + 1e-12

    fator = -G * massa_central / (r**3)
    ax = fator * x
    ay = fator * y

    if usar_1pn:
        corr = 1.0 + 3.0 * G * massa_central / (C*C * r)
        ax *= corr
        ay *= corr

    ax -= gamma_amort * vx
    ay -= gamma_amort * vy
    return ax, ay

def rk4_step(pos, vel, dt, massa_central, usar_1pn, gamma_amort):
    """Passo RK4 para o sistema de 2ª ordem (pos, vel)."""
    def acc(p, v):
        return aceleracao_relativistica(p, v, massa_central, usar_1pn, gamma_amort)

    x, y = pos
    vx, vy = vel

    ax1, ay1 = acc((x, y), (vx, vy))
    k1x, k1y = vx, vy
    l1x, l1y = ax1, ay1

    ax2, ay2 = acc((x + 0.5*dt*k1x, y + 0.5*dt*k1y),
                   (vx + 0.5*dt*l1x, vy + 0.5*dt*l1y))
    k2x, k2y = vx + 0.5*dt*l1x, vy + 0.5*dt*l1y
    l2x, l2y = ax2, ay2

    ax3, ay3 = acc((x + 0.5*dt*k2x, y + 0.5*dt*k2y),
                   (vx + 0.5*dt*l2x, vy + 0.5*dt*l2y))
    k3x, k3y = vx + 0.5*dt*l2x, vy + 0.5*dt*l2y
    l3x, l3y = ax3, ay3

    ax4, ay4 = acc((x + dt*k3x, y + dt*k3y),
                   (vx + dt*l3x, vy + dt*l3y))
    k4x, k4y = vx + dt*l3x, vy + dt*l3y
    l4x, l4y = ax4, ay4

    x_new = x + (dt/6.0)*(k1x + 2*k2x + 2*k3x + k4x)
    y_new = y + (dt/6.0)*(k1y + 2*k2y + 2*k3y + k4y)
    vx_new = vx + (dt/6.0)*(l1x + 2*l2x + 2*l3x + l4x)
    vy_new = vy + (dt/6.0)*(l1y + 2*l2y + 2*l3y + l4y)
    return (x_new, y_new), (vx_new, vy_new)

# ---------------------------
# Render do "funil" (curvatura)
# ---------------------------
def desenhar_fundo_curvatura(screen, W, H, cx, cy, escala_metros_px, massa_central):
    cor_grade = (40, 40, 60)
    cor_eixos = (60, 60, 100)
    screen.fill((10, 10, 20))

    # Deformação z ~ -k / r, com k proporcional a Rs
    Rs = 2.0 * G * massa_central / (C*C)
    k = 0.8 * Rs / escala_metros_px

    # Linhas radiais
    for ang in [i*math.pi/18 for i in range(36)]:
        pts = []
        for rho in range(5, int(0.48*min(W, H)), 6):
            x = cx + rho * math.cos(ang)
            y = cy + rho * math.sin(ang)
            r_m = math.hypot((x - cx), (y - cy)) * escala_metros_px + 1.0
            z = -k * (1.0 / r_m)
            pts.append((x, y + z))
        if len(pts) >= 2:
            pygame.draw.aalines(screen, cor_grade, False, pts)

    # Círculos
    for rho in range(20, int(0.48*min(W, H)), 20):
        circ = []
        for i in range(0, 360, 4):
            ang = math.radians(i)
            x = cx + rho * math.cos(ang)
            y = cy + rho * math.sin(ang)
            r_m = math.hypot((x - cx), (y - cy)) * escala_metros_px + 1.0
            z = -k * (1.0 / r_m)
            circ.append((x, y + z))
        if len(circ) >= 2:
            pygame.draw.aalines(screen, cor_grade, True, circ)

    pygame.draw.line(screen, cor_eixos, (0, cy), (W, cy), 1)
    pygame.draw.line(screen, cor_eixos, (cx, 0), (cx, H), 1)

# ---------------------------
# Escalas e conversões
# ---------------------------
def metros_para_px(dx_m, escala_metros_px):
    return dx_m / escala_metros_px

def pos_metros_para_pygame(pos_m, cx, cy, escala_metros_px):
    x_m, y_m = pos_m
    return (int(cx + metros_para_px(x_m, escala_metros_px)),
            int(cy - metros_para_px(y_m, escala_metros_px)))  # eixo y invertido

# ---------------------------
# Motor da animação (com correção do tempo)
# ---------------------------
def animar(massa_sol, massa_terra, dist_inicial, vel_inicial, usar_1pn, gamma_amort,
           dt_sim, escala_visual_ua, duracao_segundos, cor_trilha=(80,180,255)):
    if pygame is None:
        messagebox.showerror("Erro", "pygame não está instalado. Instale com: pip install pygame")
        return

    pygame.init()
    pygame.display.set_caption("Relatividade — Terra caindo no 'funil' do Sol (LT)")

    W, H = 900, 700
    screen = pygame.display.set_mode((W, H))
    clock = pygame.time.Clock()

    # Centro e escala (m/px)
    cx, cy = W // 2, H // 2
    escala_metros_px = (escala_visual_ua * UA) / (0.45 * min(W, H))

    # Estado inicial (m e m/s)
    pos = (dist_inicial, 0.0)
    vel = (0.0, vel_inicial)

    trilha = []
    trilha_max = 1200

    raio_sol_px = max(6, int(metros_para_px(R_SOL, escala_metros_px)))
    raio_terra_px = max(3, int(metros_para_px(R_TERRA * 4, escala_metros_px)))  # *4 p/ visibilidade

    fonte = pygame.font.SysFont("DejaVuSans", 16)
    fonte_eq = pygame.font.SysFont("DejaVuSansMono", 18, bold=True)

    tempo_sim = 0.0   # tempo de simulação (s físicos)
    tempo_real = 0.0  # tempo de relógio (s reais)
    rodando = True
    pausado = False

    while rodando:
        # Tempo real do frame (~60 FPS)
        dt_ms = clock.tick(60)
        dt_real = dt_ms / 1000.0
        tempo_real += dt_real

        # Eventos
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                rodando = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    rodando = False
                elif event.key == pygame.K_p:
                    pausado = not pausado

        # Fundo (curvatura)
        desenhar_fundo_curvatura(screen, W, H, cx, cy, escala_metros_px, massa_sol)

        # Integração (um passo por frame; ajuste dt_sim na UI)
        if not pausado:
            pos, vel = rk4_step(pos, vel, dt_sim, massa_sol, usar_1pn, gamma_amort)
            tempo_sim += dt_sim

            if len(trilha) > trilha_max:
                trilha.pop(0)
            trilha.append(pos)

        # Converter para pixels
        px_sol = pos_metros_para_pygame((0.0, 0.0), cx, cy, escala_metros_px)
        px_terra = pos_metros_para_pygame(pos, cx, cy, escala_metros_px)

        # Desenho dos corpos e trilha
        pygame.draw.circle(screen, (255, 210, 60), px_sol, raio_sol_px)
        if len(trilha) > 2:
            pts = [pos_metros_para_pygame(p, cx, cy, escala_metros_px) for p in trilha]
            pygame.draw.aalines(screen, cor_trilha, False, pts)
        pygame.draw.circle(screen, (90, 160, 255), px_terra, raio_terra_px)

        # Textos
        r = math.hypot(*pos)
        v = math.hypot(*vel)
        info1 = f"r = {r/UA:.3f} UA | v = {v/1000:.3f} km/s | t_sim = {tempo_sim/86400:.2f} dias"
        info2 = f"1-PN = {'ON' if usar_1pn else 'OFF'} | gamma = {gamma_amort:.2e} | dt_sim = {dt_sim:.0f}s | {'PAUSADO' if pausado else ''}"
        info3 = f"t_real = {tempo_real:.1f}s  (ESC sai, P pausa)"
        screen.blit(fonte.render(info1, True, (220, 220, 240)), (12, 10))
        screen.blit(fonte.render(info2, True, (210, 210, 230)), (12, 32))
        screen.blit(fonte.render(info3, True, (210, 210, 230)), (12, 54))

        # Equação de Einstein + assinatura
        eq_texto = "G_{μν} + Λ g_{μν} = (8πG / c^4) T_{μν}"
        screen.blit(fonte_eq.render(eq_texto, True, (240, 240, 255)), (12, H - 60))
        screen.blit(fonte.render("Autor: Luiz Tiago Wilcke (LT)", True, (200, 255, 200)), (12, H - 32))

        pygame.display.flip()

        # Critérios de parada
        if tempo_real >= duracao_segundos:
            rodando = False
        if r <= 1.1 * R_SOL:
            rodando = False

    pygame.quit()

# ---------------------------
# Tkinter UI
# ---------------------------
class AppRelatividade(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Relatividade Terra–Sol — Configuração (LT)")
        self.geometry("700x520")

        estilo = ttk.Style(self)
        try:
            estilo.theme_use("clam")
        except:
            pass

        # Vars (padrões)
        self.var_m_sol = tk.DoubleVar(value=M_SOL)
        self.var_m_terra = tk.DoubleVar(value=M_TERRA)
        self.var_dist_ini = tk.DoubleVar(value=1.0)  # UA
        v_circ = math.sqrt(G * M_SOL / UA)          # ~29.78 km/s
        self.var_vel_ini = tk.DoubleVar(value=v_circ)
        self.var_1pn = tk.BooleanVar(value=True)
        self.var_gamma = tk.DoubleVar(value=2.0e-7)
        self.var_dt = tk.DoubleVar(value=1200.0)           # s de simulação por frame
        self.var_escala_ua = tk.DoubleVar(value=1.5)       # UA por semieixo da tela
        self.var_duracao = tk.DoubleVar(value=120.0)       # s reais da animação

        self._montar_layout()

    def _linha(self, pai, texto, var, largura=22, ajuda=None):
        frame = ttk.Frame(pai)
        ttk.Label(frame, text=texto, width=32, anchor="w").pack(side="left", padx=6, pady=4)
        ttk.Entry(frame, textvariable=var, width=largura).pack(side="left", padx=6, pady=4)
        if ajuda:
            ttk.Label(frame, text=ajuda, foreground="#777").pack(side="left", padx=10)
        frame.pack(fill="x", padx=10)
        return frame

    def _montar_layout(self):
        ttk.Label(self,
                  text="Configuração do Modelo Relativístico Terra–Sol (1-PN + amortecimento)",
                  font=("DejaVu Sans", 12, "bold")).pack(pady=10)

        frm = ttk.Frame(self)
        frm.pack(fill="both", expand=True, padx=10)

        self._linha(frm, "Massa do Sol (kg):", self.var_m_sol, ajuda="≈ 1.98847e30")
        self._linha(frm, "Massa da Terra (kg):", self.var_m_terra, ajuda="≈ 5.9722e24")
        self._linha(frm, "Distância inicial (UA):", self.var_dist_ini, ajuda="ex.: 1.0 UA")
        self._linha(frm, "Velocidade inicial (m/s):", self.var_vel_ini, ajuda="≈ 29.78 km/s para 1 UA")

        linha_1pn = ttk.Frame(frm)
        ttk.Label(linha_1pn, text="Usar correção relativística 1-PN?").pack(side="left", padx=6, pady=4)
        ttk.Checkbutton(linha_1pn, variable=self.var_1pn).pack(side="left", padx=6)
        linha_1pn.pack(fill="x", padx=10)

        self._linha(frm, "Amortecimento γ (1/s):", self.var_gamma, ajuda="ex.: 2e-7 (0 desliga)")
        self._linha(frm, "Passo de simulação dt (s):", self.var_dt, ajuda="ex.: 1200 s (20 min por frame)")
        self._linha(frm, "Escala visual (UA por semi-eixo):", self.var_escala_ua, ajuda="ex.: 1.5")
        self._linha(frm, "Duração da animação (s):", self.var_duracao, ajuda="ex.: 120 s")

        botoes = ttk.Frame(self)
        botoes.pack(pady=12)
        ttk.Button(botoes, text="Calcular e Animar", command=self.iniciar_animacao).pack(side="left", padx=6)
        ttk.Button(botoes, text="Sair", command=self.destroy).pack(side="left", padx=6)

        ttk.Label(self,
                  text="Equação de Einstein:  G_{μν} + Λ g_{μν} = (8πG/c^4) T_{μν}    |    Autor: Luiz Tiago Wilcke (LT)",
                  foreground="#444").pack(pady=6)

    def iniciar_animacao(self):
        try:
            msol = float(self.var_m_sol.get())
            mter = float(self.var_m_terra.get())
            dist = float(self.var_dist_ini.get()) * UA
            vel0 = float(self.var_vel_ini.get())
            usar_1pn = bool(self.var_1pn.get())
            gamma = float(self.var_gamma.get())
            dt = float(self.var_dt.get())
            escala_ua = float(self.var_escala_ua.get())
            dur = float(self.var_duracao.get())

            if dist <= 1.05 * R_SOL:
                messagebox.showwarning("Atenção", "A distância inicial está muito próxima do Sol.")
                return
            if pygame is None:
                messagebox.showerror("Erro", "pygame não está instalado. Instale com: pip install pygame")
                return

            animar(msol, mter, dist, vel0, usar_1pn, gamma, dt, escala_ua, dur)
        except Exception as e:
            messagebox.showerror("Erro", f"Parâmetros inválidos: {e}")

# ---------------------------
# Execução
# ---------------------------
if __name__ == "__main__":
    app = AppRelatividade()
    app.mainloop()
