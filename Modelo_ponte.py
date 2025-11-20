# ============================================================
# MODELAGEM MATEMÁTICA DE UMA PONTE (VIGA EULER–BERNOULLI)
# EDP pesada + discretização FD + análise modal + dinâmica
# Autor: Luiz Tiago Wilcke (LT)
#
# Ponte = tabuleiro como viga simplesmente apoiada
# ρA w_tt + c w_t + EI w_xxxx = q(x,t)
# ============================================================

import numpy as np

# ------------------------------------------------------------
# 1. CONFIGURAÇÕES FÍSICAS DA PONTE
# ------------------------------------------------------------

class ConfiguracaoPonte:
    def __init__(self):
        # Geometria
        self.comprimento = 400.0         # L (m)
        self.largura = 18.0              # b (m)
        self.espessura = 2.5             # h (m)

        # Material (concreto protendido / aço equivalente)
        self.modulo_young = 35e9         # E (Pa)
        self.densidade = 2500.0          # ρ (kg/m³)
        self.coef_amortecimento = 0.015  # ζ ~ 1.5% (amortecimento modal)

        # Propriedades da seção
        self.area = self.largura * self.espessura
        self.inercia = (self.largura * self.espessura**3) / 12.0

        # Resistência
        self.tensao_escoamento = 45e6    # σ_y (Pa) (equivalente)
        self.coef_seguranca = 1.8

        # Malha espacial
        self.n_pontos = 201              # pontos ao longo do tabuleiro
        self.dx = self.comprimento / (self.n_pontos - 1)

        # Tempo (dinâmica)
        self.tempo_total = 25.0          # s
        self.dt = 0.002                  # s
        self.n_passos = int(self.tempo_total / self.dt)

        # Cargas
        self.g = 9.81

        # Carga distribuída permanente (peso próprio + pavimento)
        self.carga_permanente = self.densidade * self.area * self.g  # N/m

        # Carga móvel tipo caminhão
        self.peso_veiculo = 320e3        # N
        self.velocidade_veiculo = 22.0   # m/s (~80 km/h)

        # Vento harmônico lateral equivalente (força por metro)
        self.amplitude_vento = 9e3       # N/m
        self.frequencia_vento = 0.8      # Hz


# ------------------------------------------------------------
# 2. MATRIZES DE DIFERENÇAS FINITAS PARA w_xxxx
# ------------------------------------------------------------

def montar_matriz_quarta_derivada(n, dx):
    """
    Monta matriz D4 tal que w_xxxx ≈ D4 @ w
    Condições simplesmente apoiadas:
      w(0)=w(L)=0
      w''(0)=w''(L)=0
    Implementação via ghost nodes embutidos na matriz.
    """
    D2 = np.zeros((n, n))
    # Segunda derivada central
    for i in range(1, n-1):
        D2[i, i-1] = 1.0
        D2[i, i]   = -2.0
        D2[i, i+1] = 1.0
    D2 /= dx**2

    # Para simplesmente apoiada: w''(0)=w''(L)=0
    # Aproximações unilaterais de 2ª derivada nos extremos:
    D2[0, 0] = -2.0; D2[0, 1] = 2.0
    D2[-1,-1] = -2.0; D2[-1,-2] = 2.0
    D2 /= dx**2

    # Quarta derivada pela composição: D4 = D2 @ D2
    D4 = D2 @ D2
    return D4


# ------------------------------------------------------------
# 3. CARGAS q(x,t)
# ------------------------------------------------------------

def carga_estatica(cfg, x):
    """
    Carga total estática q(x) = permanente + veículos em posição fixa.
    Vamos considerar 2 veículos parados em 1/3 e 2/3 do vão.
    """
    q = np.ones_like(x) * cfg.carga_permanente

    pos1 = cfg.comprimento/3.0
    pos2 = 2.0*cfg.comprimento/3.0
    idx1 = np.argmin(np.abs(x - pos1))
    idx2 = np.argmin(np.abs(x - pos2))

    q[idx1] += cfg.peso_veiculo / cfg.dx
    q[idx2] += cfg.peso_veiculo / cfg.dx
    return q


def carga_dinamica(cfg, x, t):
    """
    q(x,t) = permanente + veículo em movimento + vento harmônico
    """
    q = np.ones_like(x) * cfg.carga_permanente

    # veículo móvel
    pos = cfg.velocidade_veiculo * t
    if 0.0 <= pos <= cfg.comprimento:
        idx = np.argmin(np.abs(x - pos))
        q[idx] += cfg.peso_veiculo / cfg.dx

    # vento harmônico (distribuído)
    q += cfg.amplitude_vento * np.sin(2*np.pi*cfg.frequencia_vento*t)
    return q


# ------------------------------------------------------------
# 4. ANÁLISE ESTÁTICA: EI w'''' = q(x)
# ------------------------------------------------------------

def resolver_estatico(cfg):
    n, dx = cfg.n_pontos, cfg.dx
    x = np.linspace(0, cfg.comprimento, n)

    D4 = montar_matriz_quarta_derivada(n, dx)
    K = cfg.modulo_young * cfg.inercia * D4  # rigidez

    q = carga_estatica(cfg, x)               # N/m
    f = q.copy()

    # Impõe w(0)=w(L)=0 (simplesmente apoiada)
    # Fazemos isso zerando linhas/cols e colocando 1 na diagonal
    for i in [0, n-1]:
        K[i,:] = 0.0
        K[:,i] = 0.0
        K[i,i] = 1.0
        f[i] = 0.0

    w = np.linalg.solve(K, f)
    return x, w


# ------------------------------------------------------------
# 5. ANÁLISE MODAL: (EI D4) φ = ω² (ρA) φ
# ------------------------------------------------------------

def analise_modal(cfg, n_modos=6):
    n, dx = cfg.n_pontos, cfg.dx
    D4 = montar_matriz_quarta_derivada(n, dx)
    K = cfg.modulo_young * cfg.inercia * D4
    M = (cfg.densidade * cfg.area) * np.eye(n)

    # Impõe w=0 nos extremos (tirando DOFs correspondentes)
    idx_interno = np.arange(1, n-1)
    Kii = K[np.ix_(idx_interno, idx_interno)]
    Mii = M[np.ix_(idx_interno, idx_interno)]

    # Resolve autovalor generalizado via transformação
    A = np.linalg.solve(Mii, Kii)
    autovalores, autovetores = np.linalg.eigh(A)

    autovalores = np.maximum(autovalores, 0.0)
    omegas = np.sqrt(autovalores)

    # Frequências naturais (Hz)
    frequencias = omegas / (2*np.pi)

    return frequencias[:n_modos], autovetores[:, :n_modos]


# ------------------------------------------------------------
# 6. DINÂMICA: integração explícita RK4 do sistema de 2ª ordem
# ------------------------------------------------------------

def integrar_dinamica(cfg):
    n, dx = cfg.n_pontos, cfg.dx
    x = np.linspace(0, cfg.comprimento, n)

    D4 = montar_matriz_quarta_derivada(n, dx)
    K = cfg.modulo_young * cfg.inercia * D4
    m_linha = cfg.densidade * cfg.area
    M_inv = (1.0 / m_linha) * np.eye(n)

    # amortecimento proporcional ao crítico modal aproximado:
    # c = 2 ζ sqrt(K M)
    # aqui usamos c_linha constante aproximada
    # ω1 aproximado para simples apoio: (π²)*sqrt(EI/(ρA L^4))
    omega1 = (np.pi**2) * np.sqrt(cfg.modulo_young*cfg.inercia/(cfg.densidade*cfg.area*cfg.comprimento**4))
    c_linha = 2.0 * cfg.coef_amortecimento * m_linha * omega1
    C = c_linha * np.eye(n)

    # estado: y = [w, v]
    w = np.zeros(n)
    v = np.zeros(n)

    # condições de contorno w(0)=w(L)=0 forçadas
    def aplicar_contorno(w, v):
        w[0]=0.0; w[-1]=0.0
        v[0]=0.0; v[-1]=0.0
        return w, v

    w, v = aplicar_contorno(w, v)

    # histórico de máximos
    max_deflexao = 0.0
    max_velocidade = 0.0
    max_aceleracao = 0.0
    tempo_max = 0.0
    pos_max = 0.0

    def aceleracao(w_local, v_local, t_local):
        q = carga_dinamica(cfg, x, t_local)
        f = q.copy()
        # força interna: K w
        a = M_inv @ (f - C@v_local - K@w_local)
        return a

    # RK4
    for passo in range(cfg.n_passos):
        t = passo * cfg.dt

        a1 = aceleracao(w, v, t)
        k1_w = v
        k1_v = a1

        a2 = aceleracao(w + 0.5*cfg.dt*k1_w, v + 0.5*cfg.dt*k1_v, t + 0.5*cfg.dt)
        k2_w = v + 0.5*cfg.dt*k1_v
        k2_v = a2

        a3 = aceleracao(w + 0.5*cfg.dt*k2_w, v + 0.5*cfg.dt*k2_v, t + 0.5*cfg.dt)
        k3_w = v + 0.5*cfg.dt*k2_v
        k3_v = a3

        a4 = aceleracao(w + cfg.dt*k3_w, v + cfg.dt*k3_v, t + cfg.dt)
        k4_w = v + cfg.dt*k3_v
        k4_v = a4

        w = w + (cfg.dt/6.0)*(k1_w + 2*k2_w + 2*k3_w + k4_w)
        v = v + (cfg.dt/6.0)*(k1_v + 2*k2_v + 2*k3_v + k4_v)

        w, v = aplicar_contorno(w, v)

        a_atual = aceleracao(w, v, t)

        # mede máximos
        idx = np.argmax(np.abs(w))
        if abs(w[idx]) > max_deflexao:
            max_deflexao = abs(w[idx])
            tempo_max = t
            pos_max = x[idx]

        max_velocidade = max(max_velocidade, np.max(np.abs(v)))
        max_aceleracao = max(max_aceleracao, np.max(np.abs(a_atual)))

    return {
        "x": x,
        "w_final": w,
        "v_final": v,
        "max_deflexao": max_deflexao,
        "tempo_max_deflexao": tempo_max,
        "pos_max_deflexao": pos_max,
        "max_velocidade": max_velocidade,
        "max_aceleracao": max_aceleracao,
    }


# ------------------------------------------------------------
# 7. PÓS-PROCESSAMENTO NUMÉRICO (MOMENTO, CISALHAMENTO, TENSÃO)
# ------------------------------------------------------------

def calcular_esforcos(cfg, x, w):
    dx = cfg.dx
    n = len(x)

    # segunda derivada para momento: M = -EI w''
    w2 = np.zeros(n)
    for i in range(1, n-1):
        w2[i] = (w[i-1] - 2*w[i] + w[i+1]) / dx**2
    # bordas em simples apoio: w''=0
    w2[0]=0.0; w2[-1]=0.0

    momento = -cfg.modulo_young * cfg.inercia * w2

    # terceira derivada para cisalhamento: V = dM/dx
    cisalhamento = np.zeros(n)
    for i in range(1, n-1):
        cisalhamento[i] = (momento[i+1] - momento[i-1]) / (2*dx)
    cisalhamento[0]=cisalhamento[1]
    cisalhamento[-1]=cisalhamento[-2]

    # tensão máxima na fibra extrema: σ = M*y/I
    y_max = cfg.espessura/2.0
    tensao = momento * y_max / cfg.inercia

    return momento, cisalhamento, tensao


# ------------------------------------------------------------
# 8. MAIN: RODA TUDO E IMPRIME RESULTADOS NUMÉRICOS
# ------------------------------------------------------------

def main():
    cfg = ConfiguracaoPonte()

    # ===== ESTÁTICO =====
    x, w_est = resolver_estatico(cfg)
    momento_est, cisalhamento_est, tensao_est = calcular_esforcos(cfg, x, w_est)

    deflexao_max_est = np.max(np.abs(w_est))
    pos_deflexao_max_est = x[np.argmax(np.abs(w_est))]
    momento_max_est = np.max(np.abs(momento_est))
    cisalhamento_max_est = np.max(np.abs(cisalhamento_est))
    tensao_max_est = np.max(np.abs(tensao_est))

    # segurança
    fator_utilizacao = tensao_max_est / (cfg.tensao_escoamento / cfg.coef_seguranca)

    # ===== MODAL =====
    freq, _ = analise_modal(cfg, n_modos=8)

    # ===== DINÂMICO =====
    saida_dyn = integrar_dinamica(cfg)
    w_final = saida_dyn["w_final"]
    momento_dyn, cisalhamento_dyn, tensao_dyn = calcular_esforcos(cfg, x, w_final)

    # --------------------------------------------------------
    # IMPRESSÃO DE RESULTADOS NUMÉRICOS (SÓ NÚMEROS)
    # --------------------------------------------------------

    print("\n================== RESULTADOS ESTÁTICOS ==================")
    print(f"Comprimento da ponte (m)              = {cfg.comprimento:.8f}")
    print(f"Área da seção (m^2)                   = {cfg.area:.8f}")
    print(f"Inércia da seção (m^4)                = {cfg.inercia:.8e}")
    print(f"Carga permanente (N/m)                = {cfg.carga_permanente:.8f}")
    print(f"Deflexão máxima estática (m)          = {deflexao_max_est:.8e}")
    print(f"Posição da deflexão máxima (m)        = {pos_deflexao_max_est:.8f}")
    print(f"Momento fletor máximo estático (N*m)  = {momento_max_est:.8e}")
    print(f"Cisalhamento máximo estático (N)      = {cisalhamento_max_est:.8e}")
    print(f"Tensão máxima estática (Pa)           = {tensao_max_est:.8e}")
    print(f"Fator de utilização (σ/σadm)          = {fator_utilizacao:.8f}")

    print("\n================== FREQUÊNCIAS NATURAIS ==================")
    for i, f in enumerate(freq, 1):
        print(f"Modo {i:02d} - f_n (Hz) = {f:.8f}")

    print("\n================== RESULTADOS DINÂMICOS ==================")
    print(f"Tempo total simulado (s)              = {cfg.tempo_total:.8f}")
    print(f"Passo de tempo dt (s)                 = {cfg.dt:.8f}")
    print(f"Velocidade do veículo (m/s)           = {cfg.velocidade_veiculo:.8f}")
    print(f"Peso do veículo (N)                   = {cfg.peso_veiculo:.8e}")
    print(f"Vento: amplitude (N/m)                = {cfg.amplitude_vento:.8e}")
    print(f"Vento: frequência (Hz)                = {cfg.frequencia_vento:.8f}")
    print(f"Deflexão máxima dinâmica (m)          = {saida_dyn['max_deflexao']:.8e}")
    print(f"Posição da deflexão max. dinâmica (m) = {saida_dyn['pos_max_deflexao']:.8f}")
    print(f"Tempo da deflexão max. dinâmica (s)   = {saida_dyn['tempo_max_deflexao']:.8f}")
    print(f"Velocidade máxima dinâmica (m/s)      = {saida_dyn['max_velocidade']:.8e}")
    print(f"Aceleração máxima dinâmica (m/s^2)    = {saida_dyn['max_aceleracao']:.8e}")

    print("\n================== ESFORÇOS NO FINAL (t=Tf) ===============")
    print(f"Momento máximo final (N*m)            = {np.max(np.abs(momento_dyn)):.8e}")
    print(f"Cisalhamento máximo final (N)         = {np.max(np.abs(cisalhamento_dyn)):.8e}")
    print(f"Tensão máxima final (Pa)              = {np.max(np.abs(tensao_dyn)):.8e}")

    # critérios de serviço típicos
    limite_deflexao = cfg.comprimento / 800.0   # L/800
    print("\n================== CRITÉRIOS DE SERVIÇO ==================")
    print(f"Limite deflexão L/800 (m)             = {limite_deflexao:.8e}")
    print(f"Razão deflexão estática / limite      = {(deflexao_max_est/limite_deflexao):.8f}")
    print(f"Razão deflexão dinâmica / limite      = {(saida_dyn['max_deflexao']/limite_deflexao):.8f}")

    print("\n================== AMOSTRA NUMÉRICA (w_est) ===============")
    # imprime algumas amostras ao longo do vão
    pontos_amostra = [0.0, 0.25*cfg.comprimento, 0.5*cfg.comprimento, 0.75*cfg.comprimento, cfg.comprimento]
    for p in pontos_amostra:
        i = np.argmin(np.abs(x - p))
        print(f"x={x[i]:.2f} m  ->  w_est={w_est[i]:.8e} m  |  M={momento_est[i]:.8e} N*m")

    print("\n===========================================================")


if __name__ == "__main__":
    main()
