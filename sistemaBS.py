# ============================================================
# PINN Black–Scholes — Derivativos Bancários com PyTorch
# Autor: Luiz Tiago Wilcke (LT)
#
# - Resolve a EDP de Black–Scholes usando rede neural
# - Modelo híbrido: EDP + Rede Neural (Physics-Informed NN)
# - Pode ser base para opções, swaps, futuros, seguros, etc.
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
import math

# ------------------------------------------------------------
# Configurações globais
# ------------------------------------------------------------
dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ParametrosFinanceiros:
    def __init__(self,
                 taxa_juros: float = 0.05,
                 volatilidade: float = 0.2,
                 prazo_maturidade: float = 1.0,   # anos
                 preco_exercicio: float = 100.0,
                 preco_maximo_ativo: float = 4 * 100.0):
        self.taxa_juros = taxa_juros
        self.volatilidade = volatilidade
        self.prazo_maturidade = prazo_maturidade
        self.preco_exercicio = preco_exercicio
        self.preco_maximo_ativo = preco_maximo_ativo

# ------------------------------------------------------------
# 1. Rede Neural — aproxima V(S,t)
# ------------------------------------------------------------

class RedeBScholesPINN(nn.Module):
    """
    Rede neural totalmente conectada para aproximar V(S,t).
    Entradas: [preco_ativo, tempo]
    Saída: [valor_opcao]
    """
    def __init__(self,
                 dimensao_entrada: int = 2,
                 dimensao_saida: int = 1,
                 neuronios_ocultos: int = 64,
                 camadas_ocultas: int = 5):
        super().__init__()

        camadas = []
        dim_atual = dimensao_entrada

        for _ in range(camadas_ocultas):
            camadas.append(nn.Linear(dim_atual, neuronios_ocultos))
            camadas.append(nn.Tanh())  # boa para PINNs
            dim_atual = neuronios_ocultos

        camadas.append(nn.Linear(dim_atual, dimensao_saida))
        self.rede = nn.Sequential(*camadas)

    def forward(self, preco_ativo: torch.Tensor, tempo: torch.Tensor) -> torch.Tensor:
        """
        preco_ativo: tensor shape (N, 1)
        tempo:       tensor shape (N, 1)
        """
        entrada = torch.cat([preco_ativo, tempo], dim=1)
        saida = self.rede(entrada)
        return saida  # V_theta(S,t)

# ------------------------------------------------------------
# 2. Condições de contorno e terminal
# ------------------------------------------------------------

def condicao_terminal_call(preco_ativo: torch.Tensor,
                           parametros: ParametrosFinanceiros) -> torch.Tensor:
    """
    V(S, T) = max(S - K, 0)
    """
    return torch.clamp(preco_ativo - parametros.preco_exercicio, min=0.0)

def condicao_fronteira_S_zero(tempo: torch.Tensor,
                              parametros: ParametrosFinanceiros) -> torch.Tensor:
    """
    Em S = 0, V(0,t) = 0 para call europeu sem dividendos.
    """
    return torch.zeros_like(tempo)

def condicao_fronteira_S_max(tempo: torch.Tensor,
                             parametros: ParametrosFinanceiros) -> torch.Tensor:
    """
    Em S = S_max: V(S_max, t) ≈ S_max - K * exp(-r * (T - t))
    """
    S_max = torch.full_like(tempo, parametros.preco_maximo_ativo)
    desconto = torch.exp(-parametros.taxa_juros * (parametros.prazo_maturidade - tempo))
    return S_max - parametros.preco_exercicio * desconto

# ------------------------------------------------------------
# 3. Resíduo da EDP (Physics-Informed)
# ------------------------------------------------------------

def calcular_residuo_pde(modelo: RedeBScholesPINN,
                         preco_ativo: torch.Tensor,
                         tempo: torch.Tensor,
                         parametros: ParametrosFinanceiros) -> torch.Tensor:
    """
    Calcula o resíduo da EDP de Black–Scholes:

    V_t + 0.5 * sigma^2 * S^2 * V_SS + r * S * V_S - r * V = 0
    """

    preco_ativo = preco_ativo.clone().detach().requires_grad_(True)
    tempo = tempo.clone().detach().requires_grad_(True)

    V = modelo(preco_ativo, tempo)  # shape (N,1)

    # Gradientes de primeira ordem
    grad_V = torch.autograd.grad(
        V,
        [preco_ativo, tempo],
        grad_outputs=torch.ones_like(V),
        create_graph=True
    )

    dV_dS = grad_V[0]   # ∂V/∂S
    dV_dt = grad_V[1]   # ∂V/∂t

    # Segunda derivada em relação a S
    d2V_dS2 = torch.autograd.grad(
        dV_dS,
        preco_ativo,
        grad_outputs=torch.ones_like(dV_dS),
        create_graph=True
    )[0]

    sigma = parametros.volatilidade
    r = parametros.taxa_juros

    termo_difusao = 0.5 * sigma**2 * (preco_ativo**2) * d2V_dS2
    termo_derivada_S = r * preco_ativo * dV_dS
    termo_desconto = -r * V

    residuo = dV_dt + termo_difusao + termo_derivada_S + termo_desconto
    return residuo

# ------------------------------------------------------------
# 4. Amostragem de pontos no domínio (S, t)
# ------------------------------------------------------------

def amostrar_pontos_internos(parametros: ParametrosFinanceiros,
                             quantidade: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pontos internos do domínio: 0 < S < S_max, 0 <= t < T
    """
    S = torch.rand(quantidade, 1) * parametros.preco_maximo_ativo
    t = torch.rand(quantidade, 1) * parametros.prazo_maturidade
    return S.to(dispositivo), t.to(dispositivo)

def amostrar_pontos_terminal(parametros: ParametrosFinanceiros,
                             quantidade: int) -> torch.Tensor:
    """
    Pontos em t = T (condição terminal).
    """
    S = torch.rand(quantidade, 1) * parametros.preco_maximo_ativo
    return S.to(dispositivo)

def amostrar_pontos_fronteira_S_zero(parametros: ParametrosFinanceiros,
                                     quantidade: int) -> torch.Tensor:
    t = torch.rand(quantidade, 1) * parametros.prazo_maturidade
    return t.to(dispositivo)

def amostrar_pontos_fronteira_S_max(parametros: ParametrosFinanceiros,
                                    quantidade: int) -> torch.Tensor:
    t = torch.rand(quantidade, 1) * parametros.prazo_maturidade
    return t.to(dispositivo)

# ------------------------------------------------------------
# 5. Loop de treino da PINN
# ------------------------------------------------------------

def treinar_pinn_black_scholes(
    epocas: int = 5000,
    quantidade_interno: int = 1024,
    quantidade_terminal: int = 512,
    quantidade_fronteira: int = 512,
    parametros: ParametrosFinanceiros | None = None,
    taxa_aprendizado: float = 1e-3
) -> RedeBScholesPINN:

    if parametros is None:
        parametros = ParametrosFinanceiros()

    modelo = RedeBScholesPINN().to(dispositivo)
    otimizador = optim.Adam(modelo.parameters(), lr=taxa_aprendizado)

    perda_mse = nn.MSELoss()

    for epoca in range(1, epocas + 1):
        modelo.train()
        otimizador.zero_grad()

        # ----------------------------------------
        # 5.1 Pontos internos — resíduo da EDP
        # ----------------------------------------
        S_interno, t_interno = amostrar_pontos_internos(parametros, quantidade_interno)
        residuo = calcular_residuo_pde(modelo, S_interno, t_interno, parametros)
        perda_interna = perda_mse(residuo, torch.zeros_like(residuo))

        # ----------------------------------------
        # 5.2 Condição terminal V(S,T) = payoff
        # ----------------------------------------
        S_terminal = amostrar_pontos_terminal(parametros, quantidade_terminal)
        t_terminal = torch.full_like(S_terminal, parametros.prazo_maturidade).to(dispositivo)

        V_terminal_pred = modelo(S_terminal, t_terminal)
        V_terminal_real = condicao_terminal_call(S_terminal, parametros)
        perda_terminal = perda_mse(V_terminal_pred, V_terminal_real)

        # ----------------------------------------
        # 5.3 Fronteira S = 0
        # ----------------------------------------
        t_zero = amostrar_pontos_fronteira_S_zero(parametros, quantidade_fronteira)
        S_zero = torch.zeros_like(t_zero).to(dispositivo)

        V_zero_pred = modelo(S_zero, t_zero)
        V_zero_real = condicao_fronteira_S_zero(t_zero, parametros)
        perda_fronteira_zero = perda_mse(V_zero_pred, V_zero_real)

        # ----------------------------------------
        # 5.4 Fronteira S = S_max
        # ----------------------------------------
        t_max = amostrar_pontos_fronteira_S_max(parametros, quantidade_fronteira)
        S_max = torch.full_like(t_max, parametros.preco_maximo_ativo).to(dispositivo)

        V_max_pred = modelo(S_max, t_max)
        V_max_real = condicao_fronteira_S_max(t_max, parametros)
        perda_fronteira_max = perda_mse(V_max_pred, V_max_real)

        # ----------------------------------------
        # 5.5 Perda total (combinação ponderada)
        # ----------------------------------------
        perda_total = (
            1.0 * perda_interna +
            1.0 * perda_terminal +
            0.1 * perda_fronteira_zero +
            0.1 * perda_fronteira_max
        )

        perda_total.backward()
        otimizador.step()

        if epoca % 500 == 0:
            print(
                f"Época {epoca:5d} | "
                f"Perda total: {perda_total.item():.6f} | "
                f"Interna: {perda_interna.item():.6f} | "
                f"Terminal: {perda_terminal.item():.6f}"
            )

    return modelo

# ------------------------------------------------------------
# 6. Fórmula fechada de Black–Scholes (para validar)
# ------------------------------------------------------------

def cdf_normal_padrao(x: torch.Tensor) -> torch.Tensor:
    """
    Função de distribuição acumulada da Normal padrão.
    """
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def preco_call_black_scholes_fechado(
    preco_ativo: torch.Tensor,
    tempo: torch.Tensor,
    parametros: ParametrosFinanceiros
) -> torch.Tensor:
    """
    Fórmula fechada para call europeu (sem dividendos).
    tempo aqui é t (agora), então usa T - t no tempo até a maturidade.
    """
    S = preco_ativo
    K = parametros.preco_exercicio
    r = parametros.taxa_juros
    sigma = parametros.volatilidade
    T = parametros.prazo_maturidade

    tau = T - tempo  # tempo até a maturidade
    tau = torch.clamp(tau, min=1e-8)

    d1 = (torch.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * torch.sqrt(tau))
    d2 = d1 - sigma * torch.sqrt(tau)

    N_d1 = cdf_normal_padrao(d1)
    N_d2 = cdf_normal_padrao(d2)

    return S * N_d1 - K * torch.exp(-r * tau) * N_d2

# ------------------------------------------------------------
# 7. Exemplo de uso
# ------------------------------------------------------------

if __name__ == "__main__":
    parametros = ParametrosFinanceiros(
        taxa_juros=0.05,
        volatilidade=0.2,
        prazo_maturidade=1.0,
        preco_exercicio=100.0,
        preco_maximo_ativo=400.0
    )

    print("Treinando PINN para EDP de Black–Scholes...")
    modelo_treinado = treinar_pinn_black_scholes(
        epocas=3000,  # aumente para melhor precisão
        parametros=parametros,
        quantidade_interno=2048,
        quantidade_terminal=1024,
        quantidade_fronteira=1024,
        taxa_aprendizado=1e-3
    )

    modelo_treinado.eval()

    # Avalia a opção no tempo t=0 para uma grade de preços
    with torch.no_grad():
        precos = torch.linspace(1.0, parametros.preco_maximo_ativo, 50).view(-1, 1).to(dispositivo)
        tempos = torch.zeros_like(precos).to(dispositivo)

        V_pinn = modelo_treinado(precos, tempos)
        V_fechado = preco_call_black_scholes_fechado(precos, tempos, parametros)

        erro_medio_absoluto = torch.mean(torch.abs(V_pinn - V_fechado)).item()
        print(f"Erro médio absoluto PINN vs fórmula fechada: {erro_medio_absoluto:.4f}")
