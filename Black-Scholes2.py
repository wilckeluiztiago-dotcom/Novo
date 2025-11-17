# ============================================================
# Bayesian Neural ODE — Incerteza Financeira (Black–Scholes Bayesiano)
# Autor: Luiz Tiago Wilcke 
# ============================================================
#
# Ideia:
#   - Gerar dados sintéticos de um processo log-preço X_t ~ GBM:
#       dS_t = mu * S_t dt + sigma * S_t dW_t
#       X_t = log(S_t)
#   - Treinar uma Neural ODE Bayesiana para aproximar o drift:
#       dX_t/dt = f_theta(t, X_t)
#     onde os pesos de f_theta são aleatórios (W ~ N(mu_w, sigma_w^2)).
#   - Usar inferência variacional simples:
#       KL( N(mu_w, sigma_w^2) || N(0,1) ) como regularização bayesiana.
#   - Após o treino, amostramos muitos conjuntos de pesos (cenários)
#     e integramos a ODE, gerando um leque de trajetórias de preço
#     com incerteza (fan chart).
# ============================================================

import math
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ------------------------------------------------------------
# 0) Configurações gerais
# ------------------------------------------------------------
dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Configuracoes:
    # Processo financeiro sintético (GBM)
    preco_inicial = 100.0
    mu_verdadeiro = 0.08      # drift verdadeiro anual
    sigma_verdadeiro = 0.25   # volatilidade verdadeira anual

    dias_uteis_ano = 252
    dt = 1.0 / dias_uteis_ano

    num_trajetorias_treino = 512
    num_trajetorias_teste = 128
    num_passos_tempo = 60     # horizonte em dias

    # Rede Bayesiana
    dimensao_estado = 1       # log-preço
    dimensao_oculta = 32
    num_camadas = 3

    # Treinamento
    batch_size = 64
    epocas = 200
    taxa_aprendizado = 1e-3
    fator_kl = 1e-3           # peso do termo KL (Bayes)

    # Avaliação / cenários
    num_cenarios_amostragem = 50

cfg = Configuracoes()

# ------------------------------------------------------------
# 1) Gerar dados financeiros sintéticos (GBM em log-preço)
# ------------------------------------------------------------

def simular_log_preco_gbm(
    num_trajetorias: int,
    num_passos: int,
    dt: float,
    mu: float,
    sigma: float,
    preco_inicial: float
):
    """
    Simula trajetórias de log-preço para um GBM:
        dS_t = mu S_t dt + sigma S_t dW_t
    equivalente em log:
        dX_t = (mu - 0.5 sigma^2) dt + sigma dW_t
    """
    x0 = math.log(preco_inicial)
    tempos = np.linspace(0.0, num_passos * dt, num_passos + 1, dtype=np.float32)
    trajetorias = np.zeros((num_trajetorias, num_passos + 1), dtype=np.float32)
    trajetorias[:, 0] = x0

    drift_log = mu - 0.5 * sigma**2
    for i in range(num_passos):
        z = np.random.randn(num_trajetorias).astype(np.float32)
        trajetorias[:, i+1] = (
            trajetorias[:, i]
            + drift_log * dt
            + sigma * math.sqrt(dt) * z
        )

    return tempos, trajetorias

# ------------------------------------------------------------
# 2) Camada Linear Bayesiana (peso e viés ~ N(mu, sigma²))
# ------------------------------------------------------------

class CamadaLinearBayesiana(nn.Module):
    """
    Camada fully-connected com pesos bayesianos aproximados por N(mu, sigma²).
    Usamos inferência variacional com prior N(0,1) e KL fechado.
    """
    def __init__(self, entrada: int, saida: int):
        super().__init__()
        # Parâmetros da posterior aproximada
        self.peso_loc = nn.Parameter(torch.randn(saida, entrada) * 0.1)
        self.peso_logvar = nn.Parameter(torch.full((saida, entrada), -5.0))

        self.vies_loc = nn.Parameter(torch.zeros(saida))
        self.vies_logvar = nn.Parameter(torch.full((saida,), -5.0))

        # Amostras atuais (usadas após sample_pesos())
        self.peso_amostrado = None
        self.vies_amostrado = None

    def sample_pesos(self):
        """
        Amostra pesos e viés uma vez, fixa nos atributos, e calcula KL da camada.
        """
        eps_w = torch.randn_like(self.peso_loc)
        eps_b = torch.randn_like(self.vies_loc)

        sigma_w = torch.exp(0.5 * self.peso_logvar)
        sigma_b = torch.exp(0.5 * self.vies_logvar)

        self.peso_amostrado = self.peso_loc + sigma_w * eps_w
        self.vies_amostrado = self.vies_loc + sigma_b * eps_b

        # KL( N(mu, sigma²) || N(0,1) ) = 0.5 * sum( sigma² + mu² - 1 - log(sigma²) )
        kl_peso = 0.5 * torch.sum(
            torch.exp(self.peso_logvar) + self.peso_loc**2 - 1.0 - self.peso_logvar
        )
        kl_vies = 0.5 * torch.sum(
            torch.exp(self.vies_logvar) + self.vies_loc**2 - 1.0 - self.vies_logvar
        )

        kl_total = kl_peso + kl_vies
        return kl_total

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aplica a transformação linear usando os pesos amostrados.
        É assumido que 'sample_pesos()' foi chamado antes no passo de treino ou inferência.
        """
        if self.peso_amostrado is None or self.vies_amostrado is None:
            raise RuntimeError("Chame sample_pesos() antes de usar a camada bayesiana.")

        return x @ self.peso_amostrado.t() + self.vies_amostrado


# ------------------------------------------------------------
# 3) Rede Neural Bayesiana para drift do log-preço
# ------------------------------------------------------------

class RedeDriftBayesiana(nn.Module):
    """
    Rede Neural Bayesiana que parametriza o drift da ODE:
        dX_t/dt = f_theta(t, X_t)
    Entrada: [X_t, t]  (dimensão 2)
    Saída:   drift dX_t/dt (dimensão 1)
    """
    def __init__(self, dimensao_estado: int, dimensao_oculta: int, num_camadas: int):
        super().__init__()
        self.dimensao_estado = dimensao_estado

        camadas = []
        entrada = dimensao_estado + 1  # X_t + tempo t

        for i in range(num_camadas - 1):
            camadas.append(CamadaLinearBayesiana(entrada, dimensao_oculta))
            entrada = dimensao_oculta

        camadas.append(CamadaLinearBayesiana(entrada, dimensao_estado))
        self.camadas = nn.ModuleList(camadas)
        self.ativacao = nn.Tanh()

    def sample_pesos(self):
        """
        Amostra pesos de todas as camadas e soma seus KLs.
        Isso corresponde a uma amostra de função f_theta.
        """
        kl_total = 0.0
        for camada in self.camadas:
            kl_total = kl_total + camada.sample_pesos()
        return kl_total

    def forward_drift(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Calcula dX_t/dt = f_theta(t, X_t).
        t é escalar ou tensor; x tem shape [batch, 1].
        """
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        # Criar canal de tempo com mesmo batch
        if isinstance(t, float) or isinstance(t, int):
            t_valor = float(t)
            t_tensor = torch.full_like(x, t_valor)
        else:
            # assume tensor [batch] ou [1]
            t_tensor = t.view(-1, 1).expand_as(x)

        h = torch.cat([x, t_tensor], dim=-1)

        for i, camada in enumerate(self.camadas):
            h = camada(h)
            if i < len(self.camadas) - 1:
                h = self.ativacao(h)

        # h tem shape [batch, 1] (drift)
        return h


# ------------------------------------------------------------
# 4) Integrador ODE (RK4) em PyTorch
# ------------------------------------------------------------

def integrar_ode_rk4(funcao_drift, x0: torch.Tensor, tempos: torch.Tensor):
    """
    Integra ODE:
        dX/dt = funcao_drift(t, X)
    usando método de Runge–Kutta de 4ª ordem, em PyTorch.

    x0: tensor [batch, 1]
    tempos: tensor [T] (escalares ordenados)
    retorna: trajetorias [batch, T, 1]
    """
    assert tempos.dim() == 1
    batch = x0.shape[0]
    num_passos = tempos.shape[0]

    xs = []
    x_atual = x0

    xs.append(x_atual)

    for i in range(1, num_passos):
        t0 = tempos[i-1]
        t1 = tempos[i]
        h = t1 - t0

        k1 = funcao_drift(t0, x_atual)
        k2 = funcao_drift(t0 + 0.5 * h, x_atual + 0.5 * h * k1)
        k3 = funcao_drift(t0 + 0.5 * h, x_atual + 0.5 * h * k2)
        k4 = funcao_drift(t0 + h, x_atual + h * k3)

        x_atual = x_atual + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        xs.append(x_atual)

    # [T, batch, 1] -> [batch, T, 1]
    trajetorias = torch.stack(xs, dim=1)
    return trajetorias


# ------------------------------------------------------------
# 5) Preparar dados para treino (PyTorch)
# ------------------------------------------------------------

def preparar_dados_treino():
    tempos_np, trajetorias_np = simular_log_preco_gbm(
        cfg.num_trajetorias_treino + cfg.num_trajetorias_teste,
        cfg.num_passos_tempo,
        cfg.dt,
        cfg.mu_verdadeiro,
        cfg.sigma_verdadeiro,
        cfg.preco_inicial
    )

    tempos = torch.tensor(tempos_np, dtype=torch.float32, device=dispositivo)  # [T+1]
    trajetorias = torch.tensor(trajetorias_np, dtype=torch.float32, device=dispositivo)  # [N, T+1]

    # Separar treino e teste
    trajetorias_treino = trajetorias[:cfg.num_trajetorias_treino]
    trajetorias_teste = trajetorias[cfg.num_trajetorias_treino:]

    # Dataset: cada amostra é uma trajetória completa
    dataset_treino = TensorDataset(trajetorias_treino)
    loader_treino = DataLoader(dataset_treino, batch_size=cfg.batch_size, shuffle=True)

    return tempos, trajetorias_treino, trajetorias_teste, loader_treino


# ------------------------------------------------------------
# 6) Função de treino (Bayesian Neural ODE)
# ------------------------------------------------------------

def treinar_rede_neural_ode_bayesiana():
    tempos, trajetorias_treino, trajetorias_teste, loader_treino = preparar_dados_treino()

    # Instanciar rede
    rede = RedeDriftBayesiana(
        dimensao_estado=cfg.dimensao_estado,
        dimensao_oculta=cfg.dimensao_oculta,
        num_camadas=cfg.num_camadas
    ).to(dispositivo)

    otimizador = torch.optim.Adam(rede.parameters(), lr=cfg.taxa_aprendizado)

    num_passos = tempos.shape[0]
    print("Treinando Bayesian Neural ODE...")
    print(f"Num trajetórias treino: {trajetorias_treino.shape[0]}")
    print(f"Num passos tempo: {num_passos}")
    print(f"Dispositivo: {dispositivo}")

    for epoca in range(1, cfg.epocas + 1):
        rede.train()
        perdas = []
        mse_medios = []
        kl_medios = []

        for (trajetoria_batch,) in loader_treino:
            # trajetoria_batch: [batch, T+1]
            batch_size_atual = trajetoria_batch.shape[0]

            # Separar estado inicial e restante
            x0 = trajetoria_batch[:, 0:1]     # [batch, 1]
            x_real = trajetoria_batch[:, :]   # [batch, T+1]

            # Aviso: tempos tem T+1 pontos
            tempos_torch = tempos

            # 1) Amostrar pesos (amostra de função f_theta)
            kl_pesos = rede.sample_pesos()

            # 2) Integrar ODE com RK4
            def drift_func(t, x):
                return rede.forward_drift(t, x)

            x_pred = integrar_ode_rk4(drift_func, x0, tempos_torch)  # [batch, T+1, 1]
            x_pred = x_pred.squeeze(-1)                              # [batch, T+1]

            # 3) Definir loss: MSE + KL
            mse = torch.mean((x_pred - x_real) ** 2)

            # Normalizar KL por número de amostras e passos (heurística)
            kl_normalizado = kl_pesos / (cfg.num_trajetorias_treino * num_passos)

            perda = mse + cfg.fator_kl * kl_normalizado

            otimizador.zero_grad()
            perda.backward()
            otimizador.step()

            perdas.append(perda.item())
            mse_medios.append(mse.item())
            kl_medios.append(kl_normalizado.item())

        if epoca % 10 == 0 or epoca == 1:
            print(
                f"Época {epoca:4d} | "
                f"Perda média: {np.mean(perdas):.6f} | "
                f"MSE: {np.mean(mse_medios):.6f} | "
                f"KL norm: {np.mean(kl_medios):.6f}"
            )

    print("Treino concluído.")
    return rede, tempos, trajetorias_treino, trajetorias_teste


# ------------------------------------------------------------
# 7) Amostragem Bayesiana — Cenários de preço com incerteza
# ------------------------------------------------------------

def gerar_cenarios_bayesianos(rede, tempos, traj_teste):
    """
    Gera vários cenários de preços a partir de uma trajetória de teste,
    amostrando pesos da rede (incerteza bayesiana).
    """
    rede.eval()
    with torch.no_grad():
        # Escolher uma trajetória de teste como "verdade"
        traj_verdade = traj_teste[0:1]  # [1, T+1]
        x0 = traj_verdade[:, 0:1]       # [1, 1]

        tempos_torch = tempos
        num_passos = tempos_torch.shape[0]

        cenarios_log = []

        for k in range(cfg.num_cenarios_amostragem):
            kl = rede.sample_pesos()  # amostra novos pesos (ignoramos KL aqui)

            def drift_func(t, x):
                return rede.forward_drift(t, x)

            x_pred = integrar_ode_rk4(drift_func, x0, tempos_torch)  # [1, T+1, 1]
            x_pred = x_pred.squeeze(0).squeeze(-1)                  # [T+1]
            cenarios_log.append(x_pred.cpu().numpy())

        cenarios_log = np.stack(cenarios_log, axis=0)  # [num_cenarios, T+1]

        # Converter para preço
        cenarios_preco = np.exp(cenarios_log)          # [num_cenarios, T+1]
        preco_verdade = np.exp(traj_verdade.cpu().numpy())  # [1, T+1]

        # Estatísticas
        media_preco = np.mean(cenarios_preco, axis=0)
        desvio_preco = np.std(cenarios_preco, axis=0)

        return {
            "tempos": tempos_torch.cpu().numpy(),
            "cenarios_preco": cenarios_preco,
            "preco_verdade": preco_verdade[0],
            "media_preco": media_preco,
            "desvio_preco": desvio_preco,
        }


# ------------------------------------------------------------
# 8) Visualização — Fan chart com incerteza bayesiana
# ------------------------------------------------------------

def plotar_cenarios(resultados):
    tempos = resultados["tempos"]
    cenarios_preco = resultados["cenarios_preco"]
    preco_verdade = resultados["preco_verdade"]
    media_preco = resultados["media_preco"]
    desvio_preco = resultados["desvio_preco"]

    plt.figure(figsize=(10, 6))

    # Alguns cenários individuais
    for i in range(min(10, cenarios_preco.shape[0])):
        plt.plot(tempos, cenarios_preco[i], alpha=0.3, linewidth=1)

    # Faixas de incerteza (1 e 2 desvios padrão)
    plt.fill_between(
        tempos,
        media_preco - desvio_preco,
        media_preco + desvio_preco,
        alpha=0.25,
        label="±1 desvio (incerteza bayesiana)"
    )
    plt.fill_between(
        tempos,
        media_preco - 2 * desvio_preco,
        media_preco + 2 * desvio_preco,
        alpha=0.15,
        label="±2 desvios (incerteza bayesiana)"
    )

    # Média e verdade
    plt.plot(tempos, media_preco, label="Média predita (Bayesian Neural ODE)", linewidth=2)
    plt.plot(tempos, preco_verdade, label="Trajetória real (simulada GBM)", linewidth=2, linestyle="--")

    plt.title("Incerteza Financeira — Bayesian Neural ODE no Log-Preço (Black–Scholes Bayesiano)")
    plt.xlabel("Tempo (anos)")
    plt.ylabel("Preço do ativo")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 9) Execução principal
# ------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    rede_bayesiana, tempos, traj_treino, traj_teste = treinar_rede_neural_ode_bayesiana()
    resultados = gerar_cenarios_bayesianos(rede_bayesiana, tempos, traj_teste)
    plotar_cenarios(resultados)
