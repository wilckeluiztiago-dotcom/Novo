import numpy as np
import matplotlib.pyplot as plt

class KAN:
    """
    Implementação de Rede Kolmogorov-Arnold (KAN) baseada no teorema:
    f(x₁,x₂,...,xₙ) = Σᵩ Φ_q(Σₚ Ψ_q,p(xₚ))
    
    Onde cada Ψ e Φ são funções suaves de uma variável,
    representadas por B-splines com coeficientes aprendíveis.
    """
    
    def __init__(self, dim_entrada=2, dim_saida=1, num_funcoes_ocultas=10, grau_spline=3, num_pontos_controle=5):
        self.dim_entrada = dim_entrada
        self.dim_saida = dim_saida
        self.num_funcoes_ocultas = num_funcoes_ocultas
        self.grau_spline = grau_spline
        self.num_pontos_controle = num_pontos_controle
        
        # Inicializar funções Ψ_q,p (camada de entrada)
        # Cada Ψ é uma B-spline com coeficientes aprendíveis
        self.coef_psi = np.random.randn(
            num_funcoes_ocultas, dim_entrada, num_pontos_controle + grau_spline
        ) * 0.1
        
        # Inicializar funções Φ_q (camada de saída)
        self.coef_phi = np.random.randn(
            dim_saida, num_funcoes_ocultas, num_pontos_controle + grau_spline
        ) * 0.1
        
        # Grid uniforme para as B-splines
        self.grid = np.linspace(-2, 2, num_pontos_controle + grau_spline + 1)
        
        # Pesos base lineares (como no paper original)
        self.pesos_base_psi = np.random.randn(num_funcoes_ocultas, dim_entrada) * 0.1
        self.pesos_base_phi = np.random.randn(dim_saida, num_funcoes_ocultas) * 0.1
        
        self.historico_perda = []
    
    def funcao_silu(self, x):
        """Função SiLU (Swish) - suave e diferenciável"""
        return x / (1 + np.exp(-x))
    
    def calcular_bspline(self, x, coeficientes, indice_funcao, indice_entrada):
        """Calcula o valor da B-spline para x dado os coeficientes"""
        # Para simplificação, usaremos uma aproximação por interpolação
        # Em uma implementação completa, usaríamos a fórmula recursiva de B-spline
        
        pontos_controle = self.grid[:len(coeficientes[indice_funcao, indice_entrada])]
        valores_base = coeficientes[indice_funcao, indice_entrada]
        
        # Interpolação cúbica simples como aproximação
        x_normalizado = (x + 2) / 4  # Normalizar para [0,1]
        x_normalizado = np.clip(x_normalizado, 0, 1)
        
        indice = int(x_normalizado * (len(pontos_controle) - 1))
        indice = min(indice, len(pontos_controle) - 2)
        
        t = x_normalizado * (len(pontos_controle) - 1) - indice
        return (1 - t) * valores_base[indice] + t * valores_base[indice + 1]
    
    def propagacao_direta(self, x):
        """
        Implementa: f(x) = Σ_q Φ_q(Σ_p Ψ_q,p(x_p))
        """
        self.x_entrada = x
        
        # Primeira camada: Ψ_q,p(x_p)
        self.saida_psi = np.zeros(self.num_funcoes_ocultas)
        
        for q in range(self.num_funcoes_ocultas):
            soma_psi = 0
            for p in range(self.dim_entrada):
                # Componente B-spline
                componente_spline = self.calcular_bspline(
                    x[p], self.coef_psi, q, p
                )
                # Componente linear base
                componente_linear = self.pesos_base_psi[q, p] * x[p]
                soma_psi += componente_spline + componente_linear
            
            self.saida_psi[q] = self.funcao_silu(soma_psi)
        
        # Segunda camada: Φ_q(·)
        self.saida_final = np.zeros(self.dim_saida)
        
        for saida_idx in range(self.dim_saida):
            soma_phi = 0
            for q in range(self.num_funcoes_ocultas):
                # Componente B-spline
                componente_spline = self.calcular_bspline(
                    self.saida_psi[q], self.coef_phi, saida_idx, q
                )
                # Componente linear base
                componente_linear = self.pesos_base_phi[saida_idx, q] * self.saida_psi[q]
                soma_phi += componente_spline + componente_linear
            
            self.saida_final[saida_idx] = soma_phi
        
        return self.saida_final
    
    def retropropagacao(self, alvo, taxa_aprendizado=0.01):
        """Algoritmo de retropropagação para KANs"""
        erro = self.saida_final - alvo
        
        # Gradientes para camada Φ
        grad_phi = erro
        
        # Atualizar coeficientes de Φ
        for saida_idx in range(self.dim_saida):
            for q in range(self.num_funcoes_ocultas):
                # Atualizar B-spline de Φ
                for k in range(len(self.coef_phi[saida_idx, q])):
                    # Gradiente aproximado
                    self.coef_phi[saida_idx, q, k] -= taxa_aprendizado * grad_phi[saida_idx] * 0.1
                
                # Atualizar peso base de Φ
                self.pesos_base_phi[saida_idx, q] -= (
                    taxa_aprendizado * grad_phi[saida_idx] * self.saida_psi[q]
                )
        
        # Gradientes para camada Ψ
        grad_psi = np.zeros(self.num_funcoes_ocultas)
        for q in range(self.num_funcoes_ocultas):
            for saida_idx in range(self.dim_saida):
                grad_psi[q] += grad_phi[saida_idx] * (
                    self.pesos_base_phi[saida_idx, q] + 0.1  # Aproximação do gradiente da B-spline
                )
        
        # Atualizar coeficientes de Ψ
        for q in range(self.num_funcoes_ocultas):
            for p in range(self.dim_entrada):
                # Atualizar B-spline de Ψ
                for k in range(len(self.coef_psi[q, p])):
                    self.coef_psi[q, p, k] -= (
                        taxa_aprendizado * grad_psi[q] * 0.1 * self.x_entrada[p]
                    )
                
                # Atualizar peso base de Ψ
                self.pesos_base_psi[q, p] -= (
                    taxa_aprendizado * grad_psi[q] * self.x_entrada[p]
                )
        
        return np.mean(erro ** 2)
    
    def treinar(self, dados_entrada, dados_alvo, epocas=1000, taxa_aprendizado=0.01):
        """Treina a rede KAN"""
        print("Iniciando treinamento da KAN...")
        
        for epoca in range(epocas):
            perda_total = 0
            num_amostras = len(dados_entrada)
            
            for i in range(num_amostras):
                saida = self.propagacao_direta(dados_entrada[i])
                perda = self.retropropagacao(dados_alvo[i], taxa_aprendizado)
                perda_total += perda
            
            perda_media = perda_total / num_amostras
            self.historico_perda.append(perda_media)
            
            if epoca % 100 == 0:
                print(f"Época {epoca}: Perda = {perda_media:.6f}")

# Exemplo de uso: Aprender função não-linear complexa
def gerar_dados_treinamento():
    """Gera dados para uma função não-linear complexa"""
    np.random.seed(42)
    num_amostras = 200
    dados_entrada = []
    dados_alvo = []
    
    for _ in range(num_amostras):
        x1, x2 = np.random.uniform(-1, 1, 2)
        # Função não-linear complexa: sin(πx₁) + cos(πx₂) + x₁*x₂
        y = np.sin(np.pi * x1) + np.cos(np.pi * x2) + x1 * x2
        dados_entrada.append(np.array([x1, x2]))
        dados_alvo.append(np.array([y]))
    
    return dados_entrada, dados_alvo

# Demonstração da KAN
def demonstrar_kan():
    # Gerar dados
    dados_entrada, dados_alvo = gerar_dados_treinamento()
    
    # Criar KAN com arquitetura revolucionária
    kan = KAN(
        dim_entrada=2,
        dim_saida=1, 
        num_funcoes_ocultas=10,  # 2n+1 como no teorema original
        grau_spline=3,
        num_pontos_controle=8
    )
    
    # Treinar
    kan.treinar(dados_entrada, dados_alvo, epocas=1000, taxa_aprendizado=0.01)
    
    # Testar
    print("\n--- Testando KAN ---")
    casos_teste = [
        np.array([0.0, 0.0]),
        np.array([0.5, 0.5]),
        np.array([-0.5, 0.5]),
        np.array([1.0, 0.0])
    ]
    
    for caso in casos_teste:
        pred = kan.propagacao_direta(caso)
        valor_real = np.sin(np.pi * caso[0]) + np.cos(np.pi * caso[1]) + caso[0] * caso[1]
        print(f"Entrada: {caso} -> KAN: {pred[0]:.4f}, Real: {valor_real:.4f}")
    
    # Visualizar aprendizado
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(kan.historico_perda)
    plt.title('Evolução da Perda - KAN')
    plt.xlabel('Época')
    plt.ylabel('Perda')
    plt.yscale('log')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    # Visualizar superfície de decisão
    x = np.linspace(-1, 1, 20)
    y = np.linspace(-1, 1, 20)
    X, Y = np.meshgrid(x, y)
    Z_kan = np.zeros_like(X)
    Z_real = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z_kan[i,j] = kan.propagacao_direta(np.array([X[i,j], Y[i,j]]))[0]
            Z_real[i,j] = np.sin(np.pi * X[i,j]) + np.cos(np.pi * Y[i,j]) + X[i,j] * Y[i,j]
    
    plt.contourf(X, Y, Z_kan, levels=20, cmap='viridis')
    plt.title('Superfície da KAN')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()
    
    return kan

if __name__ == "__main__":
    kan_treinada = demonstrar_kan()