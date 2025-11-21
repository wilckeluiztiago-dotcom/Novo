import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
import mpl_toolkits.mplot3d as plt3d

class DNAPoissonBoltzmann:
    def __init__(self):
        # Constantes físicas
        self.constante_boltzmann = 1.380649e-23  # J/K
        self.carga_eletron = 1.60217662e-19     # C
        self.permissividade_vacuo = 8.854187817e-12  # F/m
        self.temperatura = 298.15  # K
        self.constante_gas = 8.314462618  # J/(mol·K)
        
        # Parâmetros do DNA
        self.raio_dna = 1.0  # nm
        self.comprimento_dna = 20.0  # nm
        self.densidade_carga_dna = -0.176  # e⁻/nm² (típico para DNA)
        
        # Parâmetros da solução
        self.concentracao_sal = 0.1  # M
        self.permissividade_agua = 78.4
        
    def calcular_comprimento_debye(self):
        """Calcula o comprimento de Debye para a solução"""
        # Conversão para m^-3
        concentracao_m3 = self.concentracao_sal * 1000 * 6.022e23
        
        kappa_quadrado = (2 * self.concentracao_sal * 1000 * 
                         (self.carga_eletron**2) / 
                         (self.permissividade_agua * self.permissividade_vacuo * 
                          self.constante_boltzmann * self.temperatura))
        
        comprimento_debye = 1 / np.sqrt(kappa_quadrado) * 1e9  # em nm
        return comprimento_debye, kappa_quadrado
    
    def criar_geometria_dna(self, pontos_por_camada=50, camadas_radiais=100):
        """Cria a geometria 2D radial para o DNA"""
        raio_maximo = 10.0  # nm
        raios = np.linspace(self.raio_dna, raio_maximo, camadas_radiais)
        angulos = np.linspace(0, 2*np.pi, pontos_por_camada, endpoint=False)
        
        R, Theta = np.meshgrid(raios, angulos)
        
        # Coordenadas cartesianas
        X = R * np.cos(Theta)
        Y = R * np.sin(Theta)
        
        return X, Y, R, Theta, raios, angulos
    
    def resolver_epb_radial(self, potencial_superficie=-0.1):
        """Resolve a EPB na geometria radial do DNA"""
        comprimento_debye, kappa_quadrado = self.calcular_comprimento_debye()
        
        # Discretização radial
        num_pontos_radiais = 200
        raio_maximo = 15.0  # nm
        raios = np.linspace(self.raio_dna, raio_maximo, num_pontos_radiais)
        dr = raios[1] - raios[0]
        
        # Matriz do sistema
        diagonal_principal = np.zeros(num_pontos_radiais)
        diagonal_inferior = np.zeros(num_pontos_radiais - 1)
        diagonal_superior = np.zeros(num_pontos_radiais - 1)
        vetor_independente = np.zeros(num_pontos_radiais)
        
        # Preencher matriz para pontos internos
        for i in range(1, num_pontos_radiais - 1):
            r = raios[i]
            
            diagonal_principal[i] = -2/dr**2 - 2/(r*dr) - kappa_quadrado
            diagonal_inferior[i-1] = 1/dr**2 - 1/(r*dr)
            diagonal_superior[i] = 1/dr**2 + 1/(r*dr)
        
        # Condições de contorno
        # Superfície do DNA
        diagonal_principal[0] = 1.0
        diagonal_superior[0] = 0.0
        vetor_independente[0] = potencial_superficie
        
        # Longe do DNA (potencial zero)
        diagonal_principal[-1] = 1.0
        diagonal_inferior[-1] = 0.0
        vetor_independente[-1] = 0.0
        
        # Montar matriz esparsa
        A = sparse.diags([diagonal_inferior, diagonal_principal, diagonal_superior],
                        offsets=[-1, 0, 1], format='csr')
        
        # Resolver sistema
        potencial_radial = spsolve(A, vetor_independente)
        
        return raios, potencial_radial, comprimento_debye
    
    def calcular_distribuicao_ions(self, potencial_radial, raios):
        """Calcula a distribuição de íons ao redor do DNA"""
        kT = self.constante_boltzmann * self.temperatura
        
        # Concentração de íons positivos (cátions)
        concentracao_cations = self.concentracao_sal * np.exp(-self.carga_eletron * 
                                                             potencial_radial / kT)
        
        # Concentração de íons negativos (ânions)
        concentracao_anions = self.concentracao_sal * np.exp(self.carga_eletron * 
                                                            potencial_radial / kT)
        
        return concentracao_cations, concentracao_anions
    
    def calcular_energia_eletrostatica(self, potencial_radial, raios):
        """Calcula a energia eletrostática do sistema"""
        dr = raios[1] - raios[0]
        densidade_energia = 0.5 * self.permissividade_agua * self.permissividade_vacuo * \
                           (np.gradient(potencial_radial, dr)**2)
        
        energia_total = 2 * np.pi * self.comprimento_dna * \
                       np.trapz(densidade_energia * raios, raios)
        
        return energia_total
    
    def visualizar_resultados(self, raios, potencial_radial, concentracao_cations, 
                            concentracao_anions, comprimento_debye):
        """Visualiza os resultados da simulação"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Potencial elétrico
        ax1.plot(raios, potencial_radial * 1000, 'b-', linewidth=2)
        ax1.axvline(x=self.raio_dna, color='r', linestyle='--', label='Superfície DNA')
        ax1.axvline(x=self.raio_dna + comprimento_debye, color='g', linestyle='--', 
                   label='Camada de Debye')
        ax1.set_xlabel('Distância do centro (nm)')
        ax1.set_ylabel('Potencial Elétrico (mV)')
        ax1.set_title('Distribuição do Potencial Elétrico ao redor do DNA')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Distribuição de íons
        ax2.semilogy(raios, concentracao_cations, 'r-', linewidth=2, label='Cátions (Na⁺)')
        ax2.semilogy(raios, concentracao_anions, 'b-', linewidth=2, label='Ânions (Cl⁻)')
        ax2.axhline(y=self.concentracao_sal, color='k', linestyle='--', 
                   label='Concentração Bulk')
        ax2.axvline(x=self.raio_dna, color='r', linestyle='--')
        ax2.set_xlabel('Distância do centro (nm)')
        ax2.set_ylabel('Concentração (M)')
        ax2.set_title('Distribuição de Íons ao redor do DNA')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Campo elétrico
        campo_eletrico = -np.gradient(potencial_radial, raios[1]-raios[0])
        ax3.plot(raios, campo_eletrico * 1e9, 'purple', linewidth=2)
        ax3.axvline(x=self.raio_dna, color='r', linestyle='--')
        ax3.set_xlabel('Distância do centro (nm)')
        ax3.set_ylabel('Campo Elétrico (V/m) × 10⁹')
        ax3.set_title('Campo Elétrico ao redor do DNA')
        ax3.grid(True, alpha=0.3)
        
        # 4. Densidade de carga
        densidade_carga = self.carga_eletron * (concentracao_cations - concentracao_anions) * 1000
        ax4.plot(raios, densidade_carga, 'darkorange', linewidth=2)
        ax4.axvline(x=self.raio_dna, color='r', linestyle='--')
        ax4.axhline(y=0, color='k', linestyle='-', alpha=0.5)
        ax4.set_xlabel('Distância do centro (nm)')
        ax4.set_ylabel('Densidade de Carga (C/m³)')
        ax4.set_title('Densidade de Carga Líquida na Solução')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def simulacao_completa_dna(self, potencial_superficie=-0.1):
        """Executa a simulação completa do sistema DNA-solução"""
        print("=== SIMULAÇÃO DE POISSON-BOLTZMANN PARA DNA ===")
        print(f"Parâmetros da simulação:")
        print(f"- Concentração de sal: {self.concentracao_sal} M")
        print(f"- Temperatura: {self.temperatura - 273.15:.1f} °C")
        print(f"- Potencial na superfície: {potencial_superficie*1000:.1f} mV")
        
        # Calcular comprimento de Debye
        comprimento_debye, kappa_quadrado = self.calcular_comprimento_debye()
        print(f"- Comprimento de Debye: {comprimento_debye:.2f} nm")
        
        # Resolver EPB
        print("\nResolvendo equação de Poisson-Boltzmann...")
        raios, potencial_radial, comprimento_debye = self.resolver_epb_radial(potencial_superficie)
        
        # Calcular distribuição de íons
        print("Calculando distribuição de íons...")
        concentracao_cations, concentracao_anions = self.calcular_distribuicao_ions(
            potencial_radial, raios)
        
        # Calcular energia
        print("Calculando propriedades energéticas...")
        energia_total = self.calcular_energia_eletrostatica(potencial_radial, raios)
        
        print(f"\nResultados:")
        print(f"- Potencial a 2 nm: {potencial_radial[50]*1000:.1f} mV")
        print(f"- Concentração de cátions na superfície: {concentracao_cations[0]:.2f} M")
        print(f"- Concentração de ânions na superfície: {concentracao_anions[0]:.3f} M")
        print(f"- Energia eletrostática total: {energia_total*1e21:.2f} × 10⁻²¹ J")
        
        # Visualizar resultados
        print("\nGerando visualizações...")
        figura = self.visualizar_resultados(raios, potencial_radial, 
                                          concentracao_cations, concentracao_anions,
                                          comprimento_debye)
        
        return {
            'raios': raios,
            'potencial': potencial_radial,
            'concentracao_cations': concentracao_cations,
            'concentracao_anions': concentracao_anions,
            'energia_total': energia_total,
            'comprimento_debye': comprimento_debye
        }

# Exemplo de uso
if __name__ == "__main__":
    # Criar instância do simulador
    simulador_dna = DNAPoissonBoltzmann()
    
    # Executar simulação completa
    resultados = simulador_dna.simulacao_completa_dna(potencial_superficie=-0.08)
    
    # Simulação com diferentes concentrações de sal
    print("\n\n=== ANÁLISE COM DIFERENTES CONCENTRAÇÕES DE SAL ===")
    
    concentracoes_sal = [0.01, 0.05, 0.1, 0.2]
    cores = ['red', 'blue', 'green', 'purple']
    
    plt.figure(figsize=(12, 8))
    
    for conc, cor in zip(concentracoes_sal, cores):
        simulador_dna.concentracao_sal = conc
        raios, potencial, comprimento_debye = simulador_dna.resolver_epb_radial(-0.08)
        plt.plot(raios, potencial * 1000, color=cor, linewidth=2, 
                label=f'{conc} M (λ_D = {comprimento_debye:.1f} nm)')
        plt.axvline(x=simulador_dna.raio_dna + comprimento_debye, 
                   color=cor, linestyle=':', alpha=0.5)
    
    plt.axvline(x=simulador_dna.raio_dna, color='black', linestyle='--', 
               linewidth=2, label='Superfície DNA')
    plt.xlabel('Distância do centro (nm)')
    plt.ylabel('Potencial Elétrico (mV)')
    plt.title('Efeito da Concentração de Sal no Potencial ao redor do DNA')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()