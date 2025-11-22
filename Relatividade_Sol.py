import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class SistemaSolarRelativistico:
    def __init__(self):
        # Constantes fundamentais
        self.constante_gravitacional = 6.67430e-11  # m³ kg⁻¹ s⁻²
        self.velocidade_luz = 299792458.0  # m/s
        self.massa_sol = 1.989e30  # kg
        
        # Parâmetros do sistema solar
        self.unidade_astronomica = 1.496e11  # m
        self.raio_sol = 6.957e8  # m
        self.massa_terra = 5.972e24  # kg
        
        # Raio de Schwarzschild do Sol
        self.raio_schwarzschild_sol = (2 * self.constante_gravitacional * 
                                     self.massa_sol / self.velocidade_luz**2)
        
        # Parâmetros orbitais da Terra
        self.semi_eixo_maior_terra = self.unidade_astronomica
        self.excentricidade_terra = 0.0167
        self.periodo_orbital_terra = 3.15576e7  # segundos (1 ano)
        
    def tensor_metrico_schwarzschild(self, coordenada_radial):
        """
        Calcula o tensor métrico para a solução de Schwarzschild
        """
        fator_metrico = 1 - self.raio_schwarzschild_sol / coordenada_radial
        
        g_tt = -fator_metrico * self.velocidade_luz**2
        g_rr = 1 / fator_metrico
        g_theta_theta = coordenada_radial**2
        g_phi_phi = coordenada_radial**2
        
        return {
            'componente_temporal': g_tt,
            'componente_radial': g_rr,
            'componente_polar': g_theta_theta,
            'componente_azimutal': g_phi_phi,
            'fator_metrico': fator_metrico
        }
    
    def calcular_simbolos_christoffel(self, coordenada_radial, angulo_polar):
        """
        Calcula os símbolos de Christoffel não nulos para a métrica de Schwarzschild
        """
        fator_metrico = 1 - self.raio_schwarzschild_sol / coordenada_radial
        
        simbolos = {}
        
        # Componentes não nulos
        simbolos['Gamma^t_rt'] = simbolos['Gamma^t_tr'] = (
            self.raio_schwarzschild_sol / (2 * coordenada_radial**2 * fator_metrico)
        )
        
        simbolos['Gamma^r_tt'] = (
            self.raio_schwarzschild_sol * self.velocidade_luz**2 * fator_metrico / 
            (2 * coordenada_radial**2)
        )
        
        simbolos['Gamma^r_rr'] = -self.raio_schwarzschild_sol / (
            2 * coordenada_radial**2 * fator_metrico
        )
        
        simbolos['Gamma^r_thetatheta'] = -coordenada_radial * fator_metrico
        simbolos['Gamma^r_phiphi'] = (
            -coordenada_radial * fator_metrico * np.sin(angulo_polar)**2
        )
        
        simbolos['Gamma^theta_rtheta'] = simbolos['Gamma^theta_thetar'] = 1 / coordenada_radial
        simbolos['Gamma^theta_phiphi'] = -np.sin(angulo_polar) * np.cos(angulo_polar)
        
        simbolos['Gamma^phi_rphi'] = simbolos['Gamma^phi_phir'] = 1 / coordenada_radial
        simbolos['Gamma^phi_thetaphi'] = simbolos['Gamma^phi_phitheta'] = (
            np.cos(angulo_polar) / np.sin(angulo_polar)
        )
        
        return simbolos
    
    def tensor_curvatura_riemann(self, coordenada_radial, angulo_polar):
        """
        Calcula componentes do tensor de curvatura de Riemann
        """
        r_s = self.raio_schwarzschild_sol
        r = coordenada_radial
        theta = angulo_polar
        
        # Componentes não nulos do tensor de Riemann
        componentes = {}
        
        R_trtr = -r_s / (r**3 * (1 - r_s/r))
        R_tθtθ = r_s / (2 * r * (1 - r_s/r))
        R_tφtφ = R_tθtθ * np.sin(theta)**2
        
        R_rθrθ = -r_s / (2 * r * (1 - r_s/r))
        R_rφrφ = R_rθrθ * np.sin(theta)**2
        R_θφθφ = r_s * r * np.sin(theta)**2
        
        componentes['R^t_rtr'] = R_trtr
        componentes['R^t_θtθ'] = R_tθtθ
        componentes['R^t_φtφ'] = R_tφtφ
        componentes['R^r_θrθ'] = R_rθrθ
        componentes['R^r_φrφ'] = R_rφrφ
        componentes['R^θ_φθφ'] = R_θφθφ
        
        return componentes
    
    def tensor_ricci_e_escalar_curvatura(self, coordenada_radial):
        """
        Para o vácuo (fora do Sol), o tensor de Ricci é zero
        """
        tensor_ricci = {
            'R_tt': 0.0, 'R_rr': 0.0, 'R_thetatheta': 0.0, 'R_phiphi': 0.0
        }
        
        escalar_curvatura = 0.0  # R = 0 no vácuo
        
        return tensor_ricci, escalar_curvatura
    
    def equacao_geodesica_relativistica(self, parametro_afim, variaveis_estado):
        """
        Equação geodésica para movimento no plano equatorial (θ = π/2)
        """
        tempo_coordenado, posicao_radial, angulo_azimutal, \
        momento_temporal, momento_radial, momento_azimutal = variaveis_estado
        
        # Componentes da métrica
        metricas = self.tensor_metrico_schwarzschild(posicao_radial)
        fator_metrico = 1 - self.raio_schwarzschild_sol / posicao_radial
        
        g_tt = metricas['componente_temporal']
        g_rr = metricas['componente_radial']
        g_phiphi = metricas['componente_azimutal']
        
        # Derivadas da métrica
        dg_tt_dr = self.raio_schwarzschild_sol * self.velocidade_luz**2 / posicao_radial**2
        dg_rr_dr = self.raio_schwarzschild_sol / (posicao_radial**2 * fator_metrico**2)
        dg_phiphi_dr = 2 * posicao_radial
        
        # Equações de evolução
        derivada_tempo = -momento_temporal / g_tt
        derivada_radial = momento_radial / g_rr
        derivada_azimutal = -momento_azimutal / g_phiphi
        
        # Conservações
        derivada_momento_temporal = 0.0  # Energia conservada
        derivada_momento_azimutal = 0.0  # Momento angular conservado
        
        # Equação radial
        derivada_momento_radial = -0.5 * (
            dg_tt_dr * derivada_tempo**2 +
            dg_rr_dr * derivada_radial**2 +
            dg_phiphi_dr * derivada_azimutal**2
        )
        
        return [
            derivada_tempo, derivada_radial, derivada_azimutal,
            derivada_momento_temporal, derivada_momento_radial, derivada_momento_azimutal
        ]
    
    def resolver_orbita_relativistica(self, tempo_total_anos=1, pontos_por_ano=1000):
        """
        Resolve a órbita relativística da Terra com alta precisão
        """
        # Condições iniciais (Terra no periélio)
        posicao_radial_inicial = self.semi_eixo_maior_terra * (1 - self.excentricidade_terra)
        velocidade_orbital_inicial = np.sqrt(
            self.constante_gravitacional * self.massa_sol * 
            (1 + self.excentricidade_terra) / 
            (posicao_radial_inicial * (1 - self.excentricidade_terra))
        )
        
        # Fatores relativísticos
        fator_metrico_inicial = 1 - self.raio_schwarzschild_sol / posicao_radial_inicial
        fator_lorentz = 1 / np.sqrt(1 - (velocidade_orbital_inicial/self.velocidade_luz)**2)
        
        # Momentos iniciais generalizados
        energia_especifica = fator_lorentz * fator_metrico_inicial
        momento_angular_especifico = (
            posicao_radial_inicial * velocidade_orbital_inicial * fator_lorentz
        )
        
        momento_temporal_inicial = -energia_especifica
        momento_radial_inicial = 0.0  # Início no periélio
        momento_azimutal_inicial = momento_angular_especifico
        
        # Vetor de estado inicial
        estado_inicial = [
            0.0,  # tempo_coordenado
            posicao_radial_inicial,  # posicao_radial
            0.0,  # angulo_azimutal
            momento_temporal_inicial,  # momento_temporal
            momento_radial_inicial,  # momento_radial
            momento_azimutal_inicial  # momento_azimutal
        ]
        
        # Tempo de integração
        tempo_total_segundos = tempo_total_anos * self.periodo_orbital_terra
        tempo_pontos = np.linspace(0, tempo_total_segundos, 
                                 int(pontos_por_ano * tempo_total_anos))
        
        # Resolver equações diferenciais
        solucao = solve_ivp(
            self.equacao_geodesica_relativistica,
            [0, tempo_total_segundos],
            estado_inicial,
            t_eval=tempo_pontos,
            method='RK45',
            rtol=1e-12,
            atol=1e-15
        )
        
        return solucao
    
    def calcular_precessão_periélio(self, semi_eixo_maior, excentricidade):
        """
        Calcula a precessão do periélio usando fórmula analítica
        """
        a = semi_eixo_maior
        e = excentricidade
        
        precessao_por_orbita = (
            6 * np.pi * self.constante_gravitacional * self.massa_sol /
            (self.velocidade_luz**2 * a * (1 - e**2))
        )
        
        precessao_por_seculo = (
            precessao_por_orbita * (180 * 3600 / np.pi) *  # converter para segundos de arco
            (100 / (self.periodo_orbital_terra / 3.15576e7))  # por século
        )
        
        return precessao_por_orbita, precessao_por_seculo
    
    def analisar_resultados_numericos(self, solucao):
        """
        Analisa resultados com 5 dígitos de precisão
        """
        tempo = solucao.t
        r = solucao.y[1]
        phi = solucao.y[2]
        
        # Encontrar periélios
        indices_perihelio = []
        for i in range(1, len(r)-1):
            if r[i] < r[i-1] and r[i] < r[i+1]:
                indices_perihelio.append(i)
        
        # Precessão numérica
        if len(indices_perihelio) >= 2:
            delta_phi_numerico = (phi[indices_perihelio[1]] - phi[indices_perihelio[0]]) - 2*np.pi
            precessao_numerica_seg_arco = delta_phi_numerico * (180 * 3600 / np.pi)
        else:
            precessao_numerica_seg_arco = 0.0
        
        # Resultados formatados
        resultados = {
            'raio_minimo_metros': float(f"{np.min(r):.5e}"),
            'raio_maximo_metros': float(f"{np.max(r):.5e}"),
            'excentricidade_numerica': float(f"{((np.max(r) - np.min(r))/(np.max(r) + np.min(r))):.5f}"),
            'precessao_numerica_seg_arco_por_orbita': float(f"{precessao_numerica_seg_arco:.5f}"),
            'raio_schwarzschild_sol_metros': float(f"{self.raio_schwarzschild_sol:.5e}"),
            'fator_curvatura_superficie_sol': float(f"{self.raio_schwarzschild_sol/self.raio_sol:.5e}")
        }
        
        return resultados

# Execução e análise
if __name__ == "__main__":
    sistema = SistemaSolarRelativistico()
    
    print("=== MODELAMENTO RELATIVÍSTICO DO SISTEMA SOLAR ===")
    print(f"Raio de Schwarzschild do Sol: {sistema.raio_schwarzschild_sol:.5e} m")
    print(f"Fator de curvatura na superfície: {sistema.raio_schwarzschild_sol/sistema.raio_sol:.5e}")
    
    # Calcular precessão analítica
    precessao_orbita, precessao_seculo = sistema.calcular_precessão_periélio(
        sistema.semi_eixo_maior_terra, sistema.excentricidade_terra
    )
    
    print(f"\n=== PRECESSÃO DO PERIÉLIO ===")
    print(f"Precessão por órbita: {precessao_orbita:.5e} rad")
    print(f"Precessão por século: {precessao_seculo:.5f} segundos de arco")
    
    # Resolver numericamente
    print(f"\n=== SOLUÇÃO NUMÉRICA ===")
    solucao_numerica = sistema.resolver_orbita_relativistica(tempo_total_anos=1)
    resultados = sistema.analisar_resultados_numericos(solucao_numerica)
    
    for chave, valor in resultados.items():
        print(f"{chave}: {valor}")
    
    # Visualização
    r = solucao_numerica.y[1] / sistema.unidade_astronomica
    phi = solucao_numerica.y[2]
    
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(x, y, 'b-', linewidth=1)
    plt.plot(0, 0, 'yo', markersize=10, label='Sol')
    plt.xlabel('x (UA)')
    plt.ylabel('y (UA)')
    plt.title('Órbita Relativística da Terra')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(solucao_numerica.t / sistema.periodo_orbital_terra, r, 'g-')
    plt.xlabel('Tempo (anos)')
    plt.ylabel('Distância Radial (UA)')
    plt.title('Variação Radial da Órbita')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    tempo_anos = solucao_numerica.t / sistema.periodo_orbital_terra
    angulo_graus = np.degrees(phi) % 360
    plt.plot(tempo_anos, angulo_graus, 'r-')
    plt.xlabel('Tempo (anos)')
    plt.ylabel('Ângulo Azimutal (graus)')
    plt.title('Evolução Angular')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    # Curvatura espaço-temporal
    r_test = np.linspace(sistema.raio_sol, 2*sistema.unidade_astronomica, 1000)
    curvatura_temporal = 1 - sistema.raio_schwarzschild_sol / r_test
    plt.semilogx(r_test / sistema.unidade_astronomica, curvatura_temporal, 'purple')
    plt.axvline(sistema.raio_sol / sistema.unidade_astronomica, color='red', 
                linestyle='--', label='Superfície Solar')
    plt.axvline(1, color='orange', linestyle='--', label='Órbita Terra')
    plt.xlabel('Distância (UA)')
    plt.ylabel('Componente gₜₜ da Métrica')
    plt.title('Curvatura Temporal do Espaço-Tempo')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Tensor de Riemann na órbita da Terra
    print(f"\n=== COMPONENTES DO TENSOR DE RIEMANN ===")
    componentes_riemann = sistema.tensor_curvatura_riemann(
        sistema.unidade_astronomica, np.pi/2
    )
    for componente, valor in componentes_riemann.items():
        print(f"{componente}: {valor:.5e}")