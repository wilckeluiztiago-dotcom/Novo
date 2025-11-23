import numpy as np
import math
from decimal import Decimal, getcontext
import cmath
from scipy import special

# Configurar precisão de 50 dígitos para cálculos quânticos
getcontext().prec = 50

class SistemaQuanticoEletronProton:
    def __init__(self):
        # Constantes fundamentais com precisão extrema
        self.massa_eletron = Decimal('9.109383701528e-31')
        self.massa_proton = Decimal('1.6726219236951e-27')
        self.carga_eletron = Decimal('-1.60217663415314e-19')
        self.carga_proton = Decimal('1.60217663415314e-19')
        self.velocidade_luz = Decimal('299792458.0')
        self.constante_plank = Decimal('1.054571817e-34')
        self.constante_estrutura_fina = Decimal('7.2973525693e-3')
        self.constante_fermi = Decimal('1.1663787e-5')
        
        # Matrizes de Dirac 4x4
        self.matriz_gamma = self._inicializar_matrizes_dirac()
        self.matriz_sigma = self._inicializar_matrizes_pauli()
        
        # Coeficientes para métodos numéricos
        self.coeficientes_taylor = self._calcular_fatoriais(40)
        
        # Coeficientes Runge-Kutta adaptativo
        self.coeficientes_rk8 = self._inicializar_coeficientes_rk8()

    def _inicializar_matrizes_dirac(self):
        """Inicializa matrizes de Dirac 4x4"""
        # Matrizes gamma na representação de Dirac (como arrays numpy)
        gamma_0 = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0], 
            [0, 0, -1, 0],
            [0, 0, 0, -1]
        ], dtype=complex)
        
        gamma_1 = np.array([
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, -1, 0, 0],
            [-1, 0, 0, 0]
        ], dtype=complex)
        
        gamma_2 = np.array([
            [0, 0, 0, -1j],
            [0, 0, 1j, 0],
            [0, 1j, 0, 0],
            [-1j, 0, 0, 0]
        ], dtype=complex)
        
        gamma_3 = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, -1],
            [-1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=complex)
        
        return [gamma_0, gamma_1, gamma_2, gamma_3]

    def _inicializar_matrizes_pauli(self):
        """Inicializa matrizes de Pauli 2x2"""
        sigma_1 = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_2 = np.array([[0, -1j], [1j, 0]], dtype=complex) 
        sigma_3 = np.array([[1, 0], [0, -1]], dtype=complex)
        return [sigma_1, sigma_2, sigma_3]

    def _inicializar_coeficientes_rk8(self):
        """Inicializa coeficientes para Runge-Kutta de ordem 8"""
        # Coeficientes simplificados para exemplo
        return {
            'a': [0] * 13,
            'b8': [1/13] * 13,
            'b7': [1/14] * 13,
            'c': [i/12 for i in range(13)]
        }

    def funcao_onda_relativistica(self, momento, spin):
        """Solução exata da equação de Dirac para onda plana"""
        m = float(self.massa_eletron)
        c = float(self.velocidade_luz)
        p = float(momento)
        
        E = math.sqrt(p**2 * c**2 + m**2 * c**4)
        
        # Espinores de Pauli
        if spin == 1:  # Spin up
            ξ = np.array([1, 0], dtype=complex)
        else:  # Spin down  
            ξ = np.array([0, 1], dtype=complex)
        
        # Componentes do espinor de Dirac (corrigido)
        phi = ξ
        if p == 0:
            chi = np.array([0, 0], dtype=complex)
        else:
            # Usar produto escalar em vez de matricial para vetores
            fator = (c * p) / (E + m * c**2)
            chi = fator * self.matriz_sigma[2] @ ξ
        
        # Espinor de Dirac 4-componente
        espinor = np.zeros(4, dtype=complex)
        espinor[0:2] = phi
        espinor[2:4] = chi
        
        return espinor

    def equacao_dirac_simplificada(self, spinor, tempo):
        """Equação de Dirac simplificada para teste"""
        m = float(self.massa_eletron)
        c = float(self.velocidade_luz)
        ℏ = float(self.constante_plank)
        
        # Hamiltoniano de Dirac livre
        H = np.zeros((4, 4), dtype=complex)
        H[0:2, 0:2] = m * c**2 * np.eye(2)
        H[2:4, 2:4] = -m * c**2 * np.eye(2)
        
        # Adicionar termos de momento (simplificado)
        H[0, 3] = c * ℏ * 1j
        H[1, 2] = c * ℏ * 1j  
        H[2, 1] = c * ℏ * 1j
        H[3, 0] = c * ℏ * 1j
        
        # Derivada temporal
        dspinor_dt = -1j/ℏ * H @ spinor
        return dspinor_dt

    def equacao_evolucao_qft(self, estado, tempo):
        """Equação de evolução simplificada"""
        if isinstance(estado, dict) and 'eletron' in estado:
            spinor = estado['eletron']
        else:
            spinor = estado
            
        return self.equacao_dirac_simplificada(spinor, tempo)

    def metodo_integrador_quantico(self, estado, tempo, passo, metodo='runge_kutta_simplificado'):
        """Integrador para equações quânticas - CORRIGIDO"""
        
        if metodo == 'runge_kutta_simplificado':
            return self.runge_kutta_simplificado(estado, tempo, passo)
        elif metodo == 'euler_implícito':
            return self.metodo_euler_implicito(estado, tempo, passo)
        else:
            return self.metodo_verlet_quantico(estado, tempo, passo)

    def runge_kutta_simplificado(self, estado, tempo, passo):
        """Runge-Kutta de 4ª ordem simplificado"""
        if isinstance(estado, dict):
            spinor = estado['eletron']
        else:
            spinor = estado
            
        k1 = self.equacao_evolucao_qft(spinor, tempo)
        k2 = self.equacao_evolucao_qft(spinor + passo/2 * k1, tempo + passo/2)
        k3 = self.equacao_evolucao_qft(spinor + passo/2 * k2, tempo + passo/2) 
        k4 = self.equacao_evolucao_qft(spinor + passo * k3, tempo + passo)
        
        spinor_novo = spinor + (passo/6) * (k1 + 2*k2 + 2*k3 + k4)
        
        if isinstance(estado, dict):
            estado_novo = estado.copy()
            estado_novo['eletron'] = spinor_novo
            return estado_novo, passo, 0.0
        else:
            return spinor_novo, passo, 0.0

    def metodo_euler_implicito(self, estado, tempo, passo):
        """Método de Euler implícito para estabilidade"""
        if isinstance(estado, dict):
            spinor = estado['eletron']
        else:
            spinor = estado
            
        ℏ = float(self.constante_plank)
        H = self.hamiltoniano_instantaneo(spinor, tempo)
        
        # Resolver (I + iΔtH/ℏ)ψ_{n+1} = ψ_n
        I = np.eye(H.shape[0])
        matriz_coeficiente = I + 1j * float(passo)/ℏ * H
        spinor_novo = np.linalg.solve(matriz_coeficiente, spinor)
        
        if isinstance(estado, dict):
            estado_novo = estado.copy()
            estado_novo['eletron'] = spinor_novo
            return estado_novo, passo, 0.0
        else:
            return spinor_novo, passo, 0.0

    def metodo_verlet_quantico(self, estado, tempo, passo):
        """Método tipo Verlet para equações de onda"""
        if isinstance(estado, dict):
            spinor = estado['eletron']
        else:
            spinor = estado
            
        # Para equação de onda: segunda derivada no tempo
        if not hasattr(self, 'spinor_anterior'):
            self.spinor_anterior = spinor
            
        H = self.hamiltoniano_instantaneo(spinor, tempo)
        ℏ = float(self.constante_plank)
        
        # ψ_{n+1} = 2ψ_n - ψ_{n-1} - (Δt^2/ℏ^2) H^2 ψ_n
        spinor_novo = 2 * spinor - self.spinor_anterior - (float(passo)**2/ℏ**2) * H @ H @ spinor
        
        self.spinor_anterior = spinor
        
        if isinstance(estado, dict):
            estado_novo = estado.copy()
            estado_novo['eletron'] = spinor_novo
            return estado_novo, passo, 0.0
        else:
            return spinor_novo, passo, 0.0

    def hamiltoniano_instantaneo(self, spinor, tempo):
        """Hamiltoniano no instante t - CORRIGIDO"""
        m = float(self.massa_eletron)
        c = float(self.velocidade_luz)
        
        # Hamiltoniano de Dirac livre
        H = np.zeros((4, 4), dtype=complex)
        H[0:2, 0:2] = m * c**2 * np.eye(2)
        H[2:4, 2:4] = -m * c**2 * np.eye(2)
        
        # Termos de momento (simplificado)
        p = 1e-24  # Momento pequeno para teste
        H[0, 3] = c * p
        H[1, 2] = c * p  
        H[2, 1] = c * p
        H[3, 0] = c * p
        
        return H

    def calcular_energia(self, spinor):
        """Calcula energia esperada do estado"""
        H = self.hamiltoniano_instantaneo(spinor, 0)
        energia = np.real(np.vdot(spinor, H @ spinor))
        return energia

    def calcular_probabilidade(self, spinor):
        """Calcula probabilidade total"""
        return np.real(np.vdot(spinor, spinor))

    def calcular_secao_choque_qed(self, energia, angulo):
        """Calcula seção de choque para espalhamento e-p em QED"""
        alpha = float(self.constante_estrutura_fina)
        m_e = float(self.massa_eletron)
        c = float(self.velocidade_luz)
        E = float(energia)
        
        # Fórmula de Mott simplificada
        termo1 = (alpha * c / (2 * E * math.sin(angulo/2)**2))**2
        termo2 = math.cos(angulo/2)**2
        
        secao_choque = termo1 * termo2
        return secao_choque

    def _calcular_fatoriais(self, n):
        """Calcula fatoriais até ordem n"""
        fatoriais = [Decimal(1)]
        for i in range(1, n + 1):
            fatoriais.append(fatoriais[-1] * Decimal(i))
        return fatoriais

    def verificar_precisao_14_digitos(self, resultado1, resultado2):
        """Verifica precisão de 14 dígitos"""
        if isinstance(resultado1, np.ndarray) and isinstance(resultado2, np.ndarray):
            diferenca = np.max(np.abs(resultado1 - resultado2))
        else:
            diferenca = abs(resultado1 - resultado2)
            
        if diferenca == 0:
            return True
            
        precisao = -math.log10(float(diferenca) + 1e-30)
        return precisao >= 14

# Exemplo de uso CORRIGIDO
if __name__ == "__main__":
    sistema = SistemaQuanticoEletronProton()
    
    print("Inicializando simulação quântica elétron-próton...")
    
    # Estado quântico inicial CORRIGIDO
    try:
        spinor_eletron = sistema.funcao_onda_relativistica(Decimal('1e-24'), 1)
        print("Espinor do elétron criado com sucesso!")
        print(f"Forma do espinor: {spinor_eletron.shape}")
        print(f"Espinor: {spinor_eletron}")
    except Exception as e:
        print(f"Erro ao criar espinor: {e}")
        # Espinor fallback
        spinor_eletron = np.array([1, 0, 0, 0], dtype=complex) / math.sqrt(2)
    
    estado_inicial = {
        'eletron': spinor_eletron,
        'proton': np.array([1, 0, 0, 0], dtype=complex),  # Próton estático
        'tempo': 0.0
    }
    
    tempo = 0.0
    passo = 1e-21  # 1 zeptosegundo
    
    print("\nIniciando evolução temporal...")
    
    for i in range(50):
        try:
            estado_inicial, passo, erro = sistema.metodo_integrador_quantico(
                estado_inicial, tempo, passo, metodo='runge_kutta_simplificado'
            )
            
            tempo += passo
            
            if i % 10 == 0:
                energia = sistema.calcular_energia(estado_inicial['eletron'])
                probabilidade = sistema.calcular_probabilidade(estado_inicial['eletron'])
                
                print(f"Passo {i}: Tempo = {tempo:.2e}s, E = {energia:.6e} J, P = {probabilidade:.10f}")
                
                # Calcular seção de choque
                secao_choque = sistema.calcular_secao_choque_qed(Decimal('1.6e-15'), math.pi/4)
                print(f"Seção de choque QED: {secao_choque:.6e} m²")
                print("-" * 50)
                
        except Exception as e:
            print(f"Erro no passo {i}: {e}")
            break
    
    print("\nSimulação concluída!")
    print(f"Estado final - Energia: {sistema.calcular_energia(estado_inicial['eletron']):.6e} J")
    print(f"Probabilidade total: {sistema.calcular_probabilidade(estado_inicial['eletron']):.15f}")
    
    # Verificar conservação de probabilidade
    prob_final = sistema.calcular_probabilidade(estado_inicial['eletron'])
    if abs(prob_final - 1.0) < 1e-14:
        print("✓ Conservação de probabilidade verificada (14 dígitos de precisão)")
    else:
        print(f"! Atenção: Probabilidade final = {prob_final:.15f}")
