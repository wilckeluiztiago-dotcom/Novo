"""
Módulo de Física para Diagramas de Feynman
Autor: Luiz Tiago Wilcke

Define partículas, interações e regras de Feynman para cálculo de amplitudes.
"""

import sympy
from sympy import Symbol, Function, Matrix, I, gamma, sqrt

# Inicializar impressão bonita do sympy
sympy.init_printing()

class Particula:
    """Classe base para partículas."""
    def __init__(self, nome, massa, spin, carga, tipo):
        self.nome = nome
        self.massa = Symbol(f'm_{{{nome}}}', real=True) if isinstance(massa, str) else massa
        self.spin = spin
        self.carga = carga
        self.tipo = tipo # 'fermion', 'boson', 'gluon'
        
    def __str__(self):
        return self.nome

class Fermion(Particula):
    """Férmions (elétrons, quarks, etc)."""
    def __init__(self, nome, massa, carga):
        super().__init__(nome, massa, 1/2, carga, 'fermion')
        
class Boson(Particula):
    """Bósons vetoriais (fóton, Z, W)."""
    def __init__(self, nome, massa, carga):
        super().__init__(nome, massa, 1, carga, 'boson')

# Definição de Partículas Padrão
ELETRON = Fermion('e', 'e', -1)
MUON = Fermion('mu', 'mu', -1)
QUARK_UP = Fermion('u', 'u', 2/3)
QUARK_DOWN = Fermion('d', 'd', -1/3)
FOTON = Boson('gamma', 0, 0)
GLUON = Boson('g', 0, 0)
BOSON_Z = Boson('Z', 'Z', 0)

class RegrasFeynman:
    """Gera expressões matemáticas para elementos do diagrama."""
    
    @staticmethod
    def propagador_fermion(momento, massa):
        """
        Propagador de férmion: i(p_slash + m) / (p^2 - m^2)
        """
        p = Symbol(f'p_{{{momento}}}')
        p_slash = Symbol(f'\\slash{{p}}_{{{momento}}}') # Notação simplificada
        return (I * (p_slash + massa)) / (p**2 - massa**2)
    
    @staticmethod
    def propagador_foton(momento):
        """
        Propagador de fóton: -i g_uv / p^2
        (No gauge de Feynman)
        """
        p = Symbol(f'p_{{{momento}}}')
        g_uv = Symbol('g_{\\mu\\nu}')
        return (-I * g_uv) / (p**2)
    
    @staticmethod
    def vertice_qed(carga):
        """
        Vértice QED: -i e gamma^u
        """
        e = Symbol('e')
        gamma_u = Symbol('\\gamma^\\mu')
        return -I * e * gamma_u
    
    @staticmethod
    def spinor_entrada(particula, momento, spin):
        """u(p, s)"""
        return Symbol(f'u(p_{{{momento}}}, s_{{{spin}}})')
    
    @staticmethod
    def spinor_saida(particula, momento, spin):
        """u_bar(p, s)"""
        return Symbol(f'\\bar{{u}}(p_{{{momento}}}, s_{{{spin}}})')
    
    @staticmethod
    def antifermion_entrada(particula, momento, spin):
        """v_bar(p, s)"""
        return Symbol(f'\\bar{{v}}(p_{{{momento}}}, s_{{{spin}}})')
        
    @staticmethod
    def antifermion_saida(particula, momento, spin):
        """v(p, s)"""
        return Symbol(f'v(p_{{{momento}}}, s_{{{spin}}})')
    
    @staticmethod
    def polarizacao_foton(momento, indice_lorentz):
        """epsilon_mu(p)"""
        return Symbol(f'\\epsilon_{{{indice_lorentz}}}(p_{{{momento}}})')

def gerar_integral_exemplo():
    """Gera a integral para o espalhamento Moller (e- e- -> e- e-) como teste."""
    regras = RegrasFeynman()
    
    # Momento transferido
    q = Symbol('q')
    
    # Amplitudes (Diagrama t e u)
    # M = (-ie)^2 * [u_bar(p3) gamma^u u(p1)] * [-ig_uv/q^2] * [u_bar(p4) gamma^v u(p2)]
    
    # Simplificado para demonstração simbólica
    M1 = (regras.vertice_qed(-1) * regras.spinor_saida(ELETRON, 3, 1) * 
          regras.spinor_entrada(ELETRON, 1, 1)) * \
         regras.propagador_foton('q') * \
         (regras.vertice_qed(-1) * regras.spinor_saida(ELETRON, 4, 2) * 
          regras.spinor_entrada(ELETRON, 2, 2))
          
    return M1

if __name__ == "__main__":
    print("Teste do Motor de Física:")
    print(gerar_integral_exemplo())
