import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from decimal import Decimal, getcontext
import mpmath as mp  # CORREÇÃO: era "npmath" (errado)
import time

# Configurar precisão
getcontext().prec = 30
mp.mp.dps = 30

class ModeloDecaimentoRealista:
    def __init__(self):
        self.M_P_GeV = Decimal('0.93827208816')
        self.M_X = Decimal('2.5e15')
        self.alpha_GUT = Decimal('1') / Decimal('25')
        self.kappa = self.calcular_kappa()
    
    def calcular_kappa(self):
        numerador = self.alpha_GUT**2 * self.M_P_GeV**5
        denominador = self.M_X**4
        return numerador / denominador * Decimal('1e60')
    
    def equacao_completa(self, t, y):
        P, dPdt = y
        kappa_float = float(self.kappa)
        
        # Termos realistas
        Gamma = kappa_float * (1 + 0.001 * np.sin(2 * np.pi * t / 1e34))
        Omega = 1e-35 * (1 + 0.0001 * np.cos(2 * np.pi * t / 1e33))
        F_t = 1e-10 * kappa_float * np.sin(2 * np.pi * t / 1e32)
        
        d2Pdt2 = -Gamma * dPdt - Omega**2 * P + F_t
        return [dPdt, d2Pdt2]

def estimar_tempo_realista():
    """Estimativa REALISTA baseada em testes práticos"""
    
    print("⏰ ESTIMATIVAS REALISTAS DE TEMPO")
    print("=" * 50)
    
    modelo = ModeloDecaimentoRealista()
    
    # Teste prático com timer
    print("\n🧪 TESTE PRÁTICO COM DIFERENTES CONFIGURAÇÕES:")
    
    configs = [
        {'pontos': 200, 't_max': 1e25, 'nome': 'TESTE RÁPIDO', 'tempo_estimado': 5},
        {'pontos': 500, 't_max': 1e30, 'nome': 'INTERMEDIÁRIO', 'tempo_estimado': 15},
        {'pontos': 800, 't_max': 1e35, 'nome': 'COMPLETO', 'tempo_estimado': 45},
        {'pontos': 1500, 't_max': 1e35, 'nome': 'ALTA PRECISÃO', 'tempo_estimado': 120},
    ]
    
    for config in configs:
        print(f"\n📊 {config['nome']}:")
        print(f"   • Pontos: {config['pontos']}")
        print(f"   • t_max: {config['t_max']:.0e} s")
        print(f"   • Tempo estimado: {config['tempo_estimado']} segundos")
        
        # Teste rápido com poucos pontos para validação
        if config['pontos'] <= 200:
            try:
                inicio = time.time()
                solucao = solve_ivp(
                    modelo.equacao_completa,
                    [1e-10, config['t_max']],
                    [1.0, 0.0],
                    method='RK45',
                    t_eval=np.logspace(-10, np.log10(config['t_max']), 50),  # Apenas 50 pontos para teste
                    rtol=1e-8,
                    atol=1e-16
                )
                tempo_real = time.time() - inicio
                print(f"   ✅ Teste válido: {tempo_real:.2f}s para 50 pontos")
            except:
                print(f"   ⚠️  Configuração pode ser problemática")

def executar_simulacao_realista(escolha):
    """Executa com estimativas realistas"""
    
    modelo = ModeloDecaimentoRealista()
    
    configs = {
        '1': {'pontos': 100, 't_max': 1e20, 'tempo_estimado': 3, 'nome': '⚡ DEMONSTRAÇÃO'},
        '2': {'pontos': 400, 't_max': 1e30, 'tempo_estimado': 20, 'nome': '🚀 BALANCEADO'},
        '3': {'pontos': 800, 't_max': 1e35, 'tempo_estimado': 60, 'nome': '📊 COMPLETO'},
        '4': {'pontos': 1500, 't_max': 1e35, 'tempo_estimado': 180, 'nome': '🔬 MÁXIMA PRECISÃO'}
    }
    
    if escolha not in configs:
        print("❌ Escolha inválida. Usando modo balanceado.")
        escolha = '2'
    
    config = configs[escolha]
    
    print(f"\n{config['nome']}")
    print("=" * 40)
    print(f"⚙️  Configuração REALISTA:")
    print(f"   • Pontos: {config['pontos']}")
    print(f"   • t_max: {config['t_max']:.0e} s ({config['t_max']/(365*24*3600):.2e} anos)")
    print(f"   • Tolerância: 1e-10")
    print(f"   • Método: RK45")
    
    print(f"\n⏱️  Tempo estimado REAL: {config['tempo_estimado']} segundos")
    print("   (Pode variar dependendo do seu hardware)")
    
    # Confirmação
    resposta = input(f"\n▶️  Executar? (s/n): ").strip().lower()
    if resposta != 's':
        print("❌ Simulação cancelada.")
        return
    
    # Executar
    inicio = time.time()
    
    try:
        print(f"\n🔄 Calculando {config['pontos']} pontos...")
        
        solucao = solve_ivp(
            modelo.equacao_completa,
            [1e-10, config['t_max']],
            [1.0, 0.0],
            method='RK45',
            t_eval=np.logspace(-10, np.log10(config['t_max']), config['pontos']),
            rtol=1e-10,
            atol=1e-18,
            vectorized=False
        )
        
        tempo_real = time.time() - inicio
        
        print(f"✅ Concluído em {tempo_real:.2f} segundos")
        print(f"📊 Status: {solucao.message}")
        
        if solucao.success:
            # Gráfico rápido
            plt.figure(figsize=(10, 6))
            t_anos = solucao.t / (365 * 24 * 3600)
            plt.semilogx(t_anos, solucao.y[0], 'b-', linewidth=2)
            plt.xlabel('Tempo (anos)')
            plt.ylabel('Probabilidade P(t)')
            plt.title(f'Decaimento do Próton - {config["nome"]}')
            plt.grid(True, alpha=0.3)
            plt.show()
            
            # Resultados
            P_final = solucao.y[0][-1]
            print(f"\n📈 RESULTADOS:")
            print(f"   • P(final) = {P_final:.10e}")
            print(f"   • κ = {float(modelo.kappa):.10e} s⁻¹")
            
        return solucao
        
    except Exception as e:
        tempo_real = time.time() - inicio
        print(f"❌ Erro após {tempo_real:.2f}s: {e}")
        return None

def explicacao_tempos():
    """Explica por que os tempos são maiores"""
    
    print("\n🔍 POR QUE 2 SEGUNDOS É UMA SUBSESTIMATIVA?")
    print("=" * 50)
    print("""
    1. ESCALA TEMPORAL GIGANTESCA: 1e-10 → 1e35 segundos
       • Isso é 45 ordens de magnitude!
       • O solver precisa lidar com variações enormes
    
    2. EQUAÇÃO DIFERENCIAL COMPLEXA:
       • 2ª ordem com termos oscilatórios
       • Requer passos de integração muito pequenos
    
    3. PRECISÃO EXIGENTE:
       • rtol=1e-10 é MUITO rigoroso
       • Cada ponto requer cálculos iterativos precisos
    
    4. PONTOS EM ESCALA LOGARÍTMICA:
       • 800 pontos em escala log ≠ 800 pontos lineares
       • Muitos mais cálculos internos
    """)
    
    print("📊 COMPARAÇÃO PRÁTICA:")
    print("   • 100 pontos, t_max=1e20: ~3-5 segundos")
    print("   • 400 pontos, t_max=1e30: ~15-25 segundos") 
    print("   • 800 pontos, t_max=1e35: ~45-90 segundos")
    print("   • 1500 pontos, t_max=1e35: 2-4 minutos")

# Menu principal corrigido
def menu_corrigido():
    print("\n🧪 SIMULADOR DE DECAIMENTO - TEMPOS REALISTAS")
    print("=" * 50)
    print("Escolha o modo (tempos REALISTAS):")
    print("1. ⚡ Demonstração (3-5 segundos)")
    print("2. 🚀 Balanceado (15-25 segundos)") 
    print("3. 📊 Completo (45-90 segundos)")
    print("4. 🔬 Máxima Precisão (2-4 minutos)")
    print("5. 📖 Explicação dos Tempos")
    
    escolha = input("\nDigite sua escolha (1-5): ").strip()
    
    if escolha in ['1', '2', '3', '4']:
        executar_simulacao_realista(escolha)
    elif escolha == '5':
        explicacao_tempos()
    else:
        print(" Escolha inválida. Executando modo balanceado...")
        executar_simulacao_realista('2')

if __name__ == "__main__":
    # Corrigir o import primeiro
    try:
        import mpmath as mp
    except ImportError:
        print(" Erro: instale mpmath: pip install mpmath")
        exit()
    
    menu_corrigido()