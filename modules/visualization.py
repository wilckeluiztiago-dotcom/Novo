import matplotlib.pyplot as plt

def plotar_curvas(tensoes_V, correntes_J, potencias):
    """
    Plota as curvas J-V e P-V da célula fotovoltaica.
    
    Parâmetros:
        tensoes_V : Array de tensões [V]
        correntes_J : Array de densidades de corrente [A/m^2]
        potencias : Array de potências [W/m^2]
    """
    # Configurar estilo dos gráficos
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 10
    
    # Figura 1: Curva J-V
    plt.figure(figsize=(10, 6))
    plt.plot(tensoes_V, correntes_J * 0.1, linewidth=2, color='#2E86AB')
    plt.axhline(0, linestyle="--", color='gray', alpha=0.7)
    plt.xlabel("Tensão [V]", fontsize=12, fontweight='bold')
    plt.ylabel("Densidade de Corrente [mA/cm²]", fontsize=12, fontweight='bold')
    plt.title("Curva J–V — Célula Fotovoltaica (Simulador Modular)", 
              fontsize=14, fontweight='bold', pad=15)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Figura 2: Curva P-V
    plt.figure(figsize=(10, 6))
    plt.plot(tensoes_V, potencias, linewidth=2, color='#A23B72')
    plt.xlabel("Tensão [V]", fontsize=12, fontweight='bold')
    plt.ylabel("Potência [W/m²]", fontsize=12, fontweight='bold')
    plt.title("Curva P–V — Célula Fotovoltaica (Simulador Modular)", 
              fontsize=14, fontweight='bold', pad=15)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()
