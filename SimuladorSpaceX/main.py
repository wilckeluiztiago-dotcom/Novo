import sys
import os

def main():
    print("="*50)
    print("   SIMULADOR SPACEX - FALCON 9")
    print("   Autor: Luiz Tiago Wilcke")
    print("="*50)
    print("\nEscolha o modo de operação:")
    print("1. Calculadora de Missão (Interface Web)")
    print("2. Simulação Visual (Pygame)")
    print("3. Sair")
    
    escolha = input("\nOpção [1-3]: ")
    
    if escolha == '1':
        print("\nIniciando Calculadora Streamlit...")
        os.system("streamlit run interface/calculadora.py")
    elif escolha == '2':
        print("\nIniciando Simulação Visual...")
        from simulacao.visualizador import Visualizador
        sim = Visualizador()
        sim.executar()
    elif escolha == '3':
        print("Saindo...")
        sys.exit()
    else:
        print("Opção inválida!")

if __name__ == "__main__":
    main()
