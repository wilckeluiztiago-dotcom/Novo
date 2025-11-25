import sys
import os

def main():
    print("="*50)
    print("   SIMULADOR VAREJO BRASIL")
    print("   Autor: Luiz Tiago Wilcke")
    print("="*50)
    print("\nIniciando Dashboard Streamlit...")
    
    # Executar o Streamlit
    os.system("streamlit run interface/dashboard.py")

if __name__ == "__main__":
    main()
