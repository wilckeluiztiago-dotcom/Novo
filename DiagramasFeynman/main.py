"""
Editor de Diagramas de Feynman
Autor: Luiz Tiago Wilcke

Ponto de entrada da aplicação.
"""

import sys
import os

# Adicionar diretório atual ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import customtkinter
    import sympy
    import numpy
    import matplotlib
    import PIL
except ImportError as e:
    print("ERRO: Bibliotecas faltando.")
    print(f"Detalhe: {e}")
    print("\nPor favor, instale as dependências:")
    print("pip install customtkinter sympy numpy matplotlib pillow")
    sys.exit(1)

from gui import EditorFeynman

if __name__ == "__main__":
    app = EditorFeynman()
    app.mainloop()
