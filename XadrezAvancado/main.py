"""
Xadrez Avançado 2.0 - Entry Point
"""

import sys
import os

# Adiciona diretório atual ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from interface.gui import JogoXadrez

if __name__ == "__main__":
    print("Iniciando Xadrez 2.0...")
    jogo = JogoXadrez()
    jogo.loop()

