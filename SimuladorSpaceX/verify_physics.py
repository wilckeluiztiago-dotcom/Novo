import sys
import os
import numpy as np

# Adicionar diretório raiz ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modelo.fisica import Foguete, solver_rk4

def verificar():
    print("Verificando Modelo Físico...")
    foguete = Foguete(carga_util=5000)
    resultados = solver_rk4(foguete, t_max=300)
    
    alt_max = np.max(resultados['altitude'])
    vel_max = np.max(resultados['velocidade'])
    
    print(f"Apogeu: {alt_max/1000:.2f} km")
    print(f"Velocidade Máxima: {vel_max:.2f} m/s")
    print("Simulação concluída com sucesso.")

if __name__ == "__main__":
    verificar()
