"""
Interface de Linha de Comando (CLI) para BitcoinPreditor

Permite executar previsões rápidas diretamente do terminal.

Autor: Luiz Tiago Wilcke
"""

import argparse
import numpy as np
import sys
import os

# Adiciona raiz ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.loader import DataLoader
from models.bates import BatesModel
from engine.calibration import Calibrator

def main():
    parser = argparse.ArgumentParser(description="BitcoinPreditor CLI - Previsão Estocástica")
    parser.add_argument("--ticker", type=str, default="BTC-USD", help="Símbolo do ativo (ex: BTC-USD, ETH-USD)")
    parser.add_argument("--days", type=int, default=30, help="Horizonte de previsão em dias")
    parser.add_argument("--sims", type=int, default=1000, help="Número de simulações Monte Carlo")
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"🚀 INICIANDO PREVISÃO PARA: {args.ticker}")
    print(f"{'='*60}")
    
    # 1. Dados
    print("\n[1/3] Baixando dados históricos...")
    loader = DataLoader(args.ticker)
    try:
        df = loader.get_data()
        price = df['Price'].iloc[-1]
        print(f"      Preço Atual: ${price:,.2f}")
    except Exception as e:
        print(f"Erro: {e}")
        return

    # 2. Calibração
    print("\n[2/3] Calibrando Modelo de Bates (Heston + Jumps)...")
    calibrator = Calibrator(df)
    params = calibrator.calibrate()
    
    print(f"      Volatilidade Atual: {np.sqrt(params['v0']):.1%}")
    print(f"      Intensidade de Saltos: {params['lambda_j']:.2f}/ano")

    # 3. Simulação
    print(f"\n[3/3] Executando {args.sims} simulações para {args.days} dias...")
    model = BatesModel(params)
    T = args.days / 365
    dt = 1/365
    times, paths = model.simulate(T, dt, args.sims)
    
    # Resultados
    final_prices = paths[-1, :]
    mean_price = np.mean(final_prices)
    p5 = np.percentile(final_prices, 5)
    p95 = np.percentile(final_prices, 95)
    prob_up = np.mean(final_prices > price)
    
    print(f"\n{'='*60}")
    print(f"📊 RESULTADOS DA PREVISÃO ({args.days} dias)")
    print(f"{'='*60}")
    print(f"Preço Esperado:   ${mean_price:,.2f} ({mean_price/price - 1:+.2%})")
    print(f"Intervalo 90%:    ${p5:,.2f} a ${p95:,.2f}")
    print(f"Prob. de Alta:    {prob_up:.1%}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
