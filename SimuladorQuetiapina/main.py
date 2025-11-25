"""
Simulador Principal de Quetiapina
Autor: Luiz Tiago Wilcke
Data: 2025-11-25

Script principal para executar simulações via linha de comando
"""

import numpy as np
import matplotlib.pyplot as plt
from farmacocinetica import (ParametrosFarmacocineticos, 
                             ModeloFarmacocinetico,
                             RegimePosologico)
from farmacodinamica import ModeloFarmacodinamico
from visualizacao import VisualizadorQuetiapina
import argparse
import sys


def banner():
    """Exibe banner do programa"""
    print("=" * 80)
    print("                 SIMULADOR DE QUETIAPINA NO CÉREBRO HUMANO")
    print("              Modelo Farmacocinético e Farmacodinâmico Avançado")
    print("                        Autor: Luiz Tiago Wilcke")
    print("=" * 80)
    print()


def simular_dose_unica(peso_kg: float, dose_mg: float, via: str = "oral"):
    """
    Simula dose única de Quetiapina
    
    Args:
        peso_kg: Peso corporal (kg)
        dose_mg: Dose (mg)
        via: Via de administração
    """
    print(f"\n{'='*80}")
    print(f"SIMULAÇÃO: Dose Única")
    print(f"{'='*80}")
    print(f"Peso corporal: {peso_kg} kg")
    print(f"Dose: {dose_mg} mg")
    print(f"Via: {via}")
    print(f"{'='*80}\n")
    
    # Criar modelos
    params_pk = ParametrosFarmacocineticos(peso_corporal=peso_kg)
    modelo_pk = ModeloFarmacocinetico(params_pk)
    modelo_pd = ModeloFarmacodinamico()
    visualizador = VisualizadorQuetiapina()
    
    # Simular farmacocinética
    print("🔬 Simulando farmacocinética...")
    tempo, concentracoes = modelo_pk.simular(
        dose_mg=dose_mg,
        tempo_horas=72.0,
        num_pontos=1000,
        via=via
    )
    
    # Calcular parâmetros PK
    params_calculados = modelo_pk.calcular_parametros_pk(tempo, concentracoes[:, 1])
    
    print("\n📊 PARÂMETROS FARMACOCINÉTICOS:")
    print("-" * 80)
    for param, valor in params_calculados.items():
        print(f"  {param:30s}: {valor:10.2f}")
    
    # Simular farmacodinâmica
    print("\n🧠 Simulando farmacodinâmica...")
    resultados_pd = modelo_pd.simular_resposta_temporal(
        tempo,
        concentracoes[:, 2]
    )
    
    # Ocupação no pico
    idx_max = np.argmax(concentracoes[:, 2])
    conc_pico = concentracoes[idx_max, 2]
    tempo_pico = tempo[idx_max]
    
    ocupacoes_pico = modelo_pd.calcular_ocupacao_receptores(conc_pico)
    
    print(f"\n🎯 OCUPAÇÃO DE RECEPTORES (Pico em t={tempo_pico:.1f}h):")
    print("-" * 80)
    for receptor, ocupacao in ocupacoes_pico.items():
        nome_completo = modelo_pd.receptores[receptor].nome
        print(f"  {nome_completo:30s}: {ocupacao:5.1f}%")
    
    # Avaliar eficácia
    eficacia = modelo_pd.avaliar_eficacia_terapeutica(ocupacoes_pico)
    
    print(f"\n💊 EFICÁCIA TERAPÊUTICA:")
    print("-" * 80)
    print(f"  Score de Eficácia: {eficacia:.1f}/100")
    
    if eficacia >= 70:
        print("  Status: ✅ ADEQUADA")
    elif eficacia >= 50:
        print("  Status: ⚠️  MODERADA - Considere ajuste de dose")
    else:
        print("  Status: ❌ INSUFICIENTE - Ajuste necessário")
    
    # Efeitos colaterais
    efeitos = modelo_pd.avaliar_efeitos_colaterais(ocupacoes_pico)
    
    print(f"\n⚠️  RISCO DE EFEITOS COLATERAIS:")
    print("-" * 80)
    for efeito, risco in efeitos.items():
        if risco > 50:
            status = "🔴 ALTO"
        elif risco > 25:
            status = "🟡 MODERADO"
        else:
            status = "🟢 BAIXO"
        print(f"  {efeito.replace('_', ' '):25s}: {risco:5.1f}% {status}")
    
    # Gerar gráficos
    print("\n📈 Gerando visualizações...")
    
    fig1 = visualizador.plot_farmacocinetica_completa(
        tempo, concentracoes, params_calculados,
        salvar="resultado_farmacocinetica.png"
    )
    print("  ✓ Salvo: resultado_farmacocinetica.png")
    
    fig2 = visualizador.plot_farmacodinamica(
        tempo, resultados_pd,
        salvar="resultado_farmacodinamica.png"
    )
    print("  ✓ Salvo: resultado_farmacodinamica.png")
    
    fig3 = visualizador.plot_diagrama_cerebro(
        ocupacoes_pico,
        salvar="resultado_cerebro.png"
    )
    print("  ✓ Salvo: resultado_cerebro.png")
    
    plt.close('all')
    
    print(f"\n{'='*80}")
    print("SIMULAÇÃO CONCLUÍDA COM SUCESSO!")
    print(f"{'='*80}\n")


def simular_doses_multiplas(peso_kg: float, dose_mg: float, 
                            intervalo_h: float, num_doses: int):
    """
    Simula regime de doses múltiplas
    
    Args:
        peso_kg: Peso corporal (kg)
        dose_mg: Dose por administração (mg)
        intervalo_h: Intervalo entre doses (horas)
        num_doses: Número de doses
    """
    print(f"\n{'='*80}")
    print(f"SIMULAÇÃO: Doses Múltiplas")
    print(f"{'='*80}")
    print(f"Peso corporal: {peso_kg} kg")
    print(f"Dose: {dose_mg} mg")
    print(f"Intervalo: {intervalo_h} horas")
    print(f"Número de doses: {num_doses}")
    print(f"{'='*80}\n")
    
    # Criar modelos
    params_pk = ParametrosFarmacocineticos(peso_corporal=peso_kg)
    modelo_pk = ModeloFarmacocinetico(params_pk)
    modelo_pd = ModeloFarmacodinamico()
    visualizador = VisualizadorQuetiapina()
    
    # Simular
    print("🔬 Simulando regime de doses múltiplas...")
    regime = RegimePosologico(modelo_pk)
    
    tempo_total = intervalo_h * num_doses + 24
    tempo, concentracoes = regime.simular_doses_multiplas(
        dose_mg=dose_mg,
        intervalo_horas=intervalo_h,
        num_doses=num_doses,
        tempo_total_horas=tempo_total
    )
    
    # Analisar steady-state
    # Pegar última dose
    inicio_ultima_dose = intervalo_h * (num_doses - 1)
    idx_inicio = np.argmin(np.abs(tempo - inicio_ultima_dose))
    
    C_max_ss = np.max(concentracoes[idx_inicio:, 1])
    C_min_ss = np.min(concentracoes[idx_inicio:, 1])
    C_avg_ss = np.mean(concentracoes[idx_inicio:, 1])
    
    print(f"\n📊 ESTADO DE EQUILÍBRIO (Steady-State):")
    print("-" * 80)
    print(f"  Cmax,ss: {C_max_ss:.2f} ng/mL")
    print(f"  Cmin,ss: {C_min_ss:.2f} ng/mL")
    print(f"  Cavg,ss: {C_avg_ss:.2f} ng/mL")
    print(f"  Flutuação: {((C_max_ss - C_min_ss) / C_avg_ss * 100):.1f}%")
    
    # Gerar gráficos
    print("\n📈 Gerando visualizações...")
    
    fig = visualizador.plot_doses_multiplas(
        tempo, concentracoes, intervalo_h, num_doses,
        salvar="resultado_doses_multiplas.png"
    )
    print("  ✓ Salvo: resultado_doses_multiplas.png")
    
    plt.close('all')
    
    print(f"\n{'='*80}")
    print("SIMULAÇÃO CONCLUÍDA COM SUCESSO!")
    print(f"{'='*80}\n")


def main():
    """Função principal"""
    parser = argparse.ArgumentParser(
        description="Simulador de Quetiapina no Cérebro Humano",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:

  # Dose única de 300 mg para paciente de 70 kg
  python main.py --peso 70 --dose 300
  
  # Regime de 5 doses de 200 mg a cada 12 horas
  python main.py --peso 70 --dose 200 --multiplas --num-doses 5 --intervalo 12
  
  # Dose única intravenosa
  python main.py --peso 80 --dose 300 --via intravenosa
        """
    )
    
    parser.add_argument("--peso", type=float, default=70.0,
                       help="Peso corporal em kg (padrão: 70)")
    parser.add_argument("--dose", type=float, default=300.0,
                       help="Dose em mg (padrão: 300)")
    parser.add_argument("--via", type=str, default="oral",
                       choices=["oral", "intravenosa"],
                       help="Via de administração (padrão: oral)")
    parser.add_argument("--multiplas", action="store_true",
                       help="Simular doses múltiplas")
    parser.add_argument("--num-doses", type=int, default=5,
                       help="Número de doses (padrão: 5)")
    parser.add_argument("--intervalo", type=float, default=12.0,
                       help="Intervalo entre doses em horas (padrão: 12)")
    
    args = parser.parse_args()
    
    # Banner
    banner()
    
    # Executar simulação apropriada
    if args.multiplas:
        simular_doses_multiplas(
            peso_kg=args.peso,
            dose_mg=args.dose,
            intervalo_h=args.intervalo,
            num_doses=args.num_doses
        )
    else:
        simular_dose_unica(
            peso_kg=args.peso,
            dose_mg=args.dose,
            via=args.via
        )


if __name__ == "__main__":
    main()
