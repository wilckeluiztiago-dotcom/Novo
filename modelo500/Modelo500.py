"""
Script Principal do Sistema de Modelagem de Desemprego

Orquestra todos os módulos para realizar simulações completas,
análises e visualizações.

Autor: Luiz Tiago Wilcke
Data: 2025-11-24
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

# Importa módulos do sistema
from modelos_sde import criar_modelo, MODELOS_DISPONIVEIS
from simulador import SimuladorSDE
from gerador_dados import GeradorDados
from visualizador import Visualizador
from analise import AnalisadorEstatistico
import config


def criar_diretorio_resultados():
    """Cria diretório para salvar resultados."""
    if not os.path.exists(config.DIRETORIO_RESULTADOS):
        os.makedirs(config.DIRETORIO_RESULTADOS)
        print(f"Diretório '{config.DIRETORIO_RESULTADOS}' criado.")


def executar_simulacao_basica(
    nome_modelo: str = 'goodwin',
    T: float = None,
    N: int = None,
    num_trajetorias: int = None,
    metodo: str = None,
    seed: int = None,
    salvar_graficos: bool = True
):
    """
    Executa uma simulação básica de um modelo.
    
    Args:
        nome_modelo: Nome do modelo a simular
        T: Tempo final
        N: Número de passos
        num_trajetorias: Número de trajetórias Monte Carlo
        metodo: Método numérico
        seed: Semente aleatória
        salvar_graficos: Se True, salva gráficos em disco
    """
    print("\n" + "="*80)
    print(f"SIMULAÇÃO DO MODELO: {nome_modelo.upper()}")
    print("="*80)
    
    # Obtém configuração
    cfg = config.obter_config_simulacao(nome_modelo, T, N, num_trajetorias, metodo, seed)
    config.imprimir_configuracao(cfg)
    
    # Cria modelo
    print("Criando modelo...")
    modelo = criar_modelo(nome_modelo, cfg['parametros'])
    
    # Cria simulador
    print("Inicializando simulador...")
    sim = SimuladorSDE(modelo, seed=cfg['seed'])
    
    # Simula múltiplas trajetórias
    print(f"Simulando {cfg['num_trajetorias']} trajetórias...")
    tempos, trajetorias = sim.simular_multiplas_trajetorias(
        T=cfg['tempo_final'],
        N=cfg['numero_passos'],
        num_trajetorias=cfg['num_trajetorias'],
        metodo=cfg['metodo'],
        mostrar_progresso=True
    )
    
    # Calcula desemprego
    print("Calculando taxas de desemprego...")
    desemprego = sim.calcular_desemprego_trajetorias(trajetorias)
    
    # Estatísticas do ensemble
    print("Calculando estatísticas...")
    stats = sim.calcular_estatisticas_ensemble(trajetorias)
    desemprego_medio = np.mean(desemprego, axis=0)
    
    print(f"\nTaxa de desemprego média final: {desemprego_medio[-1]:.4%}")
    print(f"Intervalo [Q5-Q95]: [{np.percentile(desemprego[:, -1], 5):.4%}, "
          f"{np.percentile(desemprego[:, -1], 95):.4%}]")
    
    # Visualização
    print("\nGerando visualizações...")
    criar_diretorio_resultados()
    vis = Visualizador(dpi=config.DPI_PADRAO)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefixo = f"{config.DIRETORIO_RESULTADOS}/{nome_modelo}_{timestamp}"
    
    # Gráfico 1: Trajetórias
    print("  - Trajetórias de desemprego...")
    caminho = f"{prefixo}_trajetorias.png" if salvar_graficos else None
    vis.plotar_trajetorias(
        tempos, desemprego,
        titulo=f"Trajetórias de Desemprego - Modelo {nome_modelo.capitalize()}",
        salvar=caminho
    )
    
    # Gráfico 2: Distribuição final
    print("  - Distribuição da taxa de desemprego...")
    caminho = f"{prefixo}_distribuicao.png" if salvar_graficos else None
    vis.plotar_distribuicao(
        desemprego[:, -1],
        titulo=f"Distribuição do Desemprego (t={cfg['tempo_final']}) - {nome_modelo.capitalize()}",
        salvar=caminho
    )
    
    # Gráfico 3: Diagrama de fase (se aplicável)
    if nome_modelo in ['goodwin', 'phillips']:
        print("  - Diagrama de fase...")
        
        # Usa a primeira trajetória para o diagrama de fase
        if nome_modelo == 'goodwin':
            labels = ['Taxa de Emprego (u)', 'Parcela Salarial (v)']
        else:  # phillips
            labels = ['Inflação (π)', 'Desemprego (u)']
        
        caminho = f"{prefixo}_fase.png" if salvar_graficos else None
        vis.plotar_diagrama_fase(
            trajetorias[0],
            labels=labels,
            titulo=f"Diagrama de Fase - {nome_modelo.capitalize()}",
            salvar=caminho
        )
    
    # Análise estatística
    print("\nRealizando análise estatística...")
    analisador = AnalisadorEstatistico()
    df_analise = analisador.analise_completa(
        desemprego.flatten(),
        nome_serie=f"Desemprego - {nome_modelo.capitalize()}"
    )
    
    # Salva resultados numéricos
    if salvar_graficos:
        # DataFrame com resultados
        df_resultados = pd.DataFrame({
            'tempo': tempos,
            'desemprego_medio': desemprego_medio,
            'desemprego_std': np.std(desemprego, axis=0),
            'desemprego_q05': np.percentile(desemprego, 5, axis=0),
            'desemprego_q95': np.percentile(desemprego, 95, axis=0)
        })
        
        caminho_csv = f"{prefixo}_resultados.csv"
        df_resultados.to_csv(caminho_csv, index=False)
        print(f"\nResultados salvos em: {caminho_csv}")
        
        caminho_stats = f"{prefixo}_estatisticas.csv"
        df_analise.to_csv(caminho_stats, index=False)
        print(f"Estatísticas salvas em: {caminho_stats}")
    
    print("\n" + "="*80)
    print("SIMULAÇÃO CONCLUÍDA COM SUCESSO!")
    print("="*80 + "\n")
    
    return tempos, desemprego, stats


def executar_comparacao_modelos(
    modelos: list = None,
    T: float = 10.0,
    N: int = 1000,
    seed: int = 42,
    salvar_graficos: bool = True
):
    """
    Executa e compara múltiplos modelos.
    
    Args:
        modelos: Lista de nomes de modelos (usa todos se None)
        T: Tempo final
        N: Número de passos
        seed: Semente
        salvar_graficos: Se True, salva gráficos
    """
    if modelos is None:
        modelos = list(MODELOS_DISPONIVEIS.keys())
    
    print("\n" + "="*80)
    print("COMPARAÇÃO DE MODELOS")
    print("="*80)
    print(f"Modelos: {', '.join([m.capitalize() for m in modelos])}\n")
    
    dados_modelos = {}
    
    for nome_modelo in modelos:
        print(f"\nSimulando modelo: {nome_modelo.capitalize()}")
        print("-" * 40)
        
        # Cria e simula modelo
        modelo = criar_modelo(nome_modelo)
        sim = SimuladorSDE(modelo, seed=seed)
        
        tempos, trajetoria = sim.euler_maruyama(T, N)
        
        # Calcula desemprego
        desemprego = np.array([
            modelo.calcular_desemprego(trajetoria[i])
            for i in range(len(tempos))
        ])
        
        # Armazena
        df = pd.DataFrame({
            'tempo': tempos,
            'desemprego': desemprego
        })
        dados_modelos[nome_modelo] = df
        
        print(f"  Desemprego final: {desemprego[-1]:.4%}")
    
    # Visualização comparativa
    print("\nGerando visualização comparativa...")
    criar_diretorio_resultados()
    vis = Visualizador(dpi=config.DPI_ALTA_QUALIDADE)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    caminho = f"{config.DIRETORIO_RESULTADOS}/comparacao_modelos_{timestamp}.png"
    
    if salvar_graficos:
        vis.plotar_comparacao_modelos(
            dados_modelos,
            titulo="Comparação de Modelos de Desemprego",
            salvar=caminho
        )
    else:
        vis.plotar_comparacao_modelos(
            dados_modelos,
            titulo="Comparação de Modelos de Desemprego"
        )
    
    print("\n" + "="*80)
    print("COMPARAÇÃO CONCLUÍDA!")
    print("="*80 + "\n")
    
    return dados_modelos


def executar_analise_convergencia(
    nome_modelo: str = 'goodwin',
    salvar_graficos: bool = True
):
    """
    Analisa convergência do método numérico.
    
    Args:
        nome_modelo: Modelo a testar
        salvar_graficos: Se True, salva gráficos
    """
    print("\n" + "="*80)
    print(f"ANÁLISE DE CONVERGÊNCIA - {nome_modelo.upper()}")
    print("="*80 + "\n")
    
    # Cria modelo e simulador
    modelo = criar_modelo(nome_modelo)
    sim = SimuladorSDE(modelo, seed=config.SEED_PADRAO)
    
    # Testa convergência
    print("Testando convergência para diferentes valores de N...")
    N_valores, erros = sim.convergencia_forte(
        T=config.TEMPO_FINAL_PADRAO,
        N_valores=config.N_VALORES_CONVERGENCIA,
        num_trajetorias=config.N_TRAJETORIAS_CONVERGENCIA,
        metodo='euler'
    )
    
    # Estima taxa
    from simulador import AnaliseConvergencia
    taxa, _ = AnaliseConvergencia.calcular_taxa_convergencia(N_valores, erros)
    
    print(f"\nTaxa de convergência estimada: {taxa:.3f}")
    print("(Teórica para Euler-Maruyama: 0.5)")
    
    # Visualiza
    criar_diretorio_resultados()
    vis = Visualizador(dpi=config.DPI_PADRAO)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    caminho = f"{config.DIRETORIO_RESULTADOS}/convergencia_{nome_modelo}_{timestamp}.png"
    
    if salvar_graficos:
        vis.plotar_convergencia(
            N_valores, erros,
            taxa_teorica=0.5,
            titulo=f"Análise de Convergência - {nome_modelo.capitalize()}",
            salvar=caminho
        )
    else:
        vis.plotar_convergencia(
            N_valores, erros,
            taxa_teorica=0.5,
            titulo=f"Análise de Convergência - {nome_modelo.capitalize()}"
        )
    
    print("\n" + "="*80)
    print("ANÁLISE DE CONVERGÊNCIA CONCLUÍDA!")
    print("="*80 + "\n")


def main():
    """Função principal com interface de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Sistema de Modelagem de Desemprego com SDEs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python main.py --modelo goodwin --trajetorias 100
  python main.py --comparar
  python main.py --convergencia --modelo phillips
  python main.py --modelo markov --tempo 20 --passos 2000
        """
    )
    
    parser.add_argument(
        '--modelo',
        type=str,
        choices=list(MODELOS_DISPONIVEIS.keys()),
        default='goodwin',
        help='Modelo a simular'
    )
    
    parser.add_argument(
        '--tempo',
        type=float,
        default=config.TEMPO_FINAL_PADRAO,
        help='Tempo final de simulação'
    )
    
    parser.add_argument(
        '--passos',
        type=int,
        default=config.NUMERO_PASSOS_PADRAO,
        help='Número de passos de tempo'
    )
    
    parser.add_argument(
        '--trajetorias',
        type=int,
        default=config.NUMERO_TRAJETORIAS_PADRAO,
        help='Número de trajetórias Monte Carlo'
    )
    
    parser.add_argument(
        '--metodo',
        type=str,
        choices=['euler', 'milstein', 'srk'],
        default=config.METODO_NUMERICO_PADRAO,
        help='Método numérico'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=config.SEED_PADRAO,
        help='Semente aleatória'
    )
    
    parser.add_argument(
        '--comparar',
        action='store_true',
        help='Comparar todos os modelos'
    )
    
    parser.add_argument(
        '--convergencia',
        action='store_true',
        help='Análise de convergência'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Não salvar gráficos'
    )
    
    args = parser.parse_args()
    
    salvar = not args.no_save
    
    # Executa ação apropriada
    if args.comparar:
        executar_comparacao_modelos(
            T=args.tempo,
            N=args.passos,
            seed=args.seed,
            salvar_graficos=salvar
        )
    elif args.convergencia:
        executar_analise_convergencia(
            nome_modelo=args.modelo,
            salvar_graficos=salvar
        )
    else:
        executar_simulacao_basica(
            nome_modelo=args.modelo,
            T=args.tempo,
            N=args.passos,
            num_trajetorias=args.trajetorias,
            metodo=args.metodo,
            seed=args.seed,
            salvar_graficos=salvar
        )


if __name__ == '__main__':
    main()
