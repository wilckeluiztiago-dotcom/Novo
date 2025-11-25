"""
Script de Teste do Sistema de Análise Eleitoral
Testa todas as funcionalidades principais do sistema
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from utils.dados import gerar_dados_eleitorais, gerar_dados_historicos
from modelos.basicos import ModeloRegressao, ModeloARIMA, ModeloPCA
from modelos.avancados import ModeloRandomForest, ModeloGradientBoosting
from modelos.bayesianos import ModeloDirichlet
from modelos.eleitorais import QuocienteEleitoral, NumeroEfetivoPartidos, IndiceNacionalizacao
from analises.coligacoes import AnalisadorColigacoes
from analises.volatilidade import CalculadorVolatilidade
from analises.fragmentacao import AnalisadorFragmentacao
from analises.competitividade import IndiceCompetitividade

def teste_geracao_dados():
    """Testa geração de dados eleitorais"""
    print("\n" + "="*60)
    print("TESTE 1: Geração de Dados Eleitorais")
    print("="*60)
    
    dados = gerar_dados_eleitorais(n_candidatos=100, ano=2026)
    print(f"✓ Dados gerados: {len(dados)} candidatos")
    print(f"✓ Colunas: {list(dados.columns)}")
    print(f"✓ Total de votos: {dados['votos'].sum():,}")
    print(f"✓ Candidatos eleitos: {dados['eleito'].sum()}")
    print(f"✓ Partidos: {dados['partido'].nunique()}")
    return dados

def teste_regressao(dados):
    """Testa modelo de regressão linear"""
    print("\n" + "="*60)
    print("TESTE 2: Modelo de Regressão Linear")
    print("="*60)
    
    X = dados[['gasto_campanha', 'tempo_tv_segundos', 'incumbente']].values
    y = dados['votos'].values
    
    modelo = ModeloRegressao()
    modelo.treinar(X, y, features_nomes=['Gastos', 'Tempo TV', 'Incumbente'])
    
    coefs = modelo.obter_coeficientes()
    r2 = modelo.obter_r2(X, y)
    
    print(f"✓ Modelo treinado com sucesso")
    print(f"✓ R² = {r2:.4f}")
    print("\nCoeficientes:")
    print(coefs.to_string())
    
    return modelo

def teste_random_forest(dados):
    """Testa modelo Random Forest"""
    print("\n" + "="*60)
    print("TESTE 3: Modelo Random Forest")
    print("="*60)
    
    X = dados[['gasto_campanha', 'tempo_tv_segundos', 'incumbente', 'idade']].values
    y = dados['votos'].values
    
    modelo = ModeloRandomForest(n_arvores=50, tipo='regressao')
    modelo.treinar(X, y, features_nomes=['Gastos', 'Tempo TV', 'Incumbente', 'Idade'])
    
    importancias = modelo.obter_importancia_features()
    score = modelo.obter_score(X, y)
    
    print(f"✓ Modelo treinado com sucesso")
    print(f"✓ Score = {score:.4f}")
    print("\nImportância das Features:")
    print(importancias.to_string())
    
    return modelo

def teste_bayesiano(dados):
    """Testa modelo Bayesiano Dirichlet"""
    print("\n" + "="*60)
    print("TESTE 4: Modelo Bayesiano (Dirichlet)")
    print("="*60)
    
    votos_por_partido = dados.groupby('partido')['votos'].sum()
    
    modelo = ModeloDirichlet()
    modelo.treinar(votos_por_partido)
    
    proporcoes = modelo.obter_proporcoes_esperadas()
    intervalos = modelo.obter_intervalos_credibilidade()
    
    print(f"✓ Modelo Bayesiano treinado")
    print(f"✓ Partidos analisados: {len(proporcoes)}")
    print("\nProporções Esperadas (Top 5):")
    print((proporcoes.head() * 100).round(2))
    
    return modelo

def teste_quociente_eleitoral(dados):
    """Testa cálculo do quociente eleitoral"""
    print("\n" + "="*60)
    print("TESTE 5: Quociente Eleitoral (Método D'Hondt)")
    print("="*60)
    
    votos = dados.groupby('partido')['votos'].sum()
    
    qe = QuocienteEleitoral()
    resultado = qe.calcular_distribuicao(votos, n_cadeiras=50)
    
    print(f"✓ Quociente Eleitoral: {qe.obter_quociente_eleitoral():,.0f}")
    print(f"✓ Cadeiras distribuídas: {resultado['cadeiras'].sum()}")
    print("\nDistribuição de Cadeiras (Top 10):")
    print(resultado.head(10)[['partido', 'votos', 'cadeiras', 'percentual_cadeiras']].to_string())
    
    return resultado

def teste_nep(dados):
    """Testa cálculo do Número Efetivo de Partidos"""
    print("\n" + "="*60)
    print("TESTE 6: Número Efetivo de Partidos (NEP)")
    print("="*60)
    
    votos_por_partido = dados.groupby('partido')['votos'].sum()
    cadeiras_por_partido = dados[dados['eleito'] == 1].groupby('partido').size()
    
    nep = NumeroEfetivoPartidos()
    indices = nep.calcular_indices_fragmentacao(votos_por_partido, cadeiras_por_partido)
    
    print(f"✓ NEP (Votos): {indices['NEP_votos']:.2f}")
    print(f"✓ NEP (Cadeiras): {indices['NEP_cadeiras']:.2f}")
    print(f"✓ Índice de Gallagher: {indices['Indice_Gallagher']:.3f}")
    print(f"✓ HHI (Votos): {indices['HHI_votos']:.0f}")
    
    return indices

def teste_volatilidade():
    """Testa cálculo de volatilidade eleitoral"""
    print("\n" + "="*60)
    print("TESTE 7: Volatilidade Eleitoral (Índice de Pedersen)")
    print("="*60)
    
    dados_2022 = gerar_dados_eleitorais(n_candidatos=100, ano=2022)
    dados_2026 = gerar_dados_eleitorais(n_candidatos=100, ano=2026)
    
    votos_2022 = dados_2022.groupby('partido')['votos'].sum()
    votos_2026 = dados_2026.groupby('partido')['votos'].sum()
    
    calc = CalculadorVolatilidade()
    vol = calc.calcular_pedersen(votos_2022, votos_2026)
    classificacao = calc.classificar_estabilidade(vol)
    
    print(f"✓ Índice de Pedersen: {vol:.2f}")
    print(f"✓ Classificação: {classificacao}")
    
    return vol

def teste_coligacoes(dados):
    """Testa análise de coligações"""
    print("\n" + "="*60)
    print("TESTE 8: Análise de Coligações")
    print("="*60)
    
    coligacoes = {
        'Esquerda': ['PT', 'PSB', 'PCdoB'],
        'Centro': ['MDB', 'PSDB'],
        'Direita': ['PL', 'PP', 'REPUBLICANOS']
    }
    
    votos_por_partido = dados.groupby('partido')['votos'].sum().to_dict()
    cadeiras_por_partido = dados[dados['eleito'] == 1].groupby('partido').size().to_dict()
    
    analisador = AnalisadorColigacoes()
    resultado = analisador.analisar_todas_coligacoes(votos_por_partido, cadeiras_por_partido, coligacoes)
    
    print(f"✓ Coligações analisadas: {len(resultado)}")
    print("\nEficiência das Coligações:")
    print(resultado[['nome_coligacao', 'votos', 'cadeiras', 'eficiencia']].to_string())
    
    return resultado

def teste_fragmentacao(dados):
    """Testa análise de fragmentação"""
    print("\n" + "="*60)
    print("TESTE 9: Análise de Fragmentação Partidária")
    print("="*60)
    
    votos_por_partido = dados.groupby('partido')['votos'].sum()
    cadeiras_por_partido = dados[dados['eleito'] == 1].groupby('partido').size()
    
    analisador = AnalisadorFragmentacao()
    analise = analisador.analisar_fragmentacao_completa(votos_por_partido, cadeiras_por_partido)
    
    print(f"✓ NEP (Votos): {analise['NEP_votos']:.2f}")
    print(f"✓ Partidos relevantes: {analise['n_partidos_relevantes']}")
    print(f"✓ Concentração Top 3: {analise['concentracao_top3']:.1f}%")
    print(f"✓ Classificação: {analisador.classificar_fragmentacao(analise['NEP_votos'])}")
    
    return analise

def teste_competitividade(dados):
    """Testa análise de competitividade"""
    print("\n" + "="*60)
    print("TESTE 10: Análise de Competitividade")
    print("="*60)
    
    estados = dados['estado'].unique()[:5]
    resultados_estados = {}
    
    for estado in estados:
        votos_estado = dados[dados['estado'] == estado].groupby('partido')['votos'].sum()
        resultados_estados[estado] = votos_estado.to_dict()
    
    indice = IndiceCompetitividade()
    distritos = indice.identificar_distritos_competitivos(resultados_estados, threshold_margem=15)
    
    print(f"✓ Distritos analisados: {len(distritos)}")
    print(f"✓ Distritos competitivos: {distritos['competitiva'].sum()}")
    print("\nCompetitividade por Distrito:")
    print(distritos[['distrito', 'margem_percentual', 'competitiva']].to_string())
    
    return distritos

def executar_todos_testes():
    """Executa todos os testes do sistema"""
    print("\n" + "="*60)
    print("SISTEMA DE ANÁLISE ELEITORAL - BATERIA DE TESTES")
    print("="*60)
    
    try:
        # Teste 1: Geração de dados
        dados = teste_geracao_dados()
        
        # Teste 2: Regressão Linear
        teste_regressao(dados)
        
        # Teste 3: Random Forest
        teste_random_forest(dados)
        
        # Teste 4: Bayesiano
        teste_bayesiano(dados)
        
        # Teste 5: Quociente Eleitoral
        teste_quociente_eleitoral(dados)
        
        # Teste 6: NEP
        teste_nep(dados)
        
        # Teste 7: Volatilidade
        teste_volatilidade()
        
        # Teste 8: Coligações
        teste_coligacoes(dados)
        
        # Teste 9: Fragmentação
        teste_fragmentacao(dados)
        
        # Teste 10: Competitividade
        teste_competitividade(dados)
        
        print("\n" + "="*60)
        print("✅ TODOS OS TESTES CONCLUÍDOS COM SUCESSO!")
        print("="*60)
        print("\nO sistema está funcionando corretamente.")
        print("Para executar o dashboard, use:")
        print("  streamlit run dashboard/app.py")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ ERRO durante os testes: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    executar_todos_testes()
