"""
Análise de coligações e transferência de votos.
"""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


class AnalisadorColigacoes:
    """
    Análise de coligações partidárias e transferência de votos.
    
    Métricas:
    - Eficiência de coligações
    - Matriz de transferência de votos
    - Análise de sobras eleitorais
    - Impacto de coligações no resultado
    """
    
    def __init__(self):
        self.resultados = None
        
    def calcular_eficiencia_coligacao(self, votos_partidos, cadeiras_partidos, coligacao):
        """
        Calcula eficiência de uma coligação.
        
        Eficiência = Cadeiras obtidas / Cadeiras esperadas sem coligação
        
        Parâmetros:
        -----------
        votos_partidos : dict
            Votos por partido na coligação
        cadeiras_partidos : dict
            Cadeiras obtidas por partido
        coligacao : list
            Lista de partidos na coligação
        
        Retorna:
        --------
        dict
            Métricas de eficiência
        """
        # Cadeiras totais da coligação
        cadeiras_total = sum(cadeiras_partidos.get(p, 0) for p in coligacao)
        
        # Votos totais da coligação
        votos_total = sum(votos_partidos.get(p, 0) for p in coligacao)
        
        # Cadeiras esperadas (proporcional aos votos)
        votos_totais_eleicao = sum(votos_partidos.values())
        cadeiras_totais_eleicao = sum(cadeiras_partidos.values())
        
        cadeiras_esperadas = (votos_total / votos_totais_eleicao) * cadeiras_totais_eleicao
        
        # Eficiência
        eficiencia = cadeiras_total / cadeiras_esperadas if cadeiras_esperadas > 0 else 0
        
        # Ganho/perda em relação ao esperado
        ganho_cadeiras = cadeiras_total - cadeiras_esperadas
        
        return {
            'coligacao': coligacao,
            'votos_total': votos_total,
            'cadeiras_obtidas': cadeiras_total,
            'cadeiras_esperadas': cadeiras_esperadas,
            'eficiencia': eficiencia,
            'ganho_cadeiras': ganho_cadeiras,
            'percentual_ganho': (ganho_cadeiras / cadeiras_esperadas * 100) if cadeiras_esperadas > 0 else 0
        }
    
    def analisar_todas_coligacoes(self, votos_partidos, cadeiras_partidos, coligacoes):
        """
        Analisa eficiência de todas as coligações.
        
        Parâmetros:
        -----------
        votos_partidos : dict
            Votos por partido
        cadeiras_partidos : dict
            Cadeiras por partido
        coligacoes : dict
            {nome_coligacao: [lista_partidos]}
        
        Retorna:
        --------
        DataFrame
            Análise de todas as coligações
        """
        resultados = []
        
        for nome, partidos in coligacoes.items():
            eficiencia = self.calcular_eficiencia_coligacao(
                votos_partidos, cadeiras_partidos, partidos
            )
            eficiencia['nome_coligacao'] = nome
            resultados.append(eficiencia)
        
        self.resultados = pd.DataFrame(resultados).sort_values('eficiencia', ascending=False)
        
        return self.resultados
    
    def estimar_matriz_transferencia(self, votos_eleicao1, votos_eleicao2, 
                                     coligacoes_eleicao1=None, coligacoes_eleicao2=None):
        """
        Estima matriz de transferência de votos entre eleições.
        
        Usa método de fluxo ótimo para estimar como votos migraram.
        
        Parâmetros:
        -----------
        votos_eleicao1 : dict ou Series
            Votos na primeira eleição
        votos_eleicao2 : dict ou Series
            Votos na segunda eleição
        coligacoes_eleicao1 : dict, opcional
            Coligações na primeira eleição
        coligacoes_eleicao2 : dict, opcional
            Coligações na segunda eleição
        
        Retorna:
        --------
        DataFrame
            Matriz de transferência estimada
        """
        if isinstance(votos_eleicao1, dict):
            votos_eleicao1 = pd.Series(votos_eleicao1)
        if isinstance(votos_eleicao2, dict):
            votos_eleicao2 = pd.Series(votos_eleicao2)
        
        # Partidos em ambas eleições
        partidos_origem = votos_eleicao1.index.tolist()
        partidos_destino = votos_eleicao2.index.tolist()
        
        # Criar matriz de transferência
        matriz = pd.DataFrame(
            0.0,
            index=partidos_origem,
            columns=partidos_destino
        )
        
        # Método simplificado: distribuir proporcionalmente
        # Em prática, seria necessário dados de painel
        for partido_origem in partidos_origem:
            votos_origem = votos_eleicao1[partido_origem]
            
            # Se partido continua existindo, maior parte dos votos fica
            if partido_origem in partidos_destino:
                matriz.loc[partido_origem, partido_origem] = votos_origem * 0.7
                votos_restantes = votos_origem * 0.3
            else:
                votos_restantes = votos_origem
            
            # Distribuir votos restantes proporcionalmente
            if votos_restantes > 0:
                outros_partidos = [p for p in partidos_destino if p != partido_origem]
                if outros_partidos:
                    prop_outros = votos_eleicao2[outros_partidos] / votos_eleicao2[outros_partidos].sum()
                    for partido_destino in outros_partidos:
                        matriz.loc[partido_origem, partido_destino] += votos_restantes * prop_outros[partido_destino]
        
        return matriz
    
    def calcular_impacto_coligacao(self, votos_partidos, n_cadeiras, coligacoes):
        """
        Calcula impacto de coligações comparando com cenário sem coligações.
        
        Parâmetros:
        -----------
        votos_partidos : dict
            Votos por partido
        n_cadeiras : int
            Total de cadeiras
        coligacoes : dict
            Coligações existentes
        
        Retorna:
        --------
        DataFrame
            Comparação com e sem coligações
        """
        from modelos.eleitorais import QuocienteEleitoral
        
        qe = QuocienteEleitoral()
        
        # Cenário COM coligações
        resultado_com = qe.calcular_distribuicao(votos_partidos, n_cadeiras, coligacoes)
        
        # Cenário SEM coligações
        resultado_sem = qe.calcular_distribuicao(votos_partidos, n_cadeiras, None)
        
        # Comparar
        comparacao = pd.merge(
            resultado_com[['partido', 'cadeiras']],
            resultado_sem[['partido', 'cadeiras']],
            on='partido',
            suffixes=('_com_coligacao', '_sem_coligacao')
        )
        
        comparacao['diferenca'] = comparacao['cadeiras_com_coligacao'] - comparacao['cadeiras_sem_coligacao']
        comparacao['impacto_percentual'] = (comparacao['diferenca'] / comparacao['cadeiras_sem_coligacao'] * 100).fillna(0)
        
        return comparacao.sort_values('diferenca', ascending=False)
    
    def identificar_coligacoes_ideais(self, votos_partidos, ideologia_partidos, n_cadeiras, 
                                      max_distancia_ideologica=3):
        """
        Identifica coligações ideais baseado em proximidade ideológica e ganho eleitoral.
        
        Parâmetros:
        -----------
        votos_partidos : dict
            Votos por partido
        ideologia_partidos : dict
            Escala ideológica por partido (1-10)
        n_cadeiras : int
            Total de cadeiras
        max_distancia_ideologica : float
            Distância máxima permitida na escala ideológica
        
        Retorna:
        --------
        list
            Sugestões de coligações
        """
        from modelos.eleitorais import QuocienteEleitoral
        
        partidos = list(votos_partidos.keys())
        qe = QuocienteEleitoral()
        
        # Resultado sem coligações (baseline)
        resultado_base = qe.calcular_distribuicao(votos_partidos, n_cadeiras, None)
        cadeiras_base = resultado_base.set_index('partido')['cadeiras'].to_dict()
        
        sugestoes = []
        
        # Testar combinações de 2 partidos
        for i, p1 in enumerate(partidos):
            for p2 in partidos[i+1:]:
                # Verificar proximidade ideológica
                dist_ideologica = abs(ideologia_partidos.get(p1, 5) - ideologia_partidos.get(p2, 5))
                
                if dist_ideologica <= max_distancia_ideologica:
                    # Simular coligação
                    coligacao_teste = {f'{p1}+{p2}': [p1, p2]}
                    resultado_colig = qe.calcular_distribuicao(votos_partidos, n_cadeiras, coligacao_teste)
                    
                    cadeiras_p1 = resultado_colig[resultado_colig['partido'] == p1]['cadeiras'].values[0]
                    cadeiras_p2 = resultado_colig[resultado_colig['partido'] == p2]['cadeiras'].values[0]
                    cadeiras_total = cadeiras_p1 + cadeiras_p2
                    
                    # Ganho em relação ao cenário sem coligação
                    ganho = cadeiras_total - (cadeiras_base.get(p1, 0) + cadeiras_base.get(p2, 0))
                    
                    if ganho > 0:
                        sugestoes.append({
                            'partidos': [p1, p2],
                            'distancia_ideologica': dist_ideologica,
                            'cadeiras_sem_coligacao': cadeiras_base.get(p1, 0) + cadeiras_base.get(p2, 0),
                            'cadeiras_com_coligacao': cadeiras_total,
                            'ganho_cadeiras': ganho,
                            'votos_total': votos_partidos[p1] + votos_partidos[p2]
                        })
        
        # Ordenar por ganho
        sugestoes_df = pd.DataFrame(sugestoes).sort_values('ganho_cadeiras', ascending=False)
        
        return sugestoes_df
    
    def analisar_sobras_eleitorais(self, votos_partidos, n_cadeiras, coligacoes=None):
        """
        Analisa distribuição de sobras eleitorais.
        
        Parâmetros:
        -----------
        votos_partidos : dict
            Votos por partido
        n_cadeiras : int
            Total de cadeiras
        coligacoes : dict, opcional
            Coligações
        
        Retorna:
        --------
        DataFrame
            Análise de sobras
        """
        from modelos.eleitorais import QuocienteEleitoral
        
        qe = QuocienteEleitoral()
        resultado = qe.calcular_distribuicao(votos_partidos, n_cadeiras, coligacoes)
        
        # Calcular sobras
        resultado['cadeiras_qp'] = np.floor(resultado['qp'])  # Cadeiras pelo quociente
        resultado['sobras'] = resultado['qp'] - resultado['cadeiras_qp']
        resultado['cadeiras_sobras'] = resultado['cadeiras'] - resultado['cadeiras_qp']
        
        return resultado[['partido', 'votos', 'qp', 'cadeiras_qp', 'sobras', 
                         'cadeiras_sobras', 'cadeiras']].sort_values('sobras', ascending=False)
