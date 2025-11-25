"""
Análise de fragmentação partidária.
"""

import numpy as np
import pandas as pd


class AnalisadorFragmentacao:
    """
    Análise de fragmentação do sistema partidário.
    
    Métricas principais:
    - Número Efetivo de Partidos (NEP)
    - Índice de Herfindahl-Hirschman (HHI)
    - Concentração de votos
    - Evolução temporal da fragmentação
    """
    
    def __init__(self):
        self.historico = []
        
    def calcular_nep(self, proporcoes):
        """
        Calcula Número Efetivo de Partidos (Laakso-Taagepera).
        
        NEP = 1 / Σᵢ pᵢ²
        
        Parâmetros:
        -----------
        proporcoes : dict, Series ou array
            Proporções de votos ou cadeiras
        
        Retorna:
        --------
        float
            NEP
        """
        if isinstance(proporcoes, dict):
            proporcoes = pd.Series(proporcoes)
        elif isinstance(proporcoes, pd.Series):
            pass
        else:
            proporcoes = np.array(proporcoes)
        
        # Normalizar
        soma = proporcoes.sum()
        if not np.isclose(soma, 1.0):
            proporcoes = proporcoes / soma
        
        # NEP
        soma_quadrados = np.sum(proporcoes ** 2)
        return 1 / soma_quadrados if soma_quadrados > 0 else 0
    
    def calcular_hhi(self, proporcoes):
        """
        Calcula Índice de Herfindahl-Hirschman.
        
        HHI = Σᵢ pᵢ² × 10000
        
        Interpretação:
        - HHI < 1500: baixa concentração
        - 1500 ≤ HHI < 2500: concentração moderada
        - HHI ≥ 2500: alta concentração
        
        Parâmetros:
        -----------
        proporcoes : dict, Series ou array
            Proporções de votos ou cadeiras
        
        Retorna:
        --------
        float
            HHI
        """
        if isinstance(proporcoes, dict):
            proporcoes = pd.Series(proporcoes)
        elif isinstance(proporcoes, pd.Series):
            pass
        else:
            proporcoes = np.array(proporcoes)
        
        # Normalizar
        soma = proporcoes.sum()
        if not np.isclose(soma, 1.0):
            proporcoes = proporcoes / soma
        
        # HHI
        return np.sum(proporcoes ** 2) * 10000
    
    def analisar_fragmentacao_completa(self, votos_por_partido, cadeiras_por_partido=None):
        """
        Análise completa de fragmentação.
        
        Parâmetros:
        -----------
        votos_por_partido : dict ou Series
            Votos por partido
        cadeiras_por_partido : dict ou Series, opcional
            Cadeiras por partido
        
        Retorna:
        --------
        dict
            Múltiplos índices de fragmentação
        """
        if isinstance(votos_por_partido, dict):
            votos_por_partido = pd.Series(votos_por_partido)
        
        # Proporções de votos
        prop_votos = votos_por_partido / votos_por_partido.sum()
        
        # Métricas baseadas em votos
        nep_votos = self.calcular_nep(prop_votos)
        hhi_votos = self.calcular_hhi(prop_votos)
        
        # Concentração (% dos top N partidos)
        prop_votos_sorted = prop_votos.sort_values(ascending=False)
        conc_top2 = prop_votos_sorted.head(2).sum() * 100
        conc_top3 = prop_votos_sorted.head(3).sum() * 100
        conc_top5 = prop_votos_sorted.head(5).sum() * 100
        
        resultados = {
            'NEP_votos': nep_votos,
            'HHI_votos': hhi_votos,
            'concentracao_top2': conc_top2,
            'concentracao_top3': conc_top3,
            'concentracao_top5': conc_top5,
            'n_partidos_total': len(votos_por_partido),
            'n_partidos_relevantes': len(prop_votos[prop_votos >= 0.01])  # >= 1%
        }
        
        # Se há dados de cadeiras
        if cadeiras_por_partido is not None:
            if isinstance(cadeiras_por_partido, dict):
                cadeiras_por_partido = pd.Series(cadeiras_por_partido)
            
            prop_cadeiras = cadeiras_por_partido / cadeiras_por_partido.sum()
            
            nep_cadeiras = self.calcular_nep(prop_cadeiras)
            hhi_cadeiras = self.calcular_hhi(prop_cadeiras)
            
            # Desproporcionalidade
            desprop = nep_votos - nep_cadeiras
            
            resultados.update({
                'NEP_cadeiras': nep_cadeiras,
                'HHI_cadeiras': hhi_cadeiras,
                'desproporcionalidade_NEP': desprop,
                'razao_NEP': nep_cadeiras / nep_votos if nep_votos > 0 else 0
            })
        
        return resultados
    
    def analisar_evolucao_temporal(self, serie_votos_por_ano, serie_cadeiras_por_ano=None):
        """
        Analisa evolução da fragmentação ao longo do tempo.
        
        Parâmetros:
        -----------
        serie_votos_por_ano : dict
            {ano: {partido: votos}}
        serie_cadeiras_por_ano : dict, opcional
            {ano: {partido: cadeiras}}
        
        Retorna:
        --------
        DataFrame
            Evolução dos índices ao longo do tempo
        """
        anos = sorted(serie_votos_por_ano.keys())
        
        resultados = []
        
        for ano in anos:
            votos = serie_votos_por_ano[ano]
            cadeiras = serie_cadeiras_por_ano.get(ano) if serie_cadeiras_por_ano else None
            
            analise = self.analisar_fragmentacao_completa(votos, cadeiras)
            analise['ano'] = ano
            
            resultados.append(analise)
        
        df = pd.DataFrame(resultados)
        
        # Adicionar tendências
        if len(df) > 1:
            df['tendencia_NEP_votos'] = df['NEP_votos'].diff()
            df['variacao_percentual_NEP'] = df['NEP_votos'].pct_change() * 100
        
        self.historico = df
        
        return df
    
    def classificar_fragmentacao(self, nep):
        """
        Classifica nível de fragmentação baseado no NEP.
        
        Parâmetros:
        -----------
        nep : float
            Número Efetivo de Partidos
        
        Retorna:
        --------
        str
            Classificação
        """
        if nep < 2:
            return 'Sistema Unipartidário'
        elif nep < 2.5:
            return 'Bipartidarismo Forte'
        elif nep < 3:
            return 'Bipartidarismo Moderado'
        elif nep < 4:
            return 'Multipartidarismo Limitado'
        elif nep < 5:
            return 'Multipartidarismo Moderado'
        elif nep < 6:
            return 'Multipartidarismo Forte'
        else:
            return 'Fragmentação Extrema'
    
    def analisar_distribuicao_tamanhos(self, votos_por_partido):
        """
        Analisa distribuição de tamanhos dos partidos.
        
        Parâmetros:
        -----------
        votos_por_partido : dict ou Series
            Votos por partido
        
        Retorna:
        --------
        dict
            Estatísticas da distribuição
        """
        if isinstance(votos_por_partido, dict):
            votos_por_partido = pd.Series(votos_por_partido)
        
        prop_votos = (votos_por_partido / votos_por_partido.sum() * 100).sort_values(ascending=False)
        
        # Categorizar partidos
        grandes = prop_votos[prop_votos >= 10]  # >= 10%
        medios = prop_votos[(prop_votos >= 5) & (prop_votos < 10)]  # 5-10%
        pequenos = prop_votos[(prop_votos >= 1) & (prop_votos < 5)]  # 1-5%
        nanicos = prop_votos[prop_votos < 1]  # < 1%
        
        return {
            'n_partidos_grandes': len(grandes),
            'n_partidos_medios': len(medios),
            'n_partidos_pequenos': len(pequenos),
            'n_partidos_nanicos': len(nanicos),
            'votos_grandes': grandes.sum(),
            'votos_medios': medios.sum(),
            'votos_pequenos': pequenos.sum(),
            'votos_nanicos': nanicos.sum(),
            'maior_partido': prop_votos.index[0],
            'percentual_maior': prop_votos.iloc[0],
            'segundo_maior': prop_votos.index[1] if len(prop_votos) > 1 else None,
            'percentual_segundo': prop_votos.iloc[1] if len(prop_votos) > 1 else 0,
            'diferenca_top2': prop_votos.iloc[0] - prop_votos.iloc[1] if len(prop_votos) > 1 else prop_votos.iloc[0]
        }
    
    def calcular_indice_fracionalizacao(self, proporcoes):
        """
        Calcula Índice de Fracionalização (Rae).
        
        F = 1 - Σᵢ pᵢ²
        
        Interpretação:
        - F = 0: um único partido
        - F → 1: fragmentação máxima
        
        Parâmetros:
        -----------
        proporcoes : dict, Series ou array
            Proporções
        
        Retorna:
        --------
        float
            Índice de fracionalização (0-1)
        """
        if isinstance(proporcoes, dict):
            proporcoes = pd.Series(proporcoes)
        elif isinstance(proporcoes, pd.Series):
            pass
        else:
            proporcoes = np.array(proporcoes)
        
        # Normalizar
        soma = proporcoes.sum()
        if not np.isclose(soma, 1.0):
            proporcoes = proporcoes / soma
        
        return 1 - np.sum(proporcoes ** 2)
    
    def comparar_fragmentacao_regioes(self, votos_por_regiao):
        """
        Compara fragmentação entre diferentes regiões.
        
        Parâmetros:
        -----------
        votos_por_regiao : dict
            {regiao: {partido: votos}}
        
        Retorna:
        --------
        DataFrame
            Fragmentação por região
        """
        resultados = []
        
        for regiao, votos in votos_por_regiao.items():
            analise = self.analisar_fragmentacao_completa(votos)
            analise['regiao'] = regiao
            resultados.append(analise)
        
        df = pd.DataFrame(resultados).sort_values('NEP_votos', ascending=False)
        
        return df
    
    def identificar_tendencia(self):
        """
        Identifica tendência de fragmentação baseado no histórico.
        
        Retorna:
        --------
        dict
            Análise de tendência
        """
        if len(self.historico) < 2:
            return {'tendencia': 'Dados insuficientes'}
        
        # Regressão linear simples
        x = np.arange(len(self.historico))
        y = self.historico['NEP_votos'].values
        
        # Coeficiente angular
        coef = np.polyfit(x, y, 1)[0]
        
        # Variação total
        variacao_total = y[-1] - y[0]
        variacao_percentual = (variacao_total / y[0]) * 100 if y[0] > 0 else 0
        
        if abs(coef) < 0.1:
            tendencia = 'Estável'
        elif coef > 0:
            tendencia = 'Crescente (Fragmentação Aumentando)'
        else:
            tendencia = 'Decrescente (Fragmentação Diminuindo)'
        
        return {
            'tendencia': tendencia,
            'coeficiente_angular': coef,
            'variacao_total': variacao_total,
            'variacao_percentual': variacao_percentual,
            'NEP_inicial': y[0],
            'NEP_final': y[-1]
        }
