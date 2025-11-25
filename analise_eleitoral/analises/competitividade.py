"""
Índices de competitividade eleitoral.
"""

import numpy as np
import pandas as pd


class IndiceCompetitividade:
    """
    Calcula índices de competitividade eleitoral.
    
    Métricas:
    - Margem de vitória
    - Índice de competitividade distrital
    - Taxa de renovação parlamentar
    - Análise de distritos competitivos
    """
    
    def calcular_margem_vitoria(self, votos_candidatos):
        """
        Calcula margem de vitória entre primeiro e segundo colocados.
        
        Margem = (V₁ - V₂) / V_total × 100
        
        Parâmetros:
        -----------
        votos_candidatos : dict ou Series
            Votos por candidato
        
        Retorna:
        --------
        dict
            Margem de vitória e informações
        """
        if isinstance(votos_candidatos, dict):
            votos_candidatos = pd.Series(votos_candidatos)
        
        votos_sorted = votos_candidatos.sort_values(ascending=False)
        
        if len(votos_sorted) < 2:
            return {
                'margem_absoluta': votos_sorted.iloc[0] if len(votos_sorted) > 0 else 0,
                'margem_percentual': 100.0,
                'primeiro_colocado': votos_sorted.index[0] if len(votos_sorted) > 0 else None,
                'segundo_colocado': None,
                'competitiva': False
            }
        
        primeiro = votos_sorted.iloc[0]
        segundo = votos_sorted.iloc[1]
        total = votos_sorted.sum()
        
        margem_abs = primeiro - segundo
        margem_pct = (margem_abs / total) * 100
        
        # Eleição competitiva se margem < 5%
        competitiva = margem_pct < 5
        
        return {
            'margem_absoluta': margem_abs,
            'margem_percentual': margem_pct,
            'primeiro_colocado': votos_sorted.index[0],
            'votos_primeiro': primeiro,
            'segundo_colocado': votos_sorted.index[1],
            'votos_segundo': segundo,
            'competitiva': competitiva,
            'classificacao': self._classificar_competitividade(margem_pct)
        }
    
    def _classificar_competitividade(self, margem_pct):
        """Classifica competitividade baseado na margem."""
        if margem_pct < 2:
            return 'Extremamente Competitiva'
        elif margem_pct < 5:
            return 'Muito Competitiva'
        elif margem_pct < 10:
            return 'Competitiva'
        elif margem_pct < 20:
            return 'Moderadamente Competitiva'
        else:
            return 'Pouco Competitiva'
    
    def calcular_indice_competitividade_distrital(self, votos_por_partido, cadeiras_disponiveis):
        """
        Calcula índice de competitividade do distrito.
        
        ICD = (N_partidos_competitivos / N_cadeiras) × Dispersão_votos
        
        Parâmetros:
        -----------
        votos_por_partido : dict ou Series
            Votos por partido
        cadeiras_disponiveis : int
            Número de cadeiras em disputa
        
        Retorna:
        --------
        dict
            Índice e métricas relacionadas
        """
        if isinstance(votos_por_partido, dict):
            votos_por_partido = pd.Series(votos_por_partido)
        
        prop_votos = (votos_por_partido / votos_por_partido.sum() * 100).sort_values(ascending=False)
        
        # Partidos competitivos (com chance real de ganhar cadeiras)
        # Aproximação: partidos com votos >= (100 / (2 * cadeiras))%
        threshold = 100 / (2 * cadeiras_disponiveis)
        partidos_competitivos = prop_votos[prop_votos >= threshold]
        
        # Dispersão (desvio padrão normalizado)
        dispersao = prop_votos.std() / prop_votos.mean() if prop_votos.mean() > 0 else 0
        
        # ICD
        icd = (len(partidos_competitivos) / cadeiras_disponiveis) * dispersao
        
        return {
            'ICD': icd,
            'n_partidos_competitivos': len(partidos_competitivos),
            'n_partidos_total': len(votos_por_partido),
            'dispersao_votos': dispersao,
            'threshold_competitividade': threshold,
            'partidos_competitivos': partidos_competitivos.index.tolist()
        }
    
    def calcular_taxa_renovacao(self, eleitos_anterior, eleitos_atual):
        """
        Calcula taxa de renovação parlamentar.
        
        Taxa = (Novos Eleitos / Total Eleitos) × 100
        
        Parâmetros:
        -----------
        eleitos_anterior : list ou set
            IDs dos eleitos na eleição anterior
        eleitos_atual : list ou set
            IDs dos eleitos na eleição atual
        
        Retorna:
        --------
        dict
            Taxa de renovação e detalhes
        """
        eleitos_anterior = set(eleitos_anterior)
        eleitos_atual = set(eleitos_atual)
        
        # Reeleitos
        reeleitos = eleitos_anterior & eleitos_atual
        
        # Novos
        novos = eleitos_atual - eleitos_anterior
        
        # Não reeleitos
        nao_reeleitos = eleitos_anterior - eleitos_atual
        
        # Taxa de renovação
        taxa_renovacao = (len(novos) / len(eleitos_atual)) * 100 if len(eleitos_atual) > 0 else 0
        
        # Taxa de reeleição
        taxa_reeleicao = (len(reeleitos) / len(eleitos_anterior)) * 100 if len(eleitos_anterior) > 0 else 0
        
        return {
            'taxa_renovacao': taxa_renovacao,
            'taxa_reeleicao': taxa_reeleicao,
            'n_novos': len(novos),
            'n_reeleitos': len(reeleitos),
            'n_nao_reeleitos': len(nao_reeleitos),
            'n_total_atual': len(eleitos_atual),
            'n_total_anterior': len(eleitos_anterior)
        }
    
    def identificar_distritos_competitivos(self, resultados_por_distrito, threshold_margem=10):
        """
        Identifica distritos (estados) mais competitivos.
        
        Parâmetros:
        -----------
        resultados_por_distrito : dict
            {distrito: {partido: votos}}
        threshold_margem : float
            Margem máxima (%) para considerar competitivo
        
        Retorna:
        --------
        DataFrame
            Distritos ordenados por competitividade
        """
        resultados = []
        
        for distrito, votos in resultados_por_distrito.items():
            margem_info = self.calcular_margem_vitoria(votos)
            
            resultados.append({
                'distrito': distrito,
                'margem_percentual': margem_info['margem_percentual'],
                'primeiro': margem_info['primeiro_colocado'],
                'segundo': margem_info['segundo_colocado'],
                'competitiva': margem_info['margem_percentual'] <= threshold_margem,
                'classificacao': margem_info['classificacao']
            })
        
        df = pd.DataFrame(resultados).sort_values('margem_percentual')
        
        return df
    
    def analisar_competitividade_temporal(self, serie_resultados_por_ano):
        """
        Analisa evolução da competitividade ao longo do tempo.
        
        Parâmetros:
        -----------
        serie_resultados_por_ano : dict
            {ano: {partido: votos}}
        
        Retorna:
        --------
        DataFrame
            Evolução da competitividade
        """
        anos = sorted(serie_resultados_por_ano.keys())
        
        resultados = []
        
        for ano in anos:
            votos = serie_resultados_por_ano[ano]
            margem_info = self.calcular_margem_vitoria(votos)
            
            resultados.append({
                'ano': ano,
                'margem_percentual': margem_info['margem_percentual'],
                'primeiro': margem_info['primeiro_colocado'],
                'segundo': margem_info['segundo_colocado'],
                'competitiva': margem_info['competitiva'],
                'classificacao': margem_info['classificacao']
            })
        
        df = pd.DataFrame(resultados)
        
        # Adicionar tendências
        if len(df) > 1:
            df['tendencia_margem'] = df['margem_percentual'].diff()
        
        return df
    
    def calcular_indice_balanceamento(self, votos_por_partido):
        """
        Calcula índice de balanceamento (quão equilibrada é a disputa).
        
        IB = 1 - (Σᵢ |pᵢ - p̄|) / (2 × (n-1)/n)
        
        Onde:
        - pᵢ: proporção do partido i
        - p̄: proporção média (1/n)
        - n: número de partidos
        
        IB = 1: perfeitamente balanceado
        IB = 0: totalmente desbalanceado
        
        Parâmetros:
        -----------
        votos_por_partido : dict ou Series
            Votos por partido
        
        Retorna:
        --------
        float
            Índice de balanceamento (0-1)
        """
        if isinstance(votos_por_partido, dict):
            votos_por_partido = pd.Series(votos_por_partido)
        
        prop = votos_por_partido / votos_por_partido.sum()
        n = len(prop)
        
        if n <= 1:
            return 1.0
        
        media = 1 / n
        soma_desvios = np.sum(np.abs(prop - media))
        
        # Normalizar
        max_desvio = 2 * (n - 1) / n
        
        ib = 1 - (soma_desvios / max_desvio)
        
        return max(0, min(1, ib))
    
    def analisar_voto_util(self, votos_candidatos, n_vagas):
        """
        Analisa potencial de voto útil (concentração em candidatos viáveis).
        
        Parâmetros:
        -----------
        votos_candidatos : dict ou Series
            Votos por candidato
        n_vagas : int
            Número de vagas disponíveis
        
        Retorna:
        --------
        dict
            Análise de voto útil
        """
        if isinstance(votos_candidatos, dict):
            votos_candidatos = pd.Series(votos_candidatos)
        
        votos_sorted = votos_candidatos.sort_values(ascending=False)
        total_votos = votos_sorted.sum()
        
        # Candidatos viáveis (top N + margem)
        n_viaveis = min(int(n_vagas * 1.5), len(votos_sorted))
        candidatos_viaveis = votos_sorted.head(n_viaveis)
        candidatos_inviaveis = votos_sorted.iloc[n_viaveis:]
        
        # Votos em candidatos viáveis vs inviáveis
        votos_viaveis = candidatos_viaveis.sum()
        votos_inviaveis = candidatos_inviaveis.sum()
        
        # Percentual de voto útil
        pct_voto_util = (votos_viaveis / total_votos) * 100
        
        # Potencial de transferência
        potencial_transferencia = votos_inviaveis
        
        return {
            'n_candidatos_viaveis': n_viaveis,
            'n_candidatos_inviaveis': len(candidatos_inviaveis),
            'votos_em_viaveis': votos_viaveis,
            'votos_em_inviaveis': votos_inviaveis,
            'percentual_voto_util': pct_voto_util,
            'potencial_transferencia': potencial_transferencia,
            'ultimo_viavel': candidatos_viaveis.index[-1],
            'votos_ultimo_viavel': candidatos_viaveis.iloc[-1]
        }
    
    def calcular_concentracao_competitiva(self, votos_por_partido, n_cadeiras):
        """
        Calcula concentração da competição (C4, C8, etc.).
        
        C_n = Σ(i=1 até n) pᵢ
        
        Parâmetros:
        -----------
        votos_por_partido : dict ou Series
            Votos por partido
        n_cadeiras : int
            Número de cadeiras
        
        Retorna:
        --------
        dict
            Índices de concentração
        """
        if isinstance(votos_por_partido, dict):
            votos_por_partido = pd.Series(votos_por_partido)
        
        prop = (votos_por_partido / votos_por_partido.sum() * 100).sort_values(ascending=False)
        
        # C4 (top 4 partidos)
        c4 = prop.head(4).sum()
        
        # C8 (top 8 partidos)
        c8 = prop.head(8).sum()
        
        # C_n (partidos necessários para 50% dos votos)
        cumsum = prop.cumsum()
        n_para_50 = len(cumsum[cumsum <= 50]) + 1
        
        # C_cadeiras (top N partidos onde N = cadeiras)
        c_cadeiras = prop.head(n_cadeiras).sum()
        
        return {
            'C4': c4,
            'C8': c8,
            'n_partidos_para_50pct': n_para_50,
            'C_cadeiras': c_cadeiras,
            'n_cadeiras': n_cadeiras
        }
