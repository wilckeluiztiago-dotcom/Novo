"""
Cálculo de volatilidade eleitoral.
"""

import numpy as np
import pandas as pd


class CalculadorVolatilidade:
    """
    Calcula índices de volatilidade eleitoral.
    
    Volatilidade mede mudança no apoio aos partidos entre eleições.
    
    Índice de Pedersen:
    V = (1/2) Σᵢ |pᵢₜ - pᵢₜ₋₁|
    
    Onde pᵢₜ é a proporção de votos do partido i no tempo t.
    
    Interpretação:
    - V = 0: nenhuma mudança
    - V = 100: mudança completa (todos os partidos diferentes)
    - V < 10: baixa volatilidade (sistema estável)
    - 10 ≤ V < 20: volatilidade moderada
    - V ≥ 20: alta volatilidade (sistema instável)
    """
    
    def calcular_pedersen(self, votos_t1, votos_t2):
        """
        Calcula Índice de Pedersen entre duas eleições.
        
        Parâmetros:
        -----------
        votos_t1 : dict ou Series
            Votos na eleição anterior
        votos_t2 : dict ou Series
            Votos na eleição atual
        
        Retorna:
        --------
        float
            Índice de Pedersen (0-100)
        """
        if isinstance(votos_t1, dict):
            votos_t1 = pd.Series(votos_t1)
        if isinstance(votos_t2, dict):
            votos_t2 = pd.Series(votos_t2)
        
        # Normalizar para proporções
        prop_t1 = votos_t1 / votos_t1.sum()
        prop_t2 = votos_t2 / votos_t2.sum()
        
        # Todos os partidos (união)
        todos_partidos = set(prop_t1.index) | set(prop_t2.index)
        
        # Calcular diferenças absolutas
        soma_diferencas = 0
        for partido in todos_partidos:
            p1 = prop_t1.get(partido, 0)
            p2 = prop_t2.get(partido, 0)
            soma_diferencas += abs(p2 - p1)
        
        # Índice de Pedersen
        pedersen = (soma_diferencas / 2) * 100
        
        return pedersen
    
    def calcular_volatilidade_blocos(self, votos_t1, votos_t2, blocos_ideologicos):
        """
        Calcula volatilidade entre blocos ideológicos.
        
        Separa volatilidade intra-bloco (dentro do mesmo bloco) e
        inter-bloco (entre blocos diferentes).
        
        Parâmetros:
        -----------
        votos_t1 : dict ou Series
            Votos na eleição anterior
        votos_t2 : dict ou Series
            Votos na eleição atual
        blocos_ideologicos : dict
            {partido: bloco} (ex: 'esquerda', 'centro', 'direita')
        
        Retorna:
        --------
        dict
            Volatilidade total, intra-bloco e inter-bloco
        """
        if isinstance(votos_t1, dict):
            votos_t1 = pd.Series(votos_t1)
        if isinstance(votos_t2, dict):
            votos_t2 = pd.Series(votos_t2)
        
        # Normalizar
        prop_t1 = votos_t1 / votos_t1.sum()
        prop_t2 = votos_t2 / votos_t2.sum()
        
        # Agrupar por bloco
        blocos_t1 = {}
        blocos_t2 = {}
        
        for partido in set(prop_t1.index) | set(prop_t2.index):
            bloco = blocos_ideologicos.get(partido, 'outros')
            
            if bloco not in blocos_t1:
                blocos_t1[bloco] = 0
                blocos_t2[bloco] = 0
            
            blocos_t1[bloco] += prop_t1.get(partido, 0)
            blocos_t2[bloco] += prop_t2.get(partido, 0)
        
        # Volatilidade entre blocos
        vol_inter_bloco = 0
        for bloco in set(blocos_t1.keys()) | set(blocos_t2.keys()):
            vol_inter_bloco += abs(blocos_t2.get(bloco, 0) - blocos_t1.get(bloco, 0))
        
        vol_inter_bloco = (vol_inter_bloco / 2) * 100
        
        # Volatilidade total
        vol_total = self.calcular_pedersen(votos_t1, votos_t2)
        
        # Volatilidade intra-bloco
        vol_intra_bloco = vol_total - vol_inter_bloco
        
        return {
            'volatilidade_total': vol_total,
            'volatilidade_inter_bloco': vol_inter_bloco,
            'volatilidade_intra_bloco': vol_intra_bloco,
            'proporcao_inter_bloco': (vol_inter_bloco / vol_total * 100) if vol_total > 0 else 0
        }
    
    def calcular_volatilidade_regional(self, votos_por_regiao_t1, votos_por_regiao_t2):
        """
        Calcula volatilidade por região.
        
        Parâmetros:
        -----------
        votos_por_regiao_t1 : dict
            {regiao: {partido: votos}} na eleição anterior
        votos_por_regiao_t2 : dict
            {regiao: {partido: votos}} na eleição atual
        
        Retorna:
        --------
        DataFrame
            Volatilidade por região
        """
        resultados = []
        
        regioes = set(votos_por_regiao_t1.keys()) | set(votos_por_regiao_t2.keys())
        
        for regiao in regioes:
            votos_t1 = votos_por_regiao_t1.get(regiao, {})
            votos_t2 = votos_por_regiao_t2.get(regiao, {})
            
            if votos_t1 and votos_t2:
                vol = self.calcular_pedersen(votos_t1, votos_t2)
                
                resultados.append({
                    'regiao': regiao,
                    'volatilidade': vol,
                    'votos_t1': sum(votos_t1.values()),
                    'votos_t2': sum(votos_t2.values())
                })
        
        return pd.DataFrame(resultados).sort_values('volatilidade', ascending=False)
    
    def analisar_serie_temporal(self, serie_votos_por_ano):
        """
        Analisa volatilidade ao longo de múltiplas eleições.
        
        Parâmetros:
        -----------
        serie_votos_por_ano : dict
            {ano: {partido: votos}}
        
        Retorna:
        --------
        DataFrame
            Volatilidade entre cada par de eleições consecutivas
        """
        anos = sorted(serie_votos_por_ano.keys())
        
        resultados = []
        
        for i in range(len(anos) - 1):
            ano_anterior = anos[i]
            ano_atual = anos[i + 1]
            
            votos_t1 = serie_votos_por_ano[ano_anterior]
            votos_t2 = serie_votos_por_ano[ano_atual]
            
            vol = self.calcular_pedersen(votos_t1, votos_t2)
            
            resultados.append({
                'periodo': f'{ano_anterior}-{ano_atual}',
                'ano_inicial': ano_anterior,
                'ano_final': ano_atual,
                'volatilidade': vol
            })
        
        df = pd.DataFrame(resultados)
        
        # Adicionar estatísticas
        if len(df) > 0:
            df['media_movel_3'] = df['volatilidade'].rolling(window=3, min_periods=1).mean()
            df['tendencia'] = df['volatilidade'].diff()
        
        return df
    
    def calcular_volatilidade_ponderada(self, votos_t1, votos_t2, pesos=None):
        """
        Calcula volatilidade ponderada (dá mais peso a partidos maiores).
        
        Parâmetros:
        -----------
        votos_t1 : dict ou Series
            Votos na eleição anterior
        votos_t2 : dict ou Series
            Votos na eleição atual
        pesos : dict, opcional
            Pesos por partido (se None, usa média dos votos)
        
        Retorna:
        --------
        float
            Volatilidade ponderada
        """
        if isinstance(votos_t1, dict):
            votos_t1 = pd.Series(votos_t1)
        if isinstance(votos_t2, dict):
            votos_t2 = pd.Series(votos_t2)
        
        # Normalizar
        prop_t1 = votos_t1 / votos_t1.sum()
        prop_t2 = votos_t2 / votos_t2.sum()
        
        # Pesos (média das proporções se não especificado)
        if pesos is None:
            todos_partidos = set(prop_t1.index) | set(prop_t2.index)
            pesos = {}
            for partido in todos_partidos:
                p1 = prop_t1.get(partido, 0)
                p2 = prop_t2.get(partido, 0)
                pesos[partido] = (p1 + p2) / 2
        
        # Normalizar pesos
        soma_pesos = sum(pesos.values())
        pesos_norm = {p: w/soma_pesos for p, w in pesos.items()}
        
        # Volatilidade ponderada
        vol_ponderada = 0
        for partido, peso in pesos_norm.items():
            p1 = prop_t1.get(partido, 0)
            p2 = prop_t2.get(partido, 0)
            vol_ponderada += peso * abs(p2 - p1)
        
        return vol_ponderada * 100
    
    def identificar_partidos_volateis(self, votos_t1, votos_t2, threshold=5):
        """
        Identifica partidos com maior mudança de votos.
        
        Parâmetros:
        -----------
        votos_t1 : dict ou Series
            Votos na eleição anterior
        votos_t2 : dict ou Series
            Votos na eleição atual
        threshold : float
            Mudança mínima (em pontos percentuais) para considerar volátil
        
        Retorna:
        --------
        DataFrame
            Partidos com mudanças significativas
        """
        if isinstance(votos_t1, dict):
            votos_t1 = pd.Series(votos_t1)
        if isinstance(votos_t2, dict):
            votos_t2 = pd.Series(votos_t2)
        
        # Normalizar
        prop_t1 = votos_t1 / votos_t1.sum() * 100
        prop_t2 = votos_t2 / votos_t2.sum() * 100
        
        # Calcular mudanças
        todos_partidos = set(prop_t1.index) | set(prop_t2.index)
        
        mudancas = []
        for partido in todos_partidos:
            p1 = prop_t1.get(partido, 0)
            p2 = prop_t2.get(partido, 0)
            mudanca = p2 - p1
            mudanca_abs = abs(mudanca)
            
            if mudanca_abs >= threshold:
                mudancas.append({
                    'partido': partido,
                    'percentual_t1': p1,
                    'percentual_t2': p2,
                    'mudanca': mudanca,
                    'mudanca_abs': mudanca_abs,
                    'tipo': 'crescimento' if mudanca > 0 else 'declínio'
                })
        
        return pd.DataFrame(mudancas).sort_values('mudanca_abs', ascending=False)
    
    def classificar_estabilidade(self, volatilidade):
        """
        Classifica nível de estabilidade baseado na volatilidade.
        
        Parâmetros:
        -----------
        volatilidade : float
            Índice de volatilidade
        
        Retorna:
        --------
        str
            Classificação
        """
        if volatilidade < 5:
            return 'Muito Estável'
        elif volatilidade < 10:
            return 'Estável'
        elif volatilidade < 15:
            return 'Moderadamente Estável'
        elif volatilidade < 20:
            return 'Moderadamente Volátil'
        elif volatilidade < 30:
            return 'Volátil'
        else:
            return 'Muito Volátil'
