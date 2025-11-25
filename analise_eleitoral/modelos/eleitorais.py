"""
Modelos específicos do sistema eleitoral brasileiro.

Inclui:
- Quociente Eleitoral e distribuição de cadeiras
- Modelo de Markov para transição de votos
- Índice de Nacionalização Partidária
- Número Efetivo de Partidos
"""

import numpy as np
import pandas as pd
from collections import defaultdict


class QuocienteEleitoral:
    """
    Cálculo do Quociente Eleitoral e distribuição de cadeiras pelo método D'Hondt.
    
    Sistema proporcional brasileiro:
    
    1. Quociente Eleitoral (QE):
       QE = Votos Válidos / Cadeiras Disponíveis
    
    2. Quociente Partidário (QP):
       QP_partido = Votos do Partido / QE
       Cadeiras iniciais = floor(QP_partido)
    
    3. Distribuição de sobras (Método D'Hondt):
       Para cada cadeira restante, calcular:
       Média_partido = Votos do Partido / (Cadeiras já obtidas + 1)
       Atribuir à maior média
    
    Coligações:
    - Votos somados para cálculo do QP
    - Sobras distribuídas dentro da coligação
    """
    
    def __init__(self):
        self.qe = None
        self.resultados = None
        
    def calcular_distribuicao(self, votos_por_partido, n_cadeiras, coligacoes=None):
        """
        Calcula distribuição de cadeiras.
        
        Parâmetros:
        -----------
        votos_por_partido : dict ou Series
            Votos por partido
        n_cadeiras : int
            Número total de cadeiras
        coligacoes : dict, opcional
            {nome_coligacao: [lista_de_partidos]}
        
        Retorna:
        --------
        DataFrame
            Distribuição de cadeiras por partido
        """
        if isinstance(votos_por_partido, dict):
            votos_por_partido = pd.Series(votos_por_partido)
        
        votos_totais = votos_por_partido.sum()
        
        # Calcular Quociente Eleitoral
        self.qe = votos_totais / n_cadeiras
        
        # Se há coligações, agrupar votos
        if coligacoes:
            votos_coligacao = {}
            for nome_col, partidos in coligacoes.items():
                votos_coligacao[nome_col] = sum(votos_por_partido.get(p, 0) for p in partidos)
            
            # Calcular cadeiras por coligação
            cadeiras_coligacao = self._distribuir_dhondt(votos_coligacao, n_cadeiras)
            
            # Distribuir cadeiras dentro de cada coligação
            resultados = {}
            for nome_col, partidos in coligacoes.items():
                if cadeiras_coligacao[nome_col] > 0:
                    votos_col = {p: votos_por_partido.get(p, 0) for p in partidos}
                    cadeiras_col = self._distribuir_dhondt(votos_col, cadeiras_coligacao[nome_col])
                    resultados.update(cadeiras_col)
            
            # Partidos sem coligação
            partidos_coligados = set([p for partidos in coligacoes.values() for p in partidos])
            partidos_isolados = set(votos_por_partido.index) - partidos_coligados
            
            for partido in partidos_isolados:
                resultados[partido] = 0
        else:
            # Sem coligações, distribuir diretamente
            resultados = self._distribuir_dhondt(votos_por_partido.to_dict(), n_cadeiras)
        
        # Criar DataFrame de resultados
        self.resultados = pd.DataFrame([
            {
                'partido': partido,
                'votos': votos_por_partido.get(partido, 0),
                'percentual_votos': (votos_por_partido.get(partido, 0) / votos_totais) * 100,
                'cadeiras': cadeiras,
                'percentual_cadeiras': (cadeiras / n_cadeiras) * 100,
                'qp': votos_por_partido.get(partido, 0) / self.qe if self.qe > 0 else 0
            }
            for partido, cadeiras in resultados.items()
        ]).sort_values('cadeiras', ascending=False)
        
        return self.resultados
    
    def _distribuir_dhondt(self, votos_dict, n_cadeiras):
        """
        Método D'Hondt para distribuição proporcional.
        
        Parâmetros:
        -----------
        votos_dict : dict
            Votos por partido/coligação
        n_cadeiras : int
            Número de cadeiras a distribuir
        
        Retorna:
        --------
        dict
            Cadeiras por partido/coligação
        """
        cadeiras = {partido: 0 for partido in votos_dict.keys()}
        
        for _ in range(n_cadeiras):
            # Calcular média para cada partido
            medias = {}
            for partido, votos in votos_dict.items():
                if votos > 0:
                    medias[partido] = votos / (cadeiras[partido] + 1)
            
            if not medias:
                break
            
            # Atribuir cadeira ao partido com maior média
            partido_vencedor = max(medias, key=medias.get)
            cadeiras[partido_vencedor] += 1
        
        return cadeiras
    
    def obter_quociente_eleitoral(self):
        """Retorna o quociente eleitoral calculado."""
        return self.qe
    
    def calcular_eficiencia_votos(self):
        """
        Calcula eficiência de votos (votos necessários por cadeira).
        
        Retorna:
        --------
        Series
            Votos por cadeira para cada partido
        """
        if self.resultados is None:
            raise ValueError("Execute calcular_distribuicao primeiro")
        
        eficiencia = self.resultados.copy()
        eficiencia['votos_por_cadeira'] = eficiencia.apply(
            lambda row: row['votos'] / row['cadeiras'] if row['cadeiras'] > 0 else np.inf,
            axis=1
        )
        
        return eficiencia[['partido', 'votos_por_cadeira']].set_index('partido')['votos_por_cadeira']


class ModeloMarkov:
    """
    Modelo de Cadeia de Markov para transição de votos entre eleições.
    
    Matriz de Transição P:
    P_ij = P(votar em j na eleição t+1 | votou em i na eleição t)
    
    Propriedades:
    - Σⱼ P_ij = 1 (cada linha soma 1)
    - P_ij ≥ 0 (probabilidades não-negativas)
    
    Previsão:
    v_{t+1} = v_t · P
    
    Onde v_t é o vetor de proporções de votos no tempo t.
    
    Estado estacionário (equilíbrio):
    π = π · P
    """
    
    def __init__(self):
        self.matriz_transicao = None
        self.partidos = None
        
    def estimar_transicao(self, votos_t1, votos_t2):
        """
        Estima matriz de transição entre duas eleições.
        
        Parâmetros:
        -----------
        votos_t1 : dict ou Series
            Votos na eleição anterior
        votos_t2 : dict ou Series
            Votos na eleição atual
        
        Retorna:
        --------
        DataFrame
            Matriz de transição estimada
        """
        if isinstance(votos_t1, dict):
            votos_t1 = pd.Series(votos_t1)
        if isinstance(votos_t2, dict):
            votos_t2 = pd.Series(votos_t2)
        
        # Partidos presentes em ambas eleições
        self.partidos = sorted(set(votos_t1.index) | set(votos_t2.index))
        
        # Normalizar para proporções
        prop_t1 = votos_t1 / votos_t1.sum()
        prop_t2 = votos_t2 / votos_t2.sum()
        
        # Estimar matriz de transição (método simplificado)
        # Em prática, seria necessário dados de painel de eleitores
        n = len(self.partidos)
        matriz = np.zeros((n, n))
        
        for i, partido_origem in enumerate(self.partidos):
            prop_origem = prop_t1.get(partido_origem, 0)
            
            if prop_origem > 0:
                for j, partido_destino in enumerate(self.partidos):
                    prop_destino = prop_t2.get(partido_destino, 0)
                    
                    # Estimativa: maior probabilidade de manter no mesmo partido
                    if partido_origem == partido_destino:
                        matriz[i, j] = 0.7  # 70% mantém
                    else:
                        # Distribuir 30% proporcionalmente aos outros
                        matriz[i, j] = 0.3 * (prop_destino / (1 - prop_origem)) if prop_origem < 1 else 0
                
                # Normalizar linha
                soma_linha = matriz[i, :].sum()
                if soma_linha > 0:
                    matriz[i, :] /= soma_linha
        
        self.matriz_transicao = pd.DataFrame(
            matriz,
            index=self.partidos,
            columns=self.partidos
        )
        
        return self.matriz_transicao
    
    def prever_proxima_eleicao(self, votos_atual, n_periodos=1):
        """
        Prevê distribuição de votos em eleições futuras.
        
        Parâmetros:
        -----------
        votos_atual : dict ou Series
            Votos na eleição atual
        n_periodos : int
            Número de eleições à frente
        
        Retorna:
        --------
        Series
            Previsão de votos
        """
        if self.matriz_transicao is None:
            raise ValueError("Estime a matriz de transição primeiro")
        
        if isinstance(votos_atual, dict):
            votos_atual = pd.Series(votos_atual)
        
        # Normalizar para proporções
        prop_atual = votos_atual / votos_atual.sum()
        
        # Alinhar com partidos do modelo
        vetor = np.array([prop_atual.get(p, 0) for p in self.partidos])
        
        # Aplicar matriz de transição n vezes
        for _ in range(n_periodos):
            vetor = vetor @ self.matriz_transicao.values
        
        return pd.Series(vetor, index=self.partidos)
    
    def calcular_estado_estacionario(self, max_iter=1000, tol=1e-6):
        """
        Calcula distribuição estacionária (equilíbrio de longo prazo).
        
        Retorna:
        --------
        Series
            Distribuição estacionária
        """
        if self.matriz_transicao is None:
            raise ValueError("Estime a matriz de transição primeiro")
        
        # Começar com distribuição uniforme
        vetor = np.ones(len(self.partidos)) / len(self.partidos)
        
        for _ in range(max_iter):
            vetor_novo = vetor @ self.matriz_transicao.values
            
            # Verificar convergência
            if np.allclose(vetor, vetor_novo, atol=tol):
                break
            
            vetor = vetor_novo
        
        return pd.Series(vetor, index=self.partidos)


class IndiceNacionalizacao:
    """
    Índice de Nacionalização Partidária (Party Nationalization Score - PNS).
    
    Mede o grau de homogeneidade do desempenho de um partido entre regiões.
    
    PNS de Bochsler:
    PNS = 1 - √(Σᵢ (vᵢ - v̄)² · (eᵢ/E))
    
    Onde:
    - vᵢ: proporção de votos do partido na região i
    - v̄: proporção média nacional
    - eᵢ: eleitores na região i
    - E: total de eleitores
    
    Interpretação:
    - PNS próximo de 1: partido nacionalizado (desempenho uniforme)
    - PNS próximo de 0: partido regionalizado (concentrado em poucas regiões)
    """
    
    def calcular_pns(self, votos_por_regiao, eleitores_por_regiao):
        """
        Calcula PNS para um partido.
        
        Parâmetros:
        -----------
        votos_por_regiao : dict ou Series
            Votos do partido em cada região
        eleitores_por_regiao : dict ou Series
            Total de eleitores em cada região
        
        Retorna:
        --------
        float
            Índice PNS (0 a 1)
        """
        if isinstance(votos_por_regiao, dict):
            votos_por_regiao = pd.Series(votos_por_regiao)
        if isinstance(eleitores_por_regiao, dict):
            eleitores_por_regiao = pd.Series(eleitores_por_regiao)
        
        # Proporção de votos em cada região
        prop_regiao = votos_por_regiao / eleitores_por_regiao
        
        # Média nacional ponderada
        total_votos = votos_por_regiao.sum()
        total_eleitores = eleitores_por_regiao.sum()
        media_nacional = total_votos / total_eleitores
        
        # Calcular variância ponderada
        variancia_ponderada = 0
        for regiao in votos_por_regiao.index:
            diff_sq = (prop_regiao[regiao] - media_nacional) ** 2
            peso = eleitores_por_regiao[regiao] / total_eleitores
            variancia_ponderada += diff_sq * peso
        
        pns = 1 - np.sqrt(variancia_ponderada)
        
        return max(0, min(1, pns))  # Garantir intervalo [0, 1]
    
    def calcular_inp_sistema(self, dados_partidos):
        """
        Calcula Índice de Nacionalização do Sistema Partidário.
        
        INP = média ponderada dos PNS de todos os partidos
        
        Parâmetros:
        -----------
        dados_partidos : dict
            {partido: {'votos_por_regiao': ..., 'eleitores_por_regiao': ...}}
        
        Retorna:
        --------
        dict
            {'INP': valor, 'PNS_por_partido': {...}}
        """
        pns_partidos = {}
        votos_totais_partido = {}
        
        for partido, dados in dados_partidos.items():
            pns = self.calcular_pns(dados['votos_por_regiao'], dados['eleitores_por_regiao'])
            pns_partidos[partido] = pns
            votos_totais_partido[partido] = sum(dados['votos_por_regiao'].values())
        
        # INP: média ponderada pelos votos
        total_votos = sum(votos_totais_partido.values())
        inp = sum(pns * (votos / total_votos) 
                 for pns, votos in zip(pns_partidos.values(), votos_totais_partido.values()))
        
        return {
            'INP': inp,
            'PNS_por_partido': pns_partidos
        }


class NumeroEfetivoPartidos:
    """
    Número Efetivo de Partidos (Laakso-Taagepera).
    
    Mede fragmentação partidária considerando tamanho relativo dos partidos.
    
    NEP = 1 / Σᵢ pᵢ²
    
    Onde pᵢ é a proporção de votos (ou cadeiras) do partido i.
    
    Interpretação:
    - NEP = 2: sistema bipartidário perfeito
    - NEP = 3-5: multipartidarismo moderado
    - NEP > 5: alta fragmentação
    
    Variantes:
    - NEP_votos: baseado em votos
    - NEP_cadeiras: baseado em cadeiras (mede desproporcionalidade)
    """
    
    def calcular_nep(self, proporcoes):
        """
        Calcula Número Efetivo de Partidos.
        
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
        
        # Normalizar se necessário
        soma = proporcoes.sum()
        if not np.isclose(soma, 1.0):
            proporcoes = proporcoes / soma
        
        # NEP = 1 / Σ p²
        soma_quadrados = np.sum(proporcoes ** 2)
        
        return 1 / soma_quadrados if soma_quadrados > 0 else 0
    
    def calcular_indices_fragmentacao(self, votos_por_partido, cadeiras_por_partido):
        """
        Calcula múltiplos índices de fragmentação.
        
        Parâmetros:
        -----------
        votos_por_partido : dict ou Series
            Votos por partido
        cadeiras_por_partido : dict ou Series
            Cadeiras por partido
        
        Retorna:
        --------
        dict
            Índices de fragmentação
        """
        if isinstance(votos_por_partido, dict):
            votos_por_partido = pd.Series(votos_por_partido)
        if isinstance(cadeiras_por_partido, dict):
            cadeiras_por_partido = pd.Series(cadeiras_por_partido)
        
        # Proporções
        prop_votos = votos_por_partido / votos_por_partido.sum()
        prop_cadeiras = cadeiras_por_partido / cadeiras_por_partido.sum()
        
        # NEP
        nep_votos = self.calcular_nep(prop_votos)
        nep_cadeiras = self.calcular_nep(prop_cadeiras)
        
        # Índice de Herfindahl-Hirschman (concentração)
        hhi_votos = np.sum(prop_votos ** 2)
        hhi_cadeiras = np.sum(prop_cadeiras ** 2)
        
        # Índice de Gallagher (desproporcionalidade)
        gallagher = np.sqrt(0.5 * np.sum((prop_votos - prop_cadeiras) ** 2))
        
        return {
            'NEP_votos': nep_votos,
            'NEP_cadeiras': nep_cadeiras,
            'HHI_votos': hhi_votos,
            'HHI_cadeiras': hhi_cadeiras,
            'Indice_Gallagher': gallagher,
            'Desproporcionalidade': nep_votos - nep_cadeiras
        }
