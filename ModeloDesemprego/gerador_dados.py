"""
Gerador de Dados Sintéticos para Modelos de Desemprego

Gera conjuntos de dados sintéticos com características realistas incluindo
sazonalidade, tendências, e choques econômicos.

Autor: Luiz Tiago Wilcke
Data: 2025-11-24
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta


class GeradorDados:
    """
    Gerador de dados sintéticos para desemprego.
    
    Permite criar datasets realistas com múltiplas características:
    - Tendências de longo prazo
    - Sazonalidade
    - Choques econômicos (crises, booms)
    - Ruído estocástico
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Inicializa o gerador.
        
        Args:
            seed: Semente para reprodutibilidade
        """
        self.rng = np.random.RandomState(seed)
    
    def gerar_serie_temporal(
        self,
        N: int,
        taxa_base: float = 0.06,
        tendencia: float = 0.0,
        amplitude_sazonal: float = 0.01,
        volatilidade: float = 0.005,
        choques: Optional[List[Dict]] = None,
        data_inicial: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Gera uma série temporal sintética de desemprego.
        
        Args:
            N: Número de observações
            taxa_base: Taxa de desemprego base
            tendencia: Tendência linear (por período)
            amplitude_sazonal: Amplitude da componente sazonal
            volatilidade: Desvio padrão do ruído
            choques: Lista de choques econômicos
                    [{'tempo': t, 'magnitude': m, 'duracao': d}, ...]
            data_inicial: Data inicial da série (default: hoje)
            
        Returns:
            DataFrame com colunas: tempo, data, desemprego, componentes
        """
        # Gera tempos
        tempos = np.arange(N)
        
        # Componente de tendência
        comp_tendencia = tendencia * tempos
        
        # Componente sazonal (período anual = 12 meses)
        comp_sazonal = amplitude_sazonal * np.sin(2 * np.pi * tempos / 12)
        
        # Ruído estocástico
        ruido = self.rng.randn(N) * volatilidade
        
        # Componente de choques
        comp_choques = np.zeros(N)
        if choques is not None:
            for choque in choques:
                t_choque = choque['tempo']
                magnitude = choque['magnitude']
                duracao = choque.get('duracao', 6)  # Default: 6 períodos
                
                # Choque com decaimento exponencial
                for t in range(N):
                    if t >= t_choque:
                        delta_t = t - t_choque
                        comp_choques[t] += magnitude * np.exp(-delta_t / duracao)
        
        # Série completa
        desemprego = taxa_base + comp_tendencia + comp_sazonal + ruido + comp_choques
        
        # Garante que está no intervalo [0, 1]
        desemprego = np.clip(desemprego, 0.0, 1.0)
        
        # Cria datas
        if data_inicial is None:
            data_inicial = datetime.now()
        
        datas = [data_inicial + timedelta(days=30*i) for i in range(N)]
        
        # Cria DataFrame
        df = pd.DataFrame({
            'tempo': tempos,
            'data': datas,
            'desemprego': desemprego,
            'tendencia': comp_tendencia,
            'sazonal': comp_sazonal,
            'ruido': ruido,
            'choques': comp_choques
        })
        
        return df
    
    def gerar_crise_economica(
        self,
        tempo_inicio: int,
        magnitude: float = 0.05,
        duracao: int = 12,
        forma: str = 'exponencial'
    ) -> Dict:
        """
        Gera parâmetros para uma crise econômica.
        
        Args:
            tempo_inicio: Tempo de início da crise
            magnitude: Aumento no desemprego no pico
            duracao: Duração da crise em períodos
            forma: 'exponencial', 'triangular', ou 'retangular'
            
        Returns:
            Dicionário com parâmetros do choque
        """
        return {
            'tempo': tempo_inicio,
            'magnitude': magnitude,
            'duracao': duracao,
            'forma': forma
        }
    
    def adicionar_outliers(
        self,
        serie: np.ndarray,
        num_outliers: int,
        magnitude: float = 3.0
    ) -> np.ndarray:
        """
        Adiciona outliers aleatórios a uma série.
        
        Args:
            serie: Série temporal
            num_outliers: Número de outliers
            magnitude: Magnitude em termos de desvios padrão
            
        Returns:
            Série com outliers
        """
        serie_com_outliers = serie.copy()
        std = np.std(serie)
        
        indices = self.rng.choice(len(serie), num_outliers, replace=False)
        sinais = self.rng.choice([-1, 1], num_outliers)
        
        for idx, sinal in zip(indices, sinais):
            serie_com_outliers[idx] += sinal * magnitude * std
        
        return serie_com_outliers
    
    def gerar_dados_multimodelo(
        self,
        N_obs: int,
        incluir_modelos: List[str] = ['goodwin', 'phillips'],
        T_simulacao: float = 10.0
    ) -> Dict[str, pd.DataFrame]:
        """
        Gera dados sintéticos usando múltiplos modelos SDE.
        
        Args:
            N_obs: Número de observações
            incluir_modelos: Lista de modelos a usar
            T_simulacao: Tempo total de simulação
            
        Returns:
            Dicionário com DataFrames para cada modelo
        """
        from modelos_sde import criar_modelo
        from simulador import SimuladorSDE
        
        resultados = {}
        
        for nome_modelo in incluir_modelos:
            # Cria modelo
            modelo = criar_modelo(nome_modelo)
            
            # Simula
            sim = SimuladorSDE(modelo, seed=self.rng.randint(0, 2**31))
            tempos, trajetoria = sim.euler_maruyama(T_simulacao, N_obs - 1)
            
            # Calcula desemprego
            desemprego = np.array([
                modelo.calcular_desemprego(trajetoria[i]) 
                for i in range(len(tempos))
            ])
            
            # Cria DataFrame
            df = pd.DataFrame({
                'tempo': tempos,
                'desemprego': desemprego
            })
            
            # Adiciona colunas específicas do modelo
            if nome_modelo == 'goodwin':
                df['taxa_emprego'] = trajetoria[:, 0]
                df['parcela_salarial'] = trajetoria[:, 1]
            elif nome_modelo == 'phillips':
                df['inflacao'] = trajetoria[:, 0]
                df['desemprego_direto'] = trajetoria[:, 1]
            elif nome_modelo == 'populacional':
                df['forca_trabalho'] = trajetoria[:, 0]
                df['empregados'] = trajetoria[:, 1]
            elif nome_modelo == 'markov':
                df['prob_formal'] = trajetoria[:, 0]
                df['prob_informal'] = trajetoria[:, 1]
                df['prob_desemprego'] = trajetoria[:, 2]
            
            resultados[nome_modelo] = df
        
        return resultados
    
    def gerar_cenarios_comparativos(
        self,
        num_cenarios: int,
        N_obs: int,
        modelo_base: str = 'goodwin',
        variacao_parametros: float = 0.2
    ) -> List[pd.DataFrame]:
        """
        Gera múltiplos cenários variando parâmetros do modelo.
        
        Args:
            num_cenarios: Número de cenários
            N_obs: Observações por cenário
            modelo_base: Modelo a usar
            variacao_parametros: Fração de variação nos parâmetros
            
        Returns:
            Lista de DataFrames, um por cenário
        """
        from modelos_sde import criar_modelo
        from simulador import SimuladorSDE
        
        cenarios = []
        
        for i in range(num_cenarios):
            # Cria modelo com parâmetros variados
            modelo = criar_modelo(modelo_base)
            
            # Varia parâmetros aleatoriamente
            for key in modelo.parametros:
                if 'sigma' not in key and key not in ['u0', 'v0', 'pi0', 'L0', 'E0']:
                    variacao = 1 + self.rng.uniform(-variacao_parametros, variacao_parametros)
                    modelo.parametros[key] *= variacao
            
            # Simula
            sim = SimuladorSDE(modelo, seed=self.rng.randint(0, 2**31))
            tempos, trajetoria = sim.euler_maruyama(10.0, N_obs - 1)
            
            desemprego = np.array([
                modelo.calcular_desemprego(trajetoria[i]) 
                for i in range(len(tempos))
            ])
            
            df = pd.DataFrame({
                'tempo': tempos,
                'desemprego': desemprego,
                'cenario': i
            })
            
            cenarios.append(df)
        
        return cenarios
    
    def exportar_csv(
        self,
        df: pd.DataFrame,
        caminho: str,
        incluir_indice: bool = False
    ) -> None:
        """
        Exporta DataFrame para CSV.
        
        Args:
            df: DataFrame a exportar
            caminho: Caminho do arquivo
            incluir_indice: Se True, inclui índice
        """
        df.to_csv(caminho, index=incluir_indice, encoding='utf-8')
        print(f"Dados exportados para: {caminho}")
    
    def exportar_json(
        self,
        df: pd.DataFrame,
        caminho: str,
        orientacao: str = 'records'
    ) -> None:
        """
        Exporta DataFrame para JSON.
        
        Args:
            df: DataFrame a exportar
            caminho: Caminho do arquivo
            orientacao: Formato JSON ('records', 'index', 'columns')
        """
        df.to_json(caminho, orient=orientacao, date_format='iso', indent=2)
        print(f"Dados exportados para: {caminho}")
