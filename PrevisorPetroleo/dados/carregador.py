"""
Carregador de Dados para PrevisorPetroleo

Responsável por baixar dados históricos do Petróleo (Brent/WTI) e processá-los.

Autor: Luiz Tiago Wilcke
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

class CarregadorDados:
    def __init__(self, ticker="BZ=F", diretorio_cache="dados/cache"):
        self.ticker = ticker
        self.diretorio_cache = diretorio_cache
        if not os.path.exists(diretorio_cache):
            os.makedirs(diretorio_cache)
            
    def obter_dados(self, data_inicio="2010-01-01", data_fim=None, usar_cache=True):
        """
        Baixa dados históricos ou carrega do cache.
        """
        if data_fim is None:
            data_fim = datetime.now().strftime("%Y-%m-%d")
            
        arquivo_cache = os.path.join(self.diretorio_cache, f"{self.ticker}_{data_inicio}_{data_fim}.csv")
        
        if usar_cache and os.path.exists(arquivo_cache):
            # Verifica se o cache é recente (menos de 1 dia)
            tempo_arquivo = datetime.fromtimestamp(os.path.getmtime(arquivo_cache))
            if datetime.now() - tempo_arquivo < timedelta(days=1):
                print(f"Carregando dados do cache: {arquivo_cache}")
                df = pd.read_csv(arquivo_cache, index_col=0, parse_dates=True)
                return self._processar_dados(df)
        
        print(f"Baixando dados de {self.ticker}...")
        df = yf.download(self.ticker, start=data_inicio, end=data_fim, progress=False)
        
        if df.empty:
            raise ValueError(f"Nenhum dado encontrado para {self.ticker}")
            
        # Salva cache
        df.to_csv(arquivo_cache)
        
        return self._processar_dados(df)
    
    def _processar_dados(self, df):
        """
        Calcula retornos logarítmicos e métricas básicas.
        """
        # Garante que temos apenas colunas de nível único se for MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(self.ticker, axis=1, level=1, drop_level=False)
            
        # Usa 'Close' ou 'Adj Close'
        if 'Adj Close' in df.columns:
            coluna_preco = 'Adj Close'
        elif 'Close' in df.columns:
            coluna_preco = 'Close'
        else:
            coluna_preco = df.columns[0]
            
        # Limpeza básica
        df = df.copy()
        
        # Garante numérico
        df['Preco'] = pd.to_numeric(df[coluna_preco], errors='coerce')
        df.dropna(subset=['Preco'], inplace=True)
        
        # Retornos Logarítmicos: ln(S_t / S_{t-1})
        df['RetornoLog'] = np.log(df['Preco'] / df['Preco'].shift(1))
        
        # Volatilidade Histórica (Janela móvel 21 dias - ~1 mês útil)
        df['VolHist'] = df['RetornoLog'].rolling(window=21).std() * np.sqrt(252)
        
        df.dropna(inplace=True)
        return df

    def obter_preco_atual(self):
        """Obtém o preço mais recente."""
        ticker = yf.Ticker(self.ticker)
        try:
            return ticker.fast_info['last_price']
        except:
            hist = ticker.history(period="1d")
            if not hist.empty:
                return hist['Close'].iloc[-1]
            return 0.0
