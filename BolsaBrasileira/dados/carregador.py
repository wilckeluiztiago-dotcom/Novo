"""
Carregador de Dados B3 para BolsaBrasileira

Responsável por baixar dados históricos da B3 via Yahoo Finance.
Adiciona automaticamente o sufixo .SA se necessário.

Autor: Luiz Tiago Wilcke
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

class CarregadorB3:
    def __init__(self, ticker="^BVSP", diretorio_cache="dados/cache"):
        # Adiciona .SA se não for índice e não tiver sufixo
        if not ticker.startswith("^") and not ticker.endswith(".SA"):
            self.ticker = f"{ticker}.SA"
        else:
            self.ticker = ticker
            
        self.diretorio_cache = diretorio_cache
        if not os.path.exists(diretorio_cache):
            os.makedirs(diretorio_cache)
            
    def obter_dados(self, data_inicio="2015-01-01", data_fim=None, usar_cache=True):
        """
        Baixa dados históricos ou carrega do cache.
        """
        if data_fim is None:
            data_fim = datetime.now().strftime("%Y-%m-%d")
            
        nome_arquivo = self.ticker.replace("^", "") # Remove caracteres inválidos para arquivo
        arquivo_cache = os.path.join(self.diretorio_cache, f"{nome_arquivo}_{data_inicio}_{data_fim}.csv")
        
        if usar_cache and os.path.exists(arquivo_cache):
            tempo_arquivo = datetime.fromtimestamp(os.path.getmtime(arquivo_cache))
            if datetime.now() - tempo_arquivo < timedelta(days=1):
                print(f"Carregando cache: {arquivo_cache}")
                df = pd.read_csv(arquivo_cache, index_col=0, parse_dates=True)
                return self._processar(df)
        
        print(f"Baixando dados B3: {self.ticker}...")
        df = yf.download(self.ticker, start=data_inicio, end=data_fim, progress=False)
        
        if df.empty:
            # Tenta sem .SA se falhar (ex: índices globais)
            if self.ticker.endswith(".SA"):
                ticker_alt = self.ticker.replace(".SA", "")
                print(f"Tentando {ticker_alt}...")
                df = yf.download(ticker_alt, start=data_inicio, end=data_fim, progress=False)
        
        if df.empty:
            raise ValueError(f"Dados não encontrados para {self.ticker}")
            
        df.to_csv(arquivo_cache)
        return self._processar(df)
    
    def _processar(self, df):
        """Processamento e cálculo de retornos."""
        if isinstance(df.columns, pd.MultiIndex):
            # Tenta encontrar a coluna do ticker, senão pega o primeiro nível
            try:
                df = df.xs(self.ticker, axis=1, level=1, drop_level=True)
            except KeyError:
                # Se falhar (ex: ticker alternativo), pega o primeiro nível 1 disponível
                nivel1 = df.columns.get_level_values(1).unique()
                if len(nivel1) > 0:
                    df = df.xs(nivel1[0], axis=1, level=1, drop_level=True)

        # Identifica coluna de preço
        cols = df.columns
        col_preco = 'Close'
        if 'Adj Close' in cols: col_preco = 'Adj Close'
        elif 'Close' in cols: col_preco = 'Close'
        else: col_preco = cols[0]
        
        df = df.copy()
        # Conversão numérica forçada (correção de bug yfinance)
        df['Preco'] = pd.to_numeric(df[col_preco], errors='coerce')
        df.dropna(subset=['Preco'], inplace=True)
        
        # Retornos
        df['RetornoLog'] = np.log(df['Preco'] / df['Preco'].shift(1))
        
        # Volatilidade Móvel (21 dias)
        df['Volatilidade'] = df['RetornoLog'].rolling(21).std() * np.sqrt(252)
        
        df.dropna(inplace=True)
        return df
