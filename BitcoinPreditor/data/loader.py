"""
Carregador de Dados para BitcoinPreditor

Responsável por baixar dados históricos do Yahoo Finance e processá-los.

Autor: Luiz Tiago Wilcke
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

class DataLoader:
    def __init__(self, ticker="BTC-USD", cache_dir="data/cache"):
        self.ticker = ticker
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
    def get_data(self, start_date="2018-01-01", end_date=None, use_cache=True):
        """
        Baixa dados históricos ou carrega do cache.
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
            
        cache_file = os.path.join(self.cache_dir, f"{self.ticker}_{start_date}_{end_date}.csv")
        
        if use_cache and os.path.exists(cache_file):
            # Verifica se o cache é recente (menos de 1 dia)
            file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if datetime.now() - file_time < timedelta(days=1):
                print(f"Carregando dados do cache: {cache_file}")
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                return self._process_data(df)
        
        print(f"Baixando dados de {self.ticker}...")
        df = yf.download(self.ticker, start=start_date, end=end_date, progress=False)
        
        if df.empty:
            raise ValueError(f"Nenhum dado encontrado para {self.ticker}")
            
        # Salva cache
        df.to_csv(cache_file)
        
        return self._process_data(df)
    
    def _process_data(self, df):
        """
        Calcula retornos logarítmicos e métricas básicas.
        """
        # Garante que temos apenas colunas de nível único se for MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(self.ticker, axis=1, level=1, drop_level=False)
            
        # Usa 'Close' ou 'Adj Close'
        if 'Adj Close' in df.columns:
            price_col = 'Adj Close'
        elif 'Close' in df.columns:
            price_col = 'Close'
        else:
            # Tenta pegar a primeira coluna se não achar nomes padrão
            price_col = df.columns[0]
            
        # Limpeza básica
        df = df.copy()
        
        # Garante que a coluna de preço é numérica
        # 'coerce' transforma strings não numéricas em NaN
        df['Price'] = pd.to_numeric(df[price_col], errors='coerce')
        
        # Remove linhas onde o preço é NaN (falha na conversão ou dado faltante)
        df.dropna(subset=['Price'], inplace=True)
        
        # Retornos Logarítmicos: ln(S_t / S_{t-1})
        df['LogReturn'] = np.log(df['Price'] / df['Price'].shift(1))
        
        # Volatilidade Histórica (Janela móvel 30 dias)
        df['HistVol'] = df['LogReturn'].rolling(window=30).std() * np.sqrt(365)
        
        df.dropna(inplace=True)
        return df

    def get_current_price(self):
        """Obtém o preço mais recente."""
        ticker = yf.Ticker(self.ticker)
        # Tenta pegar do fast info, fallback para history
        try:
            return ticker.fast_info['last_price']
        except:
            hist = ticker.history(period="1d")
            if not hist.empty:
                return hist['Close'].iloc[-1]
            return 0.0
