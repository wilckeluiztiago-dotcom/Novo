import unittest
import pandas as pd
import numpy as np
import sys
import os

# Adiciona diretório atual ao path
sys.path.append(os.getcwd())

from gerador_dados import GeradorDados
from modelos.series_temporais import ModeloSARIMA, ModeloGARCH
from modelos.multivariada import AnaliseMultivariada
from modelos.stiglitz import ModeloShapiroStiglitz

class TesteSistemaDesemprego(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        print("Gerando dados para testes...")
        cls.gerador = GeradorDados(seed=42)
        cls.df = cls.gerador.gerar_dados_brasil_simulados(anos=5)
        
    def test_geracao_dados(self):
        """Testa se os dados gerados têm o formato correto."""
        self.assertIsInstance(self.df, pd.DataFrame)
        self.assertEqual(len(self.df), 5 * 12)
        colunas_esperadas = ['data', 'desemprego', 'inflacao', 'selic', 'pib_crescimento', 'rendimento_medio']
        for col in colunas_esperadas:
            self.assertIn(col, self.df.columns)
            
    def test_sarima(self):
        """Testa ajuste e previsão do SARIMA."""
        serie = self.df.set_index('data')['desemprego']
        modelo = ModeloSARIMA(serie, ordem=(1,0,0), ordem_sazonal=(0,0,0,12)) # Modelo simples para teste rápido
        resumo = modelo.ajustar()
        self.assertIsNotNone(resumo)
        
        previsao = modelo.prever(passos=3)
        self.assertEqual(len(previsao), 3)
        self.assertIn('previsao', previsao.columns)
        
    def test_multivariada(self):
        """Testa VAR e causalidade."""
        df_model = self.df[['desemprego', 'inflacao', 'selic']]
        analise = AnaliseMultivariada(df_model)
        
        # Teste VAR
        resumo = analise.ajustar_var(lags=2)
        self.assertIsNotNone(resumo)
        
        # Teste Granger
        p_valores = analise.teste_causalidade_granger('selic', 'desemprego', max_lags=2)
        self.assertIsInstance(p_valores, dict)
        
    def test_stiglitz(self):
        """Testa modelo de Shapiro-Stiglitz."""
        modelo = ModeloShapiroStiglitz()
        eq = modelo.calcular_equilibrio()
        
        self.assertIn('desemprego_equilibrio', eq)
        self.assertIn('salario_equilibrio', eq)
        self.assertTrue(0 < eq['desemprego_equilibrio'] < 1)
        self.assertTrue(eq['salario_equilibrio'] > 0)

if __name__ == '__main__':
    unittest.main()
