"""
Módulo de Análise Estatística para Modelos de Desemprego

Implementa testes estatísticos e análises avançadas para séries temporais
de desemprego.

Autor: Luiz Tiago Wilcke
Data: 2025-11-24
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from scipy import stats
from scipy.stats import norm, jarque_bera, skew, kurtosis


class AnalisadorEstatistico:
    """
    Análises estatísticas para dados de desemprego.
    
    Inclui testes de hipótese, análise de momentos, e medidas de risco.
    """
    
    def __init__(self):
        """Inicializa o analisador."""
        pass
    
    def calcular_momentos(self, dados: np.ndarray) -> Dict[str, float]:
        """
        Calcula momentos estatísticos da distribuição.
        
        Args:
            dados: Array com dados
            
        Returns:
            Dicionário com momentos
        """
        return {
            'media': np.mean(dados),
            'mediana': np.median(dados),
            'variancia': np.var(dados),
            'desvio_padrao': np.std(dados),
            'assimetria': skew(dados),
            'curtose': kurtosis(dados),
            'minimo': np.min(dados),
            'maximo': np.max(dados),
            'amplitude': np.ptp(dados),
            'q25': np.percentile(dados, 25),
            'q75': np.percentile(dados, 75),
            'iqr': np.percentile(dados, 75) - np.percentile(dados, 25)
        }
    
    def teste_normalidade(
        self, 
        dados: np.ndarray,
        alpha: float = 0.05
    ) -> Dict[str, any]:
        """
        Testa normalidade dos dados usando Jarque-Bera.
        
        Args:
            dados: Array com dados
            alpha: Nível de significância
            
        Returns:
            Dicionário com resultados do teste
        """
        estatistica, p_valor = jarque_bera(dados)
        
        return {
            'estatistica_JB': estatistica,
            'p_valor': p_valor,
            'rejeita_normalidade': p_valor < alpha,
            'alpha': alpha,
            'interpretacao': 'Não-normal' if p_valor < alpha else 'Normal'
        }
    
    def teste_estacionariedade_adf(
        self,
        dados: np.ndarray,
        alpha: float = 0.05
    ) -> Dict[str, any]:
        """
        Teste de Dickey-Fuller Aumentado para estacionariedade.
        
        Args:
            dados: Série temporal
            alpha: Nível de significância
            
        Returns:
            Dicionário com resultados
        """
        try:
            from statsmodels.tsa.stattools import adfuller
            
            resultado = adfuller(dados, autolag='AIC')
            
            return {
                'estatistica_ADF': resultado[0],
                'p_valor': resultado[1],
                'lags_usados': resultado[2],
                'n_obs': resultado[3],
                'valores_criticos': resultado[4],
                'estacionaria': resultado[1] < alpha,
                'alpha': alpha,
                'interpretacao': 'Estacionária' if resultado[1] < alpha else 'Não-estacionária'
            }
        except ImportError:
            return {
                'erro': 'statsmodels não disponível',
                'mensagem': 'Instale com: pip install statsmodels'
            }
    
    def teste_estacionariedade_kpss(
        self,
        dados: np.ndarray,
        alpha: float = 0.05
    ) -> Dict[str, any]:
        """
        Teste KPSS para estacionariedade.
        
        Hipótese nula: a série é estacionária (oposto do ADF).
        
        Args:
            dados: Série temporal
            alpha: Nível de significância
            
        Returns:
            Dicionário com resultados
        """
        try:
            from statsmodels.tsa.stattools import kpss
            
            estatistica, p_valor, lags, valores_criticos = kpss(dados, regression='c', nlags='auto')
            
            return {
                'estatistica_KPSS': estatistica,
                'p_valor': p_valor,
                'lags_usados': lags,
                'valores_criticos': valores_criticos,
                'estacionaria': p_valor > alpha,
                'alpha': alpha,
                'interpretacao': 'Estacionária' if p_valor > alpha else 'Não-estacionária'
            }
        except ImportError:
            return {
                'erro': 'statsmodels não disponível',
                'mensagem': 'Instale com: pip install statsmodels'
            }
    
    def calcular_autocorrelacao(
        self,
        dados: np.ndarray,
        max_lag: int = 40
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula função de autocorrelação (ACF).
        
        Args:
            dados: Série temporal
            max_lag: Máximo de lags a calcular
            
        Returns:
            (lags, acf_values): Arrays com lags e valores ACF
        """
        try:
            from statsmodels.tsa.stattools import acf
            
            acf_values = acf(dados, nlags=max_lag, fft=True)
            lags = np.arange(len(acf_values))
            
            return lags, acf_values
        except ImportError:
            # Implementação manual simples
            n = len(dados)
            media = np.mean(dados)
            var = np.var(dados)
            
            acf_values = np.zeros(max_lag + 1)
            acf_values[0] = 1.0
            
            for k in range(1, max_lag + 1):
                if k < n:
                    cov = np.mean((dados[:-k] - media) * (dados[k:] - media))
                    acf_values[k] = cov / var
            
            return np.arange(max_lag + 1), acf_values
    
    def calcular_autocorrelacao_parcial(
        self,
        dados: np.ndarray,
        max_lag: int = 40
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula função de autocorrelação parcial (PACF).
        
        Args:
            dados: Série temporal
            max_lag: Máximo de lags
            
        Returns:
            (lags, pacf_values)
        """
        try:
            from statsmodels.tsa.stattools import pacf
            
            pacf_values = pacf(dados, nlags=max_lag)
            lags = np.arange(len(pacf_values))
            
            return lags, pacf_values
        except ImportError:
            # Retorna apenas ACF se statsmodels não disponível
            print("PACF requer statsmodels. Retornando ACF.")
            return self.calcular_autocorrelacao(dados, max_lag)
    
    def calcular_var(
        self,
        dados: np.ndarray,
        nivel_confianca: float = 0.95,
        metodo: str = 'historico'
    ) -> float:
        """
        Calcula Value at Risk (VaR) para desemprego.
        
        VaR indica o nível de desemprego que não será excedido com
        uma dada probabilidade.
        
        Args:
            dados: Array com taxas de desemprego
            nivel_confianca: Nível de confiança (ex: 0.95 = 95%)
            metodo: 'historico' ou 'parametrico'
            
        Returns:
            Valor do VaR
        """
        if metodo == 'historico':
            # VaR histórico (quantil)
            alpha = 1 - nivel_confianca
            var = np.percentile(dados, 100 * (1 - alpha))
        elif metodo == 'parametrico':
            # VaR paramétrico (assume normalidade)
            media = np.mean(dados)
            desvio = np.std(dados)
            alpha = 1 - nivel_confianca
            z = norm.ppf(1 - alpha)
            var = media + z * desvio
        else:
            raise ValueError(f"Método '{metodo}' não reconhecido")
        
        return var
    
    def calcular_cvar(
        self,
        dados: np.ndarray,
        nivel_confianca: float = 0.95
    ) -> float:
        """
        Calcula Conditional Value at Risk (CVaR ou Expected Shortfall).
        
        CVaR é a média dos valores que excedem o VaR.
        
        Args:
            dados: Array com taxas
            nivel_confianca: Nível de confiança
            
        Returns:
            Valor do CVaR
        """
        var = self.calcular_var(dados, nivel_confianca, metodo='historico')
        
        # CVaR = média dos valores acima do VaR
        valores_extremos = dados[dados >= var]
        
        if len(valores_extremos) > 0:
            cvar = np.mean(valores_extremos)
        else:
            cvar = var
        
        return cvar
    
    def analise_completa(
        self,
        dados: np.ndarray,
        nome_serie: str = "Desemprego"
    ) -> pd.DataFrame:
        """
        Realiza análise estatística completa.
        
        Args:
            dados: Array com dados
            nome_serie: Nome da série para o relatório
            
        Returns:
            DataFrame com todos os resultados
        """
        print(f"\n{'='*60}")
        print(f"ANÁLISE ESTATÍSTICA COMPLETA: {nome_serie}")
        print(f"{'='*60}\n")
        
        resultados = []
        
        # Momentos
        print("1. MOMENTOS ESTATÍSTICOS")
        print("-" * 40)
        momentos = self.calcular_momentos(dados)
        for chave, valor in momentos.items():
            print(f"  {chave.capitalize():.<30} {valor:.6f}")
            resultados.append({'Métrica': chave, 'Valor': valor})
        
        # Teste de normalidade
        print("\n2. TESTE DE NORMALIDADE (Jarque-Bera)")
        print("-" * 40)
        teste_norm = self.teste_normalidade(dados)
        print(f"  Estatística JB: {teste_norm['estatistica_JB']:.4f}")
        print(f"  P-valor: {teste_norm['p_valor']:.4f}")
        print(f"  Interpretação: {teste_norm['interpretacao']}")
        
        # Estacionariedade
        print("\n3. TESTES DE ESTACIONARIEDADE")
        print("-" * 40)
        
        adf = self.teste_estacionariedade_adf(dados)
        if 'erro' not in adf:
            print(f"  ADF:")
            print(f"    Estatística: {adf['estatistica_ADF']:.4f}")
            print(f"    P-valor: {adf['p_valor']:.4f}")
            print(f"    Interpretação: {adf['interpretacao']}")
        else:
            print(f"  ADF: {adf['mensagem']}")
        
        kpss_res = self.teste_estacionariedade_kpss(dados)
        if 'erro' not in kpss_res:
            print(f"  KPSS:")
            print(f"    Estatística: {kpss_res['estatistica_KPSS']:.4f}")
            print(f"    P-valor: {kpss_res['p_valor']:.4f}")
            print(f"    Interpretação: {kpss_res['interpretacao']}")
        else:
            print(f"  KPSS: {kpss_res['mensagem']}")
        
        # Medidas de risco
        print("\n4. MEDIDAS DE RISCO")
        print("-" * 40)
        var_95 = self.calcular_var(dados, 0.95)
        var_99 = self.calcular_var(dados, 0.99)
        cvar_95 = self.calcular_cvar(dados, 0.95)
        
        print(f"  VaR (95%): {var_95:.6f}")
        print(f"  VaR (99%): {var_99:.6f}")
        print(f"  CVaR (95%): {cvar_95:.6f}")
        
        resultados.append({'Métrica': 'VaR_95', 'Valor': var_95})
        resultados.append({'Métrica': 'VaR_99', 'Valor': var_99})
        resultados.append({'Métrica': 'CVaR_95', 'Valor': cvar_95})
        
        print(f"\n{'='*60}\n")
        
        return pd.DataFrame(resultados)
    
    def plotar_acf_pacf(
        self,
        dados: np.ndarray,
        max_lag: int = 40,
        titulo: str = "Análise de Autocorrelação"
    ):
        """
        Plota ACF e PACF.
        
        Args:
            dados: Série temporal
            max_lag: Máximo de lags
            titulo: Título do gráfico
        """
        import matplotlib.pyplot as plt
        
        # Calcula ACF e PACF
        lags_acf, acf_vals = self.calcular_autocorrelacao(dados, max_lag)
        lags_pacf, pacf_vals = self.calcular_autocorrelacao_parcial(dados, max_lag)
        
        # Intervalo de confiança (95%)
        n = len(dados)
        conf_interval = 1.96 / np.sqrt(n)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # ACF
        ax1.stem(lags_acf, acf_vals, basefmt=' ')
        ax1.axhline(y=0, color='black', linewidth=0.8)
        ax1.axhline(y=conf_interval, color='red', linestyle='--', linewidth=1, 
                   label=f'IC 95%: ±{conf_interval:.3f}')
        ax1.axhline(y=-conf_interval, color='red', linestyle='--', linewidth=1)
        ax1.set_xlabel('Lag', fontweight='bold')
        ax1.set_ylabel('ACF', fontweight='bold')
        ax1.set_title('Função de Autocorrelação (ACF)', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # PACF
        ax2.stem(lags_pacf, pacf_vals, basefmt=' ')
        ax2.axhline(y=0, color='black', linewidth=0.8)
        ax2.axhline(y=conf_interval, color='red', linestyle='--', linewidth=1,
                   label=f'IC 95%: ±{conf_interval:.3f}')
        ax2.axhline(y=-conf_interval, color='red', linestyle='--', linewidth=1)
        ax2.set_xlabel('Lag', fontweight='bold')
        ax2.set_ylabel('PACF', fontweight='bold')
        ax2.set_title('Função de Autocorrelação Parcial (PACF)', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle(titulo, fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.show()
        
        return fig
