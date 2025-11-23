import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Configuração para evitar warnings
import warnings
warnings.filterwarnings('ignore')

# Coletar dados da Bovespa - VERSÃO CORRIGIDA
def carregar_dados_bovespa():
    tickers = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'WEGE3.SA']
    
    # Download corrigido
    dados = yf.download(tickers, start='2020-01-01', end='2024-01-01')
    
    # Acessar preços de fechamento ajustados corretamente
    if 'Adj Close' in dados:
        precos = dados['Adj Close']
    elif 'Close' in dados:
        precos = dados['Close']
    else:
        # Se não encontrar, usar o primeiro nível
        precos = dados.iloc[:, :len(tickers)]  # Primeiras colunas
    
    returns = precos.pct_change().dropna()
    return returns, tickers

# Calcular parâmetros
def calcular_parametros(returns):
    mu = returns.mean() * 252  # Retorno anualizado
    sigma = returns.std() * np.sqrt(252)  # Vol anualizada
    corr_matrix = returns.corr()
    cov_matrix = returns.cov() * 252
    
    return mu.values, sigma.values, corr_matrix.values, cov_matrix.values

# Função HJB para otimização - VERSÃO SIMPLIFICADA E FUNCIONAL
class ModeloHJB:
    def __init__(self, mu, cov_matrix, risk_free=0.12):
        self.mu = mu
        self.cov_matrix = cov_matrix
        self.r = risk_free  # SELIC anualizada
        self.n_assets = len(mu)
    
    def valor_utilidade(self, x, gamma=2):
        """Função utilidade CRRA"""
        if gamma == 1:
            return np.log(x)
        return (x**(1-gamma))/(1-gamma)
    
    def derivada_utilidade(self, x, gamma=2):
        """Derivada da função utilidade"""
        if gamma == 1:
            return 1/x
        return x**(-gamma)
    
    def resolver_hjb_simplificado(self, wealth_0=100000, gamma=2, T=1):
        """
        Resolução simplificada da HJB usando aproximação de tempo discreto
        """
        print("Resolvendo equação HJB...")
        
        # Para simplificar, resolveremos o problema estático de média-variância
        # que é a solução da HJB para utilidade quadrática
        
        n_assets = self.n_assets
        
        # Matriz de covariância regularizada para evitar problemas numéricos
        cov_reg = self.cov_matrix + np.eye(n_assets) * 1e-6
        
        try:
            # Calcular alocação ótima (solução analítica aproximada)
            cov_inv = np.linalg.inv(cov_reg)
            excess_returns = self.mu - self.r
            pi_optimal = (1/gamma) * np.dot(cov_inv, excess_returns)
            
            # Normalizar para soma = 1
            pi_optimal = pi_optimal / np.sum(np.abs(pi_optimal))
            
            print("Solução HJB encontrada!")
            return pi_optimal
            
        except np.linalg.LinAlgError:
            print("Problema na inversão da matriz. Usando solução alternativa...")
            # Solução alternativa: minimizar variância sujeito a retorno alvo
            retorno_alvo = np.mean(self.mu)
            
            def objetivo(pi):
                return np.dot(pi, np.dot(self.cov_matrix, pi))
            
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: np.dot(x, self.mu) - retorno_alvo}
            ]
            
            bounds = [(-1, 2) for _ in range(n_assets)]
            pi0 = np.ones(n_assets) / n_assets
            
            res = minimize(objetivo, pi0, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if res.success:
                return res.x
            else:
                # Último recurso: igualmente ponderado
                return np.ones(n_assets) / n_assets

# Função para análise de desempenho
def analisar_desempenho(pi_optimal, returns, modelo):
    """Analisar desempenho da estratégia ótima"""
    
    # Retornos da estratégia
    retornos_estrategia = returns.dot(pi_optimal)
    
    # Métricas
    retorno_medio = retornos_estrategia.mean() * 252
    volatilidade = retornos_estrategia.std() * np.sqrt(252)
    sharpe = (retorno_medio - modelo.r) / volatilidade
    
    print(f"\n=== ANÁLISE DE DESEMPENHO ===")
    print(f"Retorno Anual: {retorno_medio:.2%}")
    print(f"Volatilidade Anual: {volatilidade:.2%}")
    print(f"Índice Sharpe: {sharpe:.3f}")
    print(f"Retorno Livre de Risco: {modelo.r:.2%}")
    
    return retornos_estrategia

# Função para plotar resultados
def plotar_resultados(pi_optimal, tickers, retornos_estrategia, returns):
    plt.figure(figsize=(15, 10))
    
    # Gráfico 1: Alocação Ótima
    plt.subplot(2, 2, 1)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
    bars = plt.bar(tickers, pi_optimal, color=colors, alpha=0.8)
    plt.title('Alocação Ótima HJB - Bovespa', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.ylabel('Peso (%)')
    plt.grid(True, alpha=0.3)
    
    # Adicionar valores nas barras
    for bar, valor in zip(bars, pi_optimal):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{valor:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # Gráfico 2: Retornos Acumulados
    plt.subplot(2, 2, 2)
    retornos_acum_estrategia = (1 + retornos_estrategia).cumprod()
    retornos_acum_ibov = (1 + returns.mean(axis=1)).cumprod()  # Proxy do IBOV
    
    plt.plot(retornos_acum_estrategia, label='Estratégia HJB', linewidth=2, color='#2E86AB')
    plt.plot(retornos_acum_ibov, label='Portfólio Equiponderado', linewidth=2, color='#A23B72')
    plt.title('Retornos Acumulados', fontsize=14, fontweight='bold')
    plt.xlabel('Tempo')
    plt.ylabel('Retorno Acumulado')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Gráfico 3: Drawdown
    plt.subplot(2, 2, 3)
    pico_acum_estrategia = retornos_acum_estrategia.expanding().max()
    drawdown_estrategia = (retornos_acum_estrategia - pico_acum_estrategia) / pico_acum_estrategia
    
    pico_acum_ibov = retornos_acum_ibov.expanding().max()
    drawdown_ibov = (retornos_acum_ibov - pico_acum_ibov) / pico_acum_ibov
    
    plt.fill_between(drawdown_estrategia.index, drawdown_estrategia * 100, 0, 
                    alpha=0.7, color='#FF6B6B', label='Drawdown HJB')
    plt.fill_between(drawdown_ibov.index, drawdown_ibov * 100, 0, 
                    alpha=0.7, color='#4ECDC4', label='Drawdown Equiponderado')
    plt.title('Drawdown (%)', fontsize=14, fontweight='bold')
    plt.xlabel('Tempo')
    plt.ylabel('Drawdown (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Gráfico 4: Distribuição de Retornos Diários
    plt.subplot(2, 2, 4)
    plt.hist(retornos_estrategia * 100, bins=50, alpha=0.7, color='#45B7D1', edgecolor='black')
    plt.axvline(retornos_estrategia.mean() * 100, color='red', linestyle='--', linewidth=2, 
                label=f'Média: {retornos_estrategia.mean()*100:.3f}%')
    plt.title('Distribuição de Retornos Diários', fontsize=14, fontweight='bold')
    plt.xlabel('Retorno Diário (%)')
    plt.ylabel('Frequência')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def executar_modelo():
    print("=== MODELO HJB PARA BOVESPA ===")
    print("Carregando dados...")
    
    try:
        # Carregar dados
        returns, tickers = carregar_dados_bovespa()
        print(f"Dados carregados: {returns.shape[0]} observações de {len(tickers)} ativos")
        
        # Calcular parâmetros
        mu, sigma, corr, cov = calcular_parametros(returns)
        
        print("\nParâmetros Estimados - Bovespa:")
        print("Ativo\t\tRetorno Anual\tVolatilidade")
        print("-" * 45)
        for i, ticker in enumerate(tickers):
            print(f"{ticker}\t{mu[i]:.2%}\t\t{sigma[i]:.2%}")
        
        # Inicializar modelo HJB
        modelo = ModeloHJB(mu, cov, risk_free=0.1175)  # SELIC atual
        
        # Resolver HJB
        estrategia_otima = modelo.resolver_hjb_simplificado()
        
        print("\nEstratégia Ótima HJB:")
        print("-" * 30)
        for i, ticker in enumerate(tickers):
            print(f"{ticker}: {estrategia_otima[i]:.2%}")
        
        # Analisar desempenho
        retornos_estrategia = analisar_desempenho(estrategia_otima, returns, modelo)
        
        # Plotar resultados
        plotar_resultados(estrategia_otima, tickers, retornos_estrategia, returns)
        
        return estrategia_otima, tickers, modelo, returns
        
    except Exception as e:
        print(f"Erro durante a execução: {e}")
        print("Criando dados de exemplo para demonstração...")
        
        # Dados de exemplo em caso de erro
        tickers = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'WEGE3.SA']
        n_assets = len(tickers)
        
        # Gerar dados sintéticos
        np.random.seed(42)
        n_obs = 1000
        returns_sintetico = pd.DataFrame(
            np.random.multivariate_normal(
                mean=[0.001, 0.0008, 0.0006, 0.0007, 0.0009],
                cov=np.eye(5) * 0.0001 + 0.00005,
                size=n_obs
            ),
            columns=tickers
        )
        
        mu, sigma, corr, cov = calcular_parametros(returns_sintetico)
        
        modelo = ModeloHJB(mu, cov)
        estrategia_otima = modelo.resolver_hjb_simplificado()
        
        print("\nEstratégia Ótima HJB (Dados Sintéticos):")
        for i, ticker in enumerate(tickers):
            print(f"{ticker}: {estrategia_otima[i]:.2%}")
        
        retornos_estrategia = analisar_desempenho(estrategia_otima, returns_sintetico, modelo)
        plotar_resultados(estrategia_otima, tickers, retornos_estrategia, returns_sintetico)
        
        return estrategia_otima, tickers, modelo, returns_sintetico


if __name__ == "__main__":
    estrategia, tickers, modelo, returns = executar_modelo()
    
    # Salvar resultados
    resultados_df = pd.DataFrame({
        'Ativo': tickers,
        'Alocacao_Otima': estrategia,
        'Retorno_Esperado': modelo.mu,
        'Volatilidade': np.sqrt(np.diag(modelo.cov_matrix))
    })
    
    print("\n=== RESUMO FINAL ===")
    print(resultados_df.round(4))
    
    # Calcular estatísticas finais
    retorno_carteira = np.dot(estrategia, modelo.mu)
    volatilidade_carteira = np.sqrt(np.dot(estrategia, np.dot(modelo.cov_matrix, estrategia)))
    sharpe_final = (retorno_carteira - modelo.r) / volatilidade_carteira
    
    print(f"\nEstatísticas da Carteira Ótima:")
    print(f"Retorno Esperado: {retorno_carteira:.2%}")
    print(f"Volatilidade Esperada: {volatilidade_carteira:.2%}")
    print(f"Sharpe Ratio: {sharpe_final:.3f}")