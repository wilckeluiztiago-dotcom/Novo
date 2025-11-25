"""
Modelos Bayesianos para análise eleitoral.

Inclui:
- Modelo Bayesiano Hierárquico
- Modelo Dirichlet-Multinomial
- MCMC (Markov Chain Monte Carlo)
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gammaln
import warnings
warnings.filterwarnings('ignore')

try:
    import pymc3 as pm
    PYMC3_AVAILABLE = True
except ImportError:
    PYMC3_AVAILABLE = False


class ModeloBayesianoHierarquico:
    """
    Modelo Bayesiano Hierárquico para estimação de votos por região/estado.
    
    Estrutura hierárquica:
    
    Nível 1 (Nacional):
    μ_nacional ~ Normal(μ₀, σ₀²)
    
    Nível 2 (Estado):
    μ_estado ~ Normal(μ_nacional, τ²)
    
    Nível 3 (Observações):
    votos ~ Normal(μ_estado, σ²)
    
    Vantagens:
    - Compartilha informação entre estados (pooling parcial)
    - Melhora estimativas para estados com poucos dados
    - Quantifica incerteza naturalmente
    """
    
    def __init__(self):
        if not PYMC3_AVAILABLE:
            raise ImportError("PyMC3 não está instalado. Instale com: pip install pymc3")
        
        self.modelo = None
        self.trace = None
        self.estados = None
        
    def treinar(self, dados, coluna_estado='estado', coluna_votos='votos', 
                n_amostras=2000, tune=1000):
        """
        Treina modelo bayesiano hierárquico.
        
        Parâmetros:
        -----------
        dados : DataFrame
            Dados eleitorais com estado e votos
        coluna_estado : str
            Nome da coluna de estado
        coluna_votos : str
            Nome da coluna de votos
        n_amostras : int
            Número de amostras MCMC
        tune : int
            Número de iterações de tuning
        """
        self.estados = dados[coluna_estado].unique()
        estado_idx = pd.Categorical(dados[coluna_estado]).codes
        
        with pm.Model() as self.modelo:
            # Hiperpriors (nível nacional)
            mu_nacional = pm.Normal('mu_nacional', mu=0, sigma=10)
            tau = pm.HalfCauchy('tau', beta=5)  # Variação entre estados
            
            # Priors por estado
            mu_estado = pm.Normal('mu_estado', mu=mu_nacional, sigma=tau, shape=len(self.estados))
            
            # Likelihood
            sigma = pm.HalfCauchy('sigma', beta=5)
            votos_obs = pm.Normal('votos_obs', 
                                  mu=mu_estado[estado_idx], 
                                  sigma=sigma, 
                                  observed=dados[coluna_votos])
            
            # Amostragem MCMC
            self.trace = pm.sample(n_amostras, tune=tune, return_inferencedata=False, 
                                   progressbar=False, cores=1)
        
        return self
    
    def obter_estimativas_estados(self, intervalo_credibilidade=0.95):
        """
        Retorna estimativas por estado com intervalos de credibilidade.
        
        Parâmetros:
        -----------
        intervalo_credibilidade : float
            Nível do intervalo (padrão: 95%)
        
        Retorna:
        --------
        DataFrame
            Estimativas por estado
        """
        alpha = 1 - intervalo_credibilidade
        
        resultados = []
        for i, estado in enumerate(self.estados):
            amostras = self.trace['mu_estado'][:, i]
            
            resultados.append({
                'estado': estado,
                'media': np.mean(amostras),
                'mediana': np.median(amostras),
                'desvio_padrao': np.std(amostras),
                f'IC_{int(intervalo_credibilidade*100)}_inferior': np.percentile(amostras, alpha/2 * 100),
                f'IC_{int(intervalo_credibilidade*100)}_superior': np.percentile(amostras, (1-alpha/2) * 100)
            })
        
        return pd.DataFrame(resultados)
    
    def prever(self, estado, n_amostras=1000):
        """
        Gera amostras preditivas para um estado.
        
        Parâmetros:
        -----------
        estado : str
            Nome do estado
        n_amostras : int
            Número de amostras a gerar
        
        Retorna:
        --------
        array
            Amostras da distribuição preditiva
        """
        if estado not in self.estados:
            raise ValueError(f"Estado {estado} não encontrado nos dados de treinamento")
        
        idx = np.where(self.estados == estado)[0][0]
        
        # Amostrar da posterior
        mu_samples = self.trace['mu_estado'][:, idx]
        sigma_samples = self.trace['sigma']
        
        # Gerar previsões
        indices = np.random.choice(len(mu_samples), n_amostras)
        previsoes = np.random.normal(mu_samples[indices], sigma_samples[indices])
        
        return previsoes


class ModeloDirichlet:
    """
    Modelo Dirichlet-Multinomial para distribuição de votos entre partidos.
    
    Distribuição Dirichlet (prior conjugado para multinomial):
    
    p(θ|α) = [Γ(Σαᵢ) / Πᵢ Γ(αᵢ)] · Πᵢ θᵢ^(αᵢ-1)
    
    Onde:
    - θ = (θ₁, ..., θₖ): proporções de votos (Σθᵢ = 1)
    - α = (α₁, ..., αₖ): parâmetros de concentração
    
    Posterior (após observar dados):
    p(θ|dados) ~ Dirichlet(α + contagens)
    
    Útil para:
    - Modelar incerteza em proporções de votos
    - Incorporar conhecimento prévio
    - Fazer previsões probabilísticas
    """
    
    def __init__(self, alpha_prior=None):
        """
        Parâmetros:
        -----------
        alpha_prior : array-like, opcional
            Parâmetros α do prior Dirichlet
            Se None, usa prior uniforme (α = 1 para todos)
        """
        self.alpha_prior = alpha_prior
        self.alpha_posterior = None
        self.partidos = None
        
    def treinar(self, votos_por_partido):
        """
        Atualiza posterior com dados observados.
        
        Parâmetros:
        -----------
        votos_por_partido : dict ou Series
            Contagem de votos por partido
        """
        if isinstance(votos_por_partido, dict):
            votos_por_partido = pd.Series(votos_por_partido)
        
        self.partidos = votos_por_partido.index.tolist()
        contagens = votos_por_partido.values
        
        # Prior uniforme se não especificado
        if self.alpha_prior is None:
            self.alpha_prior = np.ones(len(self.partidos))
        
        # Posterior: α_post = α_prior + contagens
        self.alpha_posterior = self.alpha_prior + contagens
        
        return self
    
    def amostrar_proporcoes(self, n_amostras=1000):
        """
        Amostra proporções de votos da distribuição posterior.
        
        Parâmetros:
        -----------
        n_amostras : int
            Número de amostras a gerar
        
        Retorna:
        --------
        DataFrame
            Amostras de proporções por partido
        """
        amostras = np.random.dirichlet(self.alpha_posterior, n_amostras)
        
        return pd.DataFrame(amostras, columns=self.partidos)
    
    def obter_proporcoes_esperadas(self):
        """
        Retorna proporções esperadas (média da posterior).
        
        Retorna:
        --------
        Series
            Proporção esperada por partido
        """
        proporcoes = self.alpha_posterior / np.sum(self.alpha_posterior)
        return pd.Series(proporcoes, index=self.partidos)
    
    def obter_intervalos_credibilidade(self, credibilidade=0.95, n_amostras=10000):
        """
        Calcula intervalos de credibilidade para cada partido.
        
        Parâmetros:
        -----------
        credibilidade : float
            Nível de credibilidade (padrão: 95%)
        n_amostras : int
            Número de amostras para estimação
        
        Retorna:
        --------
        DataFrame
            Intervalos por partido
        """
        amostras = self.amostrar_proporcoes(n_amostras)
        alpha = 1 - credibilidade
        
        resultados = []
        for partido in self.partidos:
            resultados.append({
                'partido': partido,
                'media': amostras[partido].mean(),
                'mediana': amostras[partido].median(),
                'IC_inferior': amostras[partido].quantile(alpha/2),
                'IC_superior': amostras[partido].quantile(1 - alpha/2)
            })
        
        return pd.DataFrame(resultados)
    
    def prever_eleicao(self, n_votos_total, n_simulacoes=1000):
        """
        Simula resultados de eleição.
        
        Parâmetros:
        -----------
        n_votos_total : int
            Número total de votos a simular
        n_simulacoes : int
            Número de simulações
        
        Retorna:
        --------
        DataFrame
            Distribuição de votos por partido em cada simulação
        """
        # Amostrar proporções
        proporcoes = self.amostrar_proporcoes(n_simulacoes)
        
        # Para cada amostra de proporções, gerar votos
        resultados = []
        for i in range(n_simulacoes):
            props = proporcoes.iloc[i].values
            votos = np.random.multinomial(n_votos_total, props)
            resultados.append(votos)
        
        return pd.DataFrame(resultados, columns=self.partidos)
    
    def probabilidade_vitoria(self, n_votos_total=1000000, n_simulacoes=10000):
        """
        Calcula probabilidade de cada partido ter mais votos.
        
        Parâmetros:
        -----------
        n_votos_total : int
            Total de votos a simular
        n_simulacoes : int
            Número de simulações
        
        Retorna:
        --------
        Series
            Probabilidade de vitória por partido
        """
        simulacoes = self.prever_eleicao(n_votos_total, n_simulacoes)
        
        # Contar em quantas simulações cada partido venceu
        vitorias = (simulacoes == simulacoes.max(axis=1).values[:, None]).sum()
        probabilidades = vitorias / n_simulacoes
        
        return probabilidades.sort_values(ascending=False)


class ModeloMCMC:
    """
    Modelo genérico usando MCMC para inferência bayesiana.
    
    MCMC (Markov Chain Monte Carlo) gera amostras da distribuição posterior:
    
    p(θ|dados) ∝ p(dados|θ) · p(θ)
    
    Algoritmos:
    - Metropolis-Hastings
    - Gibbs Sampling
    - NUTS (No-U-Turn Sampler) - usado pelo PyMC3
    
    Permite inferência em modelos complexos onde solução analítica não existe.
    """
    
    def __init__(self):
        if not PYMC3_AVAILABLE:
            raise ImportError("PyMC3 não está instalado")
        
        self.modelo = None
        self.trace = None
        
    def criar_modelo_regressao_bayesiana(self, X, y):
        """
        Cria modelo de regressão bayesiana.
        
        Modelo:
        y ~ Normal(Xβ, σ²)
        β ~ Normal(0, 10²)
        σ ~ HalfCauchy(5)
        """
        n_features = X.shape[1]
        
        with pm.Model() as self.modelo:
            # Priors
            beta = pm.Normal('beta', mu=0, sigma=10, shape=n_features)
            sigma = pm.HalfCauchy('sigma', beta=5)
            
            # Modelo linear
            mu = pm.math.dot(X, beta)
            
            # Likelihood
            y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
        
        return self
    
    def amostrar(self, n_amostras=2000, tune=1000):
        """Executa amostragem MCMC."""
        with self.modelo:
            self.trace = pm.sample(n_amostras, tune=tune, return_inferencedata=False,
                                   progressbar=False, cores=1)
        return self
    
    def obter_resumo(self):
        """Retorna resumo estatístico das posteriors."""
        return pm.summary(self.trace)
    
    def prever(self, X_novo, n_amostras=1000):
        """
        Faz previsões com incerteza.
        
        Parâmetros:
        -----------
        X_novo : array-like
            Novos dados para previsão
        n_amostras : int
            Número de amostras preditivas
        
        Retorna:
        --------
        array
            Amostras preditivas (n_amostras x n_observacoes)
        """
        beta_samples = self.trace['beta']
        sigma_samples = self.trace['sigma']
        
        # Selecionar amostras aleatórias
        indices = np.random.choice(len(beta_samples), n_amostras)
        
        previsoes = []
        for idx in indices:
            mu = np.dot(X_novo, beta_samples[idx])
            y_pred = np.random.normal(mu, sigma_samples[idx])
            previsoes.append(y_pred)
        
        return np.array(previsoes)


def calcular_fator_bayes(modelo1_log_likelihood, modelo2_log_likelihood):
    """
    Calcula Fator de Bayes para comparação de modelos.
    
    BF₁₂ = p(dados|M₁) / p(dados|M₂)
    
    Interpretação:
    - BF > 10: evidência forte para M₁
    - BF > 3: evidência moderada para M₁
    - BF ≈ 1: evidência inconclusiva
    - BF < 1/3: evidência moderada para M₂
    
    Parâmetros:
    -----------
    modelo1_log_likelihood : float
        Log-likelihood do modelo 1
    modelo2_log_likelihood : float
        Log-likelihood do modelo 2
    
    Retorna:
    --------
    float
        Fator de Bayes
    """
    log_bf = modelo1_log_likelihood - modelo2_log_likelihood
    return np.exp(log_bf)
