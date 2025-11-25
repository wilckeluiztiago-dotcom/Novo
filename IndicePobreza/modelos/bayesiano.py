import pymc as pm
import numpy as np
import pandas as pd
import arviz as az

class ModeloBayesianoPobreza:
    """
    Modelo Hierárquico Bayesiano para estimar a probabilidade de pobreza.
    Considera efeitos fixos (renda, escolaridade) e efeitos aleatórios (UF).
    """
    
    def __init__(self, dados):
        """
        Inicializa o modelo.
        
        Parâmetros:
            dados (pd.DataFrame): DataFrame contendo 'is_pobre', 'renda_pc', 'anos_estudo_chefe', 'uf'.
        """
        self.dados = dados.copy()
        # Codificar UF para índices numéricos
        self.ufs = self.dados['uf'].unique()
        self.uf_idx = pd.Categorical(self.dados['uf'], categories=self.ufs).codes
        self.n_ufs = len(self.ufs)
        
    def construir_modelo(self):
        """
        Define o modelo PyMC.
        
        Modelo:
        y_i ~ Bernoulli(p_i)
        logit(p_i) = alpha + beta_renda * renda_padronizada + beta_educ * educ_padronizada + u_uf[j]
        u_uf[j] ~ Normal(0, sigma_uf)
        """
        # Padronizar variáveis contínuas para facilitar convergência
        renda_std = (self.dados['renda_pc'] - self.dados['renda_pc'].mean()) / self.dados['renda_pc'].std()
        educ_std = (self.dados['anos_estudo_chefe'] - self.dados['anos_estudo_chefe'].mean()) / self.dados['anos_estudo_chefe'].std()
        y = self.dados['is_pobre'].values
        
        with pm.Model() as model:
            # Priors para coeficientes globais
            alpha = pm.Normal('alpha', mu=0, sigma=2)
            beta_renda = pm.Normal('beta_renda', mu=0, sigma=2)
            beta_educ = pm.Normal('beta_educ', mu=0, sigma=2)
            
            # Priors para efeitos aleatórios de UF (Hierárquico)
            sigma_uf = pm.HalfNormal('sigma_uf', sigma=1)
            u_uf = pm.Normal('u_uf', mu=0, sigma=sigma_uf, shape=self.n_ufs)
            
            # Modelo Linear Generalizado (Logit)
            logit_p = alpha + beta_renda * renda_std + beta_educ * educ_std + u_uf[self.uf_idx]
            
            # Likelihood
            y_obs = pm.Bernoulli('y_obs', logit_p=logit_p, observed=y)
            
            self.model = model
            
    def amostrar(self, draws=1000, tune=500):
        """
        Realiza a amostragem MCMC.
        """
        if not hasattr(self, 'model'):
            self.construir_modelo()
            
        with self.model:
            # Usando return_inferencedata=True por padrão nas versões novas
            self.trace = pm.sample(draws=draws, tune=tune, target_accept=0.9, progressbar=False)
            
        return self.trace
    
    def resumir_resultados(self):
        """
        Retorna um resumo dos parâmetros estimados.
        """
        if not hasattr(self, 'trace'):
            raise ValueError("Execute o método amostrar() primeiro.")
            
        return az.summary(self.trace, var_names=['alpha', 'beta_renda', 'beta_educ', 'sigma_uf'])

    def probabilidades_por_uf(self):
        """
        Calcula o efeito médio de cada UF na probabilidade de pobreza (efeito aleatório).
        """
        if not hasattr(self, 'trace'):
            raise ValueError("Execute o método amostrar() primeiro.")
            
        summary_uf = az.summary(self.trace, var_names=['u_uf'])
        summary_uf['uf'] = self.ufs
        return summary_uf[['uf', 'mean', 'sd', 'hdi_3%', 'hdi_97%']]
