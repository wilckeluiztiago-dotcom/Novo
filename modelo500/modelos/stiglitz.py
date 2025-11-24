"""
Módulo Teórico de Stiglitz (Salário Eficiência).

Implementação numérica do modelo de Shapiro-Stiglitz (1984).
Analisa a relação entre desemprego, monitoramento e salários de eficiência.

Autor: Luiz Tiago Wilcke
Data: 2025-11-24
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple

class ModeloShapiroStiglitz:
    """
    Modelo de Salário Eficiência (Shapiro-Stiglitz).
    
    Determina o salário de equilíbrio que previne a "vadiagem" (shirking)
    dos trabalhadores, gerando desemprego involuntário de equilíbrio.
    """
    
    def __init__(self, parametros: Dict[str, float] = None):
        """
        Args:
            parametros: Dicionário com:
                - b: Benefício desemprego / utilidade do lazer
                - e: Esforço no trabalho
                - q: Probabilidade de ser pego vadiando (monitoramento)
                - rho: Taxa de desconto intertemporal
                - alpha: Parâmetro de produtividade (Demanda)
        """
        if parametros is None:
            self.parametros = {
                'b': 1000.0,  # Benefício/Custo oportunidade
                'e': 500.0,   # Custo do esforço
                'q': 0.2,     # Probabilidade de detecção
                'rho': 0.05,  # Taxa de desconto
                'alpha': 10000.0 # Produtividade marginal base
            }
        else:
            self.parametros = parametros
            
    def condicao_nao_vadiagem(self, u: float) -> float:
        """
        Calcula o Salário de Não-Vadiagem (No-Shirking Condition - NSC).
        
        w_NSC = b + e + (e/q) * (b/u + rho)  <-- Versão simplificada comum
        
        Formulação exata (Shapiro-Stiglitz 1984):
        w >= b + e + (e/q)(b/u + r)  (aproximação onde b é taxa de saída do emprego)
        
        Aqui usamos a forma funcional:
        w = b + e + (e * (rho + b_rate + q)) / q  ... na verdade depende de u.
        
        A equação clássica do NSC é:
        w = b + e + (e/q) * (rho + taxa_demissao_exogena + taxa_contratacao(u))
        
        Onde taxa_contratacao(u) = b_rate * (1-u)/u  (b_rate = breakup rate)
        """
        p = self.parametros
        b_rate = 0.1 # Taxa de separação exógena (breakup rate) fixa para simplificar
        
        # Taxa de encontrar emprego (job finding rate) 'a' depende de u
        # No estado estacionário: b_rate * (1-u) = a * u  => a = b_rate * (1-u)/u
        if u <= 0 or u >= 1:
            return np.inf
            
        a = b_rate * (1 - u) / u
        
        # NSC: w >= b + e + (e/q) * (rho + b_rate + a)
        w_nsc = p['b'] + p['e'] + (p['e'] / p['q']) * (p['rho'] + b_rate + a)
        
        return w_nsc
        
    def demanda_trabalho(self, L: float, N_total: float = 100.0) -> float:
        """
        Curva de Demanda de Trabalho (Produtividade Marginal).
        F'(L) = w
        
        Assumindo F(L) = alpha * ln(L) => F'(L) = alpha / L
        """
        if L <= 0:
            return np.inf
        return self.parametros['alpha'] / L
        
    def calcular_equilibrio(self) -> Dict[str, float]:
        """
        Encontra o equilíbrio (interseção NSC e Demanda).
        """
        N_total = 100.0 # Força de trabalho total normalizada
        
        # Busca binária ou otimização para encontrar u* onde NSC(u) = Demanda(L(u))
        # L = N_total * (1 - u)
        
        u_min, u_max = 0.001, 0.999
        for _ in range(50):
            u_mid = (u_min + u_max) / 2
            w_nsc = self.condicao_nao_vadiagem(u_mid)
            
            L_mid = N_total * (1 - u_mid)
            w_demanda = self.demanda_trabalho(L_mid)
            
            if w_nsc < w_demanda:
                # Salário NSC é menor que produtividade, empresas contratam mais
                # Desemprego cai -> u deve ser menor? 
                # Não, se w_nsc < w_demanda, significa que para esse u, o salário exigido é baixo
                # e a produtividade é alta. O equilíbrio está em um u menor (mais emprego)?
                # NSC é decrescente em u? Não, NSC é DECRESCENTE com u?
                # Se u aumenta, 'a' diminui (mais difícil achar emprego), punição de demissão é maior.
                # Logo trabalhador aceita salário menor. NSC é DECRESCENTE em u?
                # Vamos checar a fórmula: a = b_rate*(1-u)/u. Se u sobe, (1-u)/u desce, 'a' desce.
                # w_nsc depende de + a. Então se 'a' desce, w_nsc desce.
                # Sim, NSC é uma curva decrescente no plano (Emprego, Salário)?
                # Espere. Plano usual: Eixo X = Emprego (L), Eixo Y = Salário (w).
                # Se L aumenta, u diminui.
                # Se u diminui (L aumenta), 'a' aumenta (fácil achar emprego).
                # Se 'a' aumenta, w_nsc AUMENTA.
                # Então NSC é CRESCENTE no plano (L, w).
                # Demanda é DECRESCENTE no plano (L, w).
                
                # Voltando a u:
                # Se u aumenta (L diminui), w_nsc diminui.
                # Se u aumenta (L diminui), w_demanda aumenta (produtividade marginal sobe com escassez).
                # Queremos w_nsc = w_demanda.
                # Se w_nsc < w_demanda (no u_mid atual):
                # Significa que o salário exigido é baixo, empresas pagam w_demanda > w_nsc.
                # Isso incentiva mais emprego?
                # O equilíbrio está onde se cruzam.
                # Se w_nsc < w_demanda, estamos à direita do equilíbrio no gráfico de u?
                # Vamos pensar em L.
                # Se w_nsc(L) < w_demanda(L). Curva NSC está abaixo da Demanda.
                # Como NSC sobe com L e Demanda desce com L, precisamos AUMENTAR L para encontrar a interseção.
                # Aumentar L significa DIMINUIR u.
                u_max = u_mid # Tentar u menor (L maior)
            else:
                u_min = u_mid
                
        u_eq = (u_min + u_max) / 2
        w_eq = self.condicao_nao_vadiagem(u_eq)
        
        return {
            'desemprego_equilibrio': u_eq,
            'salario_equilibrio': w_eq,
            'emprego_equilibrio': N_total * (1 - u_eq)
        }

    def simular_curvas(self, n_pontos: int = 100) -> pd.DataFrame:
        """Gera dados das curvas para plotagem."""
        u_vals = np.linspace(0.01, 0.30, n_pontos) # Foco na faixa relevante de desemprego
        
        dados = []
        for u in u_vals:
            w_nsc = self.condicao_nao_vadiagem(u)
            L = 100.0 * (1 - u)
            w_demanda = self.demanda_trabalho(L)
            
            dados.append({
                'taxa_desemprego': u,
                'emprego': L,
                'salario_nsc': w_nsc,
                'salario_demanda': w_demanda
            })
            
        return pd.DataFrame(dados)
