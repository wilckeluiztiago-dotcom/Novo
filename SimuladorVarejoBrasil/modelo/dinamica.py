import numpy as np
import pandas as pd
from .regioes import RegioesBrasil

class SimulacaoVarejo:
    def __init__(self, regiao, preco_inicial, marketing_inicial, estoque_inicial, meta_estoque):
        self.params = RegioesBrasil.get_parametros(regiao)
        self.preco = preco_inicial
        self.marketing = marketing_inicial
        self.meta_estoque = meta_estoque
        
        # Parâmetros do Modelo
        self.r = 0.05 * self.params['fator_pib'] # Taxa de crescimento base
        self.K = self.params['populacao'] * 0.1 # Capacidade de mercado (10% da pop compra)
        self.alpha = 0.0001 # Eficiência do Marketing
        self.beta = 0.005 # Sensibilidade ao Preço
        
    def sazonalidade(self, t):
        """Retorna fator sazonal baseado no mês (t em meses)."""
        mes = int(t) % 12
        if mes == 11: return self.params['sazonalidade']['black_friday']
        if mes == 12: return self.params['sazonalidade']['natal']
        if mes in [0, 1]: return self.params['sazonalidade']['verao']
        if mes in [5, 6, 7]: return self.params['sazonalidade']['inverno']
        return 1.0

    def derivadas(self, t, estado):
        """
        Sistema de EDOs:
        D' = r*D*(1 - D/K) + alpha*Mkt - beta*Preco
        I' = Compra - Vendas
        """
        D, I = estado # Demanda, Estoque
        
        # 1. Dinâmica da Demanda (Logística Modificada)
        # Ajuste por preço e marketing
        fator_externo = (self.alpha * self.marketing) - (self.beta * self.preco)
        dD_dt = self.r * D * (1 - D / self.K) + fator_externo
        
        # Aplicar sazonalidade instantânea na demanda efetiva (não na derivada estrutural)
        # Mas para simplificar a EDO, vamos assumir que D é a demanda base.
        
        # 2. Dinâmica de Estoque
        demanda_efetiva = max(0, D * self.sazonalidade(t))
        vendas = min(demanda_efetiva, max(0, I)) # Não vende o que não tem
        
        # Política de Reabastecimento (Controle Proporcional)
        gap_estoque = self.meta_estoque - I
        compra = max(0, gap_estoque * 0.5) # Repõe 50% do gap por unidade de tempo
        
        dI_dt = compra - vendas
        
        return np.array([dD_dt, dI_dt]), vendas, compra, demanda_efetiva

    def solver(self, t_max=24, dt=0.1):
        """Simula por t_max meses."""
        t = np.arange(0, t_max, dt)
        n_steps = len(t)
        
        # Estado Inicial: [Demanda Inicial, Estoque Inicial]
        estado = np.zeros((n_steps, 2))
        estado[0] = [self.K * 0.01, self.meta_estoque] 
        
        # Métricas Auxiliares
        vendas_hist = np.zeros(n_steps)
        compra_hist = np.zeros(n_steps)
        receita_hist = np.zeros(n_steps)
        lucro_hist = np.zeros(n_steps)
        ruptura_hist = np.zeros(n_steps)
        
        for i in range(n_steps - 1):
            y_n = estado[i]
            t_n = t[i]
            
            # Euler Simples (suficiente para economia, RK4 seria overkill aqui mas ok)
            derivs, vendas, compra, dem_efetiva = self.derivadas(t_n, y_n)
            
            estado[i+1] = y_n + derivs * dt
            
            # Constraints
            if estado[i+1][0] < 0: estado[i+1][0] = 0 # Demanda não negativa
            if estado[i+1][1] < 0: estado[i+1][1] = 0 # Estoque não negativo
            
            # Salvar métricas do passo atual
            vendas_hist[i] = vendas
            compra_hist[i] = compra
            
            # Financeiro
            receita = vendas * self.preco
            custo_fixo = 10000
            custo_var = compra * (self.preco * 0.4) # Margem de 60% bruta
            custo_logistica = compra * self.params['custo_logistica']
            lucro = receita - (custo_fixo + custo_var + custo_logistica + self.marketing)
            
            receita_hist[i] = receita
            lucro_hist[i] = lucro
            
            # Ruptura (Demanda não atendida)
            ruptura_hist[i] = max(0, dem_efetiva - vendas)

        # Ajustar último ponto
        vendas_hist[-1] = vendas_hist[-2]
        
        return pd.DataFrame({
            'tempo': t,
            'demanda_base': estado[:, 0],
            'estoque': estado[:, 1],
            'vendas': vendas_hist,
            'compras': compra_hist,
            'receita': receita_hist,
            'lucro': lucro_hist,
            'ruptura': ruptura_hist
        })
