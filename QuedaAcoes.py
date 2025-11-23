"""
Sistema Avançado de Previsão de Quedas na Bolsa Brasileira
Autor: Luiz Tiago Wilcke

Modelo matemático baseado em equações diferenciais estocásticas
para previsão de riscos e quedas no mercado de ações da B3
"""

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import math
import random

class ModeloQuedaAcoes:
    def __init__(self):
        self.equacoes = {
            'principal': "dSₜ = μSₜdt + √νₜSₜdWₜ¹ + JₜSₜdNₜ",
            'volatilidade': "dνₜ = κ(θ - νₜ)dt + ξ√νₜdWₜ²",
            'sentimento': "dmₜ/dt = α(m₀ - mₜ) + β(dSₜ/Sₜ)"
        }
        
    def modelo_heston_salto(self, S0, mu, kappa, theta, xi, rho, v0, lambda_j, mu_j, sigma_j, dias=252):
        """
        Modelo Heston com saltos para a bolsa brasileira
        """
        dt = 1/252
        n_steps = dias
        
        # Arrays para armazenar resultados
        S = np.zeros(n_steps)
        v = np.zeros(n_steps)
        S[0] = S0
        v[0] = v0
        
        # Gerar correlação entre os processos
        Z1 = np.random.standard_normal(n_steps)
        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.standard_normal(n_steps)
        
        for t in range(1, n_steps):
            # Processo de saltos
            saltos = 0
            n_saltos = np.random.poisson(lambda_j * dt)
            if n_saltos > 0:
                saltos = np.sum(np.random.normal(mu_j, sigma_j, n_saltos))
            
            # Volatilidade estocástica
            dv = kappa * (theta - v[t-1]) * dt + xi * np.sqrt(max(v[t-1], 0)) * np.sqrt(dt) * Z2[t-1]
            v[t] = max(v[t-1] + dv, 0.01)
            
            # Preço do ativo com saltos
            dS = mu * S[t-1] * dt + np.sqrt(max(v[t-1], 0.01)) * S[t-1] * np.sqrt(dt) * Z1[t-1] + S[t-1] * saltos
            S[t] = max(S[t-1] + dS, 0.01)
            
        return S, v
    
    def calcular_probabilidade_queda(self, S, limiar_queda=0.10):
        """Calcula probabilidade de queda superior ao limiar"""
        retornos = np.diff(S) / S[:-1]
        prob_queda = np.mean(retornos < -limiar_queda)
        return prob_queda
    
    def calcular_var(self, S, confianca=0.95):
        """Calcula Value at Risk"""
        retornos = np.diff(S) / S[:-1]
        var = np.percentile(retornos, (1 - confianca) * 100)
        return var

    def calcular_drawdown(self, S):
        """Calcula o drawdown máximo"""
        peak = np.maximum.accumulate(S)
        drawdown = (S - peak) / peak
        return drawdown

class GraficoCanvas:
    """Classe para criar gráficos simples usando apenas Canvas do tkinter"""
    
    def __init__(self, parent, width=400, height=200):
        self.canvas = tk.Canvas(parent, width=width, height=height, bg='white')
        self.width = width
        self.height = height
        self.padding = 40
        
    def plotar_linha(self, dados, cor='blue', titulo=""):
        """Plota um gráfico de linha simples"""
        self.canvas.delete("all")
        
        if len(dados) == 0:
            return
            
        # Calcular escala
        min_val = min(dados)
        max_val = max(dados)
        range_val = max_val - min_val if max_val != min_val else 1
        
        # Desenhar eixos
        self.canvas.create_line(self.padding, self.height - self.padding, 
                               self.width - self.padding, self.height - self.padding, width=2)  # X
        self.canvas.create_line(self.padding, self.padding, 
                               self.padding, self.height - self.padding, width=2)  # Y
        
        # Título
        self.canvas.create_text(self.width // 2, 15, text=titulo, font=('Arial', 10, 'bold'))
        
        # Plotar dados
        points = []
        for i, valor in enumerate(dados):
            x = self.padding + (i / (len(dados) - 1)) * (self.width - 2 * self.padding)
            y = self.height - self.padding - ((valor - min_val) / range_val) * (self.height - 2 * self.padding)
            points.append((x, y))
        
        # Desenhar linha
        for i in range(len(points) - 1):
            self.canvas.create_line(points[i][0], points[i][1], 
                                   points[i+1][0], points[i+1][1], 
                                   fill=cor, width=2)
        
        # Valores dos eixos
        self.canvas.create_text(self.padding - 20, self.height - self.padding, 
                               text=f"{min_val:.1f}", anchor=tk.E)
        self.canvas.create_text(self.padding - 20, self.padding, 
                               text=f"{max_val:.1f}", anchor=tk.E)
        
    def plotar_histograma(self, dados, cor='green', titulo=""):
        """Plota um histograma simples"""
        self.canvas.delete("all")
        
        if len(dados) == 0:
            return
            
        # Calcular histograma
        hist, bins = np.histogram(dados, bins=20)
        max_freq = max(hist)
        
        # Desenhar eixos
        self.canvas.create_line(self.padding, self.height - self.padding, 
                               self.width - self.padding, self.height - self.padding, width=2)
        self.canvas.create_line(self.padding, self.padding, 
                               self.padding, self.height - self.padding, width=2)
        
        # Título
        self.canvas.create_text(self.width // 2, 15, text=titulo, font=('Arial', 10, 'bold'))
        
        # Plotar barras
        bin_width = (self.width - 2 * self.padding) / len(hist)
        for i, freq in enumerate(hist):
            x1 = self.padding + i * bin_width
            x2 = x1 + bin_width - 2
            height = (freq / max_freq) * (self.height - 2 * self.padding)
            y1 = self.height - self.padding - height
            y2 = self.height - self.padding
            
            self.canvas.create_rectangle(x1, y1, x2, y2, fill=cor, outline='black')

class CalculadoraBolsa:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema Avançado de Previsão de Quedas - Bolsa Brasileira - por Luiz Tiago Wilcke")
        self.root.geometry("1400x900")
        
        self.modelo = ModeloQuedaAcoes()
        self.setup_ui()
        
    def setup_ui(self):
        # Configurar grid principal
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=2)
        self.root.rowconfigure(0, weight=1)
        
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Título com autor
        title_label = ttk.Label(main_frame, 
                               text="🔻 SISTEMA DE PREVISÃO DE QUEDAS - BOLSA BRASILEIRA 🔻\npor Luiz Tiago Wilcke", 
                               font=('Arial', 16, 'bold'), foreground='darkred', justify='center')
        title_label.grid(row=0, column=0, columnspan=2, pady=15)
        
        # Frame de parâmetros (lado esquerdo)
        param_frame = ttk.LabelFrame(main_frame, text="📊 PARÂMETROS DO MODELO", padding="15")
        param_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # Variáveis de entrada
        self.variaveis = {
            'preco_inicial': tk.DoubleVar(value=35.0),
            'retorno_esperado': tk.DoubleVar(value=0.12),
            'kappa': tk.DoubleVar(value=1.5),
            'theta': tk.DoubleVar(value=0.35),
            'xi': tk.DoubleVar(value=0.3),
            'rho': tk.DoubleVar(value=-0.6),
            'vol_inicial': tk.DoubleVar(value=0.3),
            'lambda_salto': tk.DoubleVar(value=8.0),
            'mu_salto': tk.DoubleVar(value=-0.03),
            'sigma_salto': tk.DoubleVar(value=0.04),
            'dias_simulacao': tk.IntVar(value=180)
        }
        
        # Campos de entrada
        row = 0
        self.entries = {}
        
        parametros = [
            ('preco_inicial', 'Preço Inicial (R$):'),
            ('retorno_esperado', 'Retorno Esperado (%):'),
            ('kappa', 'Velocidade Reversão (κ):'),
            ('theta', 'Vol Longo Prazo (θ):'),
            ('xi', 'Vol da Vol (ξ):'),
            ('rho', 'Correlação (ρ):'),
            ('vol_inicial', 'Volatilidade Inicial:'),
            ('lambda_salto', 'Frequência Saltos (λ):'),
            ('mu_salto', 'Média Saltos (μ):'),
            ('sigma_salto', 'Vol Saltos (σ):'),
            ('dias_simulacao', 'Dias Simulação:')
        ]
        
        for var_name, label_text in parametros:
            frame = ttk.Frame(param_frame)
            frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=2)
            
            label = ttk.Label(frame, text=label_text, width=25, anchor=tk.W)
            label.grid(row=0, column=0, sticky=tk.W)
            
            entry = ttk.Entry(frame, textvariable=self.variaveis[var_name], width=15)
            entry.grid(row=0, column=1, padx=5)
            self.entries[var_name] = entry
            
            row += 1
        
        # Frame de controles
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=2, column=0, pady=15, sticky=(tk.W, tk.E))
        
        # Botões
        buttons = [
            ("🎯 SIMULAR", self.executar_simulacao),
            ("📈 MONTE CARLO", self.simulacao_monte_carlo),
            ("📚 EXPLICAR MODELO", self.explicar_modelo),
            ("🔄 VALORES PADRÃO B3", self.valores_padrao_b3),
            ("🧹 LIMPAR", self.limpar_resultados)
        ]
        
        for i, (text, command) in enumerate(buttons):
            btn = ttk.Button(control_frame, text=text, command=command, width=18)
            btn.grid(row=0, column=i, padx=3)
        
        # Frame de resultados
        result_frame = ttk.LabelFrame(main_frame, text="📋 RESULTADOS E MÉTRICAS DE RISCO", padding="10")
        result_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        self.result_text = tk.Text(result_frame, height=12, width=70, font=('Consolas', 10))
        scrollbar = ttk.Scrollbar(result_frame, orient="vertical", command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scrollbar.set)
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Frame dos gráficos (lado direito)
        graph_frame = ttk.LabelFrame(main_frame, text="📊 VISUALIZAÇÕES", padding="10")
        graph_frame.grid(row=1, column=1, rowspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=5)
        
        # Criar gráficos com Canvas
        self.grafico_preco = GraficoCanvas(graph_frame, width=500, height=200)
        self.grafico_preco.canvas.grid(row=0, column=0, padx=5, pady=5)
        
        self.grafico_vol = GraficoCanvas(graph_frame, width=500, height=200)
        self.grafico_vol.canvas.grid(row=1, column=0, padx=5, pady=5)
        
        self.grafico_drawdown = GraficoCanvas(graph_frame, width=500, height=200)
        self.grafico_drawdown.canvas.grid(row=2, column=0, padx=5, pady=5)
        
        self.grafico_hist = GraficoCanvas(graph_frame, width=500, height=200)
        self.grafico_hist.canvas.grid(row=3, column=0, padx=5, pady=5)
        
        # Configurar grid
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=2)
        main_frame.rowconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)
        
        # Inicializar com valores padrão
        self.valores_padrao_b3()

    def valores_padrao_b3(self):
        """Configura valores padrão realistas para a bolsa brasileira"""
        padroes_b3 = {
            'preco_inicial': 35.0,
            'retorno_esperado': 0.12,
            'kappa': 1.8,
            'theta': 0.32,
            'xi': 0.28,
            'rho': -0.65,
            'vol_inicial': 0.28,
            'lambda_salto': 10.0,
            'mu_salto': -0.025,
            'sigma_salto': 0.035,
            'dias_simulacao': 180
        }
        
        for var_name, valor in padroes_b3.items():
            self.variaveis[var_name].set(valor)
        
        messagebox.showinfo("Valores Padrão", "Parâmetros configurados para cenário típico da B3!")

    def executar_simulacao(self):
        try:
            # Coletar parâmetros
            params = {k: v.get() for k, v in self.variaveis.items()}
            
            # Executar simulação
            S, v = self.modelo.modelo_heston_salto(
                S0=params['preco_inicial'],
                mu=params['retorno_esperado'],
                kappa=params['kappa'],
                theta=params['theta'],
                xi=params['xi'],
                rho=params['rho'],
                v0=params['vol_inicial'],
                lambda_j=params['lambda_salto'],
                mu_j=params['mu_salto'],
                sigma_j=params['sigma_salto'],
                dias=params['dias_simulacao']
            )
            
            # Calcular métricas
            prob_queda_10 = self.modelo.calcular_probabilidade_queda(S, 0.10)
            prob_queda_20 = self.modelo.calcular_probabilidade_queda(S, 0.20)
            prob_queda_30 = self.modelo.calcular_probabilidade_queda(S, 0.30)
            var_95 = self.modelo.calcular_var(S, 0.95)
            var_99 = self.modelo.calcular_var(S, 0.99)
            drawdown = self.modelo.calcular_drawdown(S)
            max_drawdown = drawdown.min() * 100
            
            # Atualizar resultados
            self.atualizar_resultados(S, v, prob_queda_10, prob_queda_20, prob_queda_30, 
                                    var_95, var_99, max_drawdown)
            self.plotar_resultados(S, v, drawdown)
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro na simulação: {str(e)}")
    
    def simulacao_monte_carlo(self):
        try:
            params = {k: v.get() for k, v in self.variaveis.items()}
            n_simulacoes = 500  # Reduzido para performance
            precos_finais = []
            max_drawdowns = []
            
            for i in range(n_simulacoes):
                S, _ = self.modelo.modelo_heston_salto(**params)
                precos_finais.append(S[-1])
                drawdown = self.modelo.calcular_drawdown(S)
                max_drawdowns.append(drawdown.min() * 100)
                
                # Atualizar progresso a cada 50 simulações
                if i % 50 == 0:
                    self.root.update()
            
            # Análise estatística
            preco_medio = np.mean(precos_finais)
            prob_perda = np.mean(np.array(precos_finais) < params['preco_inicial'])
            var_95_preco = np.percentile(precos_finais, 5)
            var_99_preco = np.percentile(precos_finais, 1)
            drawdown_medio = np.mean(max_drawdowns)
            
            # Atualizar resultados
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, 
                f"=== ANÁLISE MONTE CARLO ({n_simulacoes} SIMULAÇÕES) ===\n\n"
                f"📊 ESTATÍSTICAS DE PREÇO:\n"
                f"   Preço Inicial: R$ {params['preco_inicial']:.2f}\n"
                f"   Preço Médio Final: R$ {preco_medio:.2f}\n"
                f"   Retorno Esperado: {(preco_medio/params['preco_inicial']-1)*100:.1f}%\n\n"
                
                f"⚠️  PROBABILIDADES DE RISCO:\n"
                f"   Prob. de Perda: {prob_perda*100:.1f}%\n"
                f"   Drawdown Médio: {drawdown_medio:.1f}%\n\n"
                
                f"🎯 VALUE AT RISK (PREÇO):\n"
                f"   VaR 95%: R$ {var_95_preco:.2f}\n"
                f"   VaR 99%: R$ {var_99_preco:.2f}\n\n"
                
                f"📈 DISTRIBUIÇÃO DE RESULTADOS:\n"
                f"   Melhor Cenário: R$ {np.max(precos_finais):.2f}\n"
                f"   Pior Cenário: R$ {np.min(precos_finais):.2f}\n"
                f"   Volatilidade Resultados: {np.std(precos_finais):.2f}"
            )
            
            # Plotar distribuição
            self.grafico_hist.plotar_histograma(precos_finais, 'skyblue', 'Distribuição Preços Finais - Monte Carlo')
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro na simulação Monte Carlo: {str(e)}")
    
    def atualizar_resultados(self, S, v, prob_10, prob_20, prob_30, var_95, var_99, max_drawdown):
        retorno_total = (S[-1] - S[0]) / S[0] * 100
        volatilidade_media = v.mean() * 100
        
        # Classificar risco
        if prob_10 > 0.15 or prob_20 > 0.05:
            classificacao_risco = "ALTO RISCO 🔴"
        elif prob_10 > 0.10:
            classificacao_risco = "RISCO MODERADO 🟡"
        else:
            classificacao_risco = "RISCO BAIZO 🟢"
        
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END,
            f"=== RESULTADOS DA SIMULAÇÃO ===\n\n"
            f"💰 DESEMPENHO:\n"
            f"   Preço Inicial: R$ {S[0]:.2f}\n"
            f"   Preço Final: R$ {S[-1]:.2f}\n"
            f"   Retorno Total: {retorno_total:+.2f}%\n"
            f"   Volatilidade Média: {volatilidade_media:.1f}%\n\n"
            
            f"📉 MÁXIMA QUEDA:\n"
            f"   Drawdown Máximo: {max_drawdown:.2f}%\n\n"
            
            f"🎲 PROBABILIDADES DE QUEDA:\n"
            f"   Queda > 10%: {prob_10*100:.1f}%\n"
            f"   Queda > 20%: {prob_20*100:.1f}%\n"
            f"   Queda > 30%: {prob_30*100:.1f}%\n\n"
            
            f"⚠️  VALUE AT RISK (1 DIA):\n"
            f"   VaR 95%: {var_95*100:.2f}%\n"
            f"   VaR 99%: {var_99*100:.2f}%\n\n"
            
            f"📊 CLASSIFICAÇÃO: {classificacao_risco}\n"
        )
        
        # Colorir baseado no risco
        if "ALTO" in classificacao_risco:
            self.result_text.insert(tk.END, "Recomendação: Reduzir posição ou hedge\n")
        elif "MODERADO" in classificacao_risco:
            self.result_text.insert(tk.END, "Recomendação: Monitorar cuidadosamente\n")
        else:
            self.result_text.insert(tk.END, "Recomendação: Posição aceitável\n")
    
    def plotar_resultados(self, S, v, drawdown):
        # Plotar preço
        self.grafico_preco.plotar_linha(S, 'blue', 'EVOLUÇÃO DO PREÇO DA AÇÃO')
        
        # Plotar volatilidade (em porcentagem)
        v_percent = v * 100
        self.grafico_vol.plotar_linha(v_percent, 'red', 'VOLATILIDADE ESTOCÁSTICA (%)')
        
        # Plotar drawdown (em porcentagem)
        drawdown_percent = drawdown * 100
        self.grafico_drawdown.plotar_linha(drawdown_percent, 'darkred', 'DRAWDOWN (%)')
        
        # Plotar histograma de retornos
        retornos = np.diff(S) / S[:-1] * 100
        self.grafico_hist.plotar_histograma(retornos, 'green', 'DISTRIBUIÇÃO DOS RETORNOS DIÁRIOS (%)')
    
    def explicar_modelo(self):
        explicacao = """
🔍 MODELO MATEMÁTICO AVANÇADO PARA PREVISÃO DE QUEDAS
Desenvolvido por: Luiz Tiago Wilcke

📈 EQUAÇÕES DIFERENCIAIS ESTOCÁSTICAS UTILIZADAS:

1. 🎯 MODELO HESTON COM SALTOS:
   dSₜ = μSₜdt + √νₜSₜdWₜ¹ + JₜSₜdNₜ
   dνₜ = κ(θ - νₜ)dt + ξ√νₜdWₜ²

ONDE:
• Sₜ = Preço da ação no tempo t
• νₜ = Volatilidade estocástica (varia no tempo)
• μ = Retorno esperado anual
• κ = Velocidade de reversão à média da volatilidade
• θ = Volatilidade de longo prazo
• ξ = Volatilidade da volatilidade
• Jₜ = Saltos (eventos raros como crises)
• dWₜ¹, dWₜ² = Processos de Wiener correlacionados

2. 📊 MÉTRICAS CALCULADAS:
• Probabilidade de Queda: Chance de perdas > 10%, 20%, 30%
• Value at Risk (VaR): Perda máxima esperada com 95%/99% confiança
• Drawdown: Queda máxima em relação ao pico histórico
• Volatilidade: Medida de risco e variabilidade dos retornos

3. 🎲 SIMULAÇÃO MONTE CARLO:
• 500 cenários diferentes
• Análise estatística completa
• Distribuição de probabilidades dos resultados

🎯 ESTE MODELO CAPTURA:
✓ Cluster de volatilidade (vol agrupa no tempo)
✓ Efeito alavancagem (correlação negativa preço-vol)
✓ Eventos extremos (saltos - crises e notícias)
✓ Reversão à média da volatilidade
✓ Realismo de mercados emergentes como Brasil

📋 PARÂMETROS TÍPICOS B3:
• Volatilidade (θ): 30-40% (mais alto que mercados desenvolvidos)
• Correlação (ρ): -0.6 a -0.7 (efeito alavancagem forte)
• Frequência saltos (λ): 8-12 (mercado mais volátil)
"""
        messagebox.showinfo("Teoria do Modelo Matemático", explicacao)
    
    def limpar_resultados(self):
        self.result_text.delete(1.0, tk.END)
        # Limpar gráficos
        for grafico in [self.grafico_preco, self.grafico_vol, self.grafico_drawdown, self.grafico_hist]:
            grafico.canvas.delete("all")
        messagebox.showinfo("Limpeza", "Resultados e gráficos limpos!")

def main():
    try:
        root = tk.Tk()
        app = CalculadoraBolsa(root)
        root.mainloop()
    except Exception as e:
        print(f"Erro ao executar aplicação: {e}")
        print("Certifique-se de ter numpy instalado: pip install numpy")

if __name__ == "__main__":
    print("=" * 60)
    print("Sistema de Previsão de Quedas - Bolsa Brasileira")
    print("Autor: Luiz Tiago Wilcke")
    print("=" * 60)
    print("Instruções:")
    print("1. Ajuste os parâmetros do modelo")
    print("2. Clique em 'SIMULAR' para uma trajetória")
    print("3. Use 'MONTE CARLO' para análise estatística")
    print("4. Consulte 'EXPLICAR MODELO' para detalhes matemáticos")
    print("=" * 60)
    
    main()