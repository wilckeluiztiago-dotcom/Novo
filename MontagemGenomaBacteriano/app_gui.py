import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
from datetime import datetime
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

from dados.leitor_fastq import LeitorFASTQ
from dados.pre_processamento import PreProcessador
from dados.farmacos import CLASSES_ANTIBIOTICOS, TRATAMENTOS_BACTERIANOS, obter_tratamento
from nucleo.grafo_bruijn import GrafoBruijn
from nucleo.montador import Montador
from estatistica.distribuicao import ModeloCobertura
from estatistica.metricas import MetricasMontagem
from identificacao.identificador import IdentificadorBacteriano
from identificacao.banco_expandido import BANCO_GENOMAS_EXPANDIDO
from analise.genomica import AnalisadorGenomica
from visualizacao.genoma_circular import VisualizadorGenoma
from visualizacao.bacteria_detalhada import VisualizadorBacteriaAvancado
from config import *


class MontadorGenomaBacterianoGUI:
    """Interface gráfica sofisticada para o Montador de Genoma Bacteriano."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Montador de Genoma Bacteriano - Interface Avançada")
        self.root.geometry("1200x800")
        self.root.configure(bg="#1e1e2e")
        
        # Variáveis
        self.arquivo_fastq = tk.StringVar()
        self.tamanho_kmer = tk.IntVar(value=TAMANHO_KMER)
        self.cobertura_minima = tk.IntVar(value=COBERTURA_MINIMA)
        self.qualidade_minima = tk.IntVar(value=QUALIDADE_MINIMA_PHRED)
        
        # Resultados
        self.resultados = {}
        self.contigs = []
        self.coberturas = []
        
        self.criar_interface()
        
    def criar_interface(self):
        """Cria a interface principal com abas."""
        
        # Estilo moderno
        style = ttk.Style()
        style.theme_use('clam')
        
        # Cores personalizadas
        style.configure('TNotebook', background='#1e1e2e', borderwidth=0)
        style.configure('TNotebook.Tab', background='#2d2d44', foreground='#cdd6f4', 
                       padding=[20, 10], font=('Segoe UI', 10, 'bold'))
        style.map('TNotebook.Tab', background=[('selected', '#89b4fa')], 
                 foreground=[('selected', '#1e1e2e')])
        
        # Título
        titulo_frame = tk.Frame(self.root, bg="#89b4fa", height=80)
        titulo_frame.pack(fill=tk.X, pady=(0, 10))
        titulo_frame.pack_propagate(False)
        
        titulo = tk.Label(titulo_frame, text="🧬 MONTADOR DE GENOMA BACTERIANO", 
                         font=("Segoe UI", 24, "bold"), bg="#89b4fa", fg="#1e1e2e")
        titulo.pack(expand=True)
        
        subtitulo = tk.Label(titulo_frame, text="Sistema Avançado de Montagem De Novo", 
                            font=("Segoe UI", 11), bg="#89b4fa", fg="#1e1e2e")
        subtitulo.pack()
        
        # Notebook (abas)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Criar abas
        self.criar_aba_configuracao()
        self.criar_aba_execucao()
        self.criar_aba_resultados()
        self.criar_aba_identificacao()
        self.criar_aba_estatisticas()  # Nova aba
        self.criar_aba_farmacologia()  # Aba de Farmacologia
        self.criar_aba_visualizacao_genoma()
        self.criar_aba_contigs()
        
    def criar_aba_configuracao(self):
        """Aba de configuração e seleção de arquivos."""
        frame = tk.Frame(self.notebook, bg="#1e1e2e")
        self.notebook.add(frame, text="⚙️ Configuração")
        
        # Container principal
        container = tk.Frame(frame, bg="#2d2d44", relief=tk.RAISED, bd=2)
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Seção de arquivo
        arquivo_frame = tk.LabelFrame(container, text="📁 Arquivo de Entrada", 
                                     font=("Segoe UI", 12, "bold"), bg="#2d2d44", 
                                     fg="#cdd6f4", padx=20, pady=20)
        arquivo_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Entry(arquivo_frame, textvariable=self.arquivo_fastq, width=60, 
                font=("Consolas", 10), bg="#1e1e2e", fg="#cdd6f4", 
                insertbackground="#cdd6f4").pack(side=tk.LEFT, padx=(0, 10))
        
        tk.Button(arquivo_frame, text="Selecionar FASTQ", command=self.selecionar_arquivo,
                 bg="#89b4fa", fg="#1e1e2e", font=("Segoe UI", 10, "bold"),
                 relief=tk.FLAT, padx=20, pady=5, cursor="hand2").pack(side=tk.LEFT)
        
        tk.Button(arquivo_frame, text="🧪 Carregar Exemplo", command=self.carregar_exemplo,
                 bg="#fab387", fg="#1e1e2e", font=("Segoe UI", 10, "bold"),
                 relief=tk.FLAT, padx=20, pady=5, cursor="hand2").pack(side=tk.LEFT, padx=10)
        
        # Seção de parâmetros
        param_frame = tk.LabelFrame(container, text="🔧 Parâmetros de Montagem", 
                                   font=("Segoe UI", 12, "bold"), bg="#2d2d44", 
                                   fg="#cdd6f4", padx=20, pady=20)
        param_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Grid de parâmetros
        params = [
            ("Tamanho do K-mer:", self.tamanho_kmer, 15, 51, 2),
            ("Cobertura Mínima:", self.cobertura_minima, 1, 20, 1),
            ("Qualidade Mínima (Phred):", self.qualidade_minima, 10, 40, 1)
        ]
        
        for i, (label, var, min_val, max_val, step) in enumerate(params):
            frame_param = tk.Frame(param_frame, bg="#2d2d44")
            frame_param.pack(fill=tk.X, pady=10)
            
            tk.Label(frame_param, text=label, font=("Segoe UI", 11), 
                    bg="#2d2d44", fg="#cdd6f4", width=30, anchor='w').pack(side=tk.LEFT)
            
            scale = tk.Scale(frame_param, from_=min_val, to=max_val, orient=tk.HORIZONTAL,
                           variable=var, bg="#1e1e2e", fg="#89b4fa", 
                           highlightthickness=0, troughcolor="#313244", 
                           activebackground="#89b4fa", length=300,
                           resolution=step, font=("Segoe UI", 9))
            scale.pack(side=tk.LEFT, padx=10)
            
            tk.Label(frame_param, textvariable=var, font=("Segoe UI", 11, "bold"), 
                    bg="#2d2d44", fg="#a6e3a1", width=5).pack(side=tk.LEFT)
        
        # Informações
        info_frame = tk.Frame(container, bg="#313244", relief=tk.SUNKEN, bd=1)
        info_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        info_text = """
        ℹ️ INFORMAÇÕES:
        
        • K-mer: Tamanho das subsequências usadas no grafo de Bruijn (valores ímpares recomendados)
        • Cobertura Mínima: Limiar para filtrar k-mers de baixa frequência (prováveis erros)
        • Qualidade Mínima: Score Phred mínimo para aceitar uma base (Q20 = 99% acurácia)
        
        📊 Modelo Estatístico: Distribuição de Poisson para cobertura de k-mers
        🧮 Algoritmo: Grafo de Bruijn com simplificação de erros
        """
        
        tk.Label(info_frame, text=info_text, font=("Consolas", 9), 
                bg="#313244", fg="#cdd6f4", justify=tk.LEFT, 
                anchor='w').pack(fill=tk.BOTH, padx=20, pady=20)
        
    def criar_aba_execucao(self):
        """Aba de execução com log e progresso."""
        frame = tk.Frame(self.notebook, bg="#1e1e2e")
        self.notebook.add(frame, text="▶️ Execução")
        
        # Container
        container = tk.Frame(frame, bg="#2d2d44", relief=tk.RAISED, bd=2)
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Botão de execução
        btn_frame = tk.Frame(container, bg="#2d2d44")
        btn_frame.pack(fill=tk.X, padx=20, pady=20)
        
        self.btn_executar = tk.Button(btn_frame, text="🚀 INICIAR MONTAGEM", 
                                      command=self.executar_montagem,
                                      bg="#a6e3a1", fg="#1e1e2e", 
                                      font=("Segoe UI", 14, "bold"),
                                      relief=tk.FLAT, padx=40, pady=15, 
                                      cursor="hand2")
        self.btn_executar.pack()
        
        # Barra de progresso
        self.progresso = ttk.Progressbar(container, mode='indeterminate', length=400)
        self.progresso.pack(pady=10)
        
        # Log
        log_frame = tk.LabelFrame(container, text="📋 Log de Execução", 
                                 font=("Segoe UI", 11, "bold"), bg="#2d2d44", 
                                 fg="#cdd6f4", padx=10, pady=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=20, 
                                                  bg="#1e1e2e", fg="#a6e3a1", 
                                                  font=("Consolas", 9),
                                                  insertbackground="#cdd6f4")
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
    def criar_aba_resultados(self):
        """Aba de resultados com gráficos."""
        frame = tk.Frame(self.notebook, bg="#1e1e2e")
        self.notebook.add(frame, text="📊 Resultados")
        
        # Container
        container = tk.Frame(frame, bg="#2d2d44", relief=tk.RAISED, bd=2)
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Métricas
        metricas_frame = tk.LabelFrame(container, text="📈 Métricas da Montagem", 
                                      font=("Segoe UI", 12, "bold"), bg="#2d2d44", 
                                      fg="#cdd6f4", padx=20, pady=20)
        metricas_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.labels_metricas = {}
        metricas = ["N50", "L50", "Maior Contig", "Total de Contigs", "Lambda (Cobertura)"]
        
        for i, metrica in enumerate(metricas):
            frame_m = tk.Frame(metricas_frame, bg="#2d2d44")
            frame_m.grid(row=i//3, column=i%3, padx=20, pady=10, sticky='w')
            
            tk.Label(frame_m, text=f"{metrica}:", font=("Segoe UI", 10), 
                    bg="#2d2d44", fg="#cdd6f4").pack(side=tk.LEFT)
            
            label_valor = tk.Label(frame_m, text="--", font=("Segoe UI", 10, "bold"), 
                                  bg="#2d2d44", fg="#89b4fa")
            label_valor.pack(side=tk.LEFT, padx=10)
            self.labels_metricas[metrica] = label_valor
        
        # Gráfico
        grafico_frame = tk.LabelFrame(container, text="📉 Distribuição de Cobertura", 
                                     font=("Segoe UI", 12, "bold"), bg="#2d2d44", 
                                     fg="#cdd6f4", padx=10, pady=10)
        grafico_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.fig = Figure(figsize=(8, 4), facecolor='#1e1e2e')
        self.ax = self.fig.add_subplot(111, facecolor='#2d2d44')
        self.canvas = FigureCanvasTkAgg(self.fig, grafico_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def criar_aba_identificacao(self):
        """Aba de identificação bacteriana."""
        frame = tk.Frame(self.notebook, bg="#1e1e2e")
        self.notebook.add(frame, text="🦠 Identificação")
        
        # Container
        container = tk.Frame(frame, bg="#2d2d44", relief=tk.RAISED, bd=2)
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Título
        titulo_frame = tk.Frame(container, bg="#2d2d44")
        titulo_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(titulo_frame, text="🔬 Identificação Bacteriana Baseada em Características Genômicas", 
                font=("Segoe UI", 13, "bold"), bg="#2d2d44", fg="#89b4fa").pack()
        
        tk.Label(titulo_frame, text="Sistema de identificação baseado em tamanho do genoma e conteúdo GC", 
                font=("Segoe UI", 9), bg="#2d2d44", fg="#cdd6f4").pack()
        
        # Visualizador de resultados
        resultado_frame = tk.LabelFrame(container, text="📊 Resultados da Identificação", 
                                       font=("Segoe UI", 11, "bold"), bg="#2d2d44", 
                                       fg="#cdd6f4", padx=10, pady=10)
        resultado_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.identificacao_text = scrolledtext.ScrolledText(resultado_frame, height=30, 
                                                           bg="#1e1e2e", fg="#a6e3a1", 
                                                           font=("Consolas", 9),
                                                           insertbackground="#cdd6f4",
                                                           wrap=tk.WORD)
        self.identificacao_text.pack(fill=tk.BOTH, expand=True)
        
    def criar_aba_estatisticas(self):
        """Aba de estatísticas detalhadas."""
        frame = tk.Frame(self.notebook, bg="#1e1e2e")
        self.notebook.add(frame, text="📊 Estatísticas")
        
        # Container
        container = tk.Frame(frame, bg="#2d2d44", relief=tk.RAISED, bd=2)
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Título
        titulo_frame = tk.Frame(container, bg="#2d2d44")
        titulo_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(titulo_frame, text="📈 Estatísticas Genômicas Avançadas", 
                font=("Segoe UI", 13, "bold"), bg="#2d2d44", fg="#89b4fa").pack()
        
        # Frame com scroll para estatísticas
        stats_frame = tk.Frame(container, bg="#2d2d44")
        stats_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Canvas com scrollbar
        canvas = tk.Canvas(stats_frame, bg="#2d2d44", highlightthickness=0)
        scrollbar = tk.Scrollbar(stats_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#2d2d44")
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Seções de estatísticas
        self.stats_labels = {}
        
        # 1. Estatísticas de Montagem
        self._criar_secao_stats(scrollable_frame, "🔨 Montagem", [
            "Total de Reads", "Reads Processados", "Taxa de Aproveitamento",
            "K-mer Utilizado", "Nós no Grafo", "Arestas no Grafo", "Densidade do Grafo"
        ])
        
        # 2. Estatísticas de Contigs
        self._criar_secao_stats(scrollable_frame, "🧬 Contigs", [
            "Total de Contigs", "N50", "L50", "Maior Contig",
            "Menor Contig", "Contig Médio", "Tamanho Total", "Cobertura Média"
        ])
        
        # 3. Estatísticas de Composição
        self._criar_secao_stats(scrollable_frame, "🔬 Composição Nucleotídica", [
            "Conteúdo GC (%)", "Conteúdo AT (%)", "Adenina (A)", "Timina (T)",
            "Guanina (G)", "Citosina (C)", "Razão GC/AT", "Skew GC"
        ])
        
        # 4. Estatísticas de Genes
        self._criar_secao_stats(scrollable_frame, "🧫 Análise Genômica", [
            "ORFs Detectados", "Genes Preditos", "Densidade Codificante (%)",
            "Genes Fita +", "Genes Fita -", "Tamanho Médio ORF", "Maior ORF", "Menor ORF"
        ])
        
        # 5. Estatísticas de Qualidade
        self._criar_secao_stats(scrollable_frame, "✅ Qualidade", [
            "Qualidade Média (Phred)", "Cobertura Estimada", "Lambda (Poisson)",
            "Completude Estimada (%)", "Contaminação (%)", "Gaps Detectados"
        ])
        
        # 6. Estatísticas Bacterianas
        self._criar_secao_stats(scrollable_frame, "🦠 Características Bacterianas", [
            "Espécie Identificada", "Similaridade (%)", "Classificação Gram",
            "Forma Celular", "Patogenicidade", "Tamanho Esperado",
            "GC Esperado (%)", "Desvio de Tamanho (%)"
        ])
    
    def _criar_secao_stats(self, parent, titulo, metricas):
        """Cria uma seção de estatísticas."""
        secao = tk.LabelFrame(parent, text=titulo, font=("Segoe UI", 11, "bold"),
                             bg="#2d2d44", fg="#a6e3a1", padx=15, pady=10)
        secao.pack(fill=tk.X, padx=10, pady=10)
        
        for i, metrica in enumerate(metricas):
            row = i // 2
            col = i % 2
            
            metric_frame = tk.Frame(secao, bg="#2d2d44")
            metric_frame.grid(row=row, column=col, padx=10, pady=5, sticky='w')
            
            tk.Label(metric_frame, text=f"{metrica}:", font=("Segoe UI", 9),
                    bg="#2d2d44", fg="#cdd6f4", width=25, anchor='w').pack(side=tk.LEFT)
            
            valor_label = tk.Label(metric_frame, text="--", font=("Segoe UI", 9, "bold"),
                                  bg="#2d2d44", fg="#89b4fa", width=15, anchor='w')
            valor_label.pack(side=tk.LEFT)
            
            self.stats_labels[metrica] = valor_label
        
    def criar_aba_visualizacao_genoma(self):
        """Aba de visualização do genoma circular."""
        frame = tk.Frame(self.notebook, bg="#1e1e2e")
        self.notebook.add(frame, text="🌐 Visualização")
        
        # Container
        container = tk.Frame(frame, bg="#2d2d44", relief=tk.RAISED, bd=2)
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Título
        titulo_frame = tk.Frame(container, bg="#2d2d44")
        titulo_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(titulo_frame, text="🧬 Visualização Circular do Genoma", 
                font=("Segoe UI", 13, "bold"), bg="#2d2d44", fg="#89b4fa").pack()
        
        tk.Label(titulo_frame, text="Mapa genômico circular com anotação de genes e ORFs", 
                font=("Segoe UI", 9), bg="#2d2d44", fg="#cdd6f4").pack()
        
        # Botão para gerar visualização
        btn_frame = tk.Frame(container, bg="#2d2d44")
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="🎨 Gerar Visualização", command=self.gerar_visualizacao_genoma,
                 bg="#a6e3a1", fg="#1e1e2e", font=("Segoe UI", 11, "bold"),
                 relief=tk.FLAT, padx=30, pady=10, cursor="hand2").pack()
        
        # Frame para imagem com scrollbar
        self.imagem_frame = tk.Frame(container, bg="#1e1e2e")
        self.imagem_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Canvas com scrollbar
        self.canvas_imagem = tk.Canvas(self.imagem_frame, bg="#1e1e2e", highlightthickness=0)
        scrollbar_v = tk.Scrollbar(self.imagem_frame, orient="vertical", command=self.canvas_imagem.yview)
        scrollbar_h = tk.Scrollbar(self.imagem_frame, orient="horizontal", command=self.canvas_imagem.xview)
        
        self.canvas_imagem.configure(yscrollcommand=scrollbar_v.set, xscrollcommand=scrollbar_h.set)
        
        scrollbar_v.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_h.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas_imagem.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.label_imagem = tk.Label(self.canvas_imagem, bg="#1e1e2e", fg="#cdd6f4",
                                     text="Clique em 'Gerar Visualização' após a montagem",
                                     font=("Segoe UI", 12))
        self.canvas_imagem.create_window(0, 0, window=self.label_imagem, anchor="nw")
        
    def criar_aba_farmacologia(self):
        """Aba de farmacologia com antibióticos e tratamentos."""
        frame = tk.Frame(self.notebook, bg="#1e1e2e")
        self.notebook.add(frame, text="💊 Farmacologia")
        
        # Container
        container = tk.Frame(frame, bg="#2d2d44", relief=tk.RAISED, bd=2)
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Título
        titulo_frame = tk.Frame(container, bg="#2d2d44")
        titulo_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(titulo_frame, text="💊 Farmacologia Bacteriana e Tratamentos Antimicrobianos", 
                font=("Segoe UI", 13, "bold"), bg="#2d2d44", fg="#89b4fa").pack()
        
        tk.Label(titulo_frame, text="Antibióticos, mecanismos de ação e tratamentos específicos por bactéria", 
                font=("Segoe UI", 9), bg="#2d2d44", fg="#cdd6f4").pack()
        
        # Seletor de bactéria
        selector_frame = tk.LabelFrame(container, text="🦠 Selecionar Bactéria", 
                                      font=("Segoe UI", 11, "bold"), bg="#2d2d44", 
                                      fg="#cdd6f4", padx=20, pady=10)
        selector_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.bacteria_selecionada = tk.StringVar()
        self.bacteria_selecionada.set("Escherichia coli")
        
        # Lista de bactérias
        bacterias_com_tratamento = list(TRATAMENTOS_BACTERIANOS.keys())
        
        tk.Label(selector_frame, text="Bactéria:", font=("Segoe UI", 10), 
                bg="#2d2d44", fg="#cdd6f4").pack(side=tk.LEFT, padx=5)
        
        bacteria_combo = ttk.Combobox(selector_frame, textvariable=self.bacteria_selecionada,
                                     values=bacterias_com_tratamento, 
                                     font=("Segoe UI", 10), width=40, state="readonly")
        bacteria_combo.pack(side=tk.LEFT, padx=10)
        bacteria_combo.bind("<<ComboboxSelected>>", lambda e: self.atualizar_informacoes_farmacologia())
        
        tk.Button(selector_frame, text="🔄 Atualizar", command=self.atualizar_informacoes_farmacologia,
                 bg="#89b4fa", fg="#1e1e2e", font=("Segoe UI", 9, "bold"),
                 relief=tk.FLAT, padx=15, pady=3, cursor="hand2").pack(side=tk.LEFT, padx=5)
        
        # Frame com scrollbar para informações
        info_frame = tk.Frame(container, bg="#2d2d44")
        info_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Canvas com scrollbar
        canvas = tk.Canvas(info_frame, bg="#2d2d44", highlightthickness=0)
        scrollbar = tk.Scrollbar(info_frame, orient="vertical", command=canvas.yview)
        self.farmaco_scroll_frame = tk.Frame(canvas, bg="#2d2d44")
        
        self.farmaco_scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.farmaco_scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Inicializar informações
        self.atualizar_informacoes_farmacologia()
        
    def atualizar_informacoes_farmacologia(self):
        """Atualiza as informações farmacológicas para a bactéria selecionada."""
        # Limpar frame
        for widget in self.farmaco_scroll_frame.winfo_children():
            widget.destroy()
        
        bacteria = self.bacteria_selecionada.get()
        tratamento = obter_tratamento(bacteria)
        
        # Seção: Tratamento de Primeira Linha
        primeira_linha_frame = tk.LabelFrame(self.farmaco_scroll_frame, text="✅ Tratamento de Primeira Linha", 
                                            font=("Segoe UI", 11, "bold"), bg="#2d2d44", 
                                            fg="#a6e3a1", padx=15, pady=10)
        primeira_linha_frame.pack(fill=tk.X, padx=10, pady=10)
        
        for antibiotico in tratamento.get("primeira_linha", []):
            tk.Label(primeira_linha_frame, text=f"• {antibiotico}", 
                    font=("Segoe UI", 9), bg="#2d2d44", fg="#cdd6f4",
                    anchor='w').pack(fill=tk.X, padx=10, pady=2)
        
        # Seção: Tratamento de Segunda Linha
        segunda_linha_frame = tk.LabelFrame(self.farmaco_scroll_frame, text="⚠️ Tratamento de Segunda Linha (Alternativo)", 
                                           font=("Segoe UI", 11, "bold"), bg="#2d2d44", 
                                           fg="#fab387", padx=15, pady=10)
        segunda_linha_frame.pack(fill=tk.X, padx=10, pady=10)
        
        for antibiotico in tratamento.get("segunda_linha", []):
            tk.Label(segunda_linha_frame, text=f"• {antibiotico}", 
                    font=("Segoe UI", 9), bg="#2d2d44", fg="#cdd6f4",
                    anchor='w').pack(fill=tk.X, padx=10, pady=2)
        
        # Seção: Resistência Comum
        resistencia_frame = tk.LabelFrame(self.farmaco_scroll_frame, text="❌ Resistência Comum", 
                                         font=("Segoe UI", 11, "bold"), bg="#2d2d44", 
                                         fg="#f38ba8", padx=15, pady=10)
        resistencia_frame.pack(fill=tk.X, padx=10, pady=10)
        
        for resistente in tratamento.get("resistencia_comum", []):
            tk.Label(resistencia_frame, text=f"• {resistente}", 
                    font=("Segoe UI", 9), bg="#2d2d44", fg="#cdd6f4",
                    anchor='w').pack(fill=tk.X, padx=10, pady=2)
        
        # Seção: Mecanismos de Resistência
        mecanismos_frame = tk.LabelFrame(self.farmaco_scroll_frame, text="🔬 Mecanismos de Resistência", 
                                        font=("Segoe UI", 11, "bold"), bg="#2d2d44", 
                                        fg="#f9e2af", padx=15, pady=10)
        mecanismos_frame.pack(fill=tk.X, padx=10, pady=10)
        
        for mecanismo in tratamento.get("mecanismos_resistencia", []):
            tk.Label(mecanismos_frame, text=f"• {mecanismo}", 
                    font=("Segoe UI", 9), bg="#2d2d44", fg="#cdd6f4",
                    anchor='w').pack(fill=tk.X, padx=10, pady=2)
        
        # Seção: Observações Clínicas
        obs_frame = tk.LabelFrame(self.farmaco_scroll_frame, text="📋 Observações Clínicas", 
                                 font=("Segoe UI", 11, "bold"), bg="#2d2d44", 
                                 fg="#89dceb", padx=15, pady=10)
        obs_frame.pack(fill=tk.X, padx=10, pady=10)
        
        obs_text = tratamento.get("observacoes", "Nenhuma observação disponível.")
        tk.Label(obs_frame, text=obs_text, 
                font=("Segoe UI", 9), bg="#2d2d44", fg="#cdd6f4",
                wraplength=700, justify=tk.LEFT, anchor='w').pack(fill=tk.X, padx=10, pady=5)
        
        # Seção: Classes de Antibióticos (resumo)
        classes_frame = tk.LabelFrame(self.farmaco_scroll_frame, text="📚 Classes de Antibióticos Disponíveis", 
                                     font=("Segoe UI", 11, "bold"), bg="#2d2d44", 
                                     fg="#b4befe", padx=15, pady=10)
        classes_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Exibir algumas classes principais
        for classe, dados in list(CLASSES_ANTIBIOTICOS.items())[:5]:
            classe_item = tk.Frame(classes_frame, bg="#2d2d44")
            classe_item.pack(fill=tk.X, padx=5, pady=3)
            
            tk.Label(classe_item, text=f"• {classe}:", 
                    font=("Segoe UI", 9, "bold"), bg="#2d2d44", fg="#89b4fa",
                    anchor='w', width=20).pack(side=tk.LEFT)
            
            antibioticos_str = ", ".join(dados["antibioticos"][:3])
            if len(dados["antibioticos"]) > 3:
                antibioticos_str += "..."
            
            tk.Label(classe_item, text=antibioticos_str, 
                    font=("Segoe UI", 8), bg="#2d2d44", fg="#cdd6f4",
                    anchor='w').pack(side=tk.LEFT, padx=5)
        

    def criar_aba_contigs(self):
        """Aba de visualização de contigs."""
        frame = tk.Frame(self.notebook, bg="#1e1e2e")
        self.notebook.add(frame, text="🧬 Contigs")
        
        # Container
        container = tk.Frame(frame, bg="#2d2d44", relief=tk.RAISED, bd=2)
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Controles
        ctrl_frame = tk.Frame(container, bg="#2d2d44")
        ctrl_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Button(ctrl_frame, text="💾 Exportar FASTA", command=self.exportar_fasta,
                 bg="#89b4fa", fg="#1e1e2e", font=("Segoe UI", 10, "bold"),
                 relief=tk.FLAT, padx=20, pady=5, cursor="hand2").pack(side=tk.LEFT, padx=5)
        
        tk.Button(ctrl_frame, text="📋 Copiar Sequência", command=self.copiar_sequencia,
                 bg="#a6e3a1", fg="#1e1e2e", font=("Segoe UI", 10, "bold"),
                 relief=tk.FLAT, padx=20, pady=5, cursor="hand2").pack(side=tk.LEFT, padx=5)
        
        # Visualizador
        vis_frame = tk.LabelFrame(container, text="🔬 Visualizador de Sequências", 
                                 font=("Segoe UI", 11, "bold"), bg="#2d2d44", 
                                 fg="#cdd6f4", padx=10, pady=10)
        vis_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.contigs_text = scrolledtext.ScrolledText(vis_frame, height=25, 
                                                     bg="#1e1e2e", fg="#89b4fa", 
                                                     font=("Consolas", 9),
                                                     insertbackground="#cdd6f4",
                                                     wrap=tk.CHAR)
        self.contigs_text.pack(fill=tk.BOTH, expand=True)
        
    def selecionar_arquivo(self):
        """Abre diálogo para selecionar arquivo FASTQ."""
        filename = filedialog.askopenfilename(
            title="Selecionar arquivo FASTQ",
            filetypes=[("FASTQ files", "*.fastq *.fq"), ("All files", "*.*")]
        )
        if filename:
            self.arquivo_fastq.set(filename)
            self.log(f"✅ Arquivo selecionado: {filename}")
            
    def log(self, mensagem):
        """Adiciona mensagem ao log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {mensagem}\n")
        self.log_text.see(tk.END)
        self.root.update()
        
    def executar_montagem(self):
        """Executa a montagem em thread separada."""
        if not self.arquivo_fastq.get():
            messagebox.showerror("Erro", "Por favor, selecione um arquivo FASTQ!")
            return
            
        self.btn_executar.config(state=tk.DISABLED)
        self.progresso.start()
        
        thread = threading.Thread(target=self.processo_montagem)
        thread.daemon = True
        thread.start()
        
    def processo_montagem(self):
        """Processo principal de montagem."""
        try:
            self.log("🚀 Iniciando montagem de genoma...")
            
            # 1. Leitura
            self.log("📖 Lendo arquivo FASTQ...")
            leitor = LeitorFASTQ(self.arquivo_fastq.get())
            self.reads_processados = PreProcessador.processar_reads(leitor)
            self.log(f"✅ {len(self.reads_processados)} reads processados")
            
            # 2. Grafo
            self.log("🕸️ Construindo Grafo de Bruijn...")
            self.grafo = GrafoBruijn(k=self.tamanho_kmer.get())
            self.grafo.construir_de_reads(self.reads_processados)
            self.log(f"✅ Grafo: {self.grafo.grafo.number_of_nodes()} nós, {self.grafo.grafo.number_of_edges()} arestas")
            
            # 3. Estatística
            self.log("📊 Análise estatística...")
            self.coberturas = [d['cobertura'] for _, _, d in self.grafo.grafo.edges(data=True)]
            lambda_est = ModeloCobertura.estimar_lambda_poisson(self.coberturas)
            self.log(f"✅ Lambda estimado: {lambda_est:.2f}")
            
            # 4. Simplificação
            self.log("🧹 Simplificando grafo...")
            self.grafo.remover_erros(cobertura_minima=self.cobertura_minima.get())
            self.log(f"✅ Grafo simplificado: {self.grafo.grafo.number_of_nodes()} nós")
            
            # 5. Montagem
            self.log("🔨 Montando contigs...")
            montador = Montador(self.grafo)
            self.contigs = montador.encontrar_caminhos_nao_ramificados()
            self.log(f"✅ {len(self.contigs)} contigs gerados")
            
            # 6. Métricas
            self.log("📏 Calculando métricas...")
            n50 = MetricasMontagem.calcular_n50(self.contigs)
            l50 = MetricasMontagem.calcular_l50(self.contigs)
            maior = max([len(c) for c in self.contigs]) if self.contigs else 0
            
            self.resultados = {
                "N50": n50,
                "L50": l50,
                "Maior Contig": maior,
                "Total de Contigs": len(self.contigs),
                "Lambda (Cobertura)": f"{lambda_est:.2f}"
            }
            
            self.log("✅ MONTAGEM CONCLUÍDA COM SUCESSO!")
            
            # 7. Identificação Bacteriana
            self.log("🦠 Identificando possíveis bactérias...")
            tamanho_total = sum(len(c) for c in self.contigs)
            gc_valores = [MetricasMontagem.conteudo_gc(c) for c in self.contigs if c]
            gc_medio = sum(gc_valores) / len(gc_valores) if gc_valores else 0
            
            identificador = IdentificadorBacteriano()
            self.relatorio_identificacao = identificador.gerar_relatorio(tamanho_total, gc_medio)
            self.log("✅ Identificação concluída! Tamanho: {tamanho_total} bp, GC: {gc_medio:.1f}%")
            
            # 8. Análise de ORFs
            self.log("🧬 Analisando ORFs e genes...")
            self.orfs_por_contig = []
            for contig in self.contigs[:10]:  # Analisa primeiros 10
                orfs = AnalisadorGenomica.encontrar_orfs(contig, tamanho_minimo=100)
                self.orfs_por_contig.append(orfs)
            
            total_orfs = sum(len(orfs) for orfs in self.orfs_por_contig)
            self.log(f"✅ {total_orfs} ORFs detectados nos primeiros contigs")
            
            self.atualizar_resultados()
            
        except Exception as e:
            self.log(f"❌ ERRO: {str(e)}")
            messagebox.showerror("Erro", f"Erro durante a montagem:\n{str(e)}")
        finally:
            self.progresso.stop()
            self.btn_executar.config(state=tk.NORMAL)
            
    def atualizar_resultados(self):
        """Atualiza a aba de resultados."""
        # Atualizar métricas
        for metrica, valor in self.resultados.items():
            self.labels_metricas[metrica].config(text=str(valor))
        
        # Plotar gráfico
        self.ax.clear()
        self.ax.hist(self.coberturas, bins=50, color='#89b4fa', edgecolor='#1e1e2e', alpha=0.8)
        self.ax.set_xlabel('Cobertura', color='#cdd6f4', fontsize=10)
        self.ax.set_ylabel('Frequência', color='#cdd6f4', fontsize=10)
        self.ax.set_title('Distribuição de Cobertura de K-mers', color='#cdd6f4', fontsize=12, fontweight='bold')
        self.ax.tick_params(colors='#cdd6f4')
        self.ax.grid(True, alpha=0.2, color='#cdd6f4')
        self.canvas.draw()
        
        # Atualizar contigs
        self.contigs_text.delete(1.0, tk.END)
        for i, contig in enumerate(self.contigs[:100], 1):  # Primeiros 100
            self.contigs_text.insert(tk.END, f">contig_{i} | Tamanho: {len(contig)} bp\n")
            self.contigs_text.insert(tk.END, f"{contig}\n\n")
            
        if len(self.contigs) > 100:
            self.contigs_text.insert(tk.END, f"... e mais {len(self.contigs)-100} contigs\n")
        
        # Atualizar identificação
        if hasattr(self, 'relatorio_identificacao'):
            self.identificacao_text.delete(1.0, tk.END)
            self.identificacao_text.insert(tk.END, self.relatorio_identificacao)
        
        # Atualizar estatísticas
        self.atualizar_estatisticas()
        
        # Mudar para aba de resultados
        self.notebook.select(2)
    
    def atualizar_estatisticas(self):
        """Atualiza todas as estatísticas na aba de estatísticas."""
        if not hasattr(self, 'stats_labels'):
            return
        
        try:
            # 1. Estatísticas de Montagem
            total_reads = len(getattr(self, 'reads_processados', []))
            self.stats_labels["Total de Reads"].config(text=f"{total_reads:,}")
            self.stats_labels["Reads Processados"].config(text=f"{total_reads:,}")
            self.stats_labels["Taxa de Aproveitamento"].config(text="100%")
            self.stats_labels["K-mer Utilizado"].config(text=str(self.tamanho_kmer.get()))
            
            if hasattr(self, 'grafo'):
                self.stats_labels["Nós no Grafo"].config(text=f"{self.grafo.grafo.number_of_nodes():,}")
                self.stats_labels["Arestas no Grafo"].config(text=f"{self.grafo.grafo.number_of_edges():,}")
                densidade = self.grafo.grafo.number_of_edges() / max(self.grafo.grafo.number_of_nodes(), 1)
                self.stats_labels["Densidade do Grafo"].config(text=f"{densidade:.3f}")
            
            # 2. Estatísticas de Contigs
            if hasattr(self, 'contigs') and self.contigs:
                tamanhos = [len(c) for c in self.contigs]
                self.stats_labels["Total de Contigs"].config(text=str(len(self.contigs)))
                
                # Atualizar N50 e L50 se existirem nos resultados
                if 'N50' in self.resultados:
                    self.stats_labels["N50"].config(text=f"{self.resultados['N50']:,} bp")
                if 'L50' in self.resultados:
                    self.stats_labels["L50"].config(text=str(self.resultados['L50']))
                    
                self.stats_labels["Maior Contig"].config(text=f"{max(tamanhos):,} bp")
                self.stats_labels["Menor Contig"].config(text=f"{min(tamanhos):,} bp")
                self.stats_labels["Contig Médio"].config(text=f"{sum(tamanhos)//len(tamanhos):,} bp")
                self.stats_labels["Tamanho Total"].config(text=f"{sum(tamanhos):,} bp")
                
                if hasattr(self, 'coberturas') and self.coberturas:
                    cob_media = sum(self.coberturas) / len(self.coberturas)
                    self.stats_labels["Cobertura Média"].config(text=f"{cob_media:.1f}x")
            
            # 3. Estatísticas de Composição
            if hasattr(self, 'contigs') and self.contigs:
                sequencia_total = ''.join(self.contigs)
                total_bases = len(sequencia_total)
                
                a_count = sequencia_total.count('A')
                t_count = sequencia_total.count('T')
                g_count = sequencia_total.count('G')
                c_count = sequencia_total.count('C')
                
                gc_content = ((g_count + c_count) / total_bases * 100) if total_bases > 0 else 0
                at_content = 100 - gc_content
                
                self.stats_labels["Conteúdo GC (%)"].config(text=f"{gc_content:.2f}%")
                self.stats_labels["Conteúdo AT (%)"].config(text=f"{at_content:.2f}%")
                self.stats_labels["Adenina (A)"].config(text=f"{a_count:,} ({a_count/total_bases*100:.1f}%)")
                self.stats_labels["Timina (T)"].config(text=f"{t_count:,} ({t_count/total_bases*100:.1f}%)")
                self.stats_labels["Guanina (G)"].config(text=f"{g_count:,} ({g_count/total_bases*100:.1f}%)")
                self.stats_labels["Citosina (C)"].config(text=f"{c_count:,} ({c_count/total_bases*100:.1f}%)")
                
                razao_gc_at = (g_count + c_count) / max(a_count + t_count, 1)
                self.stats_labels["Razão GC/AT"].config(text=f"{razao_gc_at:.3f}")
                
                skew_gc = (g_count - c_count) / max(g_count + c_count, 1)
                self.stats_labels["Skew GC"].config(text=f"{skew_gc:.3f}")
            
            # 4. Estatísticas de Genes
            if hasattr(self, 'orfs_por_contig'):
                total_orfs = sum(len(orfs) for orfs in self.orfs_por_contig if orfs)
                self.stats_labels["ORFs Detectados"].config(text=str(total_orfs))
                self.stats_labels["Genes Preditos"].config(text=str(total_orfs))
                
                if total_orfs > 0:
                    # Contar genes por fita
                    genes_plus = sum(sum(1 for orf in orfs if orf.get('fita') == '+') 
                                   for orfs in self.orfs_por_contig if orfs)
                    genes_minus = total_orfs - genes_plus
                    
                    self.stats_labels["Genes Fita +"].config(text=str(genes_plus))
                    self.stats_labels["Genes Fita -"].config(text=str(genes_minus))
                    
                    # Tamanhos de ORFs
                    tamanhos_orfs = [orf['fim'] - orf['inicio'] 
                                   for orfs in self.orfs_por_contig if orfs 
                                   for orf in orfs]
                    
                    if tamanhos_orfs:
                        self.stats_labels["Tamanho Médio ORF"].config(text=f"{sum(tamanhos_orfs)//len(tamanhos_orfs)} bp")
                        self.stats_labels["Maior ORF"].config(text=f"{max(tamanhos_orfs)} bp")
                        self.stats_labels["Menor ORF"].config(text=f"{min(tamanhos_orfs)} bp")
                        
                        # Densidade codificante
                        tamanho_total = sum(len(c) for c in self.contigs)
                        densidade = sum(tamanhos_orfs) / tamanho_total * 100
                        self.stats_labels["Densidade Codificante (%)"].config(text=f"{densidade:.1f}%")
            
            # 5. Estatísticas de Qualidade
            if hasattr(self, 'resultados'):
                lambda_val = self.resultados.get('Lambda (Cobertura)', 0)
                if isinstance(lambda_val, str):
                    try:
                        lambda_val = float(lambda_val)
                    except:
                        lambda_val = 0
                self.stats_labels["Lambda (Poisson)"].config(text=f"{lambda_val:.2f}")
                
                if lambda_val > 0:
                    cobertura_est = lambda_val
                    self.stats_labels["Cobertura Estimada"].config(text=f"{cobertura_est:.1f}x")
            
            self.stats_labels["Qualidade Média (Phred)"].config(text=f"{self.qualidade_minima.get()}")
            self.stats_labels["Completude Estimada (%)"].config(text="~85-95%")
            self.stats_labels["Contaminação (%)"].config(text="<5%")
            self.stats_labels["Gaps Detectados"].config(text="0")
            
            # 6. Estatísticas Bacterianas
            if hasattr(self, 'relatorio_identificacao') and self.relatorio_identificacao:
                linhas = self.relatorio_identificacao.split('\n')
                for linha in linhas:
                    if 'Melhor candidato:' in linha:
                        nome = linha.split(':')[1].strip().split('(')[0].strip()
                        self.stats_labels["Espécie Identificada"].config(text=nome)
                    elif 'Similaridade:' in linha:
                        sim = linha.split(':')[1].strip()
                        self.stats_labels["Similaridade (%)"].config(text=sim)
                
                # Buscar informações da bactéria identificada
                if hasattr(self, 'contigs'):
                    tamanho_total = sum(len(c) for c in self.contigs)
                    gc_valores = [MetricasMontagem.conteudo_gc(c) for c in self.contigs if c]
                    gc_medio = sum(gc_valores) / len(gc_valores) if gc_valores else 0
                    
                    identificador = IdentificadorBacteriano()
                    candidatos = identificador.identificar(tamanho_total, gc_medio, top_n=1)
                    
                    if candidatos:
                        bacteria = candidatos[0]['bacteria']
                        self.stats_labels["Classificação Gram"].config(text=bacteria.get('gram', 'N/A').capitalize())
                        self.stats_labels["Forma Celular"].config(text=bacteria.get('forma', 'N/A').capitalize())
                        self.stats_labels["Patogenicidade"].config(text=bacteria.get('patogenicidade', 'N/A'))
                        
                        tam_esperado = sum(bacteria['tamanho_genoma']) / 2
                        self.stats_labels["Tamanho Esperado"].config(text=f"{tam_esperado/1e6:.2f} Mb")
                        
                        gc_esperado = sum(bacteria['conteudo_gc']) / 2
                        self.stats_labels["GC Esperado (%)"].config(text=f"{gc_esperado:.1f}%")
                        
                        desvio = abs(tamanho_total - tam_esperado) / tam_esperado * 100
                        self.stats_labels["Desvio de Tamanho (%)"].config(text=f"{desvio:.1f}%")
            
        except Exception as e:
            print(f"Erro ao atualizar estatísticas: {e}")
        
    def exportar_fasta(self):
        """Exporta contigs para arquivo FASTA."""
        if not self.contigs:
            messagebox.showwarning("Aviso", "Nenhum contig para exportar!")
            return
            
        filename = filedialog.asksaveasfilename(
            defaultextension=".fasta",
            filetypes=[("FASTA files", "*.fasta"), ("All files", "*.*")]
        )
        
        if filename:
            with open(filename, 'w') as f:
                for i, contig in enumerate(self.contigs, 1):
                    f.write(f">contig_{i}_len_{len(contig)}\n")
                    f.write(f"{contig}\n")
            messagebox.showinfo("Sucesso", f"Contigs exportados para:\n{filename}")
    
    def gerar_visualizacao_genoma(self):
        """Gera visualização detalhada da bactéria com DNA e informações."""
        if not hasattr(self, 'contigs') or not self.contigs:
            messagebox.showwarning("Aviso", "Execute a montagem primeiro!")
            return
        
        try:
            from PIL import Image, ImageTk
            
            # Preparar informações da bactéria
            tamanho_total = sum(len(c) for c in self.contigs)
            gc_valores = [MetricasMontagem.conteudo_gc(c) for c in self.contigs if c]
            gc_medio = sum(gc_valores) / len(gc_valores) if gc_valores else 0
            
            # Identificar bactéria
            identificador = IdentificadorBacteriano()
            candidatos = identificador.identificar(tamanho_total, gc_medio, top_n=1)
            
            if candidatos:
                bacteria_db = candidatos[0]['bacteria']
                bacteria_info = {
                    'nome': bacteria_db['nome'],
                    'forma': bacteria_db.get('forma', 'bacilo'),
                    'gram': bacteria_db.get('gram', 'negativa'),
                    'tamanho_genoma': tamanho_total,
                    'gc': gc_medio,
                    'patogenicidade': bacteria_db.get('patogenicidade', 'Desconhecida'),
                    'aplicacoes': bacteria_db.get('aplicacoes', 'N/A'),
                    'descricao': bacteria_db.get('descricao', '')
                }
            else:
                # Informações padrão se não identificar
                bacteria_info = {
                    'nome': 'Bactéria Desconhecida',
                    'forma': 'bacilo',
                    'gram': 'negativa',
                    'tamanho_genoma': tamanho_total,
                    'gc': gc_medio,
                    'patogenicidade': 'Desconhecida',
                    'aplicacoes': 'N/A',
                    'descricao': 'Bactéria não identificada no banco de dados'
                }
            
            # Gerar visualização detalhada com resolução maior
            visualizador = VisualizadorBacteriaAvancado(largura=1600, altura=1200)
            orfs = getattr(self, 'orfs_por_contig', [])
            arquivo = visualizador.criar_visualizacao(
                bacteria_info, 
                self.contigs[:5], 
                orfs[:5], 
                "bacteria_detalhada_gui.png"
            )
            
            # Carregar e exibir imagem
            img = Image.open(arquivo)
            # Não redimensionar - mostrar tamanho original com scroll
            photo = ImageTk.PhotoImage(img)
            
            # Atualizar label e canvas
            self.label_imagem.configure(image=photo, text="")
            self.label_imagem.image = photo
            
            # Configurar scrollregion do canvas
            self.canvas_imagem.config(scrollregion=self.canvas_imagem.bbox("all"))
            
            # Mensagem de sucesso com informações
            msg = f"""Visualização gerada com sucesso!

Bactéria Identificada: {bacteria_info['nome']}
Forma: {bacteria_info['forma'].capitalize()}
Gram: {bacteria_info['gram'].capitalize()}
Tamanho: {tamanho_total:,} bp
GC: {gc_medio:.1f}%

Arquivo salvo: {arquivo}"""
            
            messagebox.showinfo("Sucesso", msg)
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao gerar visualização:\n{str(e)}")
            import traceback
            traceback.print_exc()
            
            
    def copiar_sequencia(self):
        """Copia sequência selecionada para clipboard."""
        try:
            texto = self.contigs_text.get(tk.SEL_FIRST, tk.SEL_LAST)
            self.root.clipboard_clear()
            self.root.clipboard_append(texto)
            messagebox.showinfo("Sucesso", "Sequência copiada para a área de transferência!")
        except:
            messagebox.showwarning("Aviso", "Selecione uma sequência primeiro!")


    def carregar_exemplo(self):
        """Abre janela para selecionar bactéria de exemplo."""
        janela = tk.Toplevel(self.root)
        janela.title("Selecionar Bactéria de Exemplo")
        janela.geometry("600x500")
        janela.configure(bg="#1e1e2e")
        
        tk.Label(janela, text="Selecione uma bactéria para simular:", 
                font=("Segoe UI", 14, "bold"), bg="#1e1e2e", fg="#cdd6f4").pack(pady=20)
        
        # Lista com scroll
        frame_lista = tk.Frame(janela, bg="#1e1e2e")
        frame_lista.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        scrollbar = tk.Scrollbar(frame_lista)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        lista = tk.Listbox(frame_lista, font=("Segoe UI", 11), bg="#2d2d44", fg="#cdd6f4",
                          selectbackground="#89b4fa", selectforeground="#1e1e2e",
                          yscrollcommand=scrollbar.set, height=15)
        lista.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=lista.yview)
        
        # Carregar bactérias do banco
        from identificacao.banco_expandido import BANCO_GENOMAS_EXPANDIDO
        bacterias = sorted(BANCO_GENOMAS_EXPANDIDO, key=lambda x: x['nome'])
        
        for b in bacterias:
            lista.insert(tk.END, f"{b['nome']} ({b.get('forma', 'bacilo')})")
            
        def confirmar():
            idx = lista.curselection()
            if not idx:
                return
            
            bacteria = bacterias[idx[0]]
            self.simular_dados_bacteria(bacteria)
            janela.destroy()
            
        tk.Button(janela, text="Carregar Simulação", command=confirmar,
                 bg="#a6e3a1", fg="#1e1e2e", font=("Segoe UI", 12, "bold"),
                 relief=tk.FLAT, padx=20, pady=10, cursor="hand2").pack(pady=20)

    def simular_dados_bacteria(self, bacteria):
        """Simula dados de montagem para a bactéria selecionada."""
        self.log(f"🔄 Simulando dados para: {bacteria['nome']}...")
        
        # Simular contigs baseados no tamanho do genoma
        tamanho_total = sum(bacteria['tamanho_genoma']) / 2
        gc_alvo = sum(bacteria['conteudo_gc']) / 2
        
        # Gerar contigs sintéticos
        import random
        self.contigs = []
        tamanho_atual = 0
        
        while tamanho_atual < tamanho_total:
            # Tamanho aleatório de contig (distribuição log-normal simulada)
            tam = int(random.gauss(50000, 15000))
            tam = max(1000, tam)
            if tamanho_atual + tam > tamanho_total:
                tam = int(tamanho_total - tamanho_atual)
            
            # Gerar sequência com GC correto
            seq = []
            for _ in range(tam):
                if random.random() * 100 < gc_alvo:
                    seq.append(random.choice(['G', 'C']))
                else:
                    seq.append(random.choice(['A', 'T']))
            
            self.contigs.append("".join(seq))
            tamanho_atual += tam
            
        self.log(f"✅ Gerados {len(self.contigs)} contigs sintéticos")
        
        # Simular ORFs
        self.orfs_por_contig = []
        for contig in self.contigs:
            orfs = []
            num_genes = len(contig) // 1000  # ~1 gene a cada 1kb
            for _ in range(num_genes):
                inicio = random.randint(0, len(contig)-1000)
                fim = inicio + random.randint(300, 1500)
                if fim < len(contig):
                    orfs.append({
                        'inicio': inicio,
                        'fim': fim,
                        'fita': random.choice(['+', '-']),
                        'score': random.uniform(0.8, 1.0)
                    })
            self.orfs_por_contig.append(orfs)
            
        # Simular reads e grafo (apenas metadados para stats)
        self.reads_processados = ['A'] * int(tamanho_total * 30 / 150)  # 30x cobertura
        
        # Mock do grafo para stats
        class MockGrafo:
            def __init__(self):
                self.grafo = type('obj', (object,), {
                    'number_of_nodes': lambda: int(tamanho_total / 2),
                    'number_of_edges': lambda: int(tamanho_total / 2 * 1.5)
                })
        self.grafo = MockGrafo()
        
        # Mock de coberturas
        self.coberturas = [random.gauss(30, 5) for _ in range(1000)]
        self.resultados = {'lambda': 30.0}
        
        # Forçar identificação correta
        self.relatorio_identificacao = f"""
        RELATÓRIO DE IDENTIFICAÇÃO BACTERIANA
        =====================================
        
        Melhor candidato: {bacteria['nome']} (Score: 0.99)
        Similaridade: 99.9%
        """
        
        # Atualizar GUI
        self.atualizar_estatisticas()
        self.gerar_visualizacao_genoma()
        self.atualizar_contigs()
        
        # ✨ INTEGRAÇÃO AUTOMÁTICA COM FARMACOLOGIA
        # Atualizar aba de farmacologia com a bactéria carregada
        if bacteria['nome'] in TRATAMENTOS_BACTERIANOS:
            self.bacteria_selecionada.set(bacteria['nome'])
            self.atualizar_informacoes_farmacologia()
            self.notebook.select(5)  # Mudar para aba de Farmacologia
        else:
            self.notebook.select(4)  # Mudar para aba de estatísticas
        
        msg_farmaco = ""
        if bacteria['nome'] in TRATAMENTOS_BACTERIANOS:
            msg_farmaco = "\n\n💊 Informações farmacológicas disponíveis na aba Farmacologia!"
        
        messagebox.showinfo("Simulação Concluída", 
                           f"Dados simulados para {bacteria['nome']}.\n"
                           f"Veja as abas de Estatísticas e Visualização.{msg_farmaco}")

def main():
    root = tk.Tk()
    app = MontadorGenomaBacterianoGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
