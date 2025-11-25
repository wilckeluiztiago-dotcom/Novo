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
from nucleo.grafo_bruijn import GrafoBruijn
from nucleo.montador import Montador
from estatistica.distribuicao import ModeloCobertura
from estatistica.metricas import MetricasMontagem
from identificacao.identificador import IdentificadorBacteriano
from analise.genomica import AnalisadorGenomica
from visualizacao.genoma_circular import VisualizadorGenoma
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
        
        # Frame para imagem
        self.imagem_frame = tk.Frame(container, bg="#1e1e2e")
        self.imagem_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.label_imagem = tk.Label(self.imagem_frame, bg="#1e1e2e", fg="#cdd6f4",
                                     text="Clique em 'Gerar Visualização' após a montagem",
                                     font=("Segoe UI", 12))
        self.label_imagem.pack(expand=True)
        
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
            reads_processados = PreProcessador.processar_reads(leitor)
            self.log(f"✅ {len(reads_processados)} reads processados")
            
            # 2. Grafo
            self.log("🕸️ Construindo Grafo de Bruijn...")
            grafo = GrafoBruijn(k=self.tamanho_kmer.get())
            grafo.construir_de_reads(reads_processados)
            self.log(f"✅ Grafo: {grafo.grafo.number_of_nodes()} nós, {grafo.grafo.number_of_edges()} arestas")
            
            # 3. Estatística
            self.log("📊 Análise estatística...")
            self.coberturas = [d['cobertura'] for _, _, d in grafo.grafo.edges(data=True)]
            lambda_est = ModeloCobertura.estimar_lambda_poisson(self.coberturas)
            self.log(f"✅ Lambda estimado: {lambda_est:.2f}")
            
            # 4. Simplificação
            self.log("🧹 Simplificando grafo...")
            grafo.remover_erros(cobertura_minima=self.cobertura_minima.get())
            self.log(f"✅ Grafo simplificado: {grafo.grafo.number_of_nodes()} nós")
            
            # 5. Montagem
            self.log("🔨 Montando contigs...")
            montador = Montador(grafo)
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
        
        # Mudar para aba de resultados
        self.notebook.select(2)
        
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
        """Gera visualização circular do genoma."""
        if not hasattr(self, 'contigs') or not self.contigs:
            messagebox.showwarning("Aviso", "Execute a montagem primeiro!")
            return
        
        try:
            from PIL import Image, ImageTk
            
            # Gera visualização
            visualizador = VisualizadorGenoma(largura=800, altura=800)
            orfs = getattr(self, 'orfs_por_contig', [])
            arquivo = visualizador.criar_visualizacao(self.contigs[:5], orfs[:5], 
                                                       "genoma_circular_gui.png")
            
            # Carrega e exibe imagem
            img = Image.open(arquivo)
            img = img.resize((700, 700), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            self.label_imagem.configure(image=photo, text="")
            self.label_imagem.image = photo  # Mantém referência
            
            messagebox.showinfo("Sucesso", f"Visualização gerada!\nSalva em: {arquivo}")
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao gerar visualização:\n{str(e)}")
            
    def copiar_sequencia(self):
        """Copia sequência selecionada para clipboard."""
        try:
            texto = self.contigs_text.get(tk.SEL_FIRST, tk.SEL_LAST)
            self.root.clipboard_clear()
            self.root.clipboard_append(texto)
            messagebox.showinfo("Sucesso", "Sequência copiada para a área de transferência!")
        except:
            messagebox.showwarning("Aviso", "Selecione uma sequência primeiro!")


def main():
    root = tk.Tk()
    app = MontadorGenomaBacterianoGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
