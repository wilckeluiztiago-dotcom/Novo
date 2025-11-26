"""
Interface Gráfica para Editor de Diagramas de Feynman
Autor: Luiz Tiago Wilcke

Usa customtkinter para uma interface moderna.
"""

import tkinter as tk
import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import io
from PIL import Image
import sympy
from desenho import calcular_pontos_foton, calcular_pontos_gluon, calcular_seta
from fisica import RegrasFeynman, ELETRON, FOTON

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class EditorFeynman(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Editor de Diagramas de Feynman - Luiz Tiago Wilcke")
        self.geometry("1000x700")
        
        # Layout principal
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # === Barra Lateral (Ferramentas) ===
        self.frame_ferramentas = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.frame_ferramentas.grid(row=0, column=0, sticky="nsew")
        
        self.label_titulo = ctk.CTkLabel(self.frame_ferramentas, text="Ferramentas", 
                                       font=ctk.CTkFont(size=20, weight="bold"))
        self.label_titulo.grid(row=0, column=0, padx=20, pady=20)
        
        # Botões de Partículas
        self.ferramenta_atual = "selecao"
        
        self.btn_fermion = ctk.CTkButton(self.frame_ferramentas, text="Férmion (e-)", 
                                       command=lambda: self.selecionar_ferramenta("fermion"))
        self.btn_fermion.grid(row=1, column=0, padx=20, pady=10)
        
        self.btn_foton = ctk.CTkButton(self.frame_ferramentas, text="Fóton (γ)", 
                                     command=lambda: self.selecionar_ferramenta("foton"))
        self.btn_foton.grid(row=2, column=0, padx=20, pady=10)
        
        self.btn_gluon = ctk.CTkButton(self.frame_ferramentas, text="Glúon (g)", 
                                     command=lambda: self.selecionar_ferramenta("gluon"))
        self.btn_gluon.grid(row=3, column=0, padx=20, pady=10)
        
        self.btn_limpar = ctk.CTkButton(self.frame_ferramentas, text="Limpar Tela", 
                                      fg_color="red", hover_color="darkred",
                                      command=self.limpar_tela)
        self.btn_limpar.grid(row=5, column=0, padx=20, pady=40)
        
        self.btn_gerar = ctk.CTkButton(self.frame_ferramentas, text="Gerar Integral", 
                                     fg_color="green", hover_color="darkgreen",
                                     command=self.gerar_integral)
        self.btn_gerar.grid(row=6, column=0, padx=20, pady=10)

        # === Área Principal ===
        self.frame_principal = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.frame_principal.grid(row=0, column=1, sticky="nsew")
        self.frame_principal.grid_rowconfigure(0, weight=3)
        self.frame_principal.grid_rowconfigure(1, weight=1)
        self.frame_principal.grid_columnconfigure(0, weight=1)
        
        # Canvas de Desenho
        self.canvas = tk.Canvas(self.frame_principal, bg="#2b2b2b", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Eventos do Canvas
        self.canvas.bind("<Button-1>", self.clique_canvas)
        self.canvas.bind("<B1-Motion>", self.arrastar_canvas)
        self.canvas.bind("<ButtonRelease-1>", self.soltar_canvas)
        
        # Área de Saída (Texto)
        self.texto_saida = ctk.CTkTextbox(self.frame_principal, height=150)
        self.texto_saida.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.texto_saida.insert("0.0", "A integral aparecerá aqui...")
        
        # Estado do Desenho
        self.inicio_linha = None
        self.linha_temporaria = None
        self.elementos = [] # Lista de dicionários {'tipo', 'pontos', 'obj_ids'}
        
    def selecionar_ferramenta(self, ferramenta):
        self.ferramenta_atual = ferramenta
        print(f"Ferramenta selecionada: {ferramenta}")
        
    def limpar_tela(self):
        self.canvas.delete("all")
        self.elementos = []
        self.texto_saida.delete("0.0", "end")
        
    def clique_canvas(self, event):
        if self.ferramenta_atual != "selecao":
            self.inicio_linha = (event.x, event.y)
            
    def arrastar_canvas(self, event):
        if self.inicio_linha:
            if self.linha_temporaria:
                self.canvas.delete(self.linha_temporaria)
            
            # Desenhar linha guia simples
            self.linha_temporaria = self.canvas.create_line(
                self.inicio_linha[0], self.inicio_linha[1], event.x, event.y,
                fill="white", dash=(4, 4)
            )
            
    def soltar_canvas(self, event):
        if self.inicio_linha:
            x1, y1 = self.inicio_linha
            x2, y2 = event.x, event.y
            
            if self.linha_temporaria:
                self.canvas.delete(self.linha_temporaria)
                self.linha_temporaria = None
            
            # Criar o elemento final
            self.criar_elemento(self.ferramenta_atual, x1, y1, x2, y2)
            self.inicio_linha = None
            
    def criar_elemento(self, tipo, x1, y1, x2, y2):
        ids = []
        cor = "white"
        width = 2
        
        if tipo == "fermion":
            # Linha reta
            l_id = self.canvas.create_line(x1, y1, x2, y2, fill="white", width=width)
            ids.append(l_id)
            
            # Seta
            pontos_seta = calcular_seta(x1, y1, x2, y2)
            s_id = self.canvas.create_polygon(pontos_seta, fill="white")
            ids.append(s_id)
            
        elif tipo == "foton":
            pontos = calcular_pontos_foton(x1, y1, x2, y2)
            # Flatten lista de pontos para create_line
            coords = [coord for p in pontos for coord in p]
            l_id = self.canvas.create_line(coords, fill="yellow", width=width, smooth=True)
            ids.append(l_id)
            cor = "yellow"
            
        elif tipo == "gluon":
            pontos = calcular_pontos_gluon(x1, y1, x2, y2)
            coords = [coord for p in pontos for coord in p]
            l_id = self.canvas.create_line(coords, fill="orange", width=width, smooth=True)
            ids.append(l_id)
            cor = "orange"
            
        # Salvar elemento
        self.elementos.append({
            'tipo': tipo,
            'p1': (x1, y1),
            'p2': (x2, y2),
            'ids': ids
        })
        
    def gerar_integral(self):
        """Gera a integral baseada nos elementos desenhados."""
        self.texto_saida.delete("0.0", "end")
        
        if not self.elementos:
            self.texto_saida.insert("0.0", "Desenhe algo primeiro!")
            return
            
        # Lógica simplificada de geração (Mockup funcional)
        texto = "Gerando amplitude para o diagrama...\n\n"
        
        num_fermions = sum(1 for e in self.elementos if e['tipo'] == 'fermion')
        num_fotons = sum(1 for e in self.elementos if e['tipo'] == 'foton')
        
        texto += f"Elementos detectados:\n- {num_fermions} Propagadores de Férmion\n- {num_fotons} Propagadores de Fóton\n\n"
        texto += "Integral (Exemplo Simbólico):\n"
        
        # Usar o motor de física para gerar um exemplo real
        from fisica import gerar_integral_exemplo
        integral = gerar_integral_exemplo()
        
        texto += str(integral)
        
        self.texto_saida.insert("0.0", texto)
        
        # Renderizar LaTeX
        self.exibir_latex(integral)
        
    def exibir_latex(self, expressao):
        """Renderiza expressão sympy como imagem LaTeX."""
        try:
            # Criar figura matplotlib
            fig = plt.figure(figsize=(6, 2), dpi=100)
            fig.patch.set_facecolor('#2b2b2b') # Cor de fundo do tema dark
            
            ax = fig.add_subplot(111)
            ax.set_axis_off()
            
            # Renderizar texto LaTeX
            latex_str = f"${sympy.latex(expressao)}$"
            ax.text(0.5, 0.5, latex_str, size=16, ha='center', va='center', color='white')
            
            # Salvar em buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1, facecolor=fig.get_facecolor())
            buf.seek(0)
            plt.close(fig)
            
            # Criar imagem CTk
            pil_img = Image.open(buf)
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=pil_img.size)
            
            # Exibir em um Label (janela pop-up ou na interface principal)
            # Vamos criar uma janela Toplevel para o resultado renderizado
            janela_resultado = ctk.CTkToplevel(self)
            janela_resultado.title("Integral Renderizada")
            janela_resultado.geometry("600x300")
            
            label_img = ctk.CTkLabel(janela_resultado, text="", image=ctk_img)
            label_img.pack(expand=True, fill="both", padx=20, pady=20)
            
        except Exception as e:
            self.texto_saida.insert("end", f"\n\nErro ao renderizar LaTeX: {e}")
            # Fallback: mostrar string latex no texto
            self.texto_saida.insert("end", f"\nLaTeX: {sympy.latex(expressao)}")

if __name__ == "__main__":
    app = EditorFeynman()
    app.mainloop()
