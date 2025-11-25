import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

from modules.quantum import SILICON, GAAS, PEROVSKITE
from modules.solar import calcular_corrente_fotogerada_limite
from modules.device import calcular_corrente_saturacao_radiativa, curva_JV_diodo
from modules.analysis import extrair_parametros


class CalculadoraFotovoltaica:
    """
    Interface gráfica sofisticada para simulação de células fotovoltaicas.
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("🌞 Calculadora Fotovoltaica - Luiz Tiago Wilcke")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Cores do tema
        self.cor_principal = '#2E86AB'
        self.cor_secundaria = '#A23B72'
        self.cor_fundo = '#f0f0f0'
        self.cor_texto = '#333333'
        
        self.criar_interface()
        
    def criar_interface(self):
        """Cria todos os elementos da interface."""
        
        # ==========================================
        # TÍTULO PRINCIPAL
        # ==========================================
        titulo_frame = tk.Frame(self.root, bg=self.cor_principal, height=60)
        titulo_frame.pack(fill='x', padx=0, pady=0)
        titulo_frame.pack_propagate(False)
        
        titulo = tk.Label(
            titulo_frame, 
            text="🌞 CALCULADORA FOTOVOLTAICA AVANÇADA",
            font=('Arial', 20, 'bold'),
            bg=self.cor_principal,
            fg='white'
        )
        titulo.pack(pady=15)
        
        subtitulo = tk.Label(
            titulo_frame,
            text="Simulador de Células Fotovoltaicas | Autor: Luiz Tiago Wilcke",
            font=('Arial', 10),
            bg=self.cor_principal,
            fg='white'
        )
        subtitulo.pack()
        
        # ==========================================
        # CONTAINER PRINCIPAL (dividido em 2 colunas)
        # ==========================================
        main_container = tk.Frame(self.root, bg=self.cor_fundo)
        main_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # COLUNA ESQUERDA: Inputs e Controles
        coluna_esquerda = tk.Frame(main_container, bg='white', relief='ridge', borderwidth=2)
        coluna_esquerda.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        # COLUNA DIREITA: Equações e Resultados
        coluna_direita = tk.Frame(main_container, bg='white', relief='ridge', borderwidth=2)
        coluna_direita.pack(side='right', fill='both', expand=True, padx=(5, 0))
        
        # ==========================================
        # COLUNA ESQUERDA - PARÂMETROS DE ENTRADA
        # ==========================================
        self.criar_secao_inputs(coluna_esquerda)
        
        # ==========================================
        # COLUNA DIREITA - EQUAÇÕES E TEORIA
        # ==========================================
        self.criar_secao_equacoes(coluna_direita)
        
        # ==========================================
        # RODAPÉ COM BOTÕES DE AÇÃO
        # ==========================================
        self.criar_rodape()
        
    def criar_secao_inputs(self, parent):
        """Cria a seção de inputs na coluna esquerda."""
        
        # Canvas com scroll para os inputs
        canvas = tk.Canvas(parent, bg='white', highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='white')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # ========== MATERIAL ==========
        self.criar_grupo("📦 MATERIAL", scrollable_frame, [
            ("Material:", "combobox", ["Silício (Si)", "Arsenieto de Gálio (GaAs)", "Perovskita (MAPI)"], "Silício (Si)", "material"),
            ("Energia de Gap (eV):", "entry", None, "1.12", "energia_gap"),
        ])
        
        # ========== TEMPERATURA ==========
        self.criar_grupo("🌡️ TEMPERATURA", scrollable_frame, [
            ("Temperatura da Célula (K):", "entry", None, "300.0", "temp_celula"),
            ("Temperatura do Sol (K):", "entry", None, "5778.0", "temp_sol"),
        ])
        
        # ========== PARÂMETROS DO DIODO ==========
        self.criar_grupo("⚡ PARÂMETROS DO DIODO", scrollable_frame, [
            ("Fator de Idealidade (n):", "entry", None, "1.0", "fator_idealidade"),
            ("Resistência Série Rs (Ω·m²):", "entry", None, "0.5", "res_serie"),
            ("Resistência Shunt Rsh (Ω·m²):", "entry", None, "10000", "res_shunt"),
        ])
        
        # ========== SIMULAÇÃO ===========
        # Botão CALCULAR (adicionado abaixo dos inputs)
        btn_calcular_esquerda = tk.Button(
            parent,
            text="🚀 CALCULAR",
            font=('Arial', 12, 'bold'),
            bg=self.cor_principal,
            fg='white',
            activebackground='#1a5a7a',
            activeforeground='white',
            relief='flat',
            padx=20,
            pady=8,
            cursor='hand2',
            command=self.calcular
        )
        btn_calcular_esquerda.pack(pady=10)
        # Área de resultados à esquerda (após o botão)
        self.criar_area_resultados_esquerda(parent)
        # ========== SIMULAÇÃO ==========

        self.criar_grupo("📊 CONFIGURAÇÃO DA SIMULAÇÃO", scrollable_frame, [
            ("Tensão Máxima (V):", "entry", None, "1.2", "v_max"),
            ("Número de Pontos:", "entry", None, "400", "num_pontos"),
        ])
        
    def criar_grupo(self, titulo, parent, campos):
        """Cria um grupo de inputs."""
        
        grupo_frame = tk.LabelFrame(
            parent,
            text=titulo,
            font=('Arial', 11, 'bold'),
            bg='white',
            fg=self.cor_principal,
            padx=15,
            pady=10
        )
        grupo_frame.pack(fill='x', padx=10, pady=10)
        
        if not hasattr(self, 'inputs'):
            self.inputs = {}
        
        for label_text, tipo, opcoes, valor_padrao, chave in campos:
            frame_campo = tk.Frame(grupo_frame, bg='white')
            frame_campo.pack(fill='x', pady=5)
            
            label = tk.Label(
                frame_campo,
                text=label_text,
                font=('Arial', 9),
                bg='white',
                anchor='w'
            )
            label.pack(side='left', fill='x', expand=False, padx=(0, 10))
            
            if tipo == "entry":
                entrada = tk.Entry(frame_campo, font=('Arial', 10), width=15)
                entrada.insert(0, valor_padrao)
                entrada.pack(side='right')
                self.inputs[chave] = entrada
            
            elif tipo == "combobox":
                entrada = ttk.Combobox(frame_campo, values=opcoes, state='readonly', width=25)
                entrada.set(valor_padrao)
                entrada.pack(side='right')
                self.inputs[chave] = entrada
    
    def criar_secao_equacoes(self, parent):
        """Cria a seção de equações e teoria na coluna direita."""
        
        # Título da seção
        titulo_eq = tk.Label(
            parent,
            text="📚 FÍSICA E EQUAÇÕES FUNDAMENTAIS",
            font=('Arial', 12, 'bold'),
            bg='white',
            fg=self.cor_secundaria,
            pady=10
        )
        titulo_eq.pack()
        
        # Área de texto com scroll para equações
        texto_frame = tk.Frame(parent, bg='white')
        texto_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.texto_equacoes = scrolledtext.ScrolledText(
            texto_frame,
            wrap=tk.WORD,
            font=('Courier New', 9),
            bg='#fafafa',
            fg=self.cor_texto,
            relief='flat',
            padx=10,
            pady=10
        )
        self.texto_equacoes.pack(fill='both', expand=True)
        
        # Inserir conteúdo teórico
        self.preencher_teoria()
        
        # Área de resultados
        self.criar_area_resultados(parent)
        
    def preencher_teoria(self):
        """Preenche a área de equações com teoria."""
        
        teoria = """
═══════════════════════════════════════════════════════════
  MODELO FÍSICO DA CÉLULA FOTOVOLTAICA
═══════════════════════════════════════════════════════════

1. CORRENTE FOTOGERADA (Limite de Shockley-Queisser)
───────────────────────────────────────────────────────────
A corrente fotogerada representa o fluxo de portadores 
criados pela absorção de fótons acima da energia de gap:

    J_ph = q · ∫[Eg→∞] Φ(E) dE

Onde:
  • q = carga elementar (1.602×10⁻¹⁹ C)
  • Φ(E) = fluxo espectral de fótons do Sol
  • Eg = energia de gap do material


2. FLUXO DE FÓTONS (Corpo Negro)
───────────────────────────────────────────────────────────
O Sol é aproximado como um corpo negro a T ≈ 5778 K:

    Φ(E) = (2π/h³c²) · E² / [exp(E/kT) - 1]

Onde:
  • h = constante de Planck (6.626×10⁻³⁴ J·s)
  • c = velocidade da luz (2.998×10⁸ m/s)
  • k = constante de Boltzmann (1.381×10⁻²³ J/K)


3. CORRENTE DE SATURAÇÃO RADIATIVA
───────────────────────────────────────────────────────────
Representa a recombinação radiativa na célula:

    J₀ = q · ∫[Eg→∞] Φ_emit(E, T_cel) dE


4. EQUAÇÃO DO DIODO (Shockley)
───────────────────────────────────────────────────────────
Modelo elétrico completo da célula fotovoltaica:

    J(V) = J_ph - J₀[exp(q(V+JRs)/nkT) - 1] - (V+JRs)/Rsh

Onde:
  • n = fator de idealidade (1-2)
  • Rs = resistência série (perdas ôhmicas)
  • Rsh = resistência shunt (correntes de fuga)


5. PARÂMETROS DE PERFORMANCE
───────────────────────────────────────────────────────────
• J_sc = Corrente de curto-circuito (V=0)
• V_oc = Tensão de circuito aberto (J=0)
        V_oc ≈ (nkT/q) · ln(J_ph/J₀ + 1)

• FF = Fator de Preenchimento
        FF = (V_mp · J_mp) / (V_oc · J_sc)

• η = Eficiência de Conversão
        η = P_max / P_solar = (V_mp · J_mp) / 1000 W/m²


6. DENSIDADE DE ESTADOS
───────────────────────────────────────────────────────────
Estados disponíveis nas bandas de condução e valência:

    Nc = 2(2πm*_n kT/h²)^(3/2)
    Nv = 2(2πm*_p kT/h²)^(3/2)

Concentração intrínseca:
    ni² = Nc · Nv · exp(-Eg/kT)

═══════════════════════════════════════════════════════════
"""
        
        self.texto_equacoes.insert('1.0', teoria)
        self.texto_equacoes.config(state='disabled')
    
    def criar_area_resultados(self, parent):
        """Cria área para exibir resultados."""
        
        resultado_frame = tk.LabelFrame(
            parent,
            text="📊 RESULTADOS DA SIMULAÇÃO",
            font=('Arial', 11, 'bold'),
            bg='white',
            fg=self.cor_secundaria,
            padx=10,
            pady=10
        )
        resultado_frame.pack(fill='x', padx=10, pady=10)
        
        self.texto_resultados = tk.Text(
            resultado_frame,
            height=12,
            font=('Courier New', 9),
            bg='#f9f9f9',
            fg=self.cor_texto,
            relief='flat',
            padx=10,
            pady=10
        )
        self.texto_resultados.pack(fill='both', expand=True)
        self.texto_resultados.insert('1.0', "Configure os parâmetros e clique em 'CALCULAR' para ver os resultados...")
        self.texto_resultados.config(state='disabled')
        
    def criar_rodape(self):
        """Cria o rodapé com botões de ação."""
        
        rodape = tk.Frame(self.root, bg=self.cor_fundo, height=70)
        rodape.pack(fill='x', padx=10, pady=(0, 10))
        rodape.pack_propagate(False)

        

    def criar_area_resultados_esquerda(self, parent):
        """Cria área para exibir resultados resumidos na coluna esquerda."""
        resultado_frame = tk.LabelFrame(
            parent,
            text="📊 RESULTADOS (ESQUERDA)",
            font=('Arial', 11, 'bold'),
            bg='white',
            fg=self.cor_secundaria,
            padx=10,
            pady=10
        )
        resultado_frame.pack(fill='both', expand=True, padx=10, pady=10)

        self.texto_resultado_esquerda = scrolledtext.ScrolledText(
            resultado_frame,
            height=12,
            font=('Courier New', 9),
            bg='#f9f9f9',
            fg=self.cor_texto,
            relief='flat',
            padx=10,
            pady=10
        )
        self.texto_resultado_esquerda.pack(fill='both', expand=True)
        self.texto_resultado_esquerda.insert('1.0', "Configure os parâmetros e clique em 'CALCULAR' para ver os resultados...")
        self.texto_resultado_esquerda.config(state='disabled')
        
    def obter_valores(self):
        """Obtém valores dos campos de entrada."""
        try:
            valores = {
                'material': self.inputs['material'].get(),
                'energia_gap': float(self.inputs['energia_gap'].get()),
                'temp_celula': float(self.inputs['temp_celula'].get()),
                'temp_sol': float(self.inputs['temp_sol'].get()),
                'fator_idealidade': float(self.inputs['fator_idealidade'].get()),
                'res_serie': float(self.inputs['res_serie'].get()),
                'res_shunt': float(self.inputs['res_shunt'].get()),
                'v_max': float(self.inputs['v_max'].get()),
                'num_pontos': int(self.inputs['num_pontos'].get()),
            }
            return valores
        except ValueError:
            messagebox.showerror("Erro", "Por favor, insira valores numéricos válidos!")
            return None
    
    def calcular(self):
        """Executa o cálculo da simulação."""
        
        valores = self.obter_valores()
        if not valores:
            return
        
        try:
            # Calcular J_ph
            J_ph = calcular_corrente_fotogerada_limite(
                energia_gap_eV=valores['energia_gap'],
                temperatura_sol=valores['temp_sol'],
                num_pontos_energia=4000
            )
            
            # Calcular J0
            J0 = calcular_corrente_saturacao_radiativa(
                energia_gap_eV=valores['energia_gap'],
                temperatura_celula=valores['temp_celula'],
                num_pontos_energia=4000
            )
            
            # Calcular curva J-V
            tensoes_V, correntes_J = curva_JV_diodo(
                J_ph=J_ph,
                J0=J0,
                temperatura_celula=valores['temp_celula'],
                fator_idealidade=valores['fator_idealidade'],
                resistencia_serie=valores['res_serie'],
                resistencia_shunt=valores['res_shunt'],
                tensao_min=0.0,
                tensao_max=valores['v_max'],
                num_pontos_tensao=valores['num_pontos']
            )
            
            # Extrair parâmetros
            resultados = extrair_parametros(
                tensoes_V, correntes_J, J_ph, J0,
                valores['temp_celula'], valores['fator_idealidade']
            )
            
            # Armazenar para plotagem
            self.tensoes = tensoes_V
            self.correntes = correntes_J
            self.potencias = resultados['Potencias']
            
            # Exibir resultados
            self.exibir_resultados(valores, J_ph, J0, resultados)
            
            messagebox.showinfo("Sucesso", "Cálculo concluído com sucesso! ✓")
            
        except Exception as e:
            messagebox.showerror("Erro no Cálculo", f"Ocorreu um erro: {str(e)}")
    
    def exibir_resultados(self, valores, J_ph, J0, resultados):
        """Exibe os resultados na área de texto."""
        
        texto = f"""
╔═══════════════════════════════════════════════════════════╗
║           RESULTADOS DA SIMULAÇÃO FOTOVOLTAICA            ║
╚═══════════════════════════════════════════════════════════╝

PARÂMETROS DE ENTRADA
─────────────────────────────────────────────────────────────
Material                    : {valores['material']}
Energia de Gap              : {valores['energia_gap']:.3f} eV
Temperatura da Célula       : {valores['temp_celula']:.1f} K
Temperatura do Sol          : {valores['temp_sol']:.1f} K
Fator de Idealidade         : {valores['fator_idealidade']:.2f}
Resistência Série (Rs)      : {valores['res_serie']:.2f} Ω·m²
Resistência Shunt (Rsh)     : {valores['res_shunt']:.1e} Ω·m²

CORRENTES FUNDAMENTAIS
─────────────────────────────────────────────────────────────
J_ph (fotogerada)           : {J_ph:.3e} A/m² ({J_ph*0.1:.2f} mA/cm²)
J₀ (saturação)              : {J0:.3e} A/m² ({J0*0.1:.4e} mA/cm²)

PARÂMETROS DE PERFORMANCE
─────────────────────────────────────────────────────────────
J_sc (curto-circuito)       : {resultados['J_sc']*0.1:.2f} mA/cm²
V_oc (circuito aberto)      : {resultados['V_oc_numerico']:.3f} V
P_max (potência máxima)     : {resultados['P_max']:.1f} W/m²
V_mp (tensão em P_max)      : {resultados['V_mp']:.3f} V
J_mp (corrente em P_max)    : {resultados['J_mp']*0.1:.2f} mA/cm²

MÉTRICAS DE QUALIDADE
─────────────────────────────────────────────────────────────
FF (fator preenchimento)    : {resultados['FF']*100:.2f} %
η (eficiência)              : {resultados['Eficiencia']*100:.2f} %

═════════════════════════════════════════════════════════════
"""
        
        # Atualiza área de resultados à direita
        self.texto_resultados.config(state='normal')
        self.texto_resultados.delete('1.0', tk.END)
        self.texto_resultados.insert('1.0', texto)
        self.texto_resultados.config(state='disabled')
        # Atualiza área de resultados resumida à esquerda (se existir)
        if hasattr(self, 'texto_resultado_esquerda'):
            self.texto_resultado_esquerda.config(state='normal')
            self.texto_resultado_esquerda.delete('1.0', tk.END)
            self.texto_resultado_esquerda.insert('1.0', texto)
            self.texto_resultado_esquerda.config(state='disabled')
    
    def plotar_graficos(self):
        """Plota os gráficos J-V e P-V."""
        
        if not hasattr(self, 'tensoes'):
            messagebox.showwarning("Aviso", "Execute o cálculo primeiro!")
            return
        
        # Criar janela para gráficos
        janela_graficos = tk.Toplevel(self.root)
        janela_graficos.title("Gráficos - Célula Fotovoltaica")
        janela_graficos.geometry("1000x600")
        
        fig = Figure(figsize=(10, 5))
        
        # Gráfico J-V
        ax1 = fig.add_subplot(121)
        ax1.plot(self.tensoes, self.correntes * 0.1, linewidth=2, color='#2E86AB')
        ax1.axhline(0, linestyle='--', color='gray', alpha=0.7)
        ax1.set_xlabel('Tensão [V]', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Densidade de Corrente [mA/cm²]', fontsize=11, fontweight='bold')
        ax1.set_title('Curva J-V', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Gráfico P-V
        ax2 = fig.add_subplot(122)
        ax2.plot(self.tensoes, self.potencias, linewidth=2, color='#A23B72')
        ax2.set_xlabel('Tensão [V]', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Potência [W/m²]', fontsize=11, fontweight='bold')
        ax2.set_title('Curva P-V', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=janela_graficos)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def resetar(self):
        """Reseta todos os campos para valores padrão."""
        resposta = messagebox.askyesno("Confirmar", "Resetar todos os valores para o padrão?")
        if resposta:
            self.inputs['material'].set("Silício (Si)")
            self.inputs['energia_gap'].delete(0, tk.END)
            self.inputs['energia_gap'].insert(0, "1.12")
            self.inputs['temp_celula'].delete(0, tk.END)
            self.inputs['temp_celula'].insert(0, "300.0")
            self.inputs['temp_sol'].delete(0, tk.END)
            self.inputs['temp_sol'].insert(0, "5778.0")
            self.inputs['fator_idealidade'].delete(0, tk.END)
            self.inputs['fator_idealidade'].insert(0, "1.0")
            self.inputs['res_serie'].delete(0, tk.END)
            self.inputs['res_serie'].insert(0, "0.5")
            self.inputs['res_shunt'].delete(0, tk.END)
            self.inputs['res_shunt'].insert(0, "10000")
            self.inputs['v_max'].delete(0, tk.END)
            self.inputs['v_max'].insert(0, "1.2")
            self.inputs['num_pontos'].delete(0, tk.END)
            self.inputs['num_pontos'].insert(0, "400")
            
            self.texto_resultados.config(state='normal')
            self.texto_resultados.delete('1.0', tk.END)
            self.texto_resultados.insert('1.0', "Configure os parâmetros e clique em 'CALCULAR' para ver os resultados...")
            self.texto_resultados.config(state='disabled')


def main():
    """Inicia a interface gráfica."""
    root = tk.Tk()
    app = CalculadoraFotovoltaica(root)
    root.mainloop()


if __name__ == "__main__":
    main()
