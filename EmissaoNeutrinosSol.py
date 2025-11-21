import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.constants import atomic_mass, c
import matplotlib.style as style

# Configuração para melhor qualidade visual
style.use('default')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True

# =============================================================================
# CONSTANTES FÍSICAS (5 dígitos de precisão)
# =============================================================================
massa_sol = 1.9890e30  # kg
luminosidade_sol = 3.8460e26  # W
raio_sol = 6.9570e8  # m
temperatura_central_sol = 1.5705e7  # K
densidade_central_sol = 1.6220e5  # kg/m³
fracao_hidrogenio = 0.7347  # Fração de hidrogênio no núcleo

# Constantes nucleares
constante_gamow = 1.3002e-14  # keV·barn
energia_limiar_pp = 1.4420  # MeV
massa_neutrino = 1.9200e-36  # kg (massa aproximada do neutrino eletrônico)
energia_media_neutrino = 0.2635  # MeV (média por neutrino)

# Conversões de unidades
MeV_para_Joule = 1.6022e-13
kg_para_MeV_c2 = 5.6096e29

class ModeloNeutrinosSolares:
    def __init__(self):
        self.resultados = None
        
    def taxa_reacao_pp(self, temperatura, densidade):
        """
        Taxa da reação próton-próton usando a equação de Salpeter
        Baseado na teoria de Gamow para reações nucleares
        """
        T6 = temperatura / 1e6  # Temperatura em milhões de K
        T9 = temperatura / 1e9  # Temperatura em bilhões de K
        
        # Fator astrofísico S(0) para a cadeia pp
        S0 = 4.0100e-22  # keV·barn
        
        # Termo de penetração de Gamow
        tau = 4.2487 / (T9**(1/3))
        
        # Fator exponencial
        f_exp = np.exp(-tau)
        
        # Correção de screening eletrônico
        f_screening = 1 + 0.123 * T6**(1/3) + 0.813 * T6**(2/3)
        
        # Taxa de reação (cm³/s)
        lambda_pp = (S0 / (1e6 * constante_gamow)) * f_exp * f_screening
        
        # Taxa por par de partículas
        taxa_por_par = densidade * lambda_pp
        
        return taxa_por_par
    
    def perfil_temperatura_solar(self, raio_normalizado):
        """
        Perfil de temperatura solar baseado em modelos padrão
        raio_normalizado: 0 (centro) a 1 (superfície)
        """
        # Modelo polinomial para temperatura
        T_central = temperatura_central_sol
        T_superficie = 5778  # K
        
        if raio_normalizado <= 0.3:  # Núcleo
            return T_central * (1 - 2.5 * raio_normalizado**2)
        else:  # Zona radiativa/convectiva
            return T_superficie + (T_central - T_superficie) * np.exp(-15 * (raio_normalizado - 0.3))
    
    def perfil_densidade_solar(self, raio_normalizado):
        """
        Perfil de densidade solar
        """
        rho_central = densidade_central_sol
        rho_superficie = 1e-4  # kg/m³ (aproximado)
        
        if raio_normalizado <= 0.3:
            return rho_central * (1 - 2.0 * raio_normalizado**2)
        else:
            return rho_superficie + (rho_central - rho_superficie) * np.exp(-12 * (raio_normalizado - 0.3))
    
    def equacao_producao_neutrinos(self, r, variaveis):
        """
        Equação diferencial para produção de neutrinos
        r: raio normalizado
        variaveis: [numero_neutrinos, fluxo_energia]
        """
        N_nu, F_nu = variaveis
        
        # Obter condições locais
        T_local = self.perfil_temperatura_solar(r)
        rho_local = self.perfil_densidade_solar(r)
        
        # Taxa de produção de neutrinos (por m³/s)
        taxa_local = self.taxa_reacao_pp(T_local, rho_local) * fracao_hidrogenio**2
        
        # Volume diferencial
        dV_dr = 4 * np.pi * (raio_sol * r)**2 * raio_sol
        
        # Produção de neutrinos
        dN_dr = taxa_local * dV_dr
        
        # Produção de fluxo de energia
        energia_por_neutrino = energia_media_neutrino * MeV_para_Joule
        dF_dr = dN_dr * energia_por_neutrino
        
        return [dN_dr, dF_dr]
    
    def resolver_modelo(self):
        """Resolver o sistema de equações diferenciais"""
        # Condições iniciais
        condicoes_iniciais = [0.0, 0.0]  # [neutrinos totais, fluxo total]
        
        # Domínio radial
        r_points = np.linspace(0, 1, 1000)
        
        # Resolver EDO
        solucao = solve_ivp(
            self.equacao_producao_neutrinos,
            [0, 1],
            condicoes_iniciais,
            t_eval=r_points,
            method='RK45',
            rtol=1e-8
        )
        
        self.resultados = solucao
        return solucao
    
    def calcular_metricas(self):
        """Calcular métricas importantes"""
        if self.resultados is None:
            self.resolver_modelo()
        
        # Valores no raio solar (r=1)
        neutrinos_totais = self.resultados.y[0, -1]
        energia_total = self.resultados.y[1, -1]
        
        # Taxa de emissão por segundo
        taxa_emissao = neutrinos_totais
        
        # Energia total perdida em neutrinos por segundo
        potencia_neutrinos = energia_total
        
        # Massa equivalente perdida por segundo (E = mc²)
        massa_perdida_segundo = potencia_neutrinos / (c**2)
        
        # Massa perdida por ano
        segundos_por_ano = 365.25 * 24 * 3600
        massa_perdida_ano = massa_perdida_segundo * segundos_por_ano
        
        return {
            'taxa_emissao_neutrinos_s': taxa_emissao,
            'potencia_neutrinos_W': potencia_neutrinos,
            'massa_perdida_kg_s': massa_perdida_segundo,
            'massa_perdida_kg_ano': massa_perdida_ano,
            'fracao_massa_perdida_ano': massa_perdida_ano / massa_sol
        }

# =============================================================================
# EXECUÇÃO E RESULTADOS
# =============================================================================

print("🌞 MODELO AVANÇADO DE EMISSÃO DE NEUTRINOS SOLARES")
print("=" * 60)

# Criar e executar modelo
modelo = ModeloNeutrinosSolares()
resultados_edo = modelo.resolver_modelo()
metricas = modelo.calcular_metricas()

# Resultados com 5 dígitos de precisão
print("\n📊 RESULTADOS NUMÉRICOS (5 dígitos):")
print(f"Taxa de emissão de neutrinos: {metricas['taxa_emissao_neutrinos_s']:.5e} neutrinos/segundo")
print(f"Potência em neutrinos: {metricas['potencia_neutrinos_W']:.5e} W")
print(f"Massa perdida em neutrinos: {metricas['massa_perdida_kg_s']:.5e} kg/segundo")
print(f"Massa perdida por ano: {metricas['massa_perdida_kg_ano']:.5e} kg/ano")
print(f"Fração da massa solar perdida/ano: {metricas['fracao_massa_perdida_ano']:.5e}")

# Comparação com valores teóricos conhecidos
taxa_teorica = 1.8e38  # neutrinos/segundo (valor teórico conhecido)
print(f"\n🔬 COMPARAÇÃO COM VALORES TEÓRICOS:")
print(f"Taxa calculada: {metricas['taxa_emissao_neutrinos_s']:.5e} ν/s")
print(f"Taxa teórica esperada: {taxa_teorica:.5e} ν/s")
print(f"Diferença relativa: {abs(metricas['taxa_emissao_neutrinos_s'] - taxa_teorica)/taxa_teorica*100:.3f}%")

# =============================================================================
# VISUALIZAÇÕES CORRIGIDAS
# =============================================================================

# Criar figura principal com subplots organizados
fig = plt.figure(figsize=(20, 12))

# 1. Perfis de temperatura e densidade
ax1 = plt.subplot(2, 3, 1)
raio = resultados_edo.t
temperatura = [modelo.perfil_temperatura_solar(r) for r in raio]
densidade = [modelo.perfil_densidade_solar(r) for r in raio]

ax1.semilogy(raio, temperatura, 'r-', linewidth=2, label='Temperatura')
ax1.set_ylabel('Temperatura (K)', fontsize=12)
ax1.set_xlabel('Raio Normalizado', fontsize=12)
ax1.set_title('Perfil de Temperatura Solar', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)

# 2. Perfil de densidade
ax2 = plt.subplot(2, 3, 2)
ax2.semilogy(raio, densidade, 'b-', linewidth=2, label='Densidade')
ax2.set_ylabel('Densidade (kg/m³)', fontsize=12)
ax2.set_xlabel('Raio Normalizado', fontsize=12)
ax2.set_title('Perfil de Densidade Solar', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)

# 3. Produção acumulada de neutrinos
ax3 = plt.subplot(2, 3, 3)
# Normalizar para melhor visualização
neutrinos_normalizados = resultados_edo.y[0] / np.max(resultados_edo.y[0])
ax3.plot(raio, neutrinos_normalizados, 'g-', linewidth=2)
ax3.set_ylabel('Neutrinos Acumulados (Normalizado)', fontsize=12)
ax3.set_xlabel('Raio Normalizado', fontsize=12)
ax3.set_title('Produção Acumulada de Neutrinos', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)

# 4. Taxa de produção diferencial
ax4 = plt.subplot(2, 3, 4)
taxa_diferencial = np.gradient(resultados_edo.y[0], raio)
# Suavizar os dados para melhor visualização
from scipy.signal import savgol_filter
taxa_suavizada = savgol_filter(taxa_diferencial, 51, 3)
ax4.plot(raio, taxa_suavizada, 'm-', linewidth=2)
ax4.set_ylabel('dN/dr (neutrinos/raio)', fontsize=12)
ax4.set_xlabel('Raio Normalizado', fontsize=12)
ax4.set_title('Taxa Diferencial de Produção', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

# 5. Energia acumulada em neutrinos
ax5 = plt.subplot(2, 3, 5)
energia_normalizada = resultados_edo.y[1] / np.max(resultados_edo.y[1])
ax5.plot(raio, energia_normalizada, 'orange', linewidth=2)
ax5.set_ylabel('Energia Acumulada (Normalizada)', fontsize=12)
ax5.set_xlabel('Raio Normalizado', fontsize=12)
ax5.set_title('Energia Total em Neutrinos', fontsize=14, fontweight='bold')
ax5.grid(True, alpha=0.3)

# 6. Gráfico de comparação de massas CORRIGIDO
ax6 = plt.subplot(2, 3, 6)
massas_comparacao = {
    'Neutrinos/s': metricas['massa_perdida_kg_s'],
    'Carro (1.5t)': 1500,
    'Estátua da\nLiberdade': 2.25e5,
    'Torre Eiffel': 1.01e7
}

nomes = list(massas_comparacao.keys())
valores = list(massas_comparacao.values())

# Criar barras com cores
cores = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
bars = ax6.bar(nomes, valores, color=cores, alpha=0.8)

ax6.set_ylabel('Massa Equivalente (kg)', fontsize=12)
ax6.set_title('Massa Perdida por Segundo\n(Comparação)', fontsize=14, fontweight='bold')
ax6.tick_params(axis='x', rotation=45)
ax6.set_yscale('log')
ax6.grid(True, alpha=0.3, axis='y')

# Adicionar valores nas barras com formatação melhorada
for bar, valor in zip(bars, valores):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height*1.1,
             f'{valor:.1e}', ha='center', va='bottom', 
             fontsize=9, fontweight='bold')

plt.tight_layout(pad=3.0)
plt.show()

# =============================================================================
# GRÁFICO ADICIONAL: EVOLUÇÃO TEMPORAL CORRIGIDO
# =============================================================================

plt.figure(figsize=(12, 8))
anos = np.linspace(0, 5e9, 1000)  # 5 bilhões de anos
massa_solar_evolucao = massa_sol - metricas['massa_perdida_kg_ano'] * anos

plt.plot(anos/1e9, massa_solar_evolucao/massa_sol, 'b-', linewidth=3, 
         label='Massa Solar Relativa')
plt.xlabel('Tempo (bilhões de anos)', fontsize=14)
plt.ylabel('Massa Solar Relativa', fontsize=14)
plt.title('Evolução da Massa Solar devido à Emissão de Neutrinos', 
          fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.axhline(y=0.99, color='r', linestyle='--', alpha=0.7, linewidth=2,
           label='99% da massa original')
plt.axhline(y=0.999, color='g', linestyle='--', alpha=0.7, linewidth=2,
           label='99.9% da massa original')
plt.legend(fontsize=12)
plt.ylim(0.998, 1.001)  # Zoom para mostrar melhor a variação
plt.tight_layout()
plt.show()

# =============================================================================
# GRÁFICO EXTRA: DISTRIBUIÇÃO RADIAL DA PRODUÇÃO
# =============================================================================

plt.figure(figsize=(10, 6))

# Calcular produção por camada
produção_por_camada = taxa_suavizada * (raio_sol / 1000)  # Converter para km

plt.plot(raio * raio_sol / 1000, produção_por_camada / np.max(produção_por_camada), 
         'purple', linewidth=2)
plt.xlabel('Raio (km)', fontsize=12)
plt.ylabel('Produção Relativa de Neutrinos', fontsize=12)
plt.title('Distribuição Radial da Produção de Neutrinos', 
          fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xlim(0, raio_sol / 1000)
plt.tight_layout()
plt.show()

print(f"\n💡 RESUMO:")
print(f"• O Sol emite aproximadamente {metricas['taxa_emissao_neutrinos_s']:.3e} neutrinos por segundo")
print(f"• Isso corresponde a ~{metricas['massa_perdida_kg_s']:.3e} kg de massa perdida por segundo")
print(f"• Em escala cósmica, esta perda é extremamente pequena")
print(f"• Mesmo após 5 bilhões de anos, a perda total é desprezível")

# Informações adicionais sobre o modelo
print(f"\n🔍 INFORMAÇÕES DO MODELO:")
print(f"Temperatura central: {temperatura_central_sol:.5e} K")
print(f"Densidade central: {densidade_central_sol:.5e} kg/m³")
print(f"Fração de H no núcleo: {fracao_hidrogenio:.5f}")
print(f"Energia média por neutrino: {energia_media_neutrino:.5f} MeV")

# Mostrar valores absolutos interessantes
print(f"\n📈 VALORES ABSOLUTOS INTERESSANTES:")
print(f"Neutrinos por segundo: {metricas['taxa_emissao_neutrinos_s']:.3e}")
print(f"Isso significa: {metricas['taxa_emissao_neutrinos_s']/1e9:.2f} bilhões de bilhões de bilhões de neutrinos/s")