import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SimuladorFoguete:
    def __init__(self):
        # Constantes físicas
        self.gravidade_terra = 9.81  # m/s²
        self.raio_terra = 6371000    # m
        self.constante_gas_ideal = 287  # J/(kg·K)
        
        # Parâmetros do foguete (baseado no Falcon 9)
        self.massa_inicial = 549000  # kg
        self.massa_combustivel = 507000  # kg
        self.impulso_especifico = 282  # s
        self.vazao_massica = 2450  # kg/s
        self.area_transversal = 10.75  # m²
        self.coeficiente_arrasto = 0.75
        
        # Condições iniciais
        self.altitude_inicial = 0  # m
        self.velocidade_inicial = 0  # m/s
        self.angulo_lancamento = 85  # graus
        
        # Ambiente atmosférico
        self.temperatura_superficie = 288.15  # K
        self.pressao_superficie = 101325  # Pa
        self.lapse_rate = -0.0065  # K/m
        
        # Controle de queima
        self.tempo_queima_total = self.massa_combustivel / self.vazao_massica
        
    def modelo_atmosfera(self, altitude):
        """Modelo ISA simplificado da atmosfera terrestre"""
        if altitude <= 11000:  # Troposfera
            temperatura = self.temperatura_superficie + self.lapse_rate * altitude
            pressao = self.pressao_superficie * (temperatura / self.temperatura_superficie) ** (-self.gravidade_terra / (self.lapse_rate * self.constante_gas_ideal))
        else:  # Estratosfera inferior
            temperatura = 216.65  # K
            pressao_11km = self.pressao_superficie * (216.65 / self.temperatura_superficie) ** (-self.gravidade_terra / (self.lapse_rate * self.constante_gas_ideal))
            pressao = pressao_11km * np.exp(-self.gravidade_terra * (altitude - 11000) / (self.constante_gas_ideal * 216.65))
        
        densidade = pressao / (self.constante_gas_ideal * temperatura)
        return densidade, pressao, temperatura
    
    def forcas_foguete(self, t, estado):
        """
        Sistema de equações diferenciais para o movimento do foguete
        estado = [x, y, vx, vy, massa]
        """
        x, y, vx, vy, massa = estado
        velocidade = np.sqrt(vx**2 + vy**2)
        
        # Altitude atual
        altitude = np.sqrt(x**2 + y**2) - self.raio_terra
        
        # Gravidade variável com altitude
        gravidade = self.gravidade_terra * (self.raio_terra / (self.raio_terra + altitude))**2
        
        # Densidade do ar
        densidade_ar, _, _ = self.modelo_atmosfera(altitude)
        
        # Empuxo do motor
        if t <= self.tempo_queima_total:
            empuxo = self.impulso_especifico * self.gravidade_terra * self.vazao_massica
            dmassa_dt = -self.vazao_massica
        else:
            empuxo = 0
            dmassa_dt = 0
        
        # Ângulo de voo
        if velocidade > 0.1:
            angulo_voo = np.arctan2(vy, vx)
        else:
            angulo_voo = np.radians(self.angulo_lancamento)
        
        # Força de arrasto
        forca_arrasto = 0.5 * densidade_ar * velocidade**2 * self.area_transversal * self.coeficiente_arrasto
        
        # Componentes das forças
        forca_gravidade_x = -gravidade * (x / np.sqrt(x**2 + y**2)) * massa
        forca_gravidade_y = -gravidade * (y / np.sqrt(x**2 + y**2)) * massa
        
        forca_empuxo_x = empuxo * np.cos(angulo_voo)
        forca_empuxo_y = empuxo * np.sin(angulo_voo)
        
        forca_arrasto_x = -forca_arrasto * (vx / velocidade) if velocidade > 0 else 0
        forca_arrasto_y = -forca_arrasto * (vy / velocidade) if velocidade > 0 else 0
        
        # Acelerações
        ax = (forca_empuxo_x + forca_arrasto_x + forca_gravidade_x) / massa
        ay = (forca_empuxo_y + forca_arrasto_y + forca_gravidade_y) / massa
        
        return [vx, vy, ax, ay, dmassa_dt]
    
    def simular_lancamento(self, tempo_simulacao=600, dt=0.1):
        """Executa a simulação completa do lançamento"""
        print("Iniciando simulação de lançamento de foguete...")
        
        # Condições iniciais
        angulo_rad = np.radians(self.angulo_lancamento)
        x0 = 0
        y0 = self.raio_terra + self.altitude_inicial
        vx0 = self.velocidade_inicial * np.cos(angulo_rad)
        vy0 = self.velocidade_inicial * np.sin(angulo_rad)
        
        estado_inicial = [x0, y0, vx0, vy0, self.massa_inicial]
        
        # Tempo de simulação
        t_eval = np.arange(0, tempo_simulacao, dt)
        
        # Resolver EDOs
        solucao = solve_ivp(
            self.forcas_foguete, 
            [0, tempo_simulacao], 
            estado_inicial, 
            t_eval=t_eval, 
            method='RK45',
            rtol=1e-8
        )
        
        self.tempo = solucao.t
        self.x = solucao.y[0]
        self.y = solucao.y[1]
        self.vx = solucao.y[2]
        self.vy = solucao.y[3]
        self.massa = solucao.y[4]
        
        # Calcular quantidades derivadas
        self.calcular_metricas()
        
        print("Simulação concluída!")
        return solucao
    
    def calcular_metricas(self):
        """Calcula métricas importantes do voo"""
        self.altitude = np.sqrt(self.x**2 + self.y**2) - self.raio_terra
        self.velocidade = np.sqrt(self.vx**2 + self.vy**2)
        self.aceleracao = np.gradient(self.velocidade, self.tempo)
        
        # Encontrar eventos importantes
        self.indice_apogeu = np.argmax(self.altitude)
        self.tempo_apogeu = self.tempo[self.indice_apogeu]
        self.altitude_apogeu = self.altitude[self.indice_apogeu]
        
        # Velocidade orbital (para referência)
        self.velocidade_orbital = np.sqrt(self.gravidade_terra * self.raio_terra**2 / 
                                        (self.raio_terra + self.altitude))
        
        print(f"Altitude máxima: {self.altitude_apogeu/1000:.2f} km")
        print(f"Tempo até apogeu: {self.tempo_apogeu:.2f} s")
        print(f"Velocidade máxima: {np.max(self.velocidade):.2f} m/s")

class VisualizadorFoguete:
    def __init__(self, simulador):
        self.sim = simulador
        self.fig = plt.figure(figsize=(16, 10))
        self.setup_plots()
        
    def setup_plots(self):
        """Configura os subplots para a visualização"""
        # Layout da figura
        gs = self.fig.add_gridspec(3, 3)
        
        # Trajetória principal
        self.ax_trajetoria = self.fig.add_subplot(gs[0:2, 0:2], projection='3d')
        
        # Gráficos de telemetria
        self.ax_altitude = self.fig.add_subplot(gs[0, 2])
        self.ax_velocidade = self.fig.add_subplot(gs[1, 2])
        self.ax_aceleracao = self.fig.add_subplot(gs[2, 0])
        self.ax_massa = self.fig.add_subplot(gs[2, 1])
        self.ax_metricas = self.fig.add_subplot(gs[2, 2])
        
        self.configurar_eixos()
        
    def configurar_eixos(self):
        """Configura a aparência dos eixos"""
        # Trajetória 3D
        self.ax_trajetoria.set_xlabel('X (km)')
        self.ax_trajetoria.set_ylabel('Y (km)')
        self.ax_trajetoria.set_zlabel('Altitude (km)')
        self.ax_trajetoria.set_title('TRAJETÓRIA DO FOGUETE', fontweight='bold')
        
        # Telemetria
        self.ax_altitude.set_title('ALTITUDE vs TEMPO')
        self.ax_altitude.set_xlabel('Tempo (s)')
        self.ax_altitude.set_ylabel('Altitude (km)')
        self.ax_altitude.grid(True, alpha=0.3)
        
        self.ax_velocidade.set_title('VELOCIDADE vs TEMPO')
        self.ax_velocidade.set_xlabel('Tempo (s)')
        self.ax_velocidade.set_ylabel('Velocidade (m/s)')
        self.ax_velocidade.grid(True, alpha=0.3)
        
        self.ax_aceleracao.set_title('ACELERAÇÃO vs TEMPO')
        self.ax_aceleracao.set_xlabel('Tempo (s)')
        self.ax_aceleracao.set_ylabel('Aceleração (m/s²)')
        self.ax_aceleracao.grid(True, alpha=0.3)
        
        self.ax_massa.set_title('MASSA vs TEMPO')
        self.ax_massa.set_xlabel('Tempo (s)')
        self.ax_massa.set_ylabel('Massa (ton)')
        self.ax_massa.grid(True, alpha=0.3)
        
        self.ax_metricas.set_title('MÉTRICAS DE VOQ')
        self.ax_metricas.axis('off')
        
    def animar_lancamento(self, salvar_animacao=False):
        """Cria animação completa do lançamento"""
        print("Preparando animação...")
        
        # Preparar dados
        x_km = self.sim.x / 1000
        y_km = self.sim.y / 1000
        altitude_km = self.sim.altitude / 1000
        massa_ton = self.sim.massa / 1000
        
        # Configurar trajetória 3D
        max_range = max(np.max(np.abs(x_km)), np.max(np.abs(y_km)), np.max(altitude_km))
        self.ax_trajetoria.set_xlim([-max_range, max_range])
        self.ax_trajetoria.set_ylim([-max_range, max_range])
        self.ax_trajetoria.set_zlim([0, max_range])
        
        # Adicionar Terra
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_terra = np.outer(np.cos(u), np.sin(v)) * (self.sim.raio_terra / 1000)
        y_terra = np.outer(np.sin(u), np.sin(v)) * (self.sim.raio_terra / 1000)
        z_terra = np.outer(np.ones(np.size(u)), np.cos(v)) * (self.sim.raio_terra / 1000)
        self.ax_trajetoria.plot_surface(x_terra, y_terra, z_terra, 
                                      color='blue', alpha=0.3, rstride=4, cstride=4)
        
        # Inicializar elementos da animação
        linha_trajetoria, = self.ax_trajetoria.plot([], [], [], 'r-', linewidth=2, alpha=0.7)
        ponto_foguete, = self.ax_trajetoria.plot([], [], [], 'ro', markersize=8)
        
        # Gráficos de telemetria
        linha_altitude, = self.ax_altitude.plot([], [], 'b-', linewidth=2)
        ponto_altitude, = self.ax_altitude.plot([], [], 'bo', markersize=4)
        
        linha_velocidade, = self.ax_velocidade.plot([], [], 'g-', linewidth=2)
        ponto_velocidade, = self.ax_velocidade.plot([], [], 'go', markersize=4)
        
        linha_aceleracao, = self.ax_aceleracao.plot([], [], 'r-', linewidth=2)
        ponto_aceleracao, = self.ax_aceleracao.plot([], [], 'ro', markersize=4)
        
        linha_massa, = self.ax_massa.plot([], [], 'purple', linewidth=2)
        ponto_massa, = self.ax_massa.plot([], [], 'o', color='purple', markersize=4)
        
        # Texto de métricas
        texto_metricas = self.ax_metricas.text(0.1, 0.9, '', transform=self.ax_metricas.transAxes,
                                             fontfamily='monospace', fontsize=10,
                                             verticalalignment='top')
        
        def animar(frame):
            # Atualizar trajetória
            linha_trajetoria.set_data(x_km[:frame], y_km[:frame])
            linha_trajetoria.set_3d_properties(altitude_km[:frame])
            
            ponto_foguete.set_data([x_km[frame]], [y_km[frame]])
            ponto_foguete.set_3d_properties([altitude_km[frame]])
            
            # Atualizar telemetria
            linha_altitude.set_data(self.sim.tempo[:frame], altitude_km[:frame])
            ponto_altitude.set_data([self.sim.tempo[frame]], [altitude_km[frame]])
            self.ax_altitude.relim()
            self.ax_altitude.autoscale_view()
            
            linha_velocidade.set_data(self.sim.tempo[:frame], self.sim.velocidade[:frame])
            ponto_velocidade.set_data([self.sim.tempo[frame]], [self.sim.velocidade[frame]])
            self.ax_velocidade.relim()
            self.ax_velocidade.autoscale_view()
            
            linha_aceleracao.set_data(self.sim.tempo[:frame], self.sim.aceleracao[:frame])
            ponto_aceleracao.set_data([self.sim.tempo[frame]], [self.sim.aceleracao[frame]])
            self.ax_aceleracao.relim()
            self.ax_aceleracao.autoscale_view()
            
            linha_massa.set_data(self.sim.tempo[:frame], massa_ton[:frame])
            ponto_massa.set_data([self.sim.tempo[frame]], [massa_ton[frame]])
            self.ax_massa.relim()
            self.ax_massa.autoscale_view()
            
            # Atualizar métricas
            metricas_texto = (
                f"TEMPO: {self.sim.tempo[frame]:.1f} s\n"
                f"ALTITUDE: {altitude_km[frame]:.1f} km\n"
                f"VELOCIDADE: {self.sim.velocidade[frame]:.1f} m/s\n"
                f"ACELERAÇÃO: {self.sim.aceleracao[frame]:.1f} m/s²\n"
                f"MASSA: {massa_ton[frame]:.0f} ton\n"
                f"FASE: {'QUEIMA' if self.sim.tempo[frame] <= self.sim.tempo_queima_total else 'BALÍSTICA'}"
            )
            texto_metricas.set_text(metricas_texto)
            
            return (linha_trajetoria, ponto_foguete, linha_altitude, ponto_altitude,
                   linha_velocidade, ponto_velocidade, linha_aceleracao, ponto_aceleracao,
                   linha_massa, ponto_massa, texto_metricas)
        
        # Criar animação
        frames = len(self.sim.tempo)
        anim = animation.FuncAnimation(
            self.fig, animar, frames=frames, interval=20, blit=False, repeat=True
        )
        
        plt.tight_layout()
        
        if salvar_animacao:
            print("Salvando animação...")
            anim.save('lancamento_foguete.mp4', writer='ffmpeg', fps=30, dpi=150)
            print("Animação salva como 'lancamento_foguete.mp4'")
        
        plt.show()
        return anim

def main():
    """Função principal para executar a simulação completa"""
    print("=" * 60)
    print("SIMULAÇÃO SOFISTICADA DE LANÇAMENTO DE FOGUETE")
    print("Desenvolvido com Equações Diferenciais Avançadas")
    print("=" * 60)
    
    # Criar e executar simulação
    simulador = SimuladorFoguete()
    simulador.simular_lancamento(tempo_simulacao=800, dt=0.5)
    
    # Criar visualização
    visualizador = VisualizadorFoguete(simulador)
    
    # Gerar animação
    animacao = visualizador.animar_lancamento(salvar_animacao=False)
    
    # Plotar resultados finais
    plotar_resultados_finais(simulador)

def plotar_resultados_finais(simulador):
    """Gera gráficos detalhados dos resultados"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Trajetória completa
    axes[0,0].plot(simulador.x/1000, simulador.y/1000, 'b-', linewidth=2)
    axes[0,0].set_xlabel('X (km)')
    axes[0,0].set_ylabel('Y (km)')
    axes[0,0].set_title('Trajetória no Plano XY')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_aspect('equal')
    
    # Altitude vs Tempo
    axes[0,1].plot(simulador.tempo, simulador.altitude/1000, 'g-', linewidth=2)
    axes[0,1].axvline(x=simulador.tempo_queima_total, color='r', linestyle='--', 
                     label='Fim da Queima')
    axes[0,1].set_xlabel('Tempo (s)')
    axes[0,1].set_ylabel('Altitude (km)')
    axes[0,1].set_title('Altitude vs Tempo')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Velocidade vs Tempo
    axes[0,2].plot(simulador.tempo, simulador.velocidade, 'r-', linewidth=2)
    axes[0,2].axvline(x=simulador.tempo_queima_total, color='r', linestyle='--')
    axes[0,2].set_xlabel('Tempo (s)')
    axes[0,2].set_ylabel('Velocidade (m/s)')
    axes[0,2].set_title('Velocidade vs Tempo')
    axes[0,2].grid(True, alpha=0.3)
    
    # Aceleração vs Tempo
    axes[1,0].plot(simulador.tempo, simulador.aceleracao, 'orange', linewidth=2)
    axes[1,0].axvline(x=simulador.tempo_queima_total, color='r', linestyle='--')
    axes[1,0].set_xlabel('Tempo (s)')
    axes[1,0].set_ylabel('Aceleração (m/s²)')
    axes[1,0].set_title('Aceleração vs Tempo')
    axes[1,0].grid(True, alpha=0.3)
    
    # Massa vs Tempo
    axes[1,1].plot(simulador.tempo, simulador.massa/1000, 'purple', linewidth=2)
    axes[1,1].axvline(x=simulador.tempo_queima_total, color='r', linestyle='--')
    axes[1,1].set_xlabel('Tempo (s)')
    axes[1,1].set_ylabel('Massa (ton)')
    axes[1,1].set_title('Massa vs Tempo')
    axes[1,1].grid(True, alpha=0.3)
    
    # Gráfico de fase
    axes[1,2].plot(simulador.altitude/1000, simulador.velocidade, 'b-', linewidth=2)
    axes[1,2].set_xlabel('Altitude (km)')
    axes[1,2].set_ylabel('Velocidade (m/s)')
    axes[1,2].set_title('Diagrama de Fase: Velocidade vs Altitude')
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()