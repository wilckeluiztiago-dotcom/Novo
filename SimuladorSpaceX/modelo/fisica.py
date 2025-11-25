import numpy as np
from .parametros import ParametrosFalcon9 as P

class Foguete:
    def __init__(self, carga_util=0, combustivel_inicial=None):
        self.carga_util = carga_util
        self.massa_combustivel = combustivel_inicial if combustivel_inicial is not None else P.MASSA_COMBUSTIVEL_MAX
        self.massa_seca = P.MASSA_SECA
        self.tempo_queima = 0
        
    @property
    def massa_total(self):
        return self.massa_seca + self.massa_combustivel + self.carga_util

    def calcular_gravidade(self, altitude):
        """g(h) = g0 * (Re / (Re + h))^2"""
        r = P.R_TERRA + altitude
        return P.G0 * (P.R_TERRA / r)**2

    def calcular_densidade_ar(self, altitude):
        """Modelo exponencial simples: rho = rho0 * exp(-h / H)"""
        if altitude < 0: return P.RHO_AR_0
        return P.RHO_AR_0 * np.exp(-altitude / P.H_ESCALA)

    def calcular_empuxo(self, altitude):
        """
        Empuxo varia com a pressão atmosférica (altitude).
        Interpolação linear simples entre nível do mar e vácuo baseada na densidade.
        """
        if self.massa_combustivel <= 0:
            return 0.0, 0.0

        # Fator de interpolação (1 no mar, 0 no vácuo)
        fator_atm = self.calcular_densidade_ar(altitude) / P.RHO_AR_0
        empuxo = P.EMPUXO_VACUO - (P.EMPUXO_VACUO - P.EMPUXO_NIVEL_MAR) * fator_atm
        isp = P.ISP_VACUO - (P.ISP_VACUO - P.ISP_NIVEL_MAR) * fator_atm
        
        return empuxo, isp

    def derivadas(self, t, estado):
        """
        Calcula as derivadas [dy/dt, dv/dt, dm/dt]
        estado = [altitude (y), velocidade (v), massa_combustivel (m)]
        """
        y, v, m_comb = estado
        
        # Se acabou o combustível ou altitude < 0 (chão), parar
        if y < 0: y = 0
        
        # Atualizar massa interna
        self.massa_combustivel = max(0, m_comb)
        m_total = self.massa_total
        
        # 1. Gravidade
        g = self.calcular_gravidade(y)
        fg = m_total * g
        
        # 2. Arrasto
        rho = self.calcular_densidade_ar(y)
        fd = 0.5 * rho * (v**2) * P.COEF_ARRASTO * P.AREA_FRONTAL * np.sign(v)
        
        # 3. Empuxo
        empuxo, isp = self.calcular_empuxo(y)
        
        # Taxa de variação da massa (dm/dt = -F / (Isp * g0))
        if self.massa_combustivel > 0:
            dm_dt = -empuxo / (isp * P.G0)
        else:
            dm_dt = 0
            empuxo = 0
            
        # Força Resultante
        f_res = empuxo - fg - fd
        
        # Aceleração (2ª Lei de Newton)
        a = f_res / m_total
        
        return np.array([v, a, dm_dt])

def solver_rk4(foguete, t_max=300, dt=0.1):
    """
    Resolve as equações diferenciais usando Runge-Kutta de 4ª Ordem.
    """
    t = np.arange(0, t_max, dt)
    n_steps = len(t)
    
    # Estado: [Altitude, Velocidade, MassaCombustivel]
    estado = np.zeros((n_steps, 3))
    estado[0] = [0.0, 0.0, foguete.massa_combustivel] # Condição inicial
    
    resultados = {
        'tempo': t,
        'altitude': np.zeros(n_steps),
        'velocidade': np.zeros(n_steps),
        'aceleracao': np.zeros(n_steps),
        'massa': np.zeros(n_steps),
        'empuxo': np.zeros(n_steps)
    }
    
    for i in range(n_steps - 1):
        h = dt
        y_n = estado[i]
        t_n = t[i]
        
        # RK4 Steps
        k1 = foguete.derivadas(t_n, y_n)
        k2 = foguete.derivadas(t_n + h/2, y_n + h*k1/2)
        k3 = foguete.derivadas(t_n + h/2, y_n + h*k2/2)
        k4 = foguete.derivadas(t_n + h, y_n + h*k3)
        
        estado[i+1] = y_n + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Evitar massa negativa e altitude negativa
        if estado[i+1][2] < 0: estado[i+1][2] = 0
        if estado[i+1][0] < 0: 
            estado[i+1][0] = 0
            estado[i+1][1] = 0 # Parar se bater no chão
            
        # Salvar dados auxiliares para plotagem
        resultados['altitude'][i] = y_n[0]
        resultados['velocidade'][i] = y_n[1]
        resultados['massa'][i] = foguete.massa_seca + foguete.carga_util + y_n[2]
        resultados['aceleracao'][i] = k1[1] # Aceleração instantânea
        
        empuxo, _ = foguete.calcular_empuxo(y_n[0])
        resultados['empuxo'][i] = empuxo

    # Preencher último ponto
    resultados['altitude'][-1] = estado[-1][0]
    resultados['velocidade'][-1] = estado[-1][1]
    resultados['massa'][-1] = foguete.massa_seca + foguete.carga_util + estado[-1][2]
    
    return resultados
