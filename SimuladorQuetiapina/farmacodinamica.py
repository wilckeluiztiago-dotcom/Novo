"""
Modelo Farmacodinâmico da Quetiapina
Autor: Luiz Tiago Wilcke
Data: 2025-11-25

Este módulo implementa modelos de ligação aos receptores cerebrais e 
efeitos terapêuticos e colaterais da Quetiapina.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List


@dataclass
class ReceptorCerebral:
    """Representa um receptor cerebral e suas propriedades de ligação"""
    
    nome: str
    ki_nM: float  # Constante de inibição (nanoMolar)
    densidade_cerebro: float  # Densidade no cérebro (pmol/g)
    efeito_terapeutico: float  # Peso do efeito terapêutico (0-1)
    efeito_colateral: float  # Peso do efeito colateral (0-1)
    
    def calcular_ocupacao(self, concentracao_cerebro_ng_g: float,
                         peso_molecular: float = 383.5) -> float:
        """
        Calcula a ocupação do receptor usando equação de Hill
        
        Ocupação = [C] / (Ki + [C])
        
        Args:
            concentracao_cerebro_ng_g: Concentração no cérebro (ng/g)
            peso_molecular: Peso molecular da Quetiapina (g/mol)
        
        Returns:
            Fração de ocupação (0-1)
        """
        # Converter ng/g para nM
        # 1 ng/g = (1e-9 g/g) * (1e6 mg/g) * (1/PM mol/mg) * (1e9 nM/mol)
        concentracao_nM = (concentracao_cerebro_ng_g / peso_molecular) * 1e3
        
        # Equação de Hill (n=1, ligação simples)
        ocupacao = concentracao_nM / (self.ki_nM + concentracao_nM)
        
        return min(ocupacao, 1.0)


class ModeloFarmacodinamico:
    """
    Modelo farmacodinâmico completo da Quetiapina
    
    Receptores principais:
    - 5-HT2A (Serotonina): Efeito antipsicótico, melhora sintomas negativos
    - D2 (Dopamina): Efeito antipsicótico, pode causar EPS em alta ocupação
    - H1 (Histamina): Sedação, ganho de peso
    - α1 (Adrenérgico): Hipotensão ortostática
    - M1 (Muscarínico): Efeitos anticolinérgicos
    """
    
    def __init__(self):
        # Definir receptores com base em dados da literatura
        self.receptores = {
            '5-HT2A': ReceptorCerebral(
                nome='Serotonina 5-HT2A',
                ki_nM=148.0,
                densidade_cerebro=15.0,
                efeito_terapeutico=0.8,  # Alto efeito terapêutico
                efeito_colateral=0.1
            ),
            'D2': ReceptorCerebral(
                nome='Dopamina D2',
                ki_nM=329.0,
                densidade_cerebro=20.0,
                efeito_terapeutico=0.9,  # Muito alto efeito terapêutico
                efeito_colateral=0.6  # EPS se ocupação > 80%
            ),
            'H1': ReceptorCerebral(
                nome='Histamina H1',
                ki_nM=11.0,  # Alta afinidade
                densidade_cerebro=10.0,
                efeito_terapeutico=0.2,  # Sedação pode ser útil
                efeito_colateral=0.7  # Sonolência, ganho de peso
            ),
            'α1': ReceptorCerebral(
                nome='Adrenérgico α1',
                ki_nM=47.0,
                densidade_cerebro=8.0,
                efeito_terapeutico=0.1,
                efeito_colateral=0.5  # Hipotensão
            ),
            'M1': ReceptorCerebral(
                nome='Muscarínico M1',
                ki_nM=1200.0,  # Baixa afinidade
                densidade_cerebro=12.0,
                efeito_terapeutico=0.0,
                efeito_colateral=0.3  # Boca seca, constipação
            ),
            '5-HT1A': ReceptorCerebral(
                nome='Serotonina 5-HT1A',
                ki_nM=430.0,
                densidade_cerebro=18.0,
                efeito_terapeutico=0.6,  # Efeito ansiolítico
                efeito_colateral=0.1
            )
        }
        
        self.peso_molecular_quetiapina = 383.5  # g/mol
    
    def calcular_ocupacao_receptores(self, 
                                     concentracao_cerebro_ng_g: float) -> Dict[str, float]:
        """
        Calcula a ocupação de todos os receptores
        
        Args:
            concentracao_cerebro_ng_g: Concentração cerebral (ng/g)
        
        Returns:
            Dicionário com ocupação de cada receptor (%)
        """
        ocupacoes = {}
        for nome, receptor in self.receptores.items():
            ocupacao = receptor.calcular_ocupacao(
                concentracao_cerebro_ng_g,
                self.peso_molecular_quetiapina
            )
            ocupacoes[nome] = ocupacao * 100  # Converter para %
        
        return ocupacoes
    
    def avaliar_eficacia_terapeutica(self, ocupacoes: Dict[str, float]) -> float:
        """
        Avalia a eficácia terapêutica baseada nas ocupações
        
        Critérios:
        - D2: 60-80% é ideal para antipsicótico
        - 5-HT2A: >80% melhora sintomas negativos
        - 5-HT1A: >50% efeito ansiolítico
        
        Args:
            ocupacoes: Dicionário com ocupações (%)
        
        Returns:
            Score de eficácia (0-100)
        """
        score = 0.0
        
        # D2: Curva gaussiana centrada em 70%
        ocupacao_d2 = ocupacoes['D2']
        if 60 <= ocupacao_d2 <= 80:
            score += 40 * (1 - ((ocupacao_d2 - 70) / 10) ** 2)
        elif ocupacao_d2 < 60:
            score += 40 * (ocupacao_d2 / 60)
        else:
            score += 20  # Acima de 80% ainda tem efeito mas com mais EPS
        
        # 5-HT2A: Linear até 80%, depois saturação
        ocupacao_5ht2a = ocupacoes['5-HT2A']
        score += 30 * min(ocupacao_5ht2a / 80, 1.0)
        
        # 5-HT1A: Efeito ansiolítico
        ocupacao_5ht1a = ocupacoes['5-HT1A']
        score += 20 * min(ocupacao_5ht1a / 60, 1.0)
        
        # H1: Sedação leve pode ser útil
        ocupacao_h1 = ocupacoes['H1']
        if 30 <= ocupacao_h1 <= 60:
            score += 10
        elif ocupacao_h1 < 30:
            score += 10 * (ocupacao_h1 / 30)
        
        return min(score, 100.0)
    
    def avaliar_efeitos_colaterais(self, ocupacoes: Dict[str, float]) -> Dict[str, float]:
        """
        Avalia a probabilidade de efeitos colaterais
        
        Args:
            ocupacoes: Dicionário com ocupações (%)
        
        Returns:
            Dicionário com scores de efeitos colaterais
        """
        efeitos = {}
        
        # Sintomas extrapiramidais (EPS) - D2 > 80%
        ocupacao_d2 = ocupacoes['D2']
        if ocupacao_d2 > 80:
            efeitos['EPS'] = min((ocupacao_d2 - 80) / 20 * 100, 100)
        else:
            efeitos['EPS'] = 0
        
        # Sedação - H1
        efeitos['Sedação'] = min(ocupacoes['H1'], 100)
        
        # Ganho de peso - H1 e 5-HT2C
        efeitos['Ganho_de_Peso'] = min(ocupacoes['H1'] * 0.8, 100)
        
        # Hipotensão ortostática - α1
        efeitos['Hipotensão'] = min(ocupacoes['α1'] * 0.9, 100)
        
        # Efeitos anticolinérgicos - M1
        efeitos['Anticolinérgicos'] = min(ocupacoes['M1'] * 0.7, 100)
        
        # Taquicardia - compensação α1
        efeitos['Taquicardia'] = min(ocupacoes['α1'] * 0.5, 100)
        
        return efeitos
    
    def simular_resposta_temporal(self, 
                                  tempo: np.ndarray,
                                  concentracoes_cerebro: np.ndarray) -> Dict:
        """
        Simula a resposta farmacodinâmica ao longo do tempo
        
        Args:
            tempo: Vetor de tempo (horas)
            concentracoes_cerebro: Concentrações no cérebro (ng/g)
        
        Returns:
            Dicionário com séries temporais de ocupações e efeitos
        """
        n_pontos = len(tempo)
        
        # Inicializar arrays
        ocupacoes_temporais = {nome: np.zeros(n_pontos) 
                              for nome in self.receptores.keys()}
        eficacia_temporal = np.zeros(n_pontos)
        efeitos_colaterais_temporais = {
            'EPS': np.zeros(n_pontos),
            'Sedação': np.zeros(n_pontos),
            'Ganho_de_Peso': np.zeros(n_pontos),
            'Hipotensão': np.zeros(n_pontos),
            'Anticolinérgicos': np.zeros(n_pontos),
            'Taquicardia': np.zeros(n_pontos)
        }
        
        # Calcular para cada ponto temporal
        for i, conc in enumerate(concentracoes_cerebro):
            # Ocupações
            ocupacoes = self.calcular_ocupacao_receptores(conc)
            for nome, valor in ocupacoes.items():
                ocupacoes_temporais[nome][i] = valor
            
            # Eficácia
            eficacia_temporal[i] = self.avaliar_eficacia_terapeutica(ocupacoes)
            
            # Efeitos colaterais
            efeitos = self.avaliar_efeitos_colaterais(ocupacoes)
            for efeito, valor in efeitos.items():
                efeitos_colaterais_temporais[efeito][i] = valor
        
        return {
            'ocupacoes': ocupacoes_temporais,
            'eficacia': eficacia_temporal,
            'efeitos_colaterais': efeitos_colaterais_temporais
        }
    
    def recomendar_dose(self, peso_corporal: float) -> Dict[str, float]:
        """
        Recomenda doses baseadas no peso e indicação
        
        Args:
            peso_corporal: Peso do paciente (kg)
        
        Returns:
            Dicionário com recomendações de dose
        """
        # Doses típicas (mg/dia)
        doses_recomendadas = {
            'Esquizofrenia_inicial': 50,  # Dia 1
            'Esquizofrenia_manutencao': 300,  # Dose alvo
            'Esquizofrenia_maxima': 800,
            'Bipolar_mania': 400,
            'Depressao_bipolar': 300,
            'Depressao_adjuvante': 150
        }
        
        # Ajustar para peso extremo
        if peso_corporal < 50:
            fator = 0.8
        elif peso_corporal > 100:
            fator = 1.2
        else:
            fator = 1.0
        
        doses_ajustadas = {k: v * fator for k, v in doses_recomendadas.items()}
        
        return doses_ajustadas


if __name__ == "__main__":
    print("=" * 60)
    print("SIMULADOR FARMACODINÂMICO DE QUETIAPINA")
    print("=" * 60)
    
    # Criar modelo
    modelo_pd = ModeloFarmacodinamico()
    
    # Simular concentração cerebral de pico
    conc_cerebro = 500.0  # ng/g (após dose de 300 mg)
    
    print(f"\nConcentração cerebral: {conc_cerebro} ng/g")
    print("\nOcupação de Receptores:")
    print("-" * 60)
    
    ocupacoes = modelo_pd.calcular_ocupacao_receptores(conc_cerebro)
    for receptor, ocupacao in ocupacoes.items():
        print(f"{modelo_pd.receptores[receptor].nome:25s}: {ocupacao:5.1f}%")
    
    print("\nEficácia Terapêutica:")
    print("-" * 60)
    eficacia = modelo_pd.avaliar_eficacia_terapeutica(ocupacoes)
    print(f"Score de Eficácia: {eficacia:.1f}/100")
    
    print("\nEfeitos Colaterais (Risco):")
    print("-" * 60)
    efeitos = modelo_pd.avaliar_efeitos_colaterais(ocupacoes)
    for efeito, risco in efeitos.items():
        print(f"{efeito:25s}: {risco:5.1f}%")
    
    print("\nRecomendações de Dose (70 kg):")
    print("-" * 60)
    doses = modelo_pd.recomendar_dose(70.0)
    for indicacao, dose in doses.items():
        print(f"{indicacao:30s}: {dose:6.0f} mg/dia")
    
    print("\n" + "=" * 60)
