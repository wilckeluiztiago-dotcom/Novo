"""
Módulo de Farmacogenômica Avançado
Autor: Luiz Tiago Wilcke
Data: 2025-11-25

Analisa perfil genético do paciente para predições ultra-personalizadas.
Inclui análise de polimorfismos genéticos que afetam metabolismo de drogas.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum


# Genes farmacogenômicos importantes para Quetiapina
class GeneFarmacogenetico(Enum):
    """Genes que afetam metabolismo/resposta da Quetiapina"""
    
    # Metabolismo (Fase I)
    CYP3A4 = "CYP3A4"      # Principal enzima metabolizadora
    CYP3A5 = "CYP3A5"      # Isoforma alternativa
    CYP2D6 = "CYP2D6"      # Metabolismo secundário
    
    # Metabolismo (Fase II)
    UGT1A1 = "UGT1A1"      # Glucuronidação
    SULT1A1 = "SULT1A1"    # Sulfatação
    
    # Transportadores
    ABCB1 = "ABCB1"        # P-glicoproteína (barreira hematoencefálica)
    SLC6A4 = "SLC6A4"      # Transportador de serotonina
    SLCO1B1 = "SLCO1B1"    # Transportador hepático
    
    # Receptores (farmacodinâmica)
    DRD2 = "DRD2"          # Receptor dopamina D2
    HTR2A = "HTR2A"        # Receptor serotonina 5-HT2A
    HTR2C = "HTR2C"        # Receptor serotonina 5-HT2C (ganho de peso)
    ADRA1A = "ADRA1A"      # Receptor alfa-1 adrenérgico
    
    # Risco de efeitos adversos
    HLA_B = "HLA-B"        # Reações de hipersensibilidade
    COMT = "COMT"          # Catecol-O-metiltransferase (resposta)
    BDNF = "BDNF"          # Fator neurotrófico (neuroplasticidade)


@dataclass
class VarianteGenetica:
    """Representa uma variante genética (SNP, indel, etc.)"""
    
    gene: GeneFarmacogenetico
    rs_id: str  # RefSeq ID (ex: rs4680)
    alelo_referencia: str
    alelo_alternativo: str
    genotipo: str  # Ex: "A/A", "A/G", "G/G"
    funcao: str  # Ex: "normal", "reduzida", "aumentada"
    nivel_evidencia: str  # "1A", "1B", "2A", "2B", "3", "4"
    impacto_metabolismo: float  # -1.0 (muito lento) a +1.0 (muito rápido)
    impacto_resposta: float  # -1.0 (pior) a +1.0 (melhor)
    frequencia_populacional: float  # 0-1


class PerfilFarmacogenomico:
    """Perfil genético completo de um paciente"""
    
    def __init__(self):
        self.variantes: Dict[str, VarianteGenetica] = {}
        self.populacao_ancestral: str = "europeia"  # europeia, africana, asiatica, etc.
        self.fenotipo_metabolizador: str = "normal"
        self.score_genetico_global: float = 0.0
        
    def adicionar_variante(self, variante: VarianteGenetica):
        """Adiciona variante genética ao perfil"""
        chave = f"{variante.gene.value}_{variante.rs_id}"
        self.variantes[chave] = variante
    
    def calcular_fenotipo_metabolizador(self) -> str:
        """
        Calcula fenótipo metabólico baseado em variantes CYP
        
        Returns:
            "ultra-rapido", "rapido", "normal", "intermediario", "lento"
        """
        # Somar impactos de genes CYP
        impacto_total = 0.0
        num_variantes = 0
        
        for variante in self.variantes.values():
            if variante.gene in [GeneFarmacogenetico.CYP3A4, 
                                GeneFarmacogenetico.CYP3A5,
                                GeneFarmacogenetico.CYP2D6]:
                impacto_total += variante.impacto_metabolismo
                num_variantes += 1
        
        if num_variantes == 0:
            return "normal"
        
        impacto_medio = impacto_total / num_variantes
        
        # Classificar
        if impacto_medio > 0.5:
            return "ultra-rapido"
        elif impacto_medio > 0.2:
            return "rapido"
        elif impacto_medio > -0.2:
            return "normal"
        elif impacto_medio > -0.5:
            return "intermediario"
        else:
            return "lento"
    
    def calcular_score_resposta_terapeutica(self) -> float:
        """
        Calcula score de resposta esperada baseado em genética
        
        Returns:
            Score 0-100
        """
        # Variantes favoráveis
        score = 50.0  # Baseline
        
        for variante in self.variantes.values():
            # Genes de receptores têm mais peso
            if variante.gene in [GeneFarmacogenetico.DRD2,
                                GeneFarmacogenetico.HTR2A,
                                GeneFarmacogenetico.HTR2C]:
                score += variante.impacto_resposta * 15
            
            # Genes de metabolismo afetam exposição
            elif variante.gene in [GeneFarmacogenetico.CYP3A4,
                                  GeneFarmacogenetico.CYP3A5]:
                # Metabolismo muito lento ou muito rápido = pior resposta
                impacto_abs = abs(variante.impacto_metabolismo)
                score -= impacto_abs * 10
            
            # Transportadores afetam entrada no cérebro
            elif variante.gene == GeneFarmacogenetico.ABCB1:
                score += variante.impacto_resposta * 8
        
        return np.clip(score, 0, 100)
    
    def prever_risco_efeitos_adversos(self) -> Dict[str, float]:
        """
        Prediz risco genético de efeitos adversos
        
        Returns:
            Dicionário com riscos (0-100)
        """
        riscos = {
            'ganho_peso': 20.0,
            'sindrome_metabolica': 15.0,
            'sedacao': 25.0,
            'discinesia_tardia': 5.0,
            'prolongamento_QT': 10.0
        }
        
        for variante in self.variantes.values():
            # HTR2C: ganho de peso
            if variante.gene == GeneFarmacogenetico.HTR2C:
                if variante.genotipo in ["C/C"]:  # Alelo de risco comum
                    riscos['ganho_peso'] += 25
                    riscos['sindrome_metabolica'] += 20
            
            # DRD2: discinesia tardia
            if variante.gene == GeneFarmacogenetico.DRD2:
                if variante.impacto_resposta < -0.3:
                    riscos['discinesia_tardia'] += 15
            
            # ABCB1: sedação (mais droga no cérebro)
            if variante.gene == GeneFarmacogenetico.ABCB1:
                if variante.impacto_metabolismo < -0.3:
                    riscos['sedacao'] += 20
        
        # Limitar a 100
        return {k: min(v, 100) for k, v in riscos.items()}
    
    def ajustar_dose_por_genetica(self, dose_base: float) -> Tuple[float, str]:
        """
        Ajusta dose baseado em perfil genético
        
        Args:
            dose_base: Dose padrão (mg)
        
        Returns:
            Tupla (dose_ajustada, justificativa)
        """
        fator_ajuste = 1.0
        justificativas = []
        
        fenotipo = self.calcular_fenotipo_metabolizador()
        
        # Ajustar por metabolismo
        if fenotipo == "ultra-rapido":
            fator_ajuste *= 1.5
            justificativas.append("Metabolizador ultra-rápido: +50%")
        elif fenotipo == "rapido":
            fator_ajuste *= 1.25
            justificativas.append("Metabolizador rápido: +25%")
        elif fenotipo == "intermediario":
            fator_ajuste *= 0.75
            justificativas.append("Metabolizador intermediário: -25%")
        elif fenotipo == "lento":
            fator_ajuste *= 0.5
            justificativas.append("Metabolizador lento: -50%")
        
        # Ajustar por transportadores
        for variante in self.variantes.values():
            if variante.gene == GeneFarmacogenetico.ABCB1:
                if variante.impacto_metabolismo < -0.4:
                    fator_ajuste *= 0.9
                    justificativas.append("ABCB1 reduzido: -10%")
        
        dose_ajustada = dose_base * fator_ajuste
        
        # Arredondar para múltiplo de 25
        dose_ajustada = round(dose_ajustada / 25) * 25
        dose_ajustada = np.clip(dose_ajustada, 25, 800)
        
        justificativa = " | ".join(justificativas) if justificativas else "Dose padrão (genética normal)"
        
        return dose_ajustada, justificativa


def criar_perfil_padrao(fenotipo_cyp3a4: str = "normal") -> PerfilFarmacogenomico:
    """
    Cria perfil farmacogenômico padrão para testes
    
    Args:
        fenotipo_cyp3a4: "lento", "normal", "rapido", "ultra-rapido"
    
    Returns:
        Perfil farmacogenômico
    """
    perfil = PerfilFarmacogenomico()
    
    # CYP3A4 - Principal enzima
    if fenotipo_cyp3a4 == "lento":
        variante = VarianteGenetica(
            gene=GeneFarmacogenetico.CYP3A4,
            rs_id="rs35599367",
            alelo_referencia="C",
            alelo_alternativo="T",
            genotipo="C/T",
            funcao="reduzida",
            nivel_evidencia="1A",
            impacto_metabolismo=-0.6,
            impacto_resposta=0.0,
            frequencia_populacional=0.05
        )
    elif fenotipo_cyp3a4 == "rapido":
        variante = VarianteGenetica(
            gene=GeneFarmacogenetico.CYP3A4,
            rs_id="rs2242480",
            alelo_referencia="C",
            alelo_alternativo="T",
            genotipo="T/T",
            funcao="aumentada",
            nivel_evidencia="1B",
            impacto_metabolismo=0.7,
            impacto_resposta=0.0,
            frequencia_populacional=0.12
        )
    else:  # normal
        variante = VarianteGenetica(
            gene=GeneFarmacogenetico.CYP3A4,
            rs_id="rs4986910",
            alelo_referencia="C",
            alelo_alternativo="C",
            genotipo="C/C",
            funcao="normal",
            nivel_evidencia="1A",
            impacto_metabolismo=0.0,
            impacto_resposta=0.0,
            frequencia_populacional=0.75
        )
    
    perfil.adicionar_variante(variante)
    
    # DRD2 - Receptor dopamina (rs1800497, Taq1A)
    variante_drd2 = VarianteGenetica(
        gene=GeneFarmacogenetico.DRD2,
        rs_id="rs1800497",
        alelo_referencia="G",
        alelo_alternativo="A",
        genotipo="G/A",
        funcao="densidade reduzida",
        nivel_evidencia="2A",
        impacto_metabolismo=0.0,
        impacto_resposta=0.3,  # Melhor resposta
        frequencia_populacional=0.45
    )
    perfil.adicionar_variante(variante_drd2)
    
    # HTR2C - Ganho de peso (rs3813929)
    variante_htr2c = VarianteGenetica(
        gene=GeneFarmacogenetico.HTR2C,
        rs_id="rs3813929",
        alelo_referencia="C",
        alelo_alternativo="T",
        genotipo="C/T",
        funcao="alterada",
        nivel_evidencia="1A",
        impacto_metabolismo=0.0,
        impacto_resposta=-0.2,  # Risco de ganho de peso
        frequencia_populacional=0.35
    )
    perfil.adicionar_variante(variante_htr2c)
    
    # ABCB1 - Transportador (rs1045642)
    variante_abcb1 = VarianteGenetica(
        gene=GeneFarmacogenetico.ABCB1,
        rs_id="rs1045642",
        alelo_referencia="C",
        alelo_alternativo="T",
        genotipo="C/C",
        funcao="normal",
        nivel_evidencia="2B",
        impacto_metabolismo=0.0,
        impacto_resposta=0.0,
        frequencia_populacional=0.50
    )
    perfil.adicionar_variante(variante_abcb1)
    
    # Calcular fenótipo
    perfil.fenotipo_metabolizador = perfil.calcular_fenotipo_metabolizador()
    perfil.score_genetico_global = perfil.calcular_score_resposta_terapeutica()
    
    return perfil


def gerar_perfil_aleatorio() -> PerfilFarmacogenomico:
    """Gera perfil farmacogenômico aleatório para simulações"""
    
    # Escolher fenotipo aleatoriamente
    fenotipos = ["lento", "normal", "normal", "normal", "rapido"]  # Normal mais comum
    fenotipo = np.random.choice(fenotipos)
    
    perfil = criar_perfil_padrao(fenotipo_cyp3a4=fenotipo)
    
    # População ancestral
    perfil.populacao_ancestral = np.random.choice(
        ["europeia", "africana", "asiatica", "americana_mista"],
        p=[0.55, 0.15, 0.20, 0.10]
    )
    
    return perfil


if __name__ == "__main__":
    print("=" * 80)
    print("MÓDULO DE FARMACOGENÔMICA")
    print("=" * 80)
    print()
    
    # Testar diferentes perfis
    for fenotipo in ["lento", "normal", "rapido"]:
        print(f"\n{'='*80}")
        print(f"PERFIL: Metabolizador {fenotipo.upper()}")
        print("=" * 80)
        
        perfil = criar_perfil_padrao(fenotipo_cyp3a4=fenotipo)
        
        print(f"\nFenótipo metabolizador: {perfil.fenotipo_metabolizador}")
        print(f"Score de resposta genético: {perfil.score_genetico_global:.1f}/100")
        
        print("\nVariantes genéticas:")
        for chave, variante in perfil.variantes.items():
            print(f"  {variante.gene.value} ({variante.rs_id}): "
                  f"{variante.genotipo} - {variante.funcao}")
        
        # Ajuste de dose
        dose_base = 300
        dose_ajustada, justificativa = perfil.ajustar_dose_por_genetica(dose_base)
        print(f"\nAjuste de dose:")
        print(f"  Base: {dose_base} mg")
        print(f"  Ajustada: {dose_ajustada:.0f} mg")
        print(f"  Justificativa: {justificativa}")
        
        # Riscos
        riscos = perfil.prever_risco_efeitos_adversos()
        print(f"\nRiscos genéticos:")
        for efeito, risco in riscos.items():
            nivel = "🔴" if risco > 50 else "🟡" if risco > 25 else "🟢"
            print(f"  {nivel} {efeito.replace('_', ' ').title()}: {risco:.1f}%")
    
    print("\n" + "=" * 80)
    print("✓ Módulo de farmacogenômica funcionando!")
    print("=" * 80)
