#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================
SIMULADOR FOTOVOLTAICO MODULAR - CALCULADORA INTERATIVA
Autor: Luiz Tiago Wilcke (LT)

Simula o comportamento de células fotovoltaicas usando
física quântica e equações de dispositivos semicondutores.
============================================================
"""

import numpy as np
from modules.quantum import SILICON, GAAS, PEROVSKITE
from modules.solar import calcular_corrente_fotogerada_limite
from modules.device import calcular_corrente_saturacao_radiativa, curva_JV_diodo
from modules.analysis import extrair_parametros
from modules.visualization import plotar_curvas


def obter_numero(prompt, valor_padrao, minimo=None, maximo=None):
    """
    Solicita entrada numérica do usuário com validação.
    """
    while True:
        entrada = input(f"{prompt} (padrão: {valor_padrao}): ").strip()
        
        if entrada == "":
            return valor_padrao
        
        try:
            valor = float(entrada)
            if minimo is not None and valor < minimo:
                print(f"❌ Valor deve ser >= {minimo}")
                continue
            if maximo is not None and valor > maximo:
                print(f"❌ Valor deve ser <= {maximo}")
                continue
            return valor
        except ValueError:
            print("❌ Digite um número válido!")


def calculadora_interativa():
    """
    Modo calculadora: permite ao usuário inserir todos os parâmetros.
    """
    print("\n" + "=" * 70)
    print(" 🧮 CALCULADORA INTERATIVA - CÉLULA FOTOVOLTAICA")
    print("=" * 70)
    print("\nConfigure os parâmetros da sua célula fotovoltaica:")
    print("(Pressione ENTER para usar o valor padrão)\n")
    
    # ========================================
    # 1. Configuração do Material
    # ========================================
    print("─" * 70)
    print("📦 PARÂMETROS DO MATERIAL")
    print("─" * 70)
    
    print("\nMateriais disponíveis:")
    print("  1 - Silício (Si)")
    print("  2 - Arsenieto de Gálio (GaAs)")
    print("  3 - Perovskita (MAPI)")
    
    material_choice = input("\nEscolha o material [1-3] (padrão: 1): ").strip()
    if material_choice == "2":
        material = GAAS
    elif material_choice == "3":
        material = PEROVSKITE
    else:
        material = SILICON
    
    print(f"\n✓ Material selecionado: {material.name}")
    
    # Parâmetros do material
    temperatura_celula = obter_numero(
        "\nTemperatura da célula [K]", 300.0, minimo=0, maximo=500
    )
    
    energia_gap_eV = obter_numero(
        "Energia de gap [eV]", 1.12, minimo=0.5, maximo=4.0
    )
    
    temperatura_sol = obter_numero(
        "Temperatura do Sol [K]", 5778.0, minimo=3000, maximo=8000
    )
    
    # ========================================
    # 2. Parâmetros do Diodo
    # ========================================
    print("\n" + "─" * 70)
    print("⚡ PARÂMETROS DO DIODO")
    print("─" * 70)
    
    fator_idealidade = obter_numero(
        "\nFator de idealidade (n)", 1.0, minimo=1.0, maximo=2.0
    )
    
    resistencia_serie = obter_numero(
        "Resistência série (Rs) [Ω·m²]", 0.5, minimo=0.0
    )
    
    resistencia_shunt = obter_numero(
        "Resistência shunt (Rsh) [Ω·m²]", 1e4, minimo=1.0
    )
    
    # ========================================
    # 3. Parâmetros de Simulação
    # ========================================
    print("\n" + "─" * 70)
    print("📊 PARÂMETROS DE SIMULAÇÃO")
    print("─" * 70)
    
    tensao_max = obter_numero(
        "\nTensão máxima [V]", 1.2, minimo=0.1, maximo=5.0
    )
    
    num_pontos = int(obter_numero(
        "Número de pontos na curva J-V", 400, minimo=50, maximo=2000
    ))
    
    # ========================================
    # 4. EXECUTAR SIMULAÇÃO
    # ========================================
    print("\n" + "=" * 70)
    print(" 🚀 EXECUTANDO SIMULAÇÃO...")
    print("=" * 70)
    
    # Cálculo da corrente fotogerada
    print("\n⏳ Calculando corrente fotogerada (J_ph)...")
    J_ph = calcular_corrente_fotogerada_limite(
        energia_gap_eV=energia_gap_eV,
        temperatura_sol=temperatura_sol,
        num_pontos_energia=4000
    )
    J_ph_mA_cm2 = J_ph * 0.1
    print(f"✓ J_ph = {J_ph:.3e} A/m² (~{J_ph_mA_cm2:.2f} mA/cm²)")
    
    # Cálculo da corrente de saturação
    print("\n⏳ Calculando corrente de saturação (J₀)...")
    J0 = calcular_corrente_saturacao_radiativa(
        energia_gap_eV=energia_gap_eV,
        temperatura_celula=temperatura_celula,
        num_pontos_energia=4000
    )
    J0_mA_cm2 = J0 * 0.1
    print(f"✓ J₀ = {J0:.3e} A/m² (~{J0_mA_cm2:.4e} mA/cm²)")
    
    # Cálculo da curva J-V
    print("\n⏳ Gerando curva J-V (Método de Newton)...")
    tensoes_V, correntes_J = curva_JV_diodo(
        J_ph=J_ph,
        J0=J0,
        temperatura_celula=temperatura_celula,
        fator_idealidade=fator_idealidade,
        resistencia_serie=resistencia_serie,
        resistencia_shunt=resistencia_shunt,
        tensao_min=0.0,
        tensao_max=tensao_max,
        num_pontos_tensao=num_pontos
    )
    print("✓ Curva J-V calculada")
    
    # Extração de parâmetros
    print("\n⏳ Extraindo parâmetros elétricos...")
    resultados = extrair_parametros(
        tensoes_V, correntes_J, J_ph, J0, temperatura_celula, fator_idealidade
    )
    
    # ========================================
    # 5. EXIBIR RESULTADOS
    # ========================================
    print("\n" + "=" * 70)
    print(" 📈 RESULTADOS DA SIMULAÇÃO")
    print("=" * 70)
    
    print(f"\n{'Parâmetro':<40} {'Valor':<30}")
    print("─" * 70)
    print(f"{'Material':<40} {material.name:<30}")
    print(f"{'Temperatura da célula':<40} {temperatura_celula:.1f} K")
    print(f"{'Energia de gap':<40} {energia_gap_eV:.3f} eV")
    print(f"{'Fator de idealidade':<40} {fator_idealidade:.2f}")
    print(f"{'Resistência série':<40} {resistencia_serie:.2f} Ω·m²")
    print(f"{'Resistência shunt':<40} {resistencia_shunt:.1e} Ω·m²")
    print("─" * 70)
    print(f"{'J_sc (curto-circuito)':<40} {resultados['J_sc']*0.1:.2f} mA/cm²")
    print(f"{'V_oc (circuito aberto)':<40} {resultados['V_oc_numerico']:.3f} V")
    print(f"{'P_max (potência máxima)':<40} {resultados['P_max']:.1f} W/m²")
    print(f"{'FF (fator de preenchimento)':<40} {resultados['FF']*100:.1f} %")
    print(f"{'η (eficiência)':<40} {resultados['Eficiencia']*100:.1f} %")
    print("=" * 70)
    
    # Plotar gráficos
    print("\n📊 Gerando gráficos...")
    plotar_curvas(tensoes_V, correntes_J, resultados['Potencias'])
    
    return resultados


def simulacao_padrao():
    """
    Executa simulação com parâmetros padrão (modo original).
    """
    material = SILICON
    temperatura_celula = 300.0
    energia_gap_eV = 1.12

    print("=" * 60)
    print(f" SIMULADOR FOTOVOLTAICO MODULAR - {material.name}")
    print("=" * 60)
    print(f"Temperatura da célula        : {temperatura_celula:.1f} K")
    print(f"Energia de gap (Eg)          : {energia_gap_eV:.3f} eV")
    print()

    print("=" * 60)
    print(" CORRENTE FOTOGERADA (Limite Quântico Shockley-Queisser)")
    print("=" * 60)
    
    J_ph = calcular_corrente_fotogerada_limite(
        energia_gap_eV=energia_gap_eV,
        temperatura_sol=5778.0,
        num_pontos_energia=4000
    )
    J_ph_mA_cm2 = J_ph * 0.1

    print(f"J_ph ~ {J_ph:.3e} A/m²  (~ {J_ph_mA_cm2:.2f} mA/cm²)")
    print()

    print("=" * 60)
    print(" CORRENTE DE SATURAÇÃO RADIATIVA (J₀)")
    print("=" * 60)
    
    J0 = calcular_corrente_saturacao_radiativa(
        energia_gap_eV=energia_gap_eV,
        temperatura_celula=temperatura_celula,
        num_pontos_energia=4000
    )
    J0_mA_cm2 = J0 * 0.1

    print(f"J₀ ~ {J0:.3e} A/m²  (~ {J0_mA_cm2:.4e} mA/cm²)")
    print()

    print("=" * 60)
    print(" CÁLCULO DA CURVA J-V (Modelo de Diodo)")
    print("=" * 60)
    
    fator_idealidade = 1.0
    resistencia_serie = 0.5
    resistencia_shunt = 1e4

    print(f"Fator de idealidade (n)      : {fator_idealidade}")
    print(f"Resistência série (Rs)       : {resistencia_serie} Ω·m²")
    print(f"Resistência shunt (Rsh)      : {resistencia_shunt} Ω·m²")
    print()

    tensoes_V, correntes_J = curva_JV_diodo(
        J_ph=J_ph,
        J0=J0,
        temperatura_celula=temperatura_celula,
        fator_idealidade=fator_idealidade,
        resistencia_serie=resistencia_serie,
        resistencia_shunt=resistencia_shunt,
        tensao_min=0.0,
        tensao_max=1.2,
        num_pontos_tensao=400
    )

    print("=" * 60)
    print(" PARÂMETROS ELÉTRICOS DA CÉLULA FOTOVOLTAICA")
    print("=" * 60)
    
    resultados = extrair_parametros(
        tensoes_V, correntes_J, J_ph, J0, temperatura_celula, fator_idealidade
    )

    print(f"J_sc (curto-circuito)        : {resultados['J_sc']:.3e} A/m² "
          f"(~ {resultados['J_sc']*0.1:.2f} mA/cm²)")
    print(f"V_oc (ideal)                 : {resultados['V_oc_ideal']:.3f} V")
    print(f"V_oc (numérico)              : {resultados['V_oc_numerico']:.3f} V")
    print(f"P_max                        : {resultados['P_max']:.1f} W/m²")
    print(f"Fator de preenchimento (FF)  : {resultados['FF']*100:.1f} %")
    print(f"Eficiência (η)               : {resultados['Eficiencia']*100:.1f} %")
    print()

    print("=" * 60)
    print(" GERANDO GRÁFICOS...")
    print("=" * 60)
    plotar_curvas(tensoes_V, correntes_J, resultados['Potencias'])


def main():
    """
    Função principal com menu de seleção.
    """
    print("\n" + "=" * 70)
    print(" 🌞 SIMULADOR FOTOVOLTAICO MODULAR")
    print(" Autor: Luiz Tiago Wilcke (LT)")
    print("=" * 70)
    
    print("\nEscolha o modo de operação:")
    print("  1 - 🧮 Calculadora Interativa (personalizar parâmetros)")
    print("  2 - ⚡ Simulação Padrão (silício, valores típicos)")
    
    escolha = input("\nSua escolha [1-2] (padrão: 2): ").strip()
    
    if escolha == "1":
        calculadora_interativa()
    else:
        simulacao_padrao()


if __name__ == "__main__":
    main()
