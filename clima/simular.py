"""
SCRIPT PRINCIPAL - Simulação do Modelo Climático
=================================================

Executa simulação completa do modelo atmosférico e gera visualizações.

Autor: Luiz Tiago Wilcke
Data: 2025-11-24
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from grid import GradeEspacial
from atmosphere import ModeloAtmosferico
from utils import kelvin_para_celsius


def criar_visualizacoes(modelo: ModeloAtmosferico, historico: dict, diretorio_saida: str):
    """
    Cria visualizações dos resultados da simulação.
    
    Args:
        modelo: Modelo atmosférico
        historico: Dicionário com histórico da simulação
        diretorio_saida: Diretório para salvar figuras
    """
    Path(diretorio_saida).mkdir(parents=True, exist_ok=True)
    
    tempo_dias = historico['tempo'] / 86400
    
    # Configurar estilo
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    
    # ========== FIGURA 1: Evolução Temporal ==========
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Temperatura atmosférica
    ax = axes[0]
    T_atm_celsius = kelvin_para_celsius(historico['temperatura_media'])
    ax.plot(tempo_dias, T_atm_celsius, 'b-', linewidth=2, label='T atmosférica')
    ax.set_ylabel('Temperatura (°C)', fontsize=11)
    ax.set_title('Evolução Temporal - Modelo Climático', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Temperatura de superfície
    ax = axes[1]
    T_surf_celsius = kelvin_para_celsius(historico['temperatura_superficie_media'])
    ax.plot(tempo_dias, T_surf_celsius, 'r-', linewidth=2, label='T superfície')
    ax.set_ylabel('Temperatura (°C)', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Velocidade do vento
    ax = axes[2]
    ax.plot(tempo_dias, historico['velocidade_vento_max'], 'g-', linewidth=2, label='Vento máximo')
    ax.set_xlabel('Tempo (dias)', fontsize=11)
    ax.set_ylabel('Velocidade (m/s)', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{diretorio_saida}/evolucao_temporal.png', dpi=150, bbox_inches='tight')
    print(f"✓ Salvo: {diretorio_saida}/evolucao_temporal.png")
    plt.close()
    
    # ========== FIGURA 2: Campos Espaciais ==========
    fig = plt.figure(figsize=(15, 10))
    
    # Temperatura de superfície
    ax1 = plt.subplot(2, 2, 1)
    temp_surf = modelo.estado.temperatura_superficie
    im1 = ax1.contourf(
        modelo.grade.longitudes,
        modelo.grade.latitudes,
        kelvin_para_celsius(temp_surf),
        levels=20,
        cmap='RdBu_r'
    )
    ax1.set_title('Temperatura de Superfície (°C)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Longitude (°)')
    ax1.set_ylabel('Latitude (°)')
    plt.colorbar(im1, ax=ax1)
    
    # Vento zonal (nível mais baixo)
    ax2 = plt.subplot(2, 2, 2)
    u_superficie = modelo.estado.u[-1, :, :]
    im2 = ax2.contourf(
        modelo.grade.longitudes,
        modelo.grade.latitudes,
        u_superficie,
        levels=20,
        cmap='RdBu_r'
    )
    ax2.set_title('Vento Zonal - Superfície (m/s)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Longitude (°)')
    ax2.set_ylabel('Latitude (°)')
    plt.colorbar(im2, ax=ax2)
    
    # Temperatura atmosférica (nível médio)
    ax3 = plt.subplot(2, 2, 3)
    nivel_medio = len(modelo.grade.niveis_pressao) // 2
    temp_medio = modelo.estado.temperatura[nivel_medio, :, :]
    im3 = ax3.contourf(
        modelo.grade.longitudes,
        modelo.grade.latitudes,
        kelvin_para_celsius(temp_medio),
        levels=20,
        cmap='RdBu_r'
    )
    p_nivel = modelo.grade.niveis_pressao[nivel_medio] / 100  # hPa
    ax3.set_title(f'Temperatura em {p_nivel:.0f} hPa (°C)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Longitude (°)')
    ax3.set_ylabel('Latitude (°)')
    plt.colorbar(im3, ax=ax3)
    
    # Perfil zonal médio de temperatura
    ax4 = plt.subplot(2, 2, 4)
    temp_zonal = np.mean(temp_surf, axis=1)
    ax4.plot(modelo.grade.latitudes, kelvin_para_celsius(temp_zonal), 'b-', linewidth=2)
    ax4.set_title('Perfil Zonal de Temperatura', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Latitude (°)')
    ax4.set_ylabel('Temperatura (°C)')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax4.axvline(0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{diretorio_saida}/campos_espaciais.png', dpi=150, bbox_inches='tight')
    print(f"✓ Salvo: {diretorio_saida}/campos_espaciais.png")
    plt.close()
    
    # ========== FIGURA 3: Estrutura Vertical ==========
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Perfil vertical de temperatura (média zonal)
    ax = axes[0]
    temp_vertical = np.mean(modelo.estado.temperatura, axis=(1, 2))
    pressao_hpa = modelo.grade.niveis_pressao / 100
    ax.plot(kelvin_para_celsius(temp_vertical), pressao_hpa, 'b-', linewidth=2, marker='o')
    ax.set_xlabel('Temperatura (°C)', fontsize=11)
    ax.set_ylabel('Pressão (hPa)', fontsize=11)
    ax.set_title('Perfil Vertical de Temperatura', fontsize=12, fontweight='bold')
    ax.invert_yaxis()  # Pressão decresce com altura
    ax.grid(True, alpha=0.3)
    
    # Perfil vertical de vento zonal
    ax = axes[1]
    u_vertical = np.mean(modelo.estado.u, axis=(1, 2))
    ax.plot(u_vertical, pressao_hpa, 'g-', linewidth=2, marker='o')
    ax.set_xlabel('Vento Zonal (m/s)', fontsize=11)
    ax.set_ylabel('Pressão (hPa)', fontsize=11)
    ax.set_title('Perfil Vertical de Vento Zonal', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.axvline(0, color='k', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f'{diretorio_saida}/estrutura_vertical.png', dpi=150, bbox_inches='tight')
    print(f"✓ Salvo: {diretorio_saida}/estrutura_vertical.png")
    plt.close()


def imprimir_resumo(modelo: ModeloAtmosferico, historico: dict):
    """Imprime resumo estatístico da simulação"""
    print("\n" + "=" * 70)
    print("RESUMO DA SIMULAÇÃO")
    print("=" * 70)
    
    T_atm_inicial = historico['temperatura_media'][0]
    T_atm_final = historico['temperatura_media'][-1]
    T_surf_inicial = historico['temperatura_superficie_media'][0]
    T_surf_final = historico['temperatura_superficie_media'][-1]
    
    print(f"\n📊 ESTATÍSTICAS GLOBAIS:")
    print(f"   Temperatura atmosférica:")
    print(f"      Inicial: {kelvin_para_celsius(T_atm_inicial):6.2f}°C ({T_atm_inicial:6.2f} K)")
    print(f"      Final:   {kelvin_para_celsius(T_atm_final):6.2f}°C ({T_atm_final:6.2f} K)")
    print(f"      Δ:       {T_atm_final - T_atm_inicial:+6.2f} K")
    
    print(f"\n   Temperatura de superfície:")
    print(f"      Inicial: {kelvin_para_celsius(T_surf_inicial):6.2f}°C ({T_surf_inicial:6.2f} K)")
    print(f"      Final:   {kelvin_para_celsius(T_surf_final):6.2f}°C ({T_surf_final:6.2f} K)")
    print(f"      Δ:       {T_surf_final - T_surf_inicial:+6.2f} K")
    
    print(f"\n   Velocidade do vento:")
    print(f"      Inicial: {historico['velocidade_vento_max'][0]:6.2f} m/s")
    print(f"      Final:   {historico['velocidade_vento_max'][-1]:6.2f} m/s")
    print(f"      Máxima:  {np.max(historico['velocidade_vento_max']):6.2f} m/s")
    
    # Distribuição de temperatura
    T_surf_final_campo = modelo.estado.temperatura_superficie
    print(f"\n   Distribuição de temperatura de superfície:")
    print(f"      Mínima:  {kelvin_para_celsius(np.min(T_surf_final_campo)):6.2f}°C")
    print(f"      Máxima:  {kelvin_para_celsius(np.max(T_surf_final_campo)):6.2f}°C")
    print(f"      Média:   {kelvin_para_celsius(np.mean(T_surf_final_campo)):6.2f}°C")
    print(f"      Desvio:  {np.std(T_surf_final_campo):6.2f} K")
    
    print("\n" + "=" * 70)


def main():
    """Função principal"""
    print("=" * 70)
    print("SIMULAÇÃO DO MODELO CLIMÁTICO")
    print("=" * 70)
    
    # Configurar simulação
    print("\n🔧 CONFIGURAÇÃO:")
    resolucao_lat = 45
    resolucao_lon = 90
    niveis_atm = 10
    dias_simulacao = 30
    dt_horas = 2.0
    
    print(f"   Resolução: {resolucao_lat} lat × {resolucao_lon} lon")
    print(f"   Níveis atmosféricos: {niveis_atm}")
    print(f"   Duração: {dias_simulacao} dias")
    print(f"   Passo de tempo: {dt_horas} horas")
    
    # Criar grade
    print("\n🌍 Criando grade espacial...")
    grade = GradeEspacial(
        num_lat=resolucao_lat,
        num_lon=resolucao_lon,
        num_niveis_atm=niveis_atm
    )
    print(f"   ✓ Grade criada: {grade.area_total/1e14:.2f} × 10¹⁴ m²")
    
    # Criar modelo
    print("\n⚙️  Inicializando modelo atmosférico...")
    modelo = ModeloAtmosferico(grade)
    print("   ✓ Modelo inicializado")
    
    # Executar simulação
    print(f"\n🚀 Executando simulação de {dias_simulacao} dias...")
    print("   (Isso pode demorar alguns minutos...)")
    historico = modelo.integrar(dias_simulacao=dias_simulacao, dt_horas=dt_horas)
    
    # Resumo
    imprimir_resumo(modelo, historico)
    
    # Visualizações
    print("\n📊 Gerando visualizações...")
    diretorio_saida = "outputs"
    criar_visualizacoes(modelo, historico, diretorio_saida)
    
    print(f"\n✅ SIMULAÇÃO CONCLUÍDA!")
    print(f"   Resultados salvos em: {diretorio_saida}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
