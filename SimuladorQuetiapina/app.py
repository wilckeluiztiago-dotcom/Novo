"""
Dashboard Interativo - Simulador de Quetiapina
Autor: Luiz Tiago Wilcke
Data: 2025-11-25

Interface web interativa usando Streamlit para simulação completa
da farmacocinética e farmacodinâmica da Quetiapina no cérebro humano.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from farmacocinetica import (ParametrosFarmacocineticos, 
                             ModeloFarmacocinetico, 
                             RegimePosologico)
from farmacodinamica import ModeloFarmacodinamico
from visualizacao import VisualizadorQuetiapina
import sys

# Configuração da página
st.set_page_config(
    page_title="Simulador de Quetiapina",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #2C3E50;
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .sub-header {
        font-size: 24px;
        font-weight: bold;
        color: #34495E;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .info-box {
        background-color: #ECF0F1;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #3498DB;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #FEF5E7;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #F39C12;
        margin: 10px 0;
    }
    .success-box {
        background-color: #E8F8F5;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #2ECC71;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


def criar_cabecalho():
    """Cria cabeçalho principal"""
    st.markdown('<div class="main-header">💊 SIMULADOR DE QUETIAPINA NO CÉREBRO HUMANO</div>', 
               unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <b>🧠 Simulação Avançada Farmacocinética & Farmacodinâmica</b><br>
        Modelo matemático completo baseado em equações diferenciais para simular
        a absorção, distribuição, metabolismo, excreção (ADME) e ocupação de 
        receptores cerebrais da Quetiapina.
    </div>
    """, unsafe_allow_html=True)


def parametros_sidebar():
    """Cria sidebar com parâmetros de entrada"""
    st.sidebar.title("⚙️ Parâmetros da Simulação")
    
    # Dados do paciente
    st.sidebar.markdown("### 👤 Dados do Paciente")
    peso_corporal = st.sidebar.number_input(
        "Peso Corporal (kg)",
        min_value=30.0,
        max_value=150.0,
        value=70.0,
        step=1.0,
        help="Peso do paciente em quilogramas"
    )
    
    # Dados da medicação
    st.sidebar.markdown("### 💊 Medicamento")
    dose_mg = st.sidebar.number_input(
        "Dose (mg)",
        min_value=25.0,
        max_value=800.0,
        value=300.0,
        step=25.0,
        help="Dose de Quetiapina em miligramas"
    )
    
    via_admin = st.sidebar.selectbox(
        "Via de Administração",
        ["oral", "intravenosa"],
        help="Via de administração do medicamento"
    )
    
    # Regime posológico
    st.sidebar.markdown("### 📅 Regime Posológico")
    tipo_regime = st.sidebar.radio(
        "Tipo de Regime",
        ["Dose Única", "Doses Múltiplas"],
        help="Simular dose única ou regime de doses múltiplas"
    )
    
    if tipo_regime == "Doses Múltiplas":
        num_doses = st.sidebar.slider(
            "Número de Doses",
            min_value=2,
            max_value=10,
            value=5,
            help="Número total de doses a administrar"
        )
        
        intervalo_horas = st.sidebar.slider(
            "Intervalo entre Doses (horas)",
            min_value=6.0,
            max_value=24.0,
            value=12.0,
            step=1.0,
            help="Intervalo de tempo entre administrações"
        )
    else:
        num_doses = 1
        intervalo_horas = 24.0
    
    # Tempo de simulação
    st.sidebar.markdown("### ⏱️ Tempo de Simulação")
    tempo_total = st.sidebar.slider(
        "Tempo Total (horas)",
        min_value=12.0,
        max_value=168.0,
        value=72.0 if tipo_regime == "Dose Única" else intervalo_horas * num_doses + 24,
        step=6.0,
        help="Duração total da simulação"
    )
    
    return {
        'peso_corporal': peso_corporal,
        'dose_mg': dose_mg,
        'via_admin': via_admin,
        'tipo_regime': tipo_regime,
        'num_doses': num_doses,
        'intervalo_horas': intervalo_horas,
        'tempo_total': tempo_total
    }


def executar_simulacao(params_entrada):
    """Executa a simulação completa"""
    
    # Criar parâmetros farmacocinéticos
    params_pk = ParametrosFarmacocineticos(
        peso_corporal=params_entrada['peso_corporal']
    )
    
    # Criar modelos
    modelo_pk = ModeloFarmacocinetico(params_pk)
    modelo_pd = ModeloFarmacodinamico()
    visualizador = VisualizadorQuetiapina()
    
    # Simular farmacocinética
    if params_entrada['tipo_regime'] == "Dose Única":
        tempo, concentracoes = modelo_pk.simular(
            dose_mg=params_entrada['dose_mg'],
            tempo_horas=params_entrada['tempo_total'],
            num_pontos=1000,
            via=params_entrada['via_admin']
        )
    else:
        regime = RegimePosologico(modelo_pk)
        tempo, concentracoes = regime.simular_doses_multiplas(
            dose_mg=params_entrada['dose_mg'],
            intervalo_horas=params_entrada['intervalo_horas'],
            num_doses=params_entrada['num_doses'],
            tempo_total_horas=params_entrada['tempo_total']
        )
    
    # Calcular parâmetros PK
    params_calculados = modelo_pk.calcular_parametros_pk(tempo, concentracoes[:, 1])
    
    # Simular farmacodinâmica
    resultados_pd = modelo_pd.simular_resposta_temporal(
        tempo,
        concentracoes[:, 2]  # Concentração cerebral
    )
    
    return {
        'tempo': tempo,
        'concentracoes': concentracoes,
        'params_pk': params_calculados,
        'resultados_pd': resultados_pd,
        'modelo_pk': modelo_pk,
        'modelo_pd': modelo_pd,
        'visualizador': visualizador
    }


def exibir_resultados(resultados, params_entrada):
    """Exibe resultados da simulação"""
    
    # Tabs para organizar visualizações
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Farmacocinética", 
        "🧠 Farmacodinâmica",
        "🎯 Ocupação de Receptores",
        "📋 Relatório Completo"
    ])
    
    # Tab 1: Farmacocinética
    with tab1:
        st.markdown('<div class="sub-header">Perfil Farmacocinético</div>', 
                   unsafe_allow_html=True)
        
        fig_pk = resultados['visualizador'].plot_farmacocinetica_completa(
            resultados['tempo'],
            resultados['concentracoes'],
            resultados['params_pk']
        )
        st.pyplot(fig_pk)
        plt.close()
        
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Cmax", f"{resultados['params_pk']['Cmax_ng_mL']:.1f} ng/mL")
        with col2:
            st.metric("Tmax", f"{resultados['params_pk']['Tmax_horas']:.1f} h")
        with col3:
            st.metric("T½", f"{resultados['params_pk']['Tmeia_vida_horas']:.1f} h")
        with col4:
            st.metric("AUC", f"{resultados['params_pk']['AUC_ng_h_mL']:.0f} ng·h/mL")
    
    # Tab 2: Farmacodinâmica
    with tab2:
        st.markdown('<div class="sub-header">Efeitos Farmacodinâmicos</div>', 
                   unsafe_allow_html=True)
        
        if params_entrada['tipo_regime'] == "Doses Múltiplas":
            fig_pd = resultados['visualizador'].plot_doses_multiplas(
                resultados['tempo'],
                resultados['concentracoes'],
                params_entrada['intervalo_horas'],
                params_entrada['num_doses']
            )
            st.pyplot(fig_pd)
            plt.close()
        
        fig_pd2 = resultados['visualizador'].plot_farmacodinamica(
            resultados['tempo'],
            resultados['resultados_pd']
        )
        st.pyplot(fig_pd2)
        plt.close()
        
        # Eficácia média
        eficacia_media = np.mean(resultados['resultados_pd']['eficacia'])
        if eficacia_media >= 70:
            st.markdown(f"""
            <div class="success-box">
                ✅ <b>Eficácia Terapêutica Adequada</b><br>
                Score médio: {eficacia_media:.1f}/100
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="warning-box">
                ⚠️ <b>Eficácia Terapêutica Subótima</b><br>
                Score médio: {eficacia_media:.1f}/100<br>
                Considere ajuste de dose.
            </div>
            """, unsafe_allow_html=True)
    
    # Tab 3: Ocupação de Receptores
    with tab3:
        st.markdown('<div class="sub-header">Mapa de Ocupação de Receptores</div>', 
                   unsafe_allow_html=True)
        
        # Ocupação no pico
        idx_max = np.argmax(resultados['concentracoes'][:, 2])
        conc_pico = resultados['concentracoes'][idx_max, 2]
        ocupacoes_pico = resultados['modelo_pd'].calcular_ocupacao_receptores(conc_pico)
        
        fig_cerebro = resultados['visualizador'].plot_diagrama_cerebro(ocupacoes_pico)
        st.pyplot(fig_cerebro)
        plt.close()
        
        # Tabela de ocupações
        st.markdown("#### Ocupação no Pico de Concentração")
        col1, col2 = st.columns(2)
        
        receptores_lista = list(ocupacoes_pico.items())
        metade = len(receptores_lista) // 2
        
        with col1:
            for receptor, ocupacao in receptores_lista[:metade]:
                st.metric(
                    resultados['modelo_pd'].receptores[receptor].nome,
                    f"{ocupacao:.1f}%"
                )
        
        with col2:
            for receptor, ocupacao in receptores_lista[metade:]:
                st.metric(
                    resultados['modelo_pd'].receptores[receptor].nome,
                    f"{ocupacao:.1f}%"
                )
        
        # Avaliação de efeitos
        efeitos_colaterais = resultados['modelo_pd'].avaliar_efeitos_colaterais(ocupacoes_pico)
        
        st.markdown("#### ⚠️ Risco de Efeitos Colaterais (no pico)")
        for efeito, risco in efeitos_colaterais.items():
            if risco > 50:
                cor = "🔴"
            elif risco > 25:
                cor = "🟡"
            else:
                cor = "🟢"
            st.write(f"{cor} **{efeito.replace('_', ' ')}**: {risco:.1f}%")
    
    # Tab 4: Relatório
    with tab4:
        st.markdown('<div class="sub-header">Relatório Completo da Simulação</div>', 
                   unsafe_allow_html=True)
        
        st.markdown("### 📝 Parâmetros de Entrada")
        st.json({
            "Peso Corporal": f"{params_entrada['peso_corporal']} kg",
            "Dose": f"{params_entrada['dose_mg']} mg",
            "Via de Administração": params_entrada['via_admin'],
            "Regime": params_entrada['tipo_regime'],
            "Tempo de Simulação": f"{params_entrada['tempo_total']} horas"
        })
        
        st.markdown("### 📊 Parâmetros Farmacocinéticos")
        st.json({k: f"{v:.2f}" for k, v in resultados['params_pk'].items()})
        
        st.markdown("### 🎯 Recomendações de Dose")
        recomendacoes = resultados['modelo_pd'].recomendar_dose(params_entrada['peso_corporal'])
        st.table({
            "Indicação": list(recomendacoes.keys()),
            "Dose Recomendada (mg/dia)": [f"{v:.0f}" for v in recomendacoes.values()]
        })
        
        st.markdown("### ℹ️ Informações sobre a Quetiapina")
        st.markdown("""
        **Nome Químico**: 2-[2-(4-dibenzo[b,f][1,4]tiazepin-11-il-1-piperazinil)etoxi]etanol
        
        **Classe**: Antipsicótico atípico (de segunda geração)
        
        **Mecanismo de Ação**:
        - Antagonista de receptores de serotonina (5-HT2A)
        - Antagonista de receptores de dopamina (D2)
        - Antagonista de receptores de histamina (H1)
        - Antagonista α-adrenérgico
        
        **Indicações Terapêuticas**:
        - Esquizofrenia
        - Transtorno bipolar (mania e depressão)
        - Depressão maior (terapia adjuvante)
        
        **Farmacocinética**:
        - Biodisponibilidade oral: ~73%
        - Ligação proteica: ~83%
        - Metabolismo: Hepático (CYP3A4)
        - Meia-vida: ~6-7 horas
        - Excreção: Renal (73%) e fecal (20%)
        """)


def main():
    """Função principal do aplicativo"""
    
    # Cabeçalho
    criar_cabecalho()
    
    # Parâmetros
    params_entrada = parametros_sidebar()
    
    # Botão de simulação
    st.sidebar.markdown("---")
    if st.sidebar.button("🚀 EXECUTAR SIMULAÇÃO", type="primary", use_container_width=True):
        with st.spinner("🔬 Executando simulação farmacocinética e farmacodinâmica..."):
            try:
                resultados = executar_simulacao(params_entrada)
                st.session_state['resultados'] = resultados
                st.session_state['params_entrada'] = params_entrada
                st.success("✅ Simulação concluída com sucesso!")
            except Exception as e:
                st.error(f"❌ Erro na simulação: {str(e)}")
                st.exception(e)
    
    # Exibir resultados se disponíveis
    if 'resultados' in st.session_state:
        exibir_resultados(st.session_state['resultados'], 
                         st.session_state['params_entrada'])
    else:
        st.info("👈 Configure os parâmetros na barra lateral e clique em 'EXECUTAR SIMULAÇÃO'")
    
    # Rodapé
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7F8C8D; padding: 20px;">
        <b>Simulador de Quetiapina no Cérebro Humano</b><br>
        Desenvolvido por <b>Luiz Tiago Wilcke</b> | 2025<br>
        Modelo baseado em princípios de farmacocinética e farmacodinâmica<br>
        ⚠️ <i>Apenas para fins educacionais e de pesquisa. Não substitui orientação médica.</i>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
 