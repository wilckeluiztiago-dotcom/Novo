"""
Gerador de Dados Sintéticos para Treinamento de Redes Neurais
Autor: Luiz Tiago Wilcke
Data: 2025-11-25

Gera dados sintéticos realistas de pacientes e suas respostas farmacocinéticas/farmacodinâmicas
para treinamento de modelos de machine learning.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from farmacocinetica import ParametrosFarmacocineticos, ModeloFarmacocinetico
from farmacodinamica import ModeloFarmacodinamico
import pickle


class GeradorDadosSinteticos:
    """Gera dados sintéticos de pacientes para treinamento de redes neurais"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.modelo_pd = ModeloFarmacodinamico()
    
    def gerar_paciente_aleatorio(self) -> Dict:
        """
        Gera características aleatórias de um paciente virtual
        
        Returns:
            Dicionário com características do paciente
        """
        # Distribuições baseadas em população real
        idade = np.clip(np.random.normal(45, 15), 18, 85)
        
        # Peso correlacionado com idade e sexo
        sexo = np.random.choice(['M', 'F'])
        if sexo == 'M':
            peso = np.clip(np.random.normal(75, 12), 45, 120)
            altura = np.clip(np.random.normal(175, 8), 155, 195)
        else:
            peso = np.clip(np.random.normal(65, 10), 40, 110)
            altura = np.clip(np.random.normal(165, 7), 150, 185)
        
        # IMC
        imc = peso / ((altura / 100) ** 2)
        
        # Função hepática (impacta metabolismo)
        funcao_hepatica = np.clip(np.random.normal(1.0, 0.2), 0.5, 1.5)
        
        # Função renal (impacta excreção)
        funcao_renal = np.clip(np.random.normal(1.0, 0.15), 0.6, 1.4)
        
        # Polimorfismo CYP3A4 (metabolizador lento/normal/rápido)
        cyp3a4 = np.random.choice(['lento', 'normal', 'rapido'], p=[0.2, 0.6, 0.2])
        
        # Diagnóstico
        diagnostico = np.random.choice([
            'esquizofrenia', 
            'bipolar_mania', 
            'bipolar_depressao',
            'depressao_maior'
        ], p=[0.4, 0.25, 0.2, 0.15])
        
        # Gravidade (1-10)
        gravidade_sintomas = np.random.uniform(3, 9)
        
        # Histórico de tratamento
        tratamento_previo = np.random.choice([True, False], p=[0.6, 0.4])
        
        # Comorbidades
        diabetes = np.random.choice([True, False], p=[0.15, 0.85])
        hipertensao = np.random.choice([True, False], p=[0.25, 0.75])
        
        return {
            'idade': float(idade),
            'sexo': sexo,
            'peso': float(peso),
            'altura': float(altura),
            'imc': float(imc),
            'funcao_hepatica': float(funcao_hepatica),
            'funcao_renal': float(funcao_renal),
            'cyp3a4': cyp3a4,
            'diagnostico': diagnostico,
            'gravidade_sintomas': float(gravidade_sintomas),
            'tratamento_previo': tratamento_previo,
            'diabetes': diabetes,
            'hipertensao': hipertensao
        }
    
    def ajustar_parametros_pk(self, paciente: Dict) -> ParametrosFarmacocineticos:
        """
        Ajusta parâmetros farmacocinéticos baseado em características do paciente
        
        Args:
            paciente: Dicionário com características do paciente
        
        Returns:
            Parâmetros farmacocinéticos ajustados
        """
        params = ParametrosFarmacocineticos(peso_corporal=paciente['peso'])
        
        # Ajustar clearance baseado em função hepática e renal
        params.clearance_por_kg *= paciente['funcao_hepatica'] * paciente['funcao_renal']
        
        # Ajustar metabolismo baseado em CYP3A4
        if paciente['cyp3a4'] == 'lento':
            params.clearance_por_kg *= 0.6
            params.k_eliminacao *= 0.7
        elif paciente['cyp3a4'] == 'rapido':
            params.clearance_por_kg *= 1.4
            params.k_eliminacao *= 1.3
        
        # Ajustar absorção baseado em IMC
        if paciente['imc'] > 30:  # Obesidade
            params.k_absorcao *= 0.85
        elif paciente['imc'] < 18.5:  # Baixo peso
            params.k_absorcao *= 1.1
        
        # Variabilidade individual
        params.k_absorcao *= np.random.normal(1.0, 0.15)
        params.volume_distribuicao_por_kg *= np.random.normal(1.0, 0.12)
        
        return params
    
    def simular_resposta_paciente(self, paciente: Dict, dose_mg: float) -> Dict:
        """
        Simula resposta completa de um paciente a uma dose
        
        Args:
            paciente: Características do paciente
            dose_mg: Dose administrada
        
        Returns:
            Dicionário com resultados da simulação
        """
        # Parâmetros PK ajustados
        params_pk = self.ajustar_parametros_pk(paciente)
        modelo_pk = ModeloFarmacocinetico(params_pk)
        
        # Simular farmacocinética
        tempo, concentracoes = modelo_pk.simular(
            dose_mg=dose_mg,
            tempo_horas=72.0,
            num_pontos=100,
            via='oral'
        )
        
        # Calcular parâmetros PK
        params_calculados = modelo_pk.calcular_parametros_pk(tempo, concentracoes[:, 1])
        
        # Simular farmacodinâmica
        resultados_pd = self.modelo_pd.simular_resposta_temporal(
            tempo,
            concentracoes[:, 2]
        )
        
        # Calcular métricas de resposta
        eficacia_media = np.mean(resultados_pd['eficacia'])
        eficacia_maxima = np.max(resultados_pd['eficacia'])
        
        # Ocupação média de D2 (principal indicador)
        ocupacao_d2_media = np.mean(resultados_pd['ocupacoes']['D2'])
        ocupacao_d2_maxima = np.max(resultados_pd['ocupacoes']['D2'])
        
        # Efeitos colaterais médios
        efeitos_medios = {
            efeito: np.mean(valores) 
            for efeito, valores in resultados_pd['efeitos_colaterais'].items()
        }
        
        # Determinar resposta clínica (baseado em critérios)
        # Boa resposta: eficácia > 70 e D2 entre 60-80% e EPS < 30%
        boa_resposta = (
            eficacia_media >= 70 and 
            60 <= ocupacao_d2_media <= 80 and 
            efeitos_medios['EPS'] < 30
        )
        
        # Resposta moderada
        resposta_moderada = (
            50 <= eficacia_media < 70 and 
            efeitos_medios['EPS'] < 50
        )
        
        # Classificar resposta
        if boa_resposta:
            resposta_clinica = 'boa'
        elif resposta_moderada:
            resposta_clinica = 'moderada'
        else:
            resposta_clinica = 'pobre'
        
        # Adicionar variabilidade de resposta baseada em fatores do paciente
        fator_resposta = 1.0
        
        # Gravidade: pacientes mais graves respondem um pouco pior
        fator_resposta *= (1.0 - 0.05 * (paciente['gravidade_sintomas'] / 10))
        
        # Tratamento prévio: pacientes já tratados podem ter resistência
        if paciente['tratamento_previo']:
            fator_resposta *= 0.9
        
        # Aplicar fator
        eficacia_media *= fator_resposta
        
        return {
            'dose': dose_mg,
            'Cmax': params_calculados['Cmax_ng_mL'],
            'Tmax': params_calculados['Tmax_horas'],
            'AUC': params_calculados['AUC_ng_h_mL'],
            'Tmeia_vida': params_calculados['Tmeia_vida_horas'],
            'eficacia_media': eficacia_media,
            'eficacia_maxima': eficacia_maxima,
            'ocupacao_d2_media': ocupacao_d2_media,
            'ocupacao_d2_maxima': ocupacao_d2_maxima,
            'efeitos_colaterais': efeitos_medios,
            'resposta_clinica': resposta_clinica,
            'concentracoes_tempo': concentracoes,
            'tempo': tempo
        }
    
    def gerar_dataset_treinamento(self, 
                                   num_pacientes: int = 1000,
                                   doses_por_paciente: int = 3) -> pd.DataFrame:
        """
        Gera dataset completo para treinamento
        
        Args:
            num_pacientes: Número de pacientes virtuais
            doses_por_paciente: Número de doses diferentes por paciente
        
        Returns:
            DataFrame com dados de treinamento
        """
        dados = []
        
        print(f"Gerando dataset com {num_pacientes} pacientes...")
        
        for i in range(num_pacientes):
            if (i + 1) % 100 == 0:
                print(f"  Progresso: {i + 1}/{num_pacientes} pacientes")
            
            # Gerar paciente
            paciente = self.gerar_paciente_aleatorio()
            
            # Testar múltiplas doses
            doses = np.random.choice([50, 100, 150, 200, 300, 400, 600, 800], 
                                    size=doses_por_paciente, 
                                    replace=False)
            
            for dose in doses:
                # Simular resposta
                resposta = self.simular_resposta_paciente(paciente, float(dose))
                
                # Combinar dados do paciente com resposta
                registro = {**paciente, **resposta}
                
                # Remover arrays (manter apenas escalares)
                del registro['concentracoes_tempo']
                del registro['tempo']
                del registro['efeitos_colaterais']  # Será desmembrado
                
                # Adicionar efeitos colaterais individualmente
                for efeito, valor in resposta['efeitos_colaterais'].items():
                    registro[f'efeito_{efeito.lower()}'] = valor
                
                dados.append(registro)
        
        df = pd.DataFrame(dados)
        
        print(f"\n✓ Dataset gerado com {len(df)} registros")
        print(f"  Variáveis: {len(df.columns)}")
        print(f"  Pacientes únicos: {num_pacientes}")
        
        return df
    
    def gerar_series_temporais_pk(self, 
                                   num_pacientes: int = 500,
                                   dose: float = 300.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gera séries temporais de concentração plasmática para LSTM
        
        Args:
            num_pacientes: Número de pacientes
            dose: Dose fixa para todos
        
        Returns:
            Tupla (características_pacientes, séries_temporais_concentração)
        """
        print(f"Gerando séries temporais PK para {num_pacientes} pacientes...")
        
        caracteristicas = []
        series_temporais = []
        
        for i in range(num_pacientes):
            paciente = self.gerar_paciente_aleatorio()
            
            # Simular
            resposta = self.simular_resposta_paciente(paciente, dose)
            
            # Características do paciente (one-hot encoding)
            feat = [
                paciente['idade'] / 100,  # Normalizar
                paciente['peso'] / 100,
                paciente['imc'] / 40,
                paciente['funcao_hepatica'],
                paciente['funcao_renal'],
                1 if paciente['sexo'] == 'M' else 0,
                1 if paciente['cyp3a4'] == 'lento' else 0,
                1 if paciente['cyp3a4'] == 'rapido' else 0,
                dose / 800  # Normalizar dose
            ]
            
            caracteristicas.append(feat)
            
            # Série temporal de concentração plasmática (100 pontos)
            series_temporais.append(resposta['concentracoes_tempo'][:, 1])
        
        print(f"✓ Séries temporais geradas")
        
        return np.array(caracteristicas), np.array(series_temporais)


def salvar_dataset(df: pd.DataFrame, caminho: str = "dataset_quetiapina.csv"):
    """Salva dataset em CSV"""
    df.to_csv(caminho, index=False)
    print(f"✓ Dataset salvo em: {caminho}")


def carregar_dataset(caminho: str = "dataset_quetiapina.csv") -> pd.DataFrame:
    """Carrega dataset de CSV"""
    return pd.read_csv(caminho)


if __name__ == "__main__":
    print("=" * 80)
    print("GERADOR DE DADOS SINTÉTICOS PARA TREINAMENTO DE REDES NEURAIS")
    print("=" * 80)
    print()
    
    # Criar gerador
    gerador = GeradorDadosSinteticos(seed=42)
    
    # Gerar dataset de treinamento
    print("\n1. Gerando dataset de treinamento...")
    df = gerador.gerar_dataset_treinamento(num_pacientes=1000, doses_por_paciente=3)
    
    # Estatísticas
    print("\n📊 Estatísticas do Dataset:")
    print("-" * 80)
    print(f"Total de registros: {len(df)}")
    print(f"Distribuição de respostas:")
    print(df['resposta_clinica'].value_counts())
    
    print(f"\nDistribuição de diagnósticos:")
    print(df['diagnostico'].value_counts())
    
    # Salvar
    salvar_dataset(df, "dataset_quetiapina_treino.csv")
    
    # Gerar séries temporais para LSTM
    print("\n2. Gerando séries temporais para LSTM...")
    X_temporal, y_temporal = gerador.gerar_series_temporais_pk(num_pacientes=500)
    
    # Salvar séries temporais
    np.save("series_temporais_features.npy", X_temporal)
    np.save("series_temporais_concentracoes.npy", y_temporal)
    print("✓ Séries temporais salvas")
    
    print("\n" + "=" * 80)
    print("GERAÇÃO DE DADOS CONCLUÍDA!")
    print("=" * 80)
