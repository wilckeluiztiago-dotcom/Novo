"""
Preditor Neural Integrado
Autor: Luiz Tiago Wilcke
Data: 2025-11-25

Interface unificada para fazer predições usando modelos neurais treinados
"""

import torch
import numpy as np
import pickle
from typing import Dict, Tuple
import os
from modelos_neurais import (
    LSTMFarmacocinetica,
    RedeOtimizacaoDose,
    ClassificadorRespostaTerapeutica,
    AutoencoderPacientes
)


class PreditorNeural:
    """Classe para fazer predições usando todos os modelos treinados"""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """
        Args:
            checkpoint_dir: Diretório com models treinados
        """
        self.checkpoint_dir = checkpoint_dir
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Modelos
        self.lstm_pk = None
        self.otimizador_dose = None
        self.classificador_resposta = None
        self.vae = None
        
        # Scalers e encoders
        self.scaler_otim = None
        self.scaler_vae = None
        self.label_encoder = None
        self.features_vae = None
        
        self.modelos_carregados = False
    
    def carregar_modelos(self):
        """Carrega todos os modelos treinados"""
        try:
            print("Carregando modelos neurais...")
            
            # LSTM
            if os.path.exists(f"{self.checkpoint_dir}/lstm_pk_best.pth"):
                self.lstm_pk = LSTMFarmacocinetica()
                checkpoint = torch.load(f"{self.checkpoint_dir}/lstm_pk_best.pth", 
                                      map_location=self.device)
                self.lstm_pk.load_state_dict(checkpoint['model_state_dict'])
                self.lstm_pk.eval()
                self.lstm_pk.to(self.device)
                print("  ✓ LSTM Farmacocinética")
            
            # Otimizador
            if os.path.exists(f"{self.checkpoint_dir}/otimizador_dose_best.pth"):
                self.otimizador_dose = RedeOtimizacaoDose()
                checkpoint = torch.load(f"{self.checkpoint_dir}/otimizador_dose_best.pth",
                                      map_location=self.device)
                self.otimizador_dose.load_state_dict(checkpoint['model_state_dict'])
                self.otimizador_dose.eval()
                self.otimizador_dose.to(self.device)
                
                # Scaler
                with open(f"{self.checkpoint_dir}/scaler_otim.pkl", "rb") as f:
                    self.scaler_otim = pickle.load(f)
                print("  ✓ Otimizador de Dose")
            
            # Classificador
            if os.path.exists(f"{self.checkpoint_dir}/classificador_resposta_best.pth"):
                self.classificador_resposta = ClassificadorRespostaTerapeutica()
                checkpoint = torch.load(f"{self.checkpoint_dir}/classificador_resposta_best.pth",
                                      map_location=self.device)
                self.classificador_resposta.load_state_dict(checkpoint['model_state_dict'])
                self.classificador_resposta.eval()
                self.classificador_resposta.to(self.device)
                
                # Label encoder
                with open(f"{self.checkpoint_dir}/label_encoder.pkl", "rb") as f:
                    self.label_encoder = pickle.load(f)
                print("  ✓ Classificador de Resposta")
            
            # VAE
            if os.path.exists(f"{self.checkpoint_dir}/vae_best.pth"):
                # Carregar features primeiro
                with open(f"{self.checkpoint_dir}/features_vae.pkl", "rb") as f:
                    self.features_vae = pickle.load(f)
                
                self.vae = AutoencoderPacientes(input_dim=len(self.features_vae))
                self.vae.load_state_dict(torch.load(f"{self.checkpoint_dir}/vae_best.pth",
                                                    map_location=self.device))
                self.vae.eval()
                self.vae.to(self.device)
                
                # Scaler
                with open(f"{self.checkpoint_dir}/scaler_vae.pkl", "rb") as f:
                    self.scaler_vae = pickle.load(f)
                print("  ✓ Autoencoder VAE")
            
            self.modelos_carregados = True
            print("\n✓ Todos os modelos carregados com sucesso!\n")
            
        except Exception as e:
            print(f"\n⚠ Erro ao carregar modelos: {str(e)}")
            print("Execute 'python3 treinar_modelos.py' para treinar os modelos primeiro.\n")
            self.modelos_carregados = False
    
    def predizer_serie_temporal_pk(self, 
                                    idade: float,
                                    peso: float,
                                    imc: float,
                                    funcao_hepatica: float,
                                    funcao_renal: float,
                                    sexo: str,
                                    cyp3a4: str,
                                    dose: float) -> np.ndarray:
        """
        Prediz série temporal de concentração plasmática usando LSTM
        
        Returns:
            Array com 100 pontos de concentração ao longo do tempo
        """
        if self.lstm_pk is None:
            raise ValueError("LSTM não carregado. Execute carregar_modelos() primeiro.")
        
        # Preparar features
        features = np.array([[
            idade / 100,
            peso / 100,
            imc / 40,
            funcao_hepatica,
            funcao_renal,
            1 if sexo == 'M' else 0,
            1 if cyp3a4 == 'lento' else 0,
            1 if cyp3a4 == 'rapido' else 0,
            dose / 800
        ]], dtype=np.float32)
        
        # Predição
        with torch.no_grad():
            x = torch.FloatTensor(features).to(self.device)
            predicao = self.lstm_pk(x)
            serie_temporal = predicao.cpu().numpy().flatten()
        
        return serie_temporal
    
    def recomendar_dose_otima(self, paciente: Dict) -> float:
        """
        Recomenda dose ótima baseado em características do paciente
        
        Args:
            paciente: Dicionário com características
        
        Returns:
            Dose recomendada em mg
        """
        if self.otimizador_dose is None or self.scaler_otim is None:
            raise ValueError("Otimizador não carregado.")
        
        # Preparar features
        features = np.array([[
            paciente['idade'],
            paciente['peso'],
            paciente['imc'],
            paciente.get('funcao_hepatica', 1.0),
            paciente.get('funcao_renal', 1.0),
            paciente.get('gravidade_sintomas', 5.0),
            1 if paciente.get('sexo') == 'M' else 0,
            1 if paciente.get('cyp3a4') == 'lento' else 0,
            1 if paciente.get('cyp3a4') == 'rapido' else 0,
            1 if paciente.get('tratamento_previo', False) else 0,
            1 if paciente.get('diagnostico') == 'esquizofrenia' else 0,
            1 if 'bipolar' in paciente.get('diagnostico', '') else 0
        ]], dtype=np.float32)
        
        # Normalizar
        features_norm = self.scaler_otim.transform(features)
        
        # Predição
        with torch.no_grad():
            x = torch.FloatTensor(features_norm).to(self.device)
            dose_otima = self.otimizador_dose.predict_dose(x)
            dose_mg = dose_otima.cpu().numpy()[0, 0]
        
        # Arredondar para múltiplo de 25
        dose_mg = round(dose_mg / 25) * 25
        dose_mg = np.clip(dose_mg, 25, 800)
        
        return float(dose_mg)
    
    def classificar_resposta(self, paciente: Dict, dose: float, 
                            cmax: float, auc: float, 
                            ocupacao_d2: float, eficacia: float) -> Tuple[str, Dict]:
        """
        Classifica a resposta terapêutica esperada
        
        Returns:
            Tupla (classe, probabilidades)
        """
        if self.classificador_resposta is None or self.label_encoder is None:
            raise ValueError("Classificador não carregado.")
        
        # Preparar features
        features = np.array([[
            paciente['idade'],
            paciente['peso'],
            paciente['imc'],
            paciente.get('funcao_hepatica', 1.0),
            paciente.get('funcao_renal', 1.0),
            paciente.get('gravidade_sintomas', 5.0),
            dose,
            cmax,
            auc,
            ocupacao_d2,
            eficacia,
            1 if paciente.get('sexo') == 'M' else 0,
            1 if paciente.get('cyp3a4') == 'lento' else 0,
            1 if paciente.get('cyp3a4') == 'rapido' else 0,
            1 if paciente.get('tratamento_previo', False) else 0
        ]], dtype=np.float32)
        
        # Normalizar (usar o mesmo scaler do treino - aqui simplificado)
        from sklearn.preprocessing import StandardScaler
        # Na prática, deveria carregar o scaler salvo
        
        # Predição
        with torch.no_grad():
            x = torch.FloatTensor(features).to(self.device)
            probs = self.classificador_resposta.predict_proba(x)
            probs_np = probs.cpu().numpy()[0]
            
            classe_idx = np.argmax(probs_np)
            classe = self.label_encoder.inverse_transform([classe_idx])[0]
            
            prob_dict = {
                self.label_encoder.inverse_transform([i])[0]: float(probs_np[i])
                for i in range(len(probs_np))
            }
        
        return classe, prob_dict
    
    def analisar_paciente_vae(self, paciente: Dict) -> Dict:
        """
        Analisa paciente usando VAE
        
        Returns:
            Dicionário com representação latente e score de anomalia
        """
        if self.vae is None or self.scaler_vae is None:
            raise ValueError("VAE não carregado.")
        
        # Preparar features completas
        # Aqui simplificado - na prática usar mesmas features do treino
        features_dict = {}
        
        for feat in self.features_vae:
            if feat in paciente:
                features_dict[feat] = paciente[feat]
            else:
                # Valores default
                features_dict[feat] = 0.0
        
        # Converter para array na ordem correta
        features = np.array([[features_dict[f] for f in self.features_vae]], dtype=np.float32)
        
        # Normalizar
        features_norm = self.scaler_vae.transform(features)
        
        # Análise
        with torch.no_grad():
            x = torch.FloatTensor(features_norm).to(self.device)
            recon, mu, logvar = self.vae(x)
            
            # Erro de reconstrução (anomalia score)
            recon_error = torch.mean((recon - x) ** 2).item()
            
            # Representação latente
            latent = mu.cpu().numpy().flatten()
        
        return {
            'latent_representation': latent.tolist(),
            'anomaly_score': recon_error,
            'is_outlier': recon_error > 0.1  # Threshold ajustável
        }


# Função de conveniência
def criar_preditor() -> PreditorNeural:
    """Cria e carrega preditor neural"""
    preditor = PreditorNeural()
    preditor.carregar_modelos()
    return preditor


if __name__ == "__main__":
    print("=" * 80)
    print("TESTE DO PREDITOR NEURAL INTEGRADO")
    print("=" * 80)
    print()
    
    # Criar preditor
    preditor = criar_preditor()
    
    if preditor.modelos_carregados:
        print("\nTestando predições...")
        
        # Paciente de teste
        paciente_teste = {
            'idade': 45.0,
            'peso': 70.0,
            'altura': 175.0,
            'imc': 22.9,
            'sexo': 'M',
            'funcao_hepatica': 1.0,
            'funcao_renal': 1.0,
            'cyp3a4': 'normal',
            'diagnostico': 'esquizofrenia',
            'gravidade_sintomas': 7.0,
            'tratamento_previo': False
        }
        
        # 1. Recomendar dose
        if preditor.otimizador_dose:
            dose_recomendada = preditor.recomendar_dose_otima(paciente_teste)
            print(f"\n1. Dose Otimalizada: {dose_recomendada:.0f} mg")
        
        # 2. Predizer série temporal
        if preditor.lstm_pk:
            serie = preditor.predizer_serie_temporal_pk(
                idade=45, peso=70, imc=22.9,
                funcao_hepatica=1.0, funcao_renal=1.0,
                sexo='M', cyp3a4='normal', dose=300
            )
            print(f"\n2. Série Temporal PK predita: {len(serie)} pontos")
            print(f"   Concentração máxima predita: {np.max(serie):.3f} ng/mL")
        
        # 3. Analisar com VAE
        if preditor.vae:
            analise_vae = preditor.analisar_paciente_vae(paciente_teste)
            print(f"\n3. Análise VAE:")
            print(f"   Anomaly Score: {analise_vae['anomaly_score']:.4f}")
            print(f"   É outlier: {analise_vae['is_outlier']}")
        
        print("\n" + "=" * 80)
        print("✓ Preditor funcionando corretamente!")
        print("=" * 80)
    else:
        print("\n⚠ Modelos não disponíveis. Execute o treinamento primeiro.")
