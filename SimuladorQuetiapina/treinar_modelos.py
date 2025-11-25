"""
Script Principal de Treinamento
Autor: Luiz Tiago Wilcke
Data: 2025-11-25

Treina todos os modelos neurais do simulador
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from modelos_neurais import (
    LSTMFarmacocinetica,
    RedeOtimizacaoDose,
    ClassificadorRespostaTerapeutica,
    AutoencoderPacientes,
    RedeMultitarefa
)
from treinamento import TreinadorRedeNeural, preparar_dados_lstm, preparar_dados_classificacao
from gerador_dados import GeradorDadosSinteticos
import os


def treinar_todos_modelos():
    """Treina todos os modelos do sistema"""
    
    print("=" * 80)
    print("TREINAMENTO COMPLETO DO SISTEMA DE REDES NEURAIS")
    print("=" * 80)
    print()
    
    # Verificar se dados existem, senão gerar
    if not os.path.exists("dataset_quetiapina_treino.csv"):
        print("📊 Gerando dataset de treinamento...")
        gerador = GeradorDadosSinteticos(seed=42)
        df = gerador.gerar_dataset_treinamento(num_pacientes=2000, doses_por_paciente=3)
        df.to_csv("dataset_quetiapina_treino.csv", index=False)
        print("✓ Dataset gerado e salvo\n")
    else:
        print("✓ Dataset encontrado\n")
        df = pd.read_csv("dataset_quetiapina_treino.csv")
    
    if not os.path.exists("series_temporais_features.npy"):
        print("📈 Gerando séries temporais...")
        gerador = GeradorDadosSinteticos(seed=42)
        X_temporal, y_temporal = gerador.gerar_series_temporais_pk(num_pacientes=1000)
        np.save("series_temporais_features.npy", X_temporal)
        np.save("series_temporais_concentracoes.npy", y_temporal)
        print("✓ Séries temporais geradas\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Dispositivo: {device}\n")
    
    # ========================
    # 1. LSTM Farmacocinética
    # ========================
    print("\n" + "=" * 80)
    print("1️⃣  TREINANDO LSTM PARA PREDIÇÃO DE SÉRIES TEMPORAIS PK")
    print("=" * 80)
    
    modelo_lstm = LSTMFarmacocinetica()
    treinador_lstm = TreinadorRedeNeural(modelo_lstm, device=device)
    
    # Preparar dados
    train_loader, val_loader = preparar_dados_lstm(
        "series_temporais_features.npy",
        "series_temporais_concentracoes.npy",
        batch_size=64
    )
    
    # Treinar
    optimizer = optim.Adam(modelo_lstm.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    treinador_lstm.treinar(
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=150,
        early_stopping_patience=20,
        nome_modelo="lstm_pk"
    )
    
    treinador_lstm.plotar_historico("lstm_training_curve.png")
    print("✓ LSTM treinada e salva\n")
    
    # ========================
    # 2. Classificador de Resposta
    # ========================
    print("\n" + "=" * 80)
    print("2️⃣  TREINANDO CLASSIFICADOR DE RESPOSTA TERAPÊUTICA")
    print("=" * 80)
    
    modelo_classificador = ClassificadorRespostaTerapeutica()
    treinador_class = TreinadorRedeNeural(modelo_classificador, device=device)
    
    # Preparar dados
    train_loader, val_loader, label_encoder = preparar_dados_classificacao(df, batch_size=64)
    
    # Salvar label encoder
    import pickle
    with open("checkpoints/label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)
    
    # Treinar
    optimizer = optim.AdamW(modelo_classificador.parameters(), lr=0.0005, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    treinador_class.treinar(
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=100,
        early_stopping_patience=15,
        nome_modelo="classificador_resposta"
    )
    
    treinador_class.plotar_historico("classificador_training_curve.png")
    print("✓ Classificador treinado e salvo\n")
    
    # ========================
    # 3. Rede de Otimização de Dose
    # ========================
    print("\n" + "=" * 80)
    print("3️⃣  TREINANDO REDE DE OTIMIZAÇÃO DE DOSE")
    print("=" * 80)
    
    modelo_otim = RedeOtimizacaoDose()
    treinador_otim = TreinadorRedeNeural(modelo_otim, device=device)
    
    # Preparar dados específicos para otimização
    # Features: características do paciente
    # Target: dose que resultou em melhor resposta
    
    # Filtrar apenas boas respostas
    df_boas = df[df['resposta_clinica'] == 'boa'].copy()
    
    features_otim = ['idade', 'peso', 'imc', 'funcao_hepatica', 'funcao_renal',
                     'gravidade_sintomas']
    
    # One-hot
    df_boas['sexo_M'] = (df_boas['sexo'] == 'M').astype(int)
    df_boas['cyp3a4_lento'] = (df_boas['cyp3a4'] == 'lento').astype(int)
    df_boas['cyp3a4_rapido'] = (df_boas['cyp3a4'] == 'rapido').astype(int)
    df_boas['tratamento_previo'] = df_boas['tratamento_previo'].astype(int)
    df_boas['diagnostico_esquiz'] = (df_boas['diagnostico'] == 'esquizofrenia').astype(int)
    df_boas['diagnostico_bipolar'] = (df_boas['diagnostico'].str.contains('bipolar')).astype(int)
    
    features_otim += ['sexo_M', 'cyp3a4_lento', 'cyp3a4_rapido', 
                      'tratamento_previo', 'diagnostico_esquiz', 'diagnostico_bipolar']
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from torch.utils.data import TensorDataset, DataLoader
    
    X_otim = df_boas[features_otim].values
    y_otim = df_boas['dose'].values.reshape(-1, 1) / 800  # Normalizar 0-1
    
    scaler_otim = StandardScaler()
    X_otim = scaler_otim.fit_transform(X_otim)
    
    X_train, X_val, y_train, y_val = train_test_split(X_otim, y_otim, test_size=0.2, random_state=42)
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    
    # Salvar scaler
    with open("checkpoints/scaler_otim.pkl", "wb") as f:
        pickle.dump(scaler_otim, f)
    
    # Treinar
    optimizer = optim.Adam(modelo_otim.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    treinador_otim.treinar(
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=100,
        early_stopping_patience=15,
        nome_modelo="otimizador_dose"
    )
    
    treinador_otim.plotar_historico("otimizador_training_curve.png")
    print("✓ Otimizador treinado e salvo\n")
    
    # ========================
    # 4. Autoencoder
    # ========================
    print("\n" + "=" * 80)
    print("4️⃣  TREINANDO AUTOENCODER VARIACIONAL")
    print("=" * 80)
    
    # Preparar dados para VAE (todas as características do paciente)
    features_vae = ['idade', 'peso', 'altura', 'imc', 'funcao_hepatica', 'funcao_renal',
                    'gravidade_sintomas']
    
    df_vae = df.copy()
    df_vae['sexo_M'] = (df_vae['sexo'] == 'M').astype(int)
    df_vae['cyp3a4_lento'] = (df_vae['cyp3a4'] == 'lento').astype(int)
    df_vae['cyp3a4_rapido'] = (df_vae['cyp3a4'] == 'rapido').astype(int)
    df_vae['tratamento_previo'] = df_vae['tratamento_previo'].astype(int)
    df_vae['diabetes'] = df_vae['diabetes'].astype(int)
    df_vae['hipertensao'] = df_vae['hipertensao'].astype(int)
    
    features_vae += ['sexo_M', 'cyp3a4_lento', 'cyp3a4_rapido', 
                     'tratamento_previo', 'diabetes', 'hipertensao']
    
    # Adicionar diagnósticos
    for diag in df_vae['diagnostico'].unique():
        df_vae[f'diag_{diag}'] = (df_vae['diagnostico'] == diag).astype(int)
        features_vae.append(f'diag_{diag}')
    
    # Remover duplicatas de pacientes (manter características únicas)
    df_vae_unique = df_vae.drop_duplicates(subset=['idade', 'peso', 'sexo'])
    
    X_vae = df_vae_unique[features_vae].values
    
    from sklearn.preprocessing import MinMaxScaler
    scaler_vae = MinMaxScaler()
    X_vae = scaler_vae.fit_transform(X_vae)
    
    X_train, X_val = train_test_split(X_vae, test_size=0.2, random_state=42)
    
    # Para VAE, usamos os mesmos dados como entrada e target
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(X_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(X_val))
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    
    # Modelo
    modelo_vae = AutoencoderPacientes(input_dim=len(features_vae))
    
    # Salvar scaler e features
    with open("checkpoints/scaler_vae.pkl", "wb") as f:
        pickle.dump(scaler_vae, f)
    with open("checkpoints/features_vae.pkl", "wb") as f:
        pickle.dump(features_vae, f)
    
    # Custom training loop para VAE
    print("Treinando VAE...")
    optimizer = optim.Adam(modelo_vae.parameters(), lr=0.001)
    
    modelo_vae = modelo_vae.to(device)
    melhor_loss = float('inf')
    
    for epoch in range(100):
        modelo_vae.train()
        train_loss = 0
        
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)
            
            optimizer.zero_grad()
            recon, mu, logvar = modelo_vae(batch_x)
            loss = modelo_vae.loss_function(recon, batch_x, mu, logvar, beta=0.5)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validação
        modelo_vae.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device)
                recon, mu, logvar = modelo_vae(batch_x)
                loss = modelo_vae.loss_function(recon, batch_x, mu, logvar, beta=0.5)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        if (epoch + 1) % 10 == 0:
            print(f"Época {epoch+1}/100 | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < melhor_loss:
            melhor_loss = val_loss
            torch.save(modelo_vae.state_dict(), "checkpoints/vae_best.pth")
    
    print("✓ VAE treinado e salvo\n")
    
    # ========================
    # Resumo Final
    # ========================
    print("\n" + "=" * 80)
    print("✅ TREINAMENTO COMPLETO!")
    print("=" * 80)
    print("\nModelos treinados e salvos:")
    print("  1. LSTM Farmacocinética → checkpoints/lstm_pk_best.pth")
    print("  2. Classificador Resposta → checkpoints/classificador_resposta_best.pth")
    print("  3. Otimizador de Dose → checkpoints/otimizador_dose_best.pth")
    print("  4. Autoencoder VAE → checkpoints/vae_best.pth")
    print("\nCurvas de aprendizado salvas:")
    print("  - lstm_training_curve.png")
    print("  - classificador_training_curve.png")
    print("  - otimizador_training_curve.png")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    treinar_todos_modelos()
