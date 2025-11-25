# 🧠 Simulador de Quetiapina com Redes Neurais Avançadas

**Autor:** Luiz Tiago Wilcke  
**Versão 2.0 - Com Machine Learning**

---

## 🎯 Visão Geral

Sistema **híbrido** que combina:
- **Modelagem matemática tradicional** (EDOs farmacocinéticas)
- **Redes neurais profundas com PyTorch** para predições personalizadas

---

## 🆕 Novos Recursos (Versão 2.0)

### 🤖 Modelos de Machine Learning

#### 1. **LSTM Farmacocinética** (484,068 parâmetros)
- Prediz séries temporais de concentração plasmática
- Input: características do paciente (idade, peso, IMC, etc.)
- Output: 100 pontos temporais de concentração

#### 2. **Otimizador de Dose** (301,569 parâmetros)
- Recomenda dose ótima personalizada
- Baseado em características individuais do paciente
- Arquitetura: Rede feedforward profunda (256→512→256→128)

#### 3. **Classificador de Resposta** (369,027 parâmetros)
- Prediz resposta terapêutica (boa/moderada/pobre)
- Arquitetura residual com mecanismo de atenção
- Probabilidades para cada classe de resposta

#### 4. **Autoencoder Variacional** (27,748 parâmetros)
- Aprende representação latente de pacientes
- Detecção de outliers e anomalias
- Clustering automático de perfis de pacientes

#### 5. **Rede Multitarefa** (402,059 parâmetros)
- Prediz simultaneamente:
  - Eficácia terapêutica
  - Ocupação de D2
  - 6 efeitos colaterais
  - Resposta clínica

**Total:** 1.584.471 parâmetros treináveis!

---

## 📁 Nova Estrutura do Projeto

```
SimuladorQuetiapina/
│
├── 🧮 MODELOS MATEMÁTICOS
│   ├── farmacocinetica.py          # EDOs compartimentais
│   ├── farmacodinamica.py          # Ocupação de receptores
│   └── visualizacao.py             # Gráficos matplotlib
│
├── 🤖 REDES NEURAIS (NOVO!)
│   ├── modelos_neurais.py          # 5 arquiteturas PyTorch
│   ├── gerador_dados.py            # Dados sintéticos
│   ├── treinamento.py              # Pipeline de treino
│   ├── treinar_modelos.py          # Script principal
│   └── preditor_neural.py          # Interface unificada
│
├── 💻 INTERFACES
│   ├── main.py                     # CLI tradicional
│   └── app.py                      # Dashboard Streamlit
│
├── 📊 DADOS
│   ├── dataset_quetiapina_treino.csv       # 3000 registros
│   ├── series_temporais_features.npy       # Features LSTM
│   └── series_temporais_concentracoes.npy  # Targets LSTM
│
└── 💾 MODELOS TREINADOS (checkpoints/)
    ├── lstm_pk_best.pth
    ├── otimizador_dose_best.pth
    ├── classificador_resposta_best.pth
    └── vae_best.pth
```

---

## 🚀 Guia de Uso Rápido

### 1️⃣ Instalação

```bash
# Dependências básicas
pip install -r requirements.txt

# Dependências de ML
pip install -r requirements_ml.txt

# Ou tudo junto
pip install numpy scipy matplotlib streamlit torch scikit-learn pandas
```

### 2️⃣ Gerar Dados Sintéticos

```bash
python3 gerador_dados.py
```

**Output:**
- `dataset_quetiapina_treino.csv` (3000 registros de pacientes)
- `series_temporais_*.npy` (500 séries temporais)

### 3️⃣ Treinar Redes Neurais

```bash
python3 treinar_modelos.py
```

**Processo:**
1. Carrega/gera dados
2. Treina LSTM (150 épocas)
3. Treina Classificador (100 épocas)
4. Treina Otimizador (100 épocas)
5. Treina VAE (100 épocas)

⏱️ **Tempo estimado:** 10-30 minutos (CPU) / 2-5 minutos (GPU)

**Output:**
- Modelos salvos em `checkpoints/`
- Curvas de aprendizado (`.png`)

### 4️⃣ Usar Simulador Tradicional

```bash
# Dose única
python3 main.py --peso 70 --dose 300

# Doses múltiplas
python3 main.py --peso 70 --dose 200 --multiplas --num-doses 5 --intervalo 12
```

### 5️⃣ Usar Predições Neurais

```python
from preditor_neural import criar_preditor

# Carregar modelos
preditor = criar_preditor()

# Definir paciente
paciente = {
    'idade': 45,
    'peso': 70,
    'imc': 22.9,
    'sexo': 'M',
    'funcao_hepatica': 1.0,
    'funcao_renal': 1.0,
    'cyp3a4': 'normal',
    'diagnostico': 'esquizofrenia',
    'gravidade_sintomas': 7.0,
    'tratamento_previo': False
}

# Recomendar dose ótima
dose_otima = preditor.recomendar_dose_otima(paciente)
print(f"Dose recomendada: {dose_otima} mg")

# Predizer série temporal
serie_pk = preditor.predizer_serie_temporal_pk(
    idade=45, peso=70, imc=22.9,
    funcao_hepatica=1.0, funcao_renal=1.0,
    sexo='M', cyp3a4='normal', dose=300
)
print(f"Concentração máxima predita: {serie_pk.max():.3f} ng/mL")
```

### 6️⃣ Dashboard Interativo

```bash
streamlit run app.py
```

Abre em `http://localhost:8501`

---

## 📊 Gerador de Dados Sintéticos

### Características Simuladas

**Demográficas:**
- Idade: 18-85 anos (μ=45, σ=15)
- Peso: 40-120 kg (ajustado por sexo)
- IMC: Calculado automaticamente
- Sexo: M/F (distribuição realista)

**Fisiológicas:**
- Função hepática: 0.5-1.5 (afeta metabolismo)
- Função renal: 0.6-1.4 (afeta excreção)
- Polimorfismo CYP3A4: lento/normal/rápido (20%/60%/20%)

**Clínicas:**
- Diagnóstico: esquizofrenia, bipolar (mania/depressão), depressão maior
- Gravidade: 3-9 (escala contínua)
- Histórico de tratamento: sim/não
- Comorbidades: diabetes, hipertensão

### Variabilidade Farmacocinética

Parâmetros ajustados por:
- **Função hepática/renal** → clearance
- **CYP3A4** → metabolismo
  - Lento: CL × 0.6
  - Rápido: CL × 1.4
- **IMC** → absorção
- **Variabilidade aleatória** (~15%)

### Critérios de Resposta

**Boa resposta:**
- Eficácia ≥ 70%
- Ocupação D2: 60-80%
- EPS < 30%

**Resposta moderada:**
- Eficácia: 50-70%
- EPS < 50%

**Resposta pobre:**
- Demais casos

---

## 🧠 Arquiteturas Neurais Detalhadas

### LSTM Farmacocinética

```
Input (9 features) 
  ↓
Feature Embedding (64 → 128)
  ↓
LSTM 3 camadas (hidden=128, dropout=0.3)
  ↓
Decoder (256 → 128 → 100)
  ↓
Output (100 pontos temporais)
```

**Loss:** MSE (Mean Squared Error)  
**Optimizer:** Adam (lr=0.001)

### Otimizador de Dose

```
Input (12 features)
  ↓
256 (BN → ReLU → Dropout)
  ↓
512 (BN → ReLU → Dropout)
  ↓
256 (BN → ReLU → Dropout)
  ↓
128 (BN → ReLU → Dropout)
  ↓
1 (Sigmoid → escalar para 25-800mg)
```

**Loss:** MSE  
**Optimizer:** Adam (lr=0.001)

### Classificador de Resposta

```
Input (15 features)
  ↓
256 (BN → ReLU → Dropout)
  ↓
Residual Block 1 (256)
  ↓
Residual Block 2 (256)
  ↓
Self-Attention Mechanism
  ↓
128 (ReLU → Dropout)
  ↓
3 classes (Softmax)
```

**Loss:** CrossEntropy  
**Optimizer:** AdamW (lr=0.0005, weight_decay=1e-4)

### Autoencoder VAE

```
Encoder:
Input (20) → 128 → 64 → 32 → 8 (latent)
           ↓
        μ, log(σ²)

Decoder:
Latent (8) → 32 → 64 → 128 → 20 (recon)
```

**Loss:** Reconstruction + β·KL-Divergence  
**Optimizer:** Adam (lr=0.001)

---

## 📈 Pipeline de Treinamento

### Funcionalidades

✅ **Early Stopping** (paciência configurável)  
✅ **Checkpoint automático** (salva melhor modelo)  
✅ **Validação cruzada** (80/20 split)  
✅ **Normalização** (StandardScaler/MinMaxScaler)  
✅ **Batch processing** (batch_size=64)  
✅ **Curvas de aprendizado** (plots automáticos)

### Métricas Monitoradas

- **Train Loss** / **Val Loss**
- **Accuracy** (classificador)
- **MSE** (regressores)
- **Reconstruction Error** (VAE)

---

## 🔬 Casos de Uso Avançados

### 1. Otimização de Dose Personalizada

```python
from preditor_neural import criar_preditor

preditor = criar_preditor()

paciente = {
    'idade': 55,
    'peso': 85,
    'imc': 28.5,
    'funcao_hepatica': 0.8,  # Comprometimento leve
    'funcao_renal': 1.0,
    'cyp3a4': 'lento',  # Metabolizador lento
    'diagnostico': 'esquizofrenia',
    'gravidade_sintomas': 8.5,
    'sexo': 'M',
    'tratamento_previo': True
}

# AI recomenda dose ajustada
dose = preditor.recomendar_dose_otima(paciente)
# Resultado: ~400-450mg (ajustado para metabolizador lento)
```

### 2. Predição de Resposta antes do Tratamento

```python
# Simular com dose proposta
dose_teste = 400
# ... executar simulação PK/PD ...

# Classificar resposta esperada
classe, probs = preditor.classificar_resposta(
    paciente, dose_teste, cmax, auc, ocupacao_d2, eficacia
)

print(f"Resposta esperada: {classe}")
print(f"Probabilidades: {probs}")
# {'boa': 0.72, 'moderada': 0.25, 'pobre': 0.03}
```

### 3. Detecção de Pacientes Atípicos

```python
# Analisar com VAE
analise = preditor.analisar_paciente_vae(paciente)

if analise['is_outlier']:
    print("⚠️ Perfil atípico detectado!")
    print("Requer monitoramento mais próximo")
    print(f"Score de anomalia: {analise['anomaly_score']:.4f}")
```

---

## 📊 Resultados e Validação

### Dataset Gerado

- **3000 registros** de paciente-dose
- **1000 pacientes únicos**
- **Distribuição realista:**
  - Esquizofrenia: 40%
  - Bipolar mania: 25%
  - Bipolar depressão: 20%
  - Depressão maior: 15%

### Performance dos Modelos

| Modelo | Val Loss | Métricas | Épocas |
|--------|----------|----------|--------|
| LSTM PK | ~0.002 MSE | R²>0.95 | 150 |
| Otimizador | ~0.015 MSE | MAE<50mg | 100 |
| Classificador | ~0.35 CE | Acc~85% | 100 |
| VAE | ~0.08 Total | Recon<0.05 | 100 |

---

## ⚡ Comparação: Tradicional vs Neural

| Aspecto | Tradicional | Neural |
|---------|-------------|--------|
| **Dose** | Baseada em tabelas | Personalizada por AI |
| **PK** | EDOs determinísticas | LSTM aprende padrões |
| **Resposta** | Regras fixas | Classificação probabilística |
| **Pacientes** | Médio populacional | Individual |
| **Adaptação** | Manual | Treina com novos dados |
| **Speed** | Rápido (~1s) | Ultra-rápido (~0.1s) |

**Melhor abordagem:** **HÍBRIDA** 🎯
- Usar EDOs para entendimento físico
- Usar redes neurais para personalização

---

## 🔧 Configuração Avançada

### Retreinar Modelos com Novos Dados

```python
from treinamento import TreinadorRedeNeural
from modelos_neurais import LSTMFarmacocinetica

# Carregar dados novos
# ... preparar train_loader/val_loader ...

# Criar modelo
modelo = LSTMFarmacocinetica()

# Treinar
treinador = TreinadorRedeNeural(modelo)
treinador.treinar(
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=nn.MSELoss(),
    optimizer=optim.Adam(modelo.parameters(), lr=0.001),
    num_epochs=200,
    nome_modelo="lstm_pk_v2"
)
```

### Hyperparameter Tuning

Ajustar em `modelos_neurais.py`:
- `hidden_size`: 64, 128, 256
- `num_layers`: 2, 3, 4
- `dropout`: 0.2, 0.3, 0.4
- `learning_rate`: 1e-4, 5e-4, 1e-3

---

## 📚 Referências Científicas

### Farmacocinética
1. DeVane CL, Nemeroff CB. Clinical pharmacokinetics of quetiapine. *Clin Pharmacokinet*. 2001.
2. Kapur S, et al. Relationship between D₂ occupancy and response. *Am J Psychiatry*. 2000.

### Machine Learning em Farmacologia
3. Carpenter KA, et al. Deep learning for patient-specific dosing. *NPJ Digital Medicine*. 2020.
4. Ryu JY, et al. Deep learning for drug response prediction. *Briefings in Bioinformatics*. 2018.
5. Zhang L, et al. Neural networks in pharmacokinetics. *Pharmaceutics*. 2021.

---

## ⚠️ Limitações e Avisos

> [!WARNING]
> **Dados Sintéticos**
> 
> Os modelos neurais foram treinados com dados sintéticos gerados matematicamente.
> Para uso clínico real, seria necessário:
> - Treinar com dados reais de pacientes
> - Validação clínica prospectiva
> - Aprovação regulatória

> [!CAUTION]
> **Uso Educacional**
> 
> Este sistema é para fins educacionais e de pesquisa.
> **NÃO** substitui orientação médica profissional.

---

## 🎓 Recursos Adicionais

### Tutoriais
- `gerador_dados.py` - Como gerar dados sintéticos
- `modelos_neurais.py` - Arquiteturas PyTorch
- `treinar_modelos.py` - Pipeline completo
- `preditor_neural.py` - Fazer predições

### Documentação Original
Ver `README.md` (versão 1.0) para:
- Equações farmacocinéticas completas
- Modelo compartimental detalhado
- Receptores cerebrais
- Interface CLI

---

## 🚀 Próximos Passos

### Melhorias Planejadas (v3.0)
- [ ] Transfer Learning com dados reais
- [ ] Graph Neural Networks para interações medicamentosas
- [ ] Reinforcement Learning para otimização dinâmica
- [ ] Explainability (SHAP, LIME)
- [ ] API REST para integração
- [ ] Mobile app (FastAPI + Flutter)

---

## 💻 Especificações Técnicas

### Requisitos de Hardware

**Mínimo:**
- CPU: 2+ cores
- RAM: 4 GB
- Disco: 2 GB

**Recomendado:**
- GPU: CUDA compatible (NVIDIA)
- RAM: 8+ GB
- SSD para dados

### Software

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ (opcional, para GPU)

---

## 👨‍💻 Contribuindo

Este é um projeto educacional. Sugestões de melhoria:
1. Adicionar mais arquiteturas (Transformers, GNNs)
2. Implementar técnicas de XAI
3. Validar com dados reais
4. Otimizar performance

---

## 📄 Licença

Uso educacional e de pesquisa.  
**Direitos Autorais © 2025 Luiz Tiago Wilcke**

---

## 🙏 Agradecimentos

- Comunidade PyTorch
- SciPy/NumPy developers
- Literatura científica de farmacologia

---

**Desenvolvido com ❤️ e 🧠 por Luiz Tiago Wilcke**

**Última atualização:** 2025-11-25 | **Versão:** 2.0 (Neural Enhanced)
