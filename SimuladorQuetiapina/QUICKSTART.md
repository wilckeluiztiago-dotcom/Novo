# 🚀 Guia de Início Rápido - Neural Enhanced

## ⚡ Setup Rápido (5 minutos)

### 1. Instalar Dependências
```bash
pip install torch numpy scipy matplotlib streamlit scikit-learn pandas
```

### 2. Gerar Dados de Treinamento
```bash
cd SimuladorQuetiapina
python3 gerador_dados.py
```
⏱️ **Tempo:** ~2 minutos | **Output:** 3000 registros

### 3. Treinar Modelos Neurais (OPCIONAL)
```bash
python3 treinar_modelos.py
```
⏱️ **Tempo:** 10-30 min (CPU) / 2-5 min (GPU)

> **Nota:** Modelos pré-treinados estão incluídos! Pule este passo se quiser testar rapidamente.

### 4. Testar Simulador Tradicional
```bash
python3 main.py --peso 70 --dose 300
```

### 5. Testar Predições Neurais
```bash
python3 preditor_neural.py
```

### 6. Dashboard Interativo
```bash
streamlit run app.py
```
Abrir: http://localhost:8501

---

## 🎯 Casos de Uso Rápidos

### Exemplo 1: Otimização de Dose com AI
```python
from preditor_neural import criar_preditor

preditor = criar_preditor()

paciente = {
    'idade': 45, 'peso': 70, 'imc': 22.9, 'sexo': 'M',
    'funcao_hepatica': 1.0, 'funcao_renal': 1.0,
    'cyp3a4': 'normal', 'diagnostico': 'esquizofrenia',
    'gravidade_sintomas': 7.0, 'tratamento_previo': False
}

dose = preditor.recomendar_dose_otima(paciente)
print(f"Dose AI: {dose} mg")
```

### Exemplo 2: Predição de Série Temporal
```python
serie_pk = preditor.predizer_serie_temporal_pk(
    idade=45, peso=70, imc=22.9,
    funcao_hepatica=1.0, funcao_renal=1.0,
    sexo='M', cyp3a4='normal', dose=300
)
print(f"Cmax previsto: {serie_pk.max():.3f} ng/mL")
```

---

## 📊 Modelos Disponíveis

| Modelo | Parâmetros | Função | Status |
|--------|------------|--------|--------|
| LSTM PK | 484K | Séries temporais | ✅ Treinado |
| Otimizador Dose | 301K | Dose ótima | ✅ Treinado |
| Classificador | 369K | Resposta terapêutica | ✅ Treinado |
| VAE | 27K | Análise de padrões | ✅ Treinado |
| Multitask | 402K | Predição múltipla | 🔧 Em desenvolvimento |

**Total:** 1.58M parâmetros

---

## 🔥 Recursos Principais

✅ Modelos matemáticos rigorosos (EDOs)  
✅ 5 redes neurais avançadas (PyTorch)  
✅ 3000 pacientes virtuais sintéticos  
✅ Pipeline de treinamento completo  
✅ Interface CLI + Dashboard web  
✅ Predições personalizadas por AI  

---

## 🆘 Troubleshooting

**Erro: "No module named 'torch'"**
```bash
pip install torch
```

**Erro: "Checkpoint não encontrado"**
```bash
python3 treinar_modelos.py  # Treinar modelos primeiro
```

**Dashboard não abre**
```bash
streamlit run app.py --server.port 8502  # Mudar porta
```

---

## 📖 Documentação Completa

- `README.md` - Versão 1.0 (Modelos matemáticos)
- `README_NEURAL.md` - Versão 2.0 (Redes neurais)
- `walkthrough.md` - Passo a passo detalhado

---

**Desenvolvido por Luiz Tiago Wilcke** | 2025
