# 📈 Modelo Neural‑Estocástico para Previsão do Dólar (USD/BRL)

Projeto **portfólio** em Python que combina:
- **Features estocásticas** derivadas de uma EDE (SDE) tipo **Geometric Brownian Motion (GBM)**  
- **Simulação Monte Carlo** como fonte de variáveis explicativas  
- **Rede Neural LSTM Bidirecional + Atenção** em PyTorch  
- Previsão **multi‑step** do USD/BRL (ex.: 5 dias à frente)

Tudo está em **um único arquivo**, ideal para GitHub/recrutadores.

---

## 🚀 Como rodar

### 1) Criar ambiente
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
```

### 2) Instalar dependências
```bash
pip install numpy pandas matplotlib yfinance
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

> Se você tiver GPU NVIDIA e CUDA configurado, pode instalar a versão GPU do torch.

### 3) Executar
```bash
python previsao_dolar_neural_sde.py
```

Com parâmetros:
```bash
python previsao_dolar_neural_sde.py --ticker BRL=X --janela 60 --horizonte 5 --epocas 40
```

---

## 🧠 O que o código faz

1. **Baixa USD/BRL** do Yahoo Finance (`BRL=X`)  
2. Calcula retornos logarítmicos  
3. Extrai **drift/volatilidade rolling**  
4. Gera **volatilidade EWMA**  
5. Simula **caminhos GBM via Monte Carlo**  
6. Usa estatísticas desses caminhos como *features*  
7. Treina LSTM + atenção para prever `h` dias à frente  
8. Salva gráficos e modelo em `saidas/`

---

## 📊 Saídas

Após rodar, será criada a pasta:

```
saidas/
 ├─ loss.png
 ├─ previsao_t1.png
 ├─ previsao_multistep.png
 └─ modelo_treinado.pt
```

---

## 📌 Por que isso impressiona recrutadores?

- Une **estatística**, **processos estocásticos**, **simulação numérica** e **deep learning**  
- Pipeline completo: dados → features matemáticas → modelo → métricas → gráficos  
- Código organizado, modular e pronto para portfólio

---

## 👤 Autor
**Luiz Tiago Wilcke (LT)**  
GitHub: https://github.com/wilckeluiztiago-dotcom/Novo  
E‑mail: wilckeluiztiago@gmail.com