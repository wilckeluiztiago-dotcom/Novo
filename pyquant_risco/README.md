# PyQuant Risco — Motor Avançado de Simulação de Risco em Python

Autor: Luiz Tiago Wilcke (LT)  
E-mail: wilckeluiztiago@gmail.com  
GitHub: https://github.com/wilckeluiztiago-dotcom

---

## Visão geral

PyQuant Risco é um projeto avançado em Python que combina:

- Modelagem estocástica com Equações Diferenciais Estocásticas (SDEs)
- Redes neurais recorrentes (LSTM) para previsão de retornos
- Motor de risco com cálculo de VaR e CVaR via simulação de Monte Carlo
- API em FastAPI para servir simulações e métricas em produção
- CLI em Typer para uso via terminal (simulações, treino de rede, API)
- Estrutura profissional de pacote Python, testes automatizados e logging

O objetivo é demonstrar competências profissionais em:

- Programação Python avançada
- Boas práticas de engenharia de software
- Estatística e finanças quantitativas
- Machine Learning com PyTorch
- Desenvolvimento de APIs e CLIs modernas

---

## Instalação

Recomendado usar Python 3.10+ em um virtualenv.

```bash
git clone https://github.com/SEU_USUARIO/pyquant_risco.git
cd pyquant_risco

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -e .
```

### Dependências principais

- numpy, pandas
- fastapi, uvicorn
- typer
- pydantic
- matplotlib (para gráficos nos exemplos)
- torch (opcional, para parte de deep learning)
- pytest (testes)

---

## Como usar

### 1) Linha de comando

O pacote registra um comando `pyquant-risco` via Typer.

Ver ajuda geral:

```bash
pyquant-risco --help
```

Simular trajetória de preço com SDE (GBM):

```bash
pyquant-risco simular-sde --preco-inicial 100 --mu 0.08 --volatilidade 0.25 --passos 252 --cenarios 1000
```

Calcular VaR e CVaR a partir de uma série histórica (CSV: data,preco):

```bash
pyquant-risco calcular-risco --caminho-csv dados/precos.csv --nivel-confianca 0.99
```

Treinar rede LSTM para prever retornos:

```bash
pyquant-risco treinar-rede --caminho-csv dados/precos.csv --epocas 30
```

Subir API local:

```bash
pyquant-risco api
# depois acessar: http://127.0.0.1:8000/docs
```

---

## 2) API (FastAPI)

Depois de rodar `pyquant-risco api`, você pode consumir endpoints REST:

- `GET /health` — health-check
- `POST /simular/sde` — simula SDE (GBM ou OU) e retorna trajetórias
- `POST /risco/var-cvar` — calcula VaR e CVaR a partir de retornos

A documentação interativa (Swagger) fica em:

- http://127.0.0.1:8000/docs

---

## 3) Exemplo em código

Veja `examples/exemplo_simulacao.py`:

- Carrega ou gera preços sintéticos
- Calibra a SDE (drift e volatilidade)
- Roda simulações de Monte Carlo
- Plota trajetórias e distribuições de retorno
- Calcula VaR e CVaR

---

## Modelo de risco

SDE básica (modelo de Black–Scholes):

- dS_t = μ S_t dt + σ S_t dW_t  
  onde μ é o drift (retorno médio) e σ a volatilidade.

Simulações são usadas para:

- Projetar distribuições de preços futuros
- Calcular VaR (Value-at-Risk) e CVaR (Conditional VaR)

A parte de deep learning usa LSTM para prever retornos:

- Entrada: janelas de retornos passados
- Saída: previsão de retorno futuro
- Rede treinada com MSE, otimizador Adam

---

## Stack técnica

- Python moderno (type hints, dataclasses, logging estruturado)
- Arquitetura modular (src/pyquant_risco/)
- CLI com Typer
- API com FastAPI + Pydantic
- PyTorch para redes neurais
- Numpy/Pandas para análise numérica
- Testes com pytest

---

## Como rodar os testes

```bash
pytest
```

---

## Por que este projeto é interessante para recrutadores?

- Demonstra domínio de Python além de scripts pontuais.
- Mostra entendimento de arquitetura de pacotes, CLI, API, testes e logging.
- Conecta teoria quantitativa (SDEs, risco, VaR/CVaR) com implementação concreta.
- Utiliza bibliotecas amplamente usadas no mercado (FastAPI, Typer, PyTorch).

---
