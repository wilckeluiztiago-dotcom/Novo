# 🗳️ Sistema Avançado de Análise Eleitoral

Sistema completo de análise estatística para eleições de Deputados Federais e Estaduais do Brasil, utilizando métodos padrão e avançados de análise eleitoral.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Completo-success.svg)

## 📋 Índice

- [Visão Geral](#visão-geral)
- [Características](#características)
- [Instalação](#instalação)
- [Uso](#uso)
- [Modelos Estatísticos](#modelos-estatísticos)
- [Análises Eleitorais](#análises-eleitorais)
- [Equações Matemáticas](#equações-matemáticas)
- [Dashboard](#dashboard)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Referências](#referências)

## 🎯 Visão Geral

Este sistema oferece uma plataforma completa para análise eleitoral, combinando:

- **Modelos Estatísticos Clássicos**: Regressão, ARIMA, PCA
- **Machine Learning Avançado**: Random Forest, Gradient Boosting, LSTM
- **Inferência Bayesiana**: Modelos Hierárquicos, Dirichlet-Multinomial, MCMC
- **Análises Eleitorais Específicas**: Quociente Eleitoral, Volatilidade, Fragmentação
- **Dashboard Interativo**: Visualizações modernas com Streamlit e Plotly

## ✨ Características

### Modelos Preditivos

- ✅ **Regressão Linear Múltipla**: Análise de fatores que influenciam votação
- ✅ **Regressão Logística**: Probabilidade de eleição de candidatos
- ✅ **ARIMA/SARIMA**: Previsão de tendências eleitorais
- ✅ **Random Forest**: Importância de features e previsões robustas
- ✅ **Gradient Boosting/XGBoost**: Otimização de predições
- ✅ **LSTM**: Redes neurais para séries temporais
- ✅ **Modelos Bayesianos**: Inferência probabilística com incerteza
- ✅ **PCA**: Redução dimensional e identificação de padrões

### Análises Eleitorais

- ✅ **Quociente Eleitoral**: Distribuição de cadeiras pelo método D'Hondt
- ✅ **Análise de Coligações**: Eficiência e transferência de votos
- ✅ **Volatilidade Eleitoral**: Índice de Pedersen e análise temporal
- ✅ **Fragmentação Partidária**: NEP, HHI, concentração
- ✅ **Competitividade**: Margem de vitória, renovação parlamentar
- ✅ **Nacionalização**: Índice PNS e homogeneidade regional
- ✅ **Cadeia de Markov**: Transição de votos entre eleições

## 🚀 Instalação

### Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)

### Passos

1. Clone ou baixe o repositório:

```bash
cd analise_eleitoral
```

2. Instale as dependências:

```bash
pip install -r requirements.txt
```

3. Execute o dashboard:

```bash
streamlit run dashboard/app.py
```

O dashboard abrirá automaticamente em seu navegador em `http://localhost:8501`

## 💻 Uso

### Exemplo Básico - Regressão Linear

```python
from modelos.basicos import ModeloRegressao
from utils.dados import gerar_dados_eleitorais
import pandas as pd

# Gerar dados
dados = gerar_dados_eleitorais(n_candidatos=500, ano=2026)

# Preparar features
X = dados[['gasto_campanha', 'tempo_tv_segundos', 'incumbente']].values
y = dados['votos'].values

# Treinar modelo
modelo = ModeloRegressao()
modelo.treinar(X, y, features_nomes=['Gastos', 'Tempo TV', 'Incumbente'])

# Obter coeficientes
coeficientes = modelo.obter_coeficientes()
print(coeficientes)

# Fazer previsões
previsoes = modelo.prever(X)
```

### Exemplo - Quociente Eleitoral

```python
from modelos.eleitorais import QuocienteEleitoral

# Votos por partido
votos = {
    'PT': 5000000,
    'PL': 4500000,
    'PP': 3000000,
    'MDB': 2500000
}

# Calcular distribuição de cadeiras
qe = QuocienteEleitoral()
resultado = qe.calcular_distribuicao(votos, n_cadeiras=50)

print(resultado)
```

### Exemplo - Análise Bayesiana

```python
from modelos.bayesianos import ModeloDirichlet

# Votos por partido
votos = {'PT': 5000000, 'PL': 4500000, 'PP': 3000000}

# Treinar modelo Dirichlet
modelo = ModeloDirichlet()
modelo.treinar(votos)

# Obter proporções esperadas
proporcoes = modelo.obter_proporcoes_esperadas()

# Simular eleições
simulacoes = modelo.prever_eleicao(n_votos_total=10000000, n_simulacoes=1000)

# Probabilidade de vitória
prob_vitoria = modelo.probabilidade_vitoria()
```

## 📊 Modelos Estatísticos

### 1. Regressão Linear Múltipla

**Equação:**

```
Y = β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ + ε
```

Onde:
- `Y`: votos do candidato
- `X₁, X₂, ..., Xₙ`: variáveis explicativas (gastos, tempo TV, etc.)
- `β₀, β₁, ..., βₙ`: coeficientes a estimar
- `ε`: erro aleatório

**Uso:** Identificar quais fatores mais influenciam a votação.

### 2. Regressão Logística

**Equação:**

```
P(Y=1|X) = 1 / (1 + e^(-(β₀ + β₁X₁ + ... + βₙXₙ)))
```

Onde:
- `P(Y=1|X)`: probabilidade de ser eleito
- `e`: número de Euler (≈2.718)

**Uso:** Prever probabilidade de eleição de candidatos.

### 3. ARIMA (p, d, q)

**Equação Geral:**

```
(1 - φ₁B - ... - φₚBᵖ)(1-B)ᵈyₜ = (1 + θ₁B + ... + θᵧBᵧ)εₜ
```

Onde:
- `yₜ`: valor no tempo t
- `B`: operador de atraso (Byₜ = yₜ₋₁)
- `φᵢ`: parâmetros autoregressivos (AR)
- `θⱼ`: parâmetros de média móvel (MA)
- `d`: ordem de diferenciação
- `εₜ`: erro no tempo t

**Uso:** Previsão de tendências eleitorais ao longo do tempo.

### 4. Random Forest

**Equação:**

```
ŷ = (1/B) Σᵢ₌₁ᴮ fᵢ(x)
```

Onde:
- `B`: número de árvores
- `fᵢ(x)`: previsão da i-ésima árvore

**Uso:** Capturar relações não-lineares e fornecer importância de features.

### 5. Gradient Boosting

**Equação:**

```
Fₘ(x) = Fₘ₋₁(x) + ν·hₘ(x)
```

Onde:
- `Fₘ(x)`: modelo na iteração m
- `hₘ(x)`: nova árvore que corrige erros
- `ν`: taxa de aprendizado

**Uso:** Otimização sequencial de predições.

### 6. LSTM (Long Short-Term Memory)

**Equações:**

```
fₜ = σ(Wf·[hₜ₋₁, xₜ] + bf)      # forget gate
iₜ = σ(Wi·[hₜ₋₁, xₜ] + bi)      # input gate
C̃ₜ = tanh(Wc·[hₜ₋₁, xₜ] + bc)   # candidate values
Cₜ = fₜ * Cₜ₋₁ + iₜ * C̃ₜ        # cell state
oₜ = σ(Wo·[hₜ₋₁, xₜ] + bo)      # output gate
hₜ = oₜ * tanh(Cₜ)              # hidden state
```

**Uso:** Capturar padrões temporais complexos em séries eleitorais.

### 7. Modelo Bayesiano Hierárquico

**Estrutura:**

```
Nível 1 (Nacional):  μ_nacional ~ Normal(μ₀, σ₀²)
Nível 2 (Estado):    μ_estado ~ Normal(μ_nacional, τ²)
Nível 3 (Dados):     votos ~ Normal(μ_estado, σ²)
```

**Uso:** Estimação de votos por região com compartilhamento de informação.

### 8. Modelo Dirichlet-Multinomial

**Distribuição Dirichlet:**

```
p(θ|α) = [Γ(Σαᵢ) / Πᵢ Γ(αᵢ)] · Πᵢ θᵢ^(αᵢ-1)
```

Onde:
- `θ = (θ₁, ..., θₖ)`: proporções de votos (Σθᵢ = 1)
- `α = (α₁, ..., αₖ)`: parâmetros de concentração

**Posterior:**

```
p(θ|dados) ~ Dirichlet(α + contagens)
```

**Uso:** Modelar incerteza em proporções de votos entre partidos.

## 🔍 Análises Eleitorais

### 1. Quociente Eleitoral (Método D'Hondt)

**Quociente Eleitoral:**

```
QE = Votos Válidos / Cadeiras Disponíveis
```

**Quociente Partidário:**

```
QP_partido = Votos do Partido / QE
Cadeiras iniciais = floor(QP_partido)
```

**Distribuição de Sobras:**

Para cada cadeira restante:

```
Média_partido = Votos do Partido / (Cadeiras já obtidas + 1)
```

Atribuir à maior média.

### 2. Volatilidade Eleitoral (Índice de Pedersen)

**Equação:**

```
V = (1/2) Σᵢ |pᵢₜ - pᵢₜ₋₁|
```

Onde:
- `pᵢₜ`: proporção de votos do partido i no tempo t
- `V`: volatilidade (0-100)

**Interpretação:**
- `V < 10`: baixa volatilidade (sistema estável)
- `10 ≤ V < 20`: volatilidade moderada
- `V ≥ 20`: alta volatilidade (sistema instável)

### 3. Número Efetivo de Partidos (NEP)

**Equação de Laakso-Taagepera:**

```
NEP = 1 / Σᵢ pᵢ²
```

Onde `pᵢ` é a proporção de votos (ou cadeiras) do partido i.

**Interpretação:**
- `NEP = 2`: sistema bipartidário
- `NEP = 3-5`: multipartidarismo moderado
- `NEP > 5`: alta fragmentação

### 4. Índice de Herfindahl-Hirschman (HHI)

**Equação:**

```
HHI = Σᵢ pᵢ² × 10000
```

**Interpretação:**
- `HHI < 1500`: baixa concentração
- `1500 ≤ HHI < 2500`: concentração moderada
- `HHI ≥ 2500`: alta concentração

### 5. Índice de Gallagher (Desproporcionalidade)

**Equação:**

```
G = √(0.5 × Σᵢ (vᵢ - cᵢ)²)
```

Onde:
- `vᵢ`: percentual de votos do partido i
- `cᵢ`: percentual de cadeiras do partido i

### 6. Índice de Nacionalização Partidária (PNS)

**Equação de Bochsler:**

```
PNS = 1 - √(Σᵢ (vᵢ - v̄)² · (eᵢ/E))
```

Onde:
- `vᵢ`: proporção de votos do partido na região i
- `v̄`: proporção média nacional
- `eᵢ`: eleitores na região i
- `E`: total de eleitores

**Interpretação:**
- `PNS → 1`: partido nacionalizado (desempenho uniforme)
- `PNS → 0`: partido regionalizado

### 7. Cadeia de Markov (Transição de Votos)

**Matriz de Transição P:**

```
P_ij = P(votar em j no tempo t+1 | votou em i no tempo t)
```

**Propriedades:**
- `Σⱼ P_ij = 1` (cada linha soma 1)
- `P_ij ≥ 0` (probabilidades não-negativas)

**Previsão:**

```
v_{t+1} = v_t · P
```

**Estado Estacionário:**

```
π = π · P
```

### 8. Índice de Fracionalização (Rae)

**Equação:**

```
F = 1 - Σᵢ pᵢ²
```

**Interpretação:**
- `F = 0`: um único partido
- `F → 1`: fragmentação máxima

## 🎨 Dashboard

O dashboard oferece 7 seções principais:

### 1. 📊 Visão Geral
- Métricas principais (candidatos, votos, eleitos, partidos)
- Gráficos de votos e cadeiras por partido
- Número Efetivo de Partidos (NEP)

### 2. 🤖 Modelos Preditivos
- Seleção e execução de modelos
- Visualização de coeficientes e importância de features
- Métricas de performance (R², acurácia)

### 3. 🤝 Coligações
- Análise de eficiência de coligações
- Distribuição de sobras eleitorais
- Impacto de coligações no resultado

### 4. 📈 Volatilidade
- Índice de Pedersen
- Identificação de partidos voláteis
- Evolução temporal da volatilidade

### 5. 🔀 Fragmentação
- NEP, HHI, índices de concentração
- Distribuição de tamanhos dos partidos
- Análise temporal

### 6. ⚔️ Competitividade
- Margem de vitória por estado
- Taxa de renovação parlamentar
- Identificação de distritos competitivos

### 7. 🎯 Simulador
- Simulação de cenários eleitorais
- Ajuste de parâmetros (candidatos, cadeiras, coligações)
- Visualização de resultados e métricas

## 📁 Estrutura do Projeto

```
analise_eleitoral/
├── modelos/
│   ├── __init__.py
│   ├── basicos.py          # Regressão, ARIMA, PCA
│   ├── avancados.py        # Random Forest, XGBoost, LSTM
│   ├── bayesianos.py       # Modelos Bayesianos, MCMC
│   └── eleitorais.py       # Quociente, Markov, NEP, PNS
├── analises/
│   ├── __init__.py
│   ├── coligacoes.py       # Análise de coligações
│   ├── volatilidade.py     # Índice de Pedersen
│   ├── fragmentacao.py     # NEP, HHI
│   └── competitividade.py  # Margem, renovação
├── utils/
│   ├── __init__.py
│   ├── dados.py            # Geração de dados simulados
│   └── metricas.py         # Métricas de avaliação
├── dashboard/
│   └── app.py              # Dashboard Streamlit
├── README.md
└── requirements.txt
```

## 📚 Referências

### Livros e Artigos

1. **Nicolau, J.** (2012). *Sistemas Eleitorais*. FGV Editora.

2. **Pedersen, M. N.** (1979). "The Dynamics of European Party Systems: Changing Patterns of Electoral Volatility". *European Journal of Political Research*, 7(1), 1-26.

3. **Laakso, M., & Taagepera, R.** (1979). "Effective Number of Parties: A Measure with Application to West Europe". *Comparative Political Studies*, 12(1), 3-27.

4. **Gallagher, M.** (1991). "Proportionality, Disproportionality and Electoral Systems". *Electoral Studies*, 10(1), 33-51.

5. **Bochsler, D.** (2010). "Measuring Party Nationalisation: A New Gelman-King Index". *Electoral Studies*, 29(1), 155-168.

### Métodos Estatísticos

6. **James, G., Witten, D., Hastie, T., & Tibshirani, R.** (2013). *An Introduction to Statistical Learning*. Springer.

7. **Hastie, T., Tibshirani, R., & Friedman, J.** (2009). *The Elements of Statistical Learning*. Springer.

8. **Gelman, A., & Hill, J.** (2006). *Data Analysis Using Regression and Multilevel/Hierarchical Models*. Cambridge University Press.

9. **McElreath, R.** (2020). *Statistical Rethinking: A Bayesian Course with Examples in R and Stan*. CRC Press.

### Machine Learning

10. **Goodfellow, I., Bengio, Y., & Courville, A.** (2016). *Deep Learning*. MIT Press.

11. **Chen, T., & Guestrin, C.** (2016). "XGBoost: A Scalable Tree Boosting System". *KDD '16*.

## 📝 Licença

Este projeto é distribuído sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.

## 🤝 Contribuições

Contribuições são bem-vindas! Por favor:

1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/NovaFeature`)
3. Commit suas mudanças (`git commit -m 'Adiciona NovaFeature'`)
4. Push para a branch (`git push origin feature/NovaFeature`)
5. Abra um Pull Request

## 📧 Contato

Para questões, sugestões ou colaborações, entre em contato.

## 🙏 Agradecimentos

- Tribunal Superior Eleitoral (TSE) - dados eleitorais brasileiros
- Comunidade Python e desenvolvedores de bibliotecas open-source
- Pesquisadores em ciência política e análise eleitoral

---

**Desenvolvido com ❤️ usando Python, Streamlit, Scikit-learn, TensorFlow, PyMC3 e Plotly**
