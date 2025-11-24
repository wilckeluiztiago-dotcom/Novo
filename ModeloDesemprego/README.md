# Modelo de Previsão de Desemprego com SDEs

Um sistema avançado para modelagem e previsão de desemprego utilizando Equações Diferenciais Estocásticas (SDEs). Este projeto implementa múltiplos modelos econômicos matemáticos com solvers numéricos robustos.

Autor: **Luiz Tiago Wilcke**

## 🚀 Funcionalidades

- **Múltiplos Modelos Matemáticos**:
  - **Goodwin Estocástico**: Ciclos de crescimento e distribuição de renda (Predador-Presa).
  - **Curva de Phillips Estocástica**: Dinâmica entre inflação e desemprego.
  - **Crescimento Populacional**: Dinâmica de força de trabalho com choques.
  - **Markov Estocástico**: Transições entre estados de emprego (Formal/Informal/Desempregado).

- **Simulação Numérica Avançada**:
  - Métodos: Euler-Maruyama, Milstein e Runge-Kutta Estocástico (SRK).
  - Simulações de Monte Carlo para intervalos de confiança.
  - Análise de convergência forte.

- **Análise e Visualização**:
  - Gráficos de trajetórias, distribuições e diagramas de fase.
  - Testes estatísticos (Normalidade, Estacionariedade ADF/KPSS).
  - Medidas de risco (VaR, CVaR).

## 📦 Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/ModeloDesemprego.git
cd ModeloDesemprego
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

## 🛠️ Como Usar

### Simulação Básica
Execute o modelo padrão (Goodwin) por 10 anos:
```bash
python main.py
```

### Configuração Personalizada
Simule o modelo de Phillips com 500 trajetórias:
```bash
python main.py --modelo phillips --trajetorias 500 --tempo 20
```

### Comparação de Modelos
Compare todos os modelos disponíveis:
```bash
python main.py --comparar
```

### Análise de Convergência
Verifique a precisão numérica do solver:
```bash
python main.py --convergencia
```

## 📊 Modelos Implementados

### 1. Modelo de Goodwin Estocástico
Baseado nas equações de Lotka-Volterra, modela a luta de classes entre trabalhadores (emprego) e capitalistas (salários).
$$
\begin{cases}
du = u(\gamma - \alpha v)dt + \sigma_u u dW_1 \\
dv = v(\beta u - \delta)dt + \sigma_v v dW_2
\end{cases}
$$

### 2. Curva de Phillips Estocástica
Modela a relação trade-off entre inflação e desemprego com reversão à média.

## 📂 Estrutura do Projeto

- `modelos_sde.py`: Definição matemática das equações.
- `simulador.py`: Solvers numéricos (Euler, Milstein, SRK).
- `visualizador.py`: Geração de gráficos profissionais.
- `analise.py`: Testes estatísticos e métricas.
- `config.py`: Parâmetros globais e calibração.
- `main.py`: Interface de linha de comando.

## 📝 Licença

Este projeto está sob a licença MIT.
