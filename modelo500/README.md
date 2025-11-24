# Sistema Avançado de Análise de Desemprego

Este projeto implementa um sistema completo para modelagem, previsão e análise do desemprego no Brasil, combinando técnicas avançadas de econometria (Séries Temporais e Análise Multivariada) com teoria econômica (Modelo de Salário Eficiência de Shapiro-Stiglitz).

## 🚀 Funcionalidades

### 1. Dashboard Interativo
Uma interface web moderna construída com **Streamlit** que permite:
- Visualizar indicadores macroeconômicos em tempo real (simulado).
- Realizar previsões de desemprego.
- Analisar choques econômicos (Impulso-Resposta).
- Simular o equilíbrio do mercado de trabalho.

### 2. Modelagem Matemática
O sistema inclui três módulos principais de modelagem:

*   **Séries Temporais (`modelos/series_temporais.py`)**:
    *   **SARIMA**: Para previsões de curto e médio prazo, capturando sazonalidade e tendências.
    *   **GARCH**: Para modelar a volatilidade e incerteza do mercado de trabalho.

*   **Análise Multivariada (`modelos/multivariada.py`)**:
    *   **VAR (Vetores Autorregressivos)**: Captura a dinâmica entre Desemprego, Inflação, Selic e PIB.
    *   **Causalidade de Granger**: Identifica relações de precedência temporal entre variáveis.

*   **Teoria Econômica (`modelos/stiglitz.py`)**:
    *   **Modelo de Shapiro-Stiglitz**: Implementação numérica da teoria de Salário Eficiência, explicando o desemprego involuntário como um dispositivo de disciplina.

### 3. Dados Brasileiros Simulados
O sistema utiliza um gerador de dados robusto (`gerador_dados.py`) calibrado com parâmetros da economia brasileira:
- Taxa de Desemprego (PNADC) com sazonalidade típica.
- Inflação (IPCA) e sua relação com o desemprego (Curva de Phillips).
- Taxa Selic reagindo à inflação (Regra de Taylor).

## 🛠️ Instalação e Execução

### Pré-requisitos
- Python 3.8+
- Bibliotecas listadas em `requirements.txt` (pandas, numpy, statsmodels, streamlit, plotly, etc.)

### Executando o Dashboard
Para iniciar a aplicação, execute o comando abaixo na raiz do projeto:

```bash
streamlit run dashboard/app.py
```

O dashboard será aberto automaticamente no seu navegador padrão.

## 📂 Estrutura do Projeto

```
modelo500/
├── dashboard/              # Aplicação Streamlit
│   ├── app.py              # Ponto de entrada
│   └── pages/              # Páginas do dashboard
├── modelos/                # Módulos matemáticos
│   ├── series_temporais.py # SARIMA/GARCH
│   ├── multivariada.py     # VAR/VECM
│   └── stiglitz.py         # Teoria Econômica
├── gerador_dados.py        # Simulação de dados macroeconômicos
├── test_modelos.py         # Testes automatizados
└── README.md               # Documentação
```

## 🧪 Testes
Para verificar a integridade dos modelos, execute:

```bash
python3 test_modelos.py
```

## 📝 Autor
Desenvolvido por Luiz Tiago Wilcke.
