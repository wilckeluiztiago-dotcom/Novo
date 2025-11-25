# IndicePobreza

**Autor:** Luiz Tiago Wilcke  
**Versão:** 1.0.0

## Visão Geral

O **IndicePobreza** é um sistema avançado e modular desenvolvido em Python para a análise e cálculo de índices de pobreza e desigualdade no Brasil. O projeto utiliza dados simulados baseados na PNAD Contínua e implementa métodos estatísticos robustos, desde indicadores clássicos até modelagem bayesiana hierárquica.

## Estrutura do Projeto

- `dados/`: Geração e processamento de dados socioeconômicos.
- `indicadores/`: Implementação de índices clássicos (FGT, Gini, Theil).
- `modelos/`: Modelos avançados (MPI Alkire-Foster, Bayesiano).
- `visualizacao/`: Gráficos estáticos e dashboard interativo.
- `main.py`: Script principal para execução e relatórios.

## Metodologia e Equações

### 1. Índices Foster-Greer-Thorbecke (FGT)

A família de índices FGT é definida por:

$$ P_\alpha = \frac{1}{N} \sum_{i=1}^{q} \left( \frac{z - y_i}{z} \right)^\alpha $$

Onde:
- $N$: População total.
- $q$: Número de pobres ($y_i < z$).
- $z$: Linha de pobreza.
- $y_i$: Renda do indivíduo $i$.
- $\alpha$: Parâmetro de aversão à pobreza.
    - $\alpha=0$: Incidência (Proporção de pobres).
    - $\alpha=1$: Hiato (Profundidade média da pobreza).
    - $\alpha=2$: Severidade (Considera a desigualdade entre os pobres).

### 2. Coeficiente de Gini

O Coeficiente de Gini mede a desigualdade de renda:

$$ G = \frac{2 \sum_{i=1}^{n} i y_i}{n \sum_{i=1}^{n} y_i} - \frac{n+1}{n} $$

Onde $y_i$ são as rendas ordenadas de forma crescente.

### 3. Índice de Pobreza Multidimensional (MPI) - Método Alkire-Foster

O MPI considera privações em múltiplas dimensões (saúde, educação, padrão de vida).

1.  **Matriz de Privação ($g_{ij}$)**: 1 se o domicílio $i$ é privado na dimensão $j$, 0 caso contrário.
2.  **Escore de Privação ($c_i$)**: Soma ponderada das privações.
    $$ c_i = \sum_{j=1}^{d} w_j g_{ij} $$
3.  **Identificação**: Domicílio é pobre se $c_i \geq k$ (corte, ex: 0.33).
4.  **Cálculo do MPI ($M_0$)**:
    $$ MPI = H \times A $$
    - $H = q/n$ (Incidência multidimensional).
    - $A = \frac{1}{q} \sum_{i=1}^{q} c_i$ (Intensidade média).

### 4. Modelo Hierárquico Bayesiano

Utilizamos um modelo Logit Hierárquico para estimar a probabilidade de pobreza, considerando efeitos fixos (renda, escolaridade) e efeitos aleatórios por Unidade Federativa (UF).

$$ y_i \sim \text{Bernoulli}(p_i) $$
$$ \text{logit}(p_i) = \alpha + \beta_1 \cdot \text{Renda}_i + \beta_2 \cdot \text{Educ}_i + u_{UF[i]} $$
$$ u_{UF} \sim \mathcal{N}(0, \sigma_{UF}^2) $$

Isso permite estimativas mais robustas para estados com menos observações (shrinkage).

## Como Executar

### Instalação das Dependências

```bash
pip install -r requirements.txt
```

### Execução do Relatório

```bash
python main.py
```

### Dashboard Interativo

Para visualizar os dados e interagir com os modelos:

```bash
streamlit run visualizacao/dashboard.py
```

## Licença

Este projeto é de autoria de Luiz Tiago Wilcke.
