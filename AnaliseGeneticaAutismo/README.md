# NeuroGen: Sistema Avançado de Análise Genética (Autismo/TEA)

Este software é uma plataforma completa e modular para simulação, análise e predição de riscos genéticos associados ao Transtorno do Espectro Autista (TEA) e síndromes relacionadas. Desenvolvido em Python, o sistema integra estatística clássica, biologia de sistemas e inteligência artificial.

## 🚀 Funcionalidades Principais

1.  **Simulação de Genomas Complexos**: Gera dados sintéticos realistas de SNPs (Single Nucleotide Polymorphisms) e expressão gênica (RNA-seq) para genes de alto risco como *SHANK3*, *MECP2* e *CHD8*.
2.  **Análise de Associação (GWAS)**: Realiza testes estatísticos para identificar variantes associadas ao fenótipo.
3.  **Score de Risco Poligênico (PRS)**: Calcula o risco genético cumulativo de cada indivíduo.
4.  **Análise Multivariada (PCA)**: Decompõe a variância genética em componentes principais para visualização 3D da estrutura populacional.
5.  **Inteligência Artificial (Deep Learning)**: Utiliza Redes Neurais (MLP) e Random Forest para prever o risco de desenvolvimento da síndrome com base em perfis genéticos e moleculares.
6.  **Visualização Avançada**: Inclui Manhattan Plots interativos, Heatmaps de expressão, Redes de Interação Proteica e Plots 3D.

## 🧠 Fundamentação Matemática e Científica

### 1. Análise de Componentes Principais (PCA)
Utilizamos decomposição de valores singulares (SVD) na matriz de genótipos normalizada $X$ para encontrar os autovetores que maximizam a variância.

$$ X = U \Sigma V^T $$

Os componentes principais (PCs) são projeções das amostras nesses autovetores, permitindo visualizar clusters populacionais em 3D.

### 2. Score de Risco Poligênico (PRS)
O PRS é calculado como a soma ponderada dos alelos de risco que um indivíduo carrega.

$$ PRS_i = \sum_{j=1}^{M} \beta_j \cdot G_{ij} $$

Onde:
*   $PRS_i$ é o score de risco para o indivíduo $i$.
*   $M$ é o número total de variantes genéticas (SNPs).
*   $\beta_j$ é o peso de efeito (log odds ratio) da variante $j$.
*   $G_{ij}$ é o genótipo do indivíduo $i$ para a variante $j$ (0, 1 ou 2 alelos de risco).

### 3. Redes Neurais Artificiais (MLP)
O sistema implementa um Perceptron Multicamadas para capturar não-linearidades. A saída de cada neurônio $j$ na camada $l$ é dada por:

$$ a_j^l = \sigma(\sum_k w_{jk}^l a_k^{l-1} + b_j^l) $$

Onde $\sigma$ é a função de ativação (ReLU/Sigmoid).

### 2. Teste de Associação (Qui-quadrado)
Para cada variante, testamos a hipótese nula de não associação entre o genótipo e o fenótipo (Caso vs Controle).

$$ \chi^2 = \sum \frac{(O - E)^2}{E} $$

Onde $O$ é a frequência observada e $E$ é a frequência esperada sob a hipótese nula.

### 3. Modelo de Predição (Random Forest)
O sistema utiliza um ensemble de árvores de decisão para classificar o risco. A probabilidade de classe é dada por:

$$ P(y=1|x) = \frac{1}{T} \sum_{t=1}^{T} P_t(y=1|x) $$

Onde $T$ é o número de árvores na floresta e $P_t$ é a probabilidade predita pela árvore $t$.

## 🛠️ Estrutura do Projeto

*   `dados/`: Módulo de geração de dados sintéticos (Genótipos, Fenótipos, Expressão).
*   `analise/`: Módulos estatísticos (GWAS, PRS, Frequências Alélicas).
*   `modelos/`: Algoritmos de Machine Learning (Random Forest).
*   `visualizacao/`: Geração de gráficos complexos (Plotly, NetworkX).
*   `interface/`: Dashboard interativo (Streamlit).
*   `configuracao.py`: Parâmetros globais e listas de genes.

## 📦 Como Executar

1.  Instale as dependências:
    ```bash
    pip install streamlit pandas numpy scipy scikit-learn plotly networkx
    ```

2.  Execute a aplicação:
    ```bash
    streamlit run app.py
    ```

## 🧬 Genes Analisados
O sistema foca em genes com forte evidência de associação ao TEA, incluindo:
*   **SHANK3**: Proteína de scaffolding sináptico.
*   **MECP2**: Regulador transcricional (Síndrome de Rett).
*   **CHD8**: Remodelador de cromatina.
*   **PTEN**, **ADNP**, **SYNGAP1**, entre outros.

---
Desenvolvido por **Luiz Tiago Wilcke**
