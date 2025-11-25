# Montador de Genoma Bacteriano *De Novo*

Este software implementa um montador de genomas bacterianos *de novo* utilizando Grafos de Bruijn, com foco em modelagem estatística rigorosa e código modular em Python.

## Funcionalidades

*   **Leitura e QC**: Suporte a arquivos FASTQ, filtragem por qualidade (Phred) e trimagem.
*   **Grafo de Bruijn**: Construção eficiente de grafos a partir de k-mers.
*   **Estatística**: Modelagem de cobertura via distribuição de Poisson e detecção de erros.
*   **Montagem**: Algoritmos para resolução de caminhos e geração de contigs.
*   **Visualização**: Gráficos de distribuição de cobertura e estrutura do grafo.
*   **Interface Gráfica**: GUI sofisticada com design moderno e gráficos interativos.

## Modelos Matemáticos

### 1. Distribuição de Cobertura (Poisson)

A chegada de reads em uma determinada posição do genoma é modelada como um processo de Poisson. A probabilidade de observar $k$ reads cobrindo uma base (ou k-mer) é dada por:

$$ P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!} $$

Onde $\lambda$ é a cobertura média esperada do genoma ($C = \frac{N \times L}{G}$), sendo $N$ o número de reads, $L$ o tamanho do read e $G$ o tamanho do genoma.

### 2. Probabilidade de Erro (Phred Score)

A qualidade de cada base é dada pelo score Phred $Q$. A probabilidade de erro $P_{erro}$ é calculada como:

$$ P_{erro} = 10^{\frac{-Q}{10}} $$

Para um k-mer de tamanho $k$, a probabilidade dele estar correto (assumindo independência entre erros de base) é:

$$ P(kmer\_correto) = \prod_{i=1}^{k} (1 - P_{erro, i}) $$

### 3. Métricas de Montagem (N50)

O N50 é uma métrica estatística ponderada que descreve o comprimento dos contigs. É definido como o comprimento do menor contig tal que a soma dos comprimentos de todos os contigs maiores ou iguais a ele representa pelo menos 50% do tamanho total da montagem.

## Estrutura do Projeto

*   `dados/`: Módulos para leitura e pré-processamento de sequências.
*   `nucleo/`: Algoritmos centrais (K-mers, Grafo de Bruijn, Montador).
*   `estatistica/`: Modelos probabilísticos e métricas.
*   `visualizacao/`: Geração de gráficos.
*   `main.py`: Script principal de execução (linha de comando).
*   `app_gui.py`: Interface gráfica sofisticada.

## Como Executar

### Interface Gráfica (Recomendado) 🎨

1.  Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```
2.  Execute a interface:
    ```bash
    python app_gui.py
    ```
3.  Use a interface para:
    - Selecionar arquivo FASTQ
    - Ajustar parâmetros (K-mer, cobertura, qualidade)
    - Iniciar montagem com um clique
    - Visualizar resultados e gráficos em tempo real
    - Exportar contigs em FASTA

### Linha de Comando 💻

1.  Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```
2.  Execute o montador:
    ```bash
    python main.py seu_arquivo.fastq
    ```

## Recursos da Interface Gráfica

✨ **Design Moderno**: Tema escuro profissional com cores vibrantes

📊 **4 Abas Principais**:
- **Configuração**: Seleção de arquivo e ajuste de parâmetros
- **Execução**: Log em tempo real e barra de progresso
- **Resultados**: Métricas e gráficos interativos
- **Contigs**: Visualização e exportação de sequências

🎯 **Recursos Avançados**:
- Execução em thread separada (interface não trava)
- Gráficos matplotlib integrados
- Exportação de FASTA com um clique
- Cópia de sequências para clipboard

## Autor

Desenvolvido como parte de um projeto de bioinformática avançada.

