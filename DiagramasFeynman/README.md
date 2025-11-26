# Editor de Diagramas de Feynman

Desenvolvido por **Luiz Tiago Wilcke**.

Uma ferramenta poderosa para desenhar diagramas de Feynman e gerar automaticamente as expressões matemáticas correspondentes (integrais e amplitudes) usando a biblioteca `sympy`.

## Funcionalidades

- **Interface Gráfica Moderna**: Construída com `customtkinter`.
- **Desenho Intuitivo**:
  - **Férmions (e-)**: Linhas retas com setas.
  - **Fótons (γ)**: Linhas onduladas (senoides).
  - **Glúons (g)**: Linhas encaracoladas (loops).
- **Motor de Física**:
  - Definição de partículas do Modelo Padrão.
  - Regras de Feynman para QED e QCD.
  - Geração simbólica de integrais.

## Como Usar

1. **Instale as dependências**:
   ```bash
   pip install customtkinter sympy numpy
   ```

2. **Execute o programa**:
   ```bash
   python3 main.py
   ```

3. **Desenhe**:
   - Selecione uma partícula na barra lateral.
   - Clique e arraste no canvas para desenhar a linha.
   - Use "Limpar Tela" para recomeçar.

4. **Gere a Integral**:
   - Clique em "Gerar Integral" para ver a expressão matemática correspondente ao diagrama (atualmente mostra um exemplo de espalhamento Moller).

## Estrutura do Projeto

- `main.py`: Ponto de entrada.
- `gui.py`: Interface gráfica.
- `desenho.py`: Algoritmos de desenho vetorial.
- `fisica.py`: Definições de física de partículas e regras de Feynman.
