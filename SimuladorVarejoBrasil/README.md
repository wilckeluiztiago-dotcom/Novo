# Simulador Varejo Brasil

**Autor:** Luiz Tiago Wilcke  
**Versão:** 1.0.0

## Visão Geral

O **Simulador Varejo Brasil** é um sistema avançado para modelagem e análise do setor varejista brasileiro. Ele utiliza equações diferenciais para simular a dinâmica de demanda, estoque e lucratividade, considerando as especificidades regionais do Brasil.

## Estrutura do Projeto

- `modelo/`: Núcleo matemático (EDOs e Parâmetros Regionais).
- `interface/`: Dashboard interativo (Streamlit).
- `main.py`: Script principal.

## Modelo Econômico

O simulador baseia-se em um sistema de Equações Diferenciais Ordinárias (EDOs).

### 1. Dinâmica da Demanda ($D$)

A demanda segue um modelo logístico modificado por fatores externos:

$$ \frac{dD}{dt} = r \cdot D \left(1 - \frac{D}{K}\right) + (\alpha \cdot \text{Marketing} - \beta \cdot \text{Preço}) $$

- **$r$**: Taxa de crescimento intrínseca (ajustada pelo PIB regional).
- **$K$**: Capacidade de suporte do mercado (População regional).
- **Sazonalidade**: Fatores multiplicativos aplicados à demanda efetiva (Natal, Black Friday, etc.).

### 2. Dinâmica de Estoque ($I$)

O estoque varia conforme o fluxo de entrada (compras) e saída (vendas):

$$ \frac{dI}{dt} = \text{Compras}(t) - \text{Vendas}(t) $$

- **Vendas**: $\min(\text{Demanda Efetiva}, \text{Estoque Atual})$.
- **Compras**: Sistema de controle proporcional para manter o estoque próximo à meta definida.

### 3. Análise Financeira

- **Receita**: Vendas $\times$ Preço.
- **Lucro**: Receita - (Custos Fixos + Custos Variáveis + Logística Regional + Marketing).

## Parâmetros Regionais

O sistema inclui dados específicos para as 5 regiões do Brasil:
- **Norte**: Logística complexa, menor densidade populacional.
- **Nordeste**: Grande mercado consumidor, sensibilidade a preço.
- **Centro-Oeste**: Forte influência do agronegócio no PIB.
- **Sudeste**: Maior concentração de renda e eficiência logística.
- **Sul**: Alta sazonalidade climática (inverno).

## Como Executar

1. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

2. Execute o simulador:
   ```bash
   python main.py
   ```
