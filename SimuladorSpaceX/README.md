# Simulador SpaceX - Falcon 9

**Autor:** Luiz Tiago Wilcke  
**Versão:** 1.0.0

## Visão Geral

Este projeto é um simulador completo de voo para o foguete Falcon 9 da SpaceX, desenvolvido em Python. Ele integra um modelo físico avançado, uma calculadora de missão interativa e uma simulação visual em tempo real.

## Estrutura do Projeto

- `modelo/`: Núcleo matemático (Física e Parâmetros).
- `interface/`: Calculadora de Missão (Streamlit).
- `simulacao/`: Visualizador Gráfico (Pygame).
- `main.py`: Ponto de entrada do sistema.

## Modelo Matemático

O simulador resolve as equações diferenciais do movimento de um foguete de massa variável na atmosfera.

### 1. Forças Atuantes

A força resultante $F_{res}$ é dada por:

$$ F_{res} = F_{empuxo} - F_{gravidade} - F_{arrasto} $$

#### Empuxo ($T$)
O empuxo varia com a pressão atmosférica (altitude):

$$ T(h) = \dot{m} v_e + (p_e - p_{atm}(h)) A_e $$

No código, interpolamos entre o empuxo no nível do mar e no vácuo baseado na densidade do ar.

#### Gravidade ($g$)
A gravidade diminui com o quadrado da distância ao centro da Terra:

$$ g(h) = g_0 \left( \frac{R_E}{R_E + h} \right)^2 $$

#### Arrasto ($D$)
A resistência do ar é calculada pela equação do arrasto quadrático:

$$ D = \frac{1}{2} \rho(h) v^2 C_d A $$

Onde a densidade do ar $\rho(h)$ segue um modelo exponencial:

$$ \rho(h) = \rho_0 e^{-h/H} $$

### 2. Equação do Foguete (Tsiolkovsky)

A variação de massa é governada pela taxa de queima de combustível:

$$ \frac{dm}{dt} = -\frac{T}{I_{sp} g_0} $$

### 3. Solver Numérico

Utilizamos o método de **Runge-Kutta de 4ª Ordem (RK4)** para integrar o sistema de equações diferenciais ordinárias (EDOs), garantindo alta precisão na simulação da trajetória.

## Como Executar

### Pré-requisitos

Instale as dependências:

```bash
pip install -r requirements.txt
```

### Executando o Sistema

Para acessar o menu principal:

```bash
python main.py
```

Você terá duas opções:
1.  **Calculadora de Missão**: Abre uma interface web para planejar lançamentos.
2.  **Simulação Visual**: Abre uma janela gráfica mostrando o lançamento em tempo real.

## Autor

Desenvolvido por **Luiz Tiago Wilcke**.
