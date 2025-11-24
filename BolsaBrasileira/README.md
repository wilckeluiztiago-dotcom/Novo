# Terminal B3 - Modelagem Estocástica 🇧🇷

Sistema de modelagem quantitativa para a Bolsa de Valores Brasileira (B3) utilizando o modelo de **Merton Jump Diffusion (MJD)**.

## 🚫 Sem "AI" Caixa-Preta
Este projeto rejeita o uso de redes neurais opacas. Utilizamos **Matemática Financeira Pura** (Equações Diferenciais Estocásticas) para modelar o comportamento dos ativos, garantindo transparência e robustez teórica.

## 🧠 O Modelo Matemático (Merton)

O mercado brasileiro é caracterizado por alta volatilidade e choques frequentes (políticos, fiscais, externos). O modelo de Black-Scholes é insuficiente pois assume distribuição normal.

Utilizamos o **Merton Jump Diffusion**:
$$dS_t = (\mu - \lambda k)S_t dt + \sigma S_t dW_t + S_t (e^J - 1) dN_t$$

Isso permite modelar:
1.  **Difusão Contínua**: O "ruído" normal do mercado diário.
2.  **Saltos de Poisson**: Eventos raros mas impactantes ("Cisnes Negros").

## 💻 Interface "Terminal"

O dashboard foi desenhado com estética de **Terminal Financeiro** (fundo preto, fonte monoespaçada, alto contraste), focado em dados e eficiência para traders e analistas quantitativos.

## 📦 Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/BolsaBrasileira.git
cd BolsaBrasileira
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

## 🚀 Como Usar

Execute o terminal:

```bash
streamlit run app/terminal.py
```

Digite o ticker do ativo desejado (ex: `PETR4`, `VALE3`, `WEGE3`, `^BVSP`) e clique em **EXECUTAR ANÁLISE**. O sistema adiciona `.SA` automaticamente se necessário.

## 📊 Estrutura

- `modelos/`: Implementação do MJD com Numba.
- `motor/`: Calibração via Método dos Momentos.
- `dados/`: Integração B3 via Yahoo Finance.
- `app/`: Interface Streamlit.

## ⚠️ Aviso Legal

Este software é para fins educacionais e de pesquisa. **Não é uma recomendação de investimento.** O mercado de renda variável envolve riscos significativos.
