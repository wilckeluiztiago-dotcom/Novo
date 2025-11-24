# PrevisorPetroleo AI 🛢️

Sistema avançado para previsão de preços de commodities (Petróleo Brent e WTI) utilizando Equações Diferenciais Estocásticas (SDEs) com Reversão à Média.

## 🧠 O Modelo Matemático (MRSVJ)

Diferente de criptoativos ou ações de tecnologia que podem crescer indefinidamente, commodities tendem a oscilar em torno de um preço de equilíbrio (custo marginal de produção). Por isso, utilizamos o modelo **Mean-Reverting Stochastic Volatility with Jumps**.

### Equações
O log-preço $X_t = \ln S_t$ e a variância $v_t$ evoluem como:

$$
\begin{cases}
dX_t = \kappa_S (\theta_S - X_t)dt + \sqrt{v_t} dW_t^S + J dN_t \\
dv_t = \kappa_v (\theta_v - v_t)dt + \xi \sqrt{v_t} dW_t^v
\end{cases}
$$

Onde:
- **Reversão à Média ($\kappa_S, \theta_S$)**: Força o preço a voltar para o equilíbrio de longo prazo quando está muito alto ou muito baixo.
- **Volatilidade Estocástica ($\kappa_v, \theta_v, \xi$)**: A incerteza do mercado muda com o tempo (ex: períodos de guerra vs paz).
- **Saltos ($J$)**: Choques repentinos (ex: decisões da OPEP, conflitos geopolíticos).

## ⚡ Funcionalidades

- **Calibração Automática**: O sistema calcula o "Preço Justo" de equilíbrio baseado no histórico de 10+ anos.
- **Simulação de Crises**: Permite injetar choques geopolíticos artificiais para testes de estresse ("Stress Testing").
- **Dashboard Profissional**: Interface em Streamlit com cones de probabilidade.

## 📦 Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/PrevisorPetroleo.git
cd PrevisorPetroleo
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

## 🚀 Como Usar

Execute o painel de controle:

```bash
streamlit run app/painel.py
```

O sistema abrirá automaticamente no seu navegador, permitindo escolher entre **Brent (Global)** e **WTI (EUA)**.

## 📊 Estrutura do Projeto

- `modelos/`: Implementação matemática (MRSVJ).
- `motor/`: Motores de calibração e simulação Monte Carlo.
- `dados/`: Integração com Yahoo Finance (BZ=F, CL=F).
- `app/`: Interface Streamlit em Português.

## ⚠️ Aviso Legal

Este software é para fins educacionais e de pesquisa. **Não é uma recomendação de investimento.** Commodities são ativos de alto risco.
