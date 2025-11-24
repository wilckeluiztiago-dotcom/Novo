# BitcoinPreditor AI 🚀

Um sistema avançado de previsão de preços de criptoativos utilizando Equações Diferenciais Estocásticas (SDEs) de última geração. O sistema implementa o **Modelo de Bates** (Heston Stochastic Volatility + Merton Jump Diffusion) para capturar a dinâmica complexa do Bitcoin.

## 🧠 O Modelo Matemático

O sistema não usa "Machine Learning" caixa-preta, mas sim modelagem financeira quantitativa robusta.

### Equações (Modelo de Bates)
O preço $S_t$ e a variância $v_t$ evoluem de acordo com o sistema de SDEs:

$$
\begin{cases}
\frac{dS_t}{S_t} = (r - \lambda \bar{k})dt + \sqrt{v_t} dW_t^S + dZ_t \\
dv_t = \kappa(\theta - v_t)dt + \xi \sqrt{v_t} dW_t^v
\end{cases}
$$

Onde:
- **Heston**: Volatilidade estocástica com reversão à média ($\kappa, \theta, \xi$).
- **Merton**: Saltos de Poisson ($dZ_t$) para modelar crashes e pumps repentinos.
- **Correlação**: $dW_t^S$ e $dW_t^v$ têm correlação $\rho$ (efeito alavancagem).

## ⚡ Funcionalidades

- **Calibração Automática**: O sistema baixa dados históricos e usa algoritmos de otimização para encontrar os parâmetros ($\kappa, \theta, \xi, \rho, \lambda$) que melhor explicam o comportamento recente do mercado.
- **Simulação Monte Carlo Acelerada**: Usa compilador JIT (`numba`) para simular milhares de cenários em milissegundos.
- **Dashboard Interativo**: Interface web completa para análise de risco e projeções.

## 📦 Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/BitcoinPreditor.git
cd BitcoinPreditor
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

## 🚀 Como Usar

### Dashboard Web (Recomendado)
Para iniciar a interface visual:

```bash
streamlit run app/dashboard.py
```
O navegador abrirá automaticamente com o sistema.

### Linha de Comando (CLI)
Para uma previsão rápida no terminal:

```bash
python cli.py --ticker BTC-USD --days 30 --sims 1000
```

## 📊 Estrutura do Projeto

- `models/`: Implementação matemática (Bates, Heston).
- `engine/`: Motores de calibração e simulação Monte Carlo.
- `data/`: Integração com Yahoo Finance e cache.
- `app/`: Interface Streamlit.

## ⚠️ Aviso Legal

Este software é para fins educacionais e de pesquisa. **Não é uma recomendação de investimento.** Criptoativos são extremamente voláteis.
