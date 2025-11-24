# Simulador Neural‑Estocástico de Valorização de Ações (Heston‑Jump)

Projeto em Python (arquivo único) para **simular a valorização de ações** de uma empresa usando **equações diferenciais estocásticas avançadas** + Monte Carlo, com opção de correção via rede neural.

> **Aviso importante:** este projeto é **didático/portfólio**. Não constitui recomendação de investimento.

---

## 1) Modelo matemático

Modelamos o preço da ação \(S_t\) sob medida “real‑world” (P), com:

### 1.1 Preço com volatilidade estocástica + saltos

\[
\frac{dS_t}{S_t} = \mu_t\,dt + \sqrt{v_t}\,dW_{1,t} + (J-1)\,dN_t
\]

- \(v_t\) é a variância estocástica (Heston).
- \(N_t\) é um processo de Poisson com intensidade \(\lambda_J\).
- O tamanho do salto é multiplicativo: \(J=\exp(Y)\), com \(Y\sim \mathcal{N}(\mu_J,\sigma_J^2)\).

### 1.2 Dinâmica da variância (Heston)

\[
dv_t = \kappa_v(\theta_v - v_t)\,dt + \sigma_v\sqrt{v_t}\,dW_{2,t}
\]

com correlação:

\[
\text{corr}(dW_{1,t}, dW_{2,t})=\rho
\]

### 1.3 Drift “fundamentalista + momentum”

\[
\mu_t = \mu_{base} + \lambda_{fund}
\Big[
\kappa_{valor}\,\frac{V_t - S_t}{S_t}
+
\gamma_{mom}\,M_t
\Big]
\cdot \frac{1}{1+2\,\sigma_{real,t}}
\]

- \(V_t\): “valor justo” latente via EMA (média móvel exponencial).
- \(M_t\): momentum (retorno acumulado em janela).
- amortecimento reduz drift em alta volatilidade realizada.

### 1.4 Correção neural opcional

Uma MLP aprende um ajuste extra \(\Delta\mu_t\) a partir de janelas temporais de features:

\[
\mu_t \leftarrow \mu_t + \Delta \mu_t^{(NN)}
\]

---

## 2) O que o código faz

1. Baixa dados do Yahoo Finance via `yfinance` (se disponível).  
   Se não encontrar, gera série sintética para você rodar offline.
2. Extrai features (retorno log, vol realizada, momentum, valor justo).
3. Constrói drift fundamentalista.
4. (Opcional) treina rede neural para correção de drift.
5. Simula **milhares de trajetórias Monte Carlo** com Heston‑Jump.
6. Exibe:
   - Trajetórias simuladas.
   - Distribuição do preço final.
   - Distribuição do retorno.
   - Probabilidades de cenários (+10% / −10%).
   - VaR/CVaR.

---

## 3) Requisitos

- Python 3.10+
- numpy, pandas, matplotlib
- (Opcional) yfinance
- (Opcional) torch

Instalação:

```bash
pip install numpy pandas matplotlib
pip install yfinance            # opcional
pip install torch               # opcional (rede neural)
```

---

## 4) Como rodar

Rodando padrão:

```bash
python simulador_valorizacao_acoes.py
```

Escolher ticker e parâmetros:

```bash
python simulador_valorizacao_acoes.py --ticker AAPL --horizonte 252 --trajetorias 10000
```

Ativar rede neural:

```bash
python simulador_valorizacao_acoes.py --ticker PETR4.SA --usar_rede_neural
```

---

## 5) Estrutura do projeto

```
projeto_simulador_acoes/
 ├─ simulador_valorizacao_acoes.py
 └─ README.md
```

---

## 6) Ideias para você evoluir (nível recrutador)

- Calibração de Heston por máxima verossimilhança.
- Inclusão de fatores macro (SELIC, IPCA, dólar) no drift.
- Backtesting walk‑forward com validação temporal.
- Comparar com Black‑Scholes e GARCH.
- Dashboard web (Streamlit) para sliders de parâmetros.

---

Se quiser, posso adaptar este modelo para um ticker específico, incluir calibração automática real e transformar em dashboard interativo.
