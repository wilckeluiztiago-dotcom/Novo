# Modelo Probabilístico‑Neural para Prever Votos — Deputado Estadual

Projeto **portfolio-friendly**: um único script Python que mistura **estatística avançada**, **SDEs** e **redes neurais** para estimar distribuições de votos para vários candidatos e gerar gráficos de probabilidade.

> **Foco:** mostrar capacidade técnica (modelagem probabilística moderna + PyTorch + simulação) sem fazer propaganda política.  
> **É um modelo analítico/didático.**

---

## Ideia matemática (curta e bonita)

### 1) Baseline neural com incerteza (MC Dropout)

Queremos um voto esperado por candidato:

\[
\log(1+V_i) = f_\theta(x_i) + \varepsilon_i
\]

- \(x_i\): features (gastos, seguidores, histórico, etc.)  
- \(f_\theta\): rede neural MLP.  
- Dropout ativo na inferência gera amostras \(f_\theta^{(s)}\Rightarrow\) **incerteza**.

### 2) Dinâmica estocástica no simplex (shares eleitorais)

Trabalhamos em um espaço latente \(z\) (log‑votos).  
A evolução ao longo da campanha é modelada por um processo de **Ornstein‑Uhlenbeck**:

\[
dz_t = \kappa (m - z_t) dt + \sigma dW_t
\]

- \(m\): alvo estocástico vindo da rede (baseline)  
- \(W_t\): ruído Browniano (choques de campanha)

Convertendo para **shares**:

\[
p_t = \text{softmax}(z_t)
\]

e votos absolutos:

\[
V_{i,t} = p_{i,t}\, V_{\text{total}}
\]

Esse tipo de difusão em simplex é coerente com a literatura de **Wright‑Fisher / logistic‑normal** para múltiplas categorias (voto multipartidário).citeturn0academia20turn0search5

### 3) Probabilidades finais

Simulamos milhares de trajetórias (Monte Carlo):

- \(P(\Delta V_i>0)\): chance de crescer vs. base  
- \(P(\text{top‑K})\): chance de “ganhar” (entrar no top‑K)  
- Percentis (fan chart) ao longo do tempo

Modelos Dirichlet/Hierárquicos são padrão em previsão eleitoral multi‑candidato; aqui usamos uma versão prática via latent softmax + SDE. citeturn0academia21turn0search1turn0search19

---

## Dados reais (TSE)

O TSE possui **dados abertos oficiais** de eleições desde 1933, incluindo resultados por candidato:  
Portal de Dados Abertos do TSE. citeturn0search8turn0search0turn0search2

Você pode baixar um `CSV` (ou juntar vários) e rodar o modelo.

### Exemplo de colunas mínimas

```
candidato, partido, uf, votos_2022, gastos_campanha, seguidores, incumbente, ...
```

Qualquer coluna numérica extra vira feature automaticamente.

---

## Como rodar

### 1) Instale dependências

```bash
pip install numpy pandas matplotlib torch scikit-learn
```

### 2) Rode com base sintética (demo)

```bash
python ModeloVotosDeputado.py --gerar_sintetico
```

### 3) Rode com seus dados reais

```bash
python ModeloVotosDeputado.py --csv seus_dados.csv --col_votos votos_2022
```

Parâmetro opcional de vitória:

```bash
python ModeloVotosDeputado.py --csv seus_dados.csv --top_k 7
```

---

## Saídas

No diretório `saida_modelo_votos/`:

- `resumo_probabilistico.csv`  
- `densidades_votos.png`  
- `prob_crescer.png`  
- `prob_topK.png`  
- `fan_chart_top6.png`  
- `metadados.json`

---

## O que isso mostra para recrutadores

- Forecast probabilístico real (distribuições, não só ponto)  
- Simulação estocástica (SDE / difusão em simplex)  
- PyTorch com incerteza (MC Dropout)  
- Pipeline completo → dados, treino, Monte Carlo, plots

---

## Limitações naturais

- Sem dados de pesquisa/tempo real o modelo só extrapola features históricas.
- Para elevar a acurácia: inclua pesquisas por semana, dados regionais e redes sociais.

---

Se quiser, posso te ajudar a:
1) montar um notebook de limpeza dos dados do TSE,  
2) colocar isso como README “premium” no GitHub com imagens,  
3) transformar em dashboard interativo.
