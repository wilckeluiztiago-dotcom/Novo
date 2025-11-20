# Projeto Avançado (Seguro) — Fissão U-235 com EDP Multigrupo + Nêutrons Atrasados

**Autor:** Luiz Tiago Wilcke

Este projeto implementa a **forma real das equações diferenciais**
usadas em física de reatores (difusão multigrupo) e a **cinética
pontual com nêutrons atrasados**, porém com **parâmetros fictícios/
normalizados**, por segurança.

> **Aviso de segurança**
> - Não há seções de choque reais, enriquecimento real, geometria de reator,
>   dados de criticidade operacional, nem validação para uso industrial.
> - Toy model para estudo e portfólio.

---

## Equações implementadas

### Difusão multigrupo (2 grupos)
Para cada grupo g:

\[
\frac{\partial \phi_g}{\partial t}
= D_g\nabla^2\phi_g
- \Sigma_{a,g}\phi_g
+ \sum_{g'\neq g}\Sigma_{s,g'\to g}\phi_{g'}
+ \chi_g \frac{1-\beta}{\Lambda}P(t)\,F(\phi)
\]

onde o termo de fissão total é:

\[
F(\phi)=\sum_{g}\nu\Sigma_{f,g}\phi_g
\]

### Cinética pontual com nêutrons atrasados (6 grupos)

\[
\frac{dP}{dt}=\frac{\rho-\beta}{\Lambda}P+\sum_{i=1}^6\lambda_i C_i
\]

\[
\frac{dC_i}{dt}=\frac{\beta_i}{\Lambda}P-\lambda_i C_i
\]

### k_eff monitorado (simbólico toy)

\[
k_{eff}(t)=\frac{\int \nu\Sigma_f \phi\,dV}{\int\Sigma_a\phi\,dV}
\]

---

## Como rodar

### Dependências
```bash
python -m pip install numpy pygame
```

### Executar
```bash
python FissaoU235_MultiGrupoSeguro.py
```

Sem pygame:
```bash
python FissaoU235_MultiGrupoSeguro.py --sem-animacao
```

---

## Ajustando parâmetros (didático)

No bloco `ParametrosMultiGrupo` você encontra:
- `D_g`
- `Sigma_a,g`
- `nu_Sigma_f,g`
- `Sigma_s,g->g'`
- `chi_g`

Você pode substituir por valores de material didático **sob sua responsabilidade**.

---

## O que você vê

- Heatmap do fluxo de nêutrons por grupo (TAB troca rápido/térmico).
- Núcleos toy que acumulam fluxo térmico e “fissionam” visualmente.
- Painel com P(t), k_eff simbólico, energia toy.

---

## Licença
Livre para estudo e portfólio, com atribuição ao autor.
