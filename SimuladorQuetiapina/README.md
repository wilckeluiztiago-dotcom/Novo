# 💊 Simulador de Quetiapina no Cérebro Humano

**Autor:** Luiz Tiago Wilcke  
**Data:** 2025-11-25

## 📋 Descrição

Sistema avançado de simulação farmacocinética e farmacodinâmica da **Quetiapina** (antipsicótico atípico) no cérebro humano. O simulador utiliza modelos matemáticos baseados em equações diferenciais ordinárias (EDOs) para simular:

- **Farmacocinética (ADME)**: Absorção, Distribuição, Metabolismo e Excreção
- **Farmacodinâmica**: Ocupação de receptores cerebrais e efeitos terapêuticos/colaterais
- **Visualizações**: Gráficos avançados e mapas de ocupação cerebral
- **Interface Interativa**: Dashboard web com Streamlit

---

## 🧬 Sobre a Quetiapina

### Informações Farmacológicas

- **Nome Químico**: 2-[2-(4-dibenzo[b,f][1,4]tiazepin-11-il-1-piperazinil)etoxi]etanol
- **Fórmula Molecular**: C₂₁H₂₅N₃O₂S
- **Peso Molecular**: 383.5 g/mol
- **Classe**: Antipsicótico atípico (segunda geração)

### Mecanismo de Ação

A Quetiapina atua como antagonista de múltiplos receptores:

| Receptor | Ki (nM) | Efeito Principal |
|----------|---------|------------------|
| 5-HT₂A (Serotonina) | 148 | Antipsicótico, melhora sintomas negativos |
| D₂ (Dopamina) | 329 | Antipsicótico primário |
| H₁ (Histamina) | 11 | Sedação, ganho de peso |
| α₁ (Adrenérgico) | 47 | Hipotensão ortostática |
| M₁ (Muscarínico) | 1200 | Efeitos anticolinérgicos |

### Parâmetros Farmacocinéticos

- **Biodisponibilidade Oral**: ~73%
- **Ligação Proteica**: ~83%
- **Volume de Distribuição**: ~10 L/kg
- **Clearance**: ~1.2 L/h/kg
- **Meia-vida de eliminação**: ~6-7 horas
- **Metabolismo**: Hepático (CYP3A4)
- **Excreção**: Renal (73%) + Fecal (20%)

---

## 🔬 Modelo Matemático

### 1. Modelo Farmacocinético Compartimental

O sistema é descrito por um modelo de **4 compartimentos**:

1. **TGI** (Trato Gastrointestinal) - Absorção
2. **Plasma** - Circulação central
3. **Cérebro** - Alvo terapêutico (SNC)
4. **Periférico** - Tecidos periféricos

#### Equações Diferenciais

```math
dA_TGI/dt = -k_abs · A_TGI + R(t)
```

```math
dA_plasma/dt = k_abs · F · A_TGI - (CL/V_d) · A_plasma - k_cb · A_plasma + k_ret_cb · A_cerebro - k_per · A_plasma + k_ret_per · A_periferico
```

```math
dA_cerebro/dt = k_cb · A_plasma - k_ret_cb · A_cerebro
```

```math
dA_periferico/dt = k_per · A_plasma - k_ret_per · A_periferico
```

**Onde:**
- `A_i` = Quantidade no compartimento i (mg)
- `k_abs` = Constante de absorção (1/h)
- `F` = Biodisponibilidade (0-1)
- `CL` = Clearance total (L/h)
- `V_d` = Volume de distribuição (L)
- `k_cb` = Constante de distribuição cérebro
- `R(t)` = Taxa de infusão (mg/h)

#### Parâmetros Ajustados por Peso

- `V_d = 10 L/kg × Peso_corporal`
- `CL = 1.2 L/h/kg × Peso_corporal`

### 2. Modelo Farmacodinâmico

#### Ocupação de Receptores (Equação de Hill)

```math
θ = [C] / (K_i + [C])
```

**Onde:**
- `θ` = Fração de ocupação (0-1)
- `[C]` = Concentração cerebral (nM)
- `K_i` = Constante de inibição (nM)

#### Score de Eficácia Terapêutica

Baseado em critérios clínicos:

- **D₂**: Ocupação ideal entre 60-80% (antipsicótico sem EPS)
- **5-HT₂A**: >80% para melhora de sintomas negativos
- **5-HT₁A**: >50% para efeito ansiolítico

```math
Eficácia = w_D2 · f(θ_D2) + w_5HT2A · f(θ_5HT2A) + w_5HT1A · f(θ_5HT1A) + w_H1 · f(θ_H1)
```

#### Efeitos Colaterais

- **EPS** (Sintomas Extrapiramidais): θ_D2 > 80%
- **Sedação**: proporcional a θ_H1
- **Ganho de Peso**: θ_H1 × 0.8
- **Hipotensão**: θ_α1 × 0.9
- **Anticolinérgicos**: θ_M1 × 0.7

---

## 🚀 Instalação e Uso

### Requisitos

- Python 3.8+
- Bibliotecas: NumPy, SciPy, Matplotlib, Streamlit

### Instalação

```bash
# Clonar ou baixar o projeto
cd SimuladorQuetiapina

# Instalar dependências
pip install -r requirements.txt
```

### Uso via Linha de Comando

```bash
# Dose única de 300 mg para paciente de 70 kg
python main.py --peso 70 --dose 300

# Regime de 5 doses de 200 mg a cada 12 horas
python main.py --peso 70 --dose 200 --multiplas --num-doses 5 --intervalo 12

# Dose única intravenosa
python main.py --peso 80 --dose 300 --via intravenosa
```

**Parâmetros disponíveis:**
- `--peso`: Peso corporal em kg (padrão: 70)
- `--dose`: Dose em mg (padrão: 300)
- `--via`: Via de administração - oral ou intravenosa (padrão: oral)
- `--multiplas`: Flag para doses múltiplas
- `--num-doses`: Número de doses (padrão: 5)
- `--intervalo`: Intervalo entre doses em horas (padrão: 12)

### Interface Web Interativa

```bash
# Iniciar dashboard Streamlit
streamlit run app.py
```

O navegador abrirá automaticamente em `http://localhost:8501`

---

## 📊 Funcionalidades

### Dashboard Interativo (Streamlit)

#### 1. Configuração de Parâmetros
- Peso corporal do paciente
- Dose do medicamento
- Via de administração
- Regime posológico (dose única ou múltipla)
- Tempo de simulação

#### 2. Visualizações

**Farmacocinética:**
- Perfil de concentração plasmática
- Distribuição nos compartimentos (plasma, cérebro, tecidos)
- Curva de absorção gastrointestinal
- Tabela de parâmetros PK (Cmax, Tmax, T½, AUC, CL, Vd)

**Farmacodinâmica:**
- Ocupação temporal de receptores
- Score de eficácia terapêutica
- Perfil de efeitos colaterais
- Estado de equilíbrio (steady-state)

**Mapa Cerebral:**
- Diagrama visual do cérebro
- Representação da ocupação de cada receptor
- Cores e tamanhos proporcionais à ocupação

#### 3. Relatório Completo
- Resumo de todos os parâmetros
- Recomendações de dose por indicação
- Informações farmacológicas da Quetiapina

### Script CLI (main.py)

- Execução rápida via terminal
- Geração automática de gráficos (PNG)
- Resultados formatados no console
- Análise de steady-state para doses múltiplas

---

## 📁 Estrutura do Projeto

```
SimuladorQuetiapina/
│
├── farmacocinetica.py      # Modelo PK (ADME)
├── farmacodinamica.py      # Modelo PD (receptores)
├── visualizacao.py         # Gráficos e visualizações
├── app.py                  # Dashboard Streamlit
├── main.py                 # Interface CLI
├── requirements.txt        # Dependências
└── README.md              # Documentação
```

---

## 🎯 Casos de Uso

### Indicações Terapêuticas

| Indicação | Dose Inicial | Dose de Manutenção | Dose Máxima |
|-----------|--------------|-------------------|-------------|
| **Esquizofrenia** | 50 mg/dia | 300-400 mg/dia | 800 mg/dia |
| **Mania Bipolar** | 100 mg/dia | 400-800 mg/dia | 800 mg/dia |
| **Depressão Bipolar** | 50 mg/dia | 300 mg/dia | 600 mg/dia |
| **Depressão Maior (adjuvante)** | 50 mg/dia | 150-300 mg/dia | 300 mg/dia |

### Ajustes Posológicos

- **Insuficiência hepática**: Reduzir dose em 25-50%
- **Idosos**: Iniciar com 25-50 mg/dia
- **Baixo peso (<50 kg)**: Reduzir dose em ~20%
- **Alto peso (>100 kg)**: Pode necessitar doses maiores

---

## ⚠️ Avisos Importantes

### Limitações do Modelo

1. **Simulação Teórica**: Baseado em parâmetros médios da população
2. **Variabilidade Individual**: Não considera polimorfismos genéticos (CYP3A4)
3. **Interações Medicamentosas**: Não modeladas
4. **Condições Patológicas**: Não ajusta para doenças hepáticas/renais
5. **Apenas Educacional**: Não substitui avaliação clínica

### Uso Responsável

⚠️ **Este simulador é para fins educacionais e de pesquisa.**

- Não use para prescrição médica
- Consulte sempre um profissional de saúde
- Decisões terapêuticas devem ser individualizadas
- Mantenha medicamentos fora do alcance de crianças

---

## 📚 Referências Científicas

1. **Farmacocinética da Quetiapina:**
   - DeVane CL, Nemeroff CB. Clinical pharmacokinetics of quetiapine. *Clin Pharmacokinet*. 2001;40(7):509-522.

2. **Ocupação de Receptores:**
   - Kapur S, et al. Relationship between dopamine D₂ occupancy, clinical response, and side effects. *Am J Psychiatry*. 2000;157(4):514-520.

3. **Farmacodinâmica:**
   - Riedel M, et al. Quetiapine in the treatment of schizophrenia and related disorders. *Neuropsychiatr Dis Treat*. 2007;3(2):219-235.

4. **Modelo Compartimental:**
   - Gabrielsson J, Weiner D. *Pharmacokinetic and Pharmacodynamic Data Analysis*. 5th ed. Swedish Pharmaceutical Press; 2016.

---

## 🔧 Desenvolvimento Futuro

### Melhorias Planejadas

- [ ] Modelo de metabolismo CYP3A4 com variantes genéticas
- [ ] Simulação de interações medicamentosas
- [ ] Modelo populacional (Monte Carlo)
- [ ] Integração com dados reais de pacientes
- [ ] Visualização 3D da molécula (RDKit)
- [ ] Predição de resposta terapêutica individualizada
- [ ] Exportação de relatórios PDF
- [ ] API REST para integração

---

## 📄 Licença

Este projeto é distribuído para fins educacionais e de pesquisa.

**Direitos Autorais © 2025 Luiz Tiago Wilcke**

---

## 👨‍💻 Autor

**Luiz Tiago Wilcke**

Simulador desenvolvido como ferramenta educacional para demonstração de princípios de farmacocinética e farmacodinâmica aplicados.

---

## 🙏 Agradecimentos

Agradecimentos especiais à comunidade científica de farmacologia clínica e aos desenvolvedores das bibliotecas de código aberto utilizadas neste projeto.

---

## 📞 Suporte

Para questões técnicas ou sugestões de melhoria, considere:
- Documentar issues detalhadamente
- Incluir parâmetros de entrada e saída esperada
- Anexar screenshots quando relevante

---

**Última atualização:** 2025-11-25
