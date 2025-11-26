# 🦠 Visualização Avançada de Bactérias - Montador de Genoma

## Novo Recurso Implementado

Sistema de visualização detalhada de bactérias mostrando:
- **Forma real da bactéria** (bacilo, coco, espiral)
- **DNA interno visível** (dupla hélice ou circular)
- **Genes mapeados** (ORFs nas fitas + e -)
- **Informações completas** (Gram, patogenicidade, aplicações)

## Características

### 1. Formas Bacterianas Suportadas

#### 🔴 Bacilo (Bastão)
- Formato alongado
- DNA em dupla hélice interna
- Parede celular Gram+/- com cores diferentes
- Flagelo opcional
- Exemplos: E. coli, Bacillus subtilis

#### 🔵 Coco (Esférico)
- Formato circular
- DNA circular (plasmídeo)
- Genes ao redor do círculo
- Exemplos: Staphylococcus aureus, Streptococcus

#### 🌀 Espiral
- Formato helicoidal
- DNA ao longo da espiral
- Parede celular colorida
- Exemplos: Helicobacter pylori, Treponema pallidum

### 2. Informações Exibidas

**Painel Lateral Completo:**
- Nome científico
- Classificação Gram (positiva/negativa)
- Forma celular
- Tamanho do genoma montado
- Conteúdo GC%
- Número de contigs
- ORFs detectados
- Patogenicidade
- Aplicações práticas

### 3. Elementos Visuais

- **Parede Celular**: Roxa (Gram+) ou Rosa (Gram-)
- **Membrana**: Azul
- **Citoplasma**: Azul claro
- **DNA**: Vermelho (dupla hélice ou circular)
- **Genes fita +**: Verde
- **Genes fita -**: Laranja
- **Flagelo**: Cinza (quando presente)

## Como Usar

### Teste Rápido

```bash
cd MontagemGenomaBacteriano
python teste_visualizacao_bacteria.py
```

Gera 3 visualizações:
- `bacteria_bacilo.png` - E. coli
- `bacteria_coco.png` - Staphylococcus
- `bacteria_espiral.png` - Helicobacter

### Integração com GUI

A visualização será automaticamente gerada após a montagem e identificação.

### Uso Programático

```python
from visualizacao.bacteria_detalhada import VisualizadorBacteriaAvancado
from identificacao.banco_expandido import BACTERIAS_EXPANDIDO

# Preparar informações
bacteria_info = {
    'nome': 'Escherichia coli',
    'forma': 'bacilo',
    'gram': 'negativa',
    'tamanho_genoma': 5000000,
    'gc': 50.5,
    'patogenicidade': 'Algumas cepas patogênicas',
    'aplicacoes': 'Biotecnologia'
}

# Criar visualização
visualizador = VisualizadorBacteriaAvancado()
visualizador.criar_visualizacao(bacteria_info, contigs, orfs, "saida.png")
```

## Banco de Dados Atualizado

Todas as 61 bactérias agora incluem:
- ✅ Campo `"gram"`: "positiva" ou "negativa"
- ✅ Campo `"forma"`: "bacilo", "coco", "espiral", etc.
- ✅ Informações completas de patogenicidade
- ✅ Aplicações práticas

## Diferenças Gram

### Gram-Positiva (Roxa)
- Parede celular **espessa** (8px)
- Cor: Violeta
- Exemplos: Bacillus, Staphylococcus, Streptococcus

### Gram-Negativa (Rosa)
- Parede celular **fina** (4px)
- Cor: Rosa
- Exemplos: E. coli, Salmonella, Pseudomonas

## Arquivos Criados

```
MontagemGenomaBacteriano/
├── visualizacao/
│   └── bacteria_detalhada.py      ✅ Visualizador avançado
├── identificacao/
│   └── banco_expandido.py         ✅ Atualizado com Gram
└── teste_visualizacao_bacteria.py ✅ Script de teste
```

## Exemplo de Saída

A visualização mostra:

```
┌─────────────────────────────────────────────────────────┐
│  🦠 Escherichia coli                                    │
├──────────────────────────┬──────────────────────────────┤
│                          │ INFORMAÇÕES BACTERIANAS      │
│     ╔════════════╗       │                              │
│     ║  Parede    ║       │ Nome: Escherichia coli       │
│     ║  Celular   ║       │                              │
│     ║            ║       │ CLASSIFICAÇÃO:               │
│     ║  ~~~~~~~~  ║       │ Gram: Negativa               │
│     ║   DNA      ║       │ Forma: Bacilo                │
│     ║  ~~~~~~~~  ║       │                              │
│     ║            ║       │ CARACTERÍSTICAS:             │
│     ║  Citoplasma║       │ Tamanho: 5,000,000 bp        │
│     ╚════════════╝       │ GC: 50.5%                    │
│                          │ Contigs: 196                 │
│  Genes: ● ● ● ●         │ ORFs: 45                     │
│                          │                              │
│                          │ PATOGENICIDADE:              │
│                          │ Algumas cepas patogênicas    │
│                          │                              │
│                          │ APLICAÇÕES:                  │
│                          │ Biotecnologia, produção...   │
└──────────────────────────┴──────────────────────────────┘
```

## Próximos Passos

1. Execute o teste: `python teste_visualizacao_bacteria.py`
2. Veja as imagens geradas
3. Integre na GUI (já preparado)

---

**Autor:** Luiz Tiago Wilcke  
**Projeto:** Montador de Genoma Bacteriano
