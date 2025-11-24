# Xadrez Avançado 2.0 ♟️

Uma implementação robusta e modular de um motor de Xadrez em Python, apresentando uma Inteligência Artificial avançada e interface gráfica limpa.

## 🚀 Novidades da Versão 2.0

- **Arquitetura Modular**: Código totalmente refatorado em pacotes (`motor`, `interface`, `utils`).
- **IA Aprimorada**:
    - **PVS (Principal Variation Search)**: Otimização sobre o Alpha-Beta tradicional.
    - **Iterative Deepening**: Busca progressiva para melhor gerenciamento de tempo.
    - **Move Ordering**: Heurística MVV-LVA para podar a árvore de busca mais cedo.
    - **Tabela de Transposição**: Cache de posições usando Zobrist Hashing.
- **Interface Gráfica**: Painel informativo com avaliação da IA em tempo real.

## 🛠️ Instalação

Necessário apenas Python 3.8+ e Pygame.

```bash
pip install pygame
```

## 🎮 Como Jogar

Execute o arquivo principal:

```bash
python XadrezAvancado/main.py
```

### Controles
- **Clique**: Selecionar e mover peças.
- **F**: Inverter lado (Jogar como Pretas/Brancas).
- **R**: Reiniciar partida.
- **ESC**: Sair.

## 🧠 Estrutura do Código

```
XadrezAvancado/
├── config.py           # Constantes e Tabelas PST
├── main.py             # Ponto de entrada
├── utils/              # Tipos e Hashing
├── motor/
│   ├── tabuleiro.py    # Regras, Geração de Movimentos
│   ├── avaliacao.py    # Função de Avaliação Estática
│   └── ia.py           # Motor de Busca (PVS/Negamax)
└── interface/
    └── gui.py          # Renderização Pygame
```

## 🤖 Sobre a IA

A IA utiliza uma busca **Negamax** com **Poda Alpha-Beta**. Para eficiência, emprega **Principal Variation Search (PVS)**, assumindo que o primeiro movimento (ordenado) é provavelmente o melhor, realizando buscas com janela nula (Null Window) nos subsequentes. A **Tabela de Transposição** evita re-calcular posições idênticas alcançadas por ordens diferentes de movimentos.
