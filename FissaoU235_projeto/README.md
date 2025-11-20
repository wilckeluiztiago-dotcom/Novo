# Simulador Didático de Fissão do Urânio-235 (Toy Model)

**Autor:** Luiz Tiago Wilcke  

Este projeto apresenta uma **simulação educacional** da fissão do U-235 usando um modelo Monte Carlo simplificado, além de uma animação em `pygame` mostrando:
1. Nêutrons se movendo em um meio com núcleos U-235.
2. Captura de nêutron por um núcleo.
3. Evento de fissão.
4. Emissão de novos nêutrons (reação em cadeia toy).

> **Aviso de segurança**  
> Isso é um **toy model didático** para fins de aprendizado.  
> Não contém parâmetros reais nem qualquer instrução para projeto de reatores ou armas.

---

## Como rodar

### Dependências
```bash
python -m pip install numpy pygame matplotlib
```

### Executar
```bash
python FissaoU235.py
```

Se o `pygame` não estiver instalado, ainda é possível rodar a simulação numérica desligando a animação:
```bash
python FissaoU235.py --sem-animacao
```

---

## O que o modelo faz (visão conceitual)

- O domínio é 2D e contém núcleos de U-235 distribuídos aleatoriamente.
- Cada nêutron anda com velocidade fixa e direção que pode espalhar aleatoriamente.
- Quando um nêutron chega perto de um núcleo, há uma chance de captura.
- Após captura, há alta chance de fissão, liberando energia e `k` novos nêutrons.
- Isso gera um processo estocástico de ramificação (reação em cadeia toy).

---

## Saídas

Durante a animação, o painel mostra:
- Neutrons ativos
- Núcleos restantes
- Fissões totais
- Energia acumulada (unidade arbitrária)
- Sparkline com histórico de nêutrons

---

## Estrutura

- **`FissaoU235.py`**  
  Código único contendo:
  - Simulador numérico
  - Visualizador `pygame`
  - README em string

---

## Licença

Uso livre para estudos e portfólio acadêmico, com atribuição ao autor.
