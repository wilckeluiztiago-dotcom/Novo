# 🌞 Simulador Fotovoltaico Modular

**Autor:** Luiz Tiago Wilcke (LT)

Simulador avançado de células fotovoltaicas baseado em física quântica e equações de dispositivos semicondutores.

## 📋 Descrição

Este simulador implementa um modelo matemático completo para células fotovoltaicas de silício, incluindo:

- **Nível Quântico do Material:**
  - Estrutura de bandas (via massa efetiva)
  - Densidade de estados Nc, Nv
  - Concentração intrínseca ni

- **Interação Luz-Matéria:**
  - Espectro solar aproximado por corpo negro (T_sol ~ 5778 K)
  - Fator geométrico Sol-Terra
  - Corrente fotogerada J_ph (limite de Shockley-Queisser)

- **Nível de Dispositivo/Circuito:**
  - Corrente de saturação J₀ (recombinação radiativa)
  - Equação de diodo com fator de idealidade n
  - Resistência série Rs e shunt Rsh
  - Curvas J-V, P-V, J_sc, V_oc, FF, eficiência

## 🗂️ Estrutura do Projeto

```
SimuladorFotovoltaico/
├── main.py                    # Script principal
├── requirements.txt           # Dependências
├── modules/
│   ├── constants.py          # Constantes físicas fundamentais
│   ├── quantum.py            # Parâmetros quânticos dos materiais
│   ├── solar.py              # Espectro solar e corrente fotogerada
│   ├── device.py             # Modelo de diodo e equações do dispositivo
│   ├── analysis.py           # Extração de parâmetros (Jsc, Voc, FF, η)
│   └── visualization.py      # Plotagem de gráficos
└── README.md                 # Este arquivo
```

## 🚀 Como Usar

### 1. Instalar Dependências

```bash
pip install -r requirements.txt
```

### 2. Executar o Simulador

```bash
python3 main.py
```

O programa irá:
1. Calcular a corrente fotogerada (J_ph)
2. Calcular a corrente de saturação radiativa (J₀)
3. Gerar a curva J-V usando o modelo de diodo
4. Extrair os parâmetros elétricos (J_sc, V_oc, P_max, FF, η)
5. Plotar as curvas J-V e P-V

## 📊 Resultados Típicos (Silício)

Para uma célula de silício a 300 K:

| Parâmetro | Valor |
|-----------|-------|
| J_ph | ~52.45 mA/cm² |
| J₀ | ~8.23×10⁻¹⁴ mA/cm² |
| J_sc | ~52.19 mA/cm² |
| V_oc | ~0.881 V |
| Eficiência (η) | ~7.9% |

## 🔧 Personalização

### Alterar Material

No arquivo `main.py`, você pode escolher entre diferentes materiais:

```python
from modules.quantum import SILICON, GAAS, PEROVSKITE

material = GAAS  # Ou SILICON, PEROVSKITE
```

### Ajustar Parâmetros do Diodo

```python
fator_idealidade = 1.0    # Fator de idealidade
resistencia_serie = 0.5   # Ω·m²
resistencia_shunt = 1e4   # Ω·m²
```

## 📚 Física Implementada

### Equação de Shockley-Queisser

$$J_{ph} = q \\int_{E_g}^{\\infty} \\Phi_{inc}(E) dE$$

### Modelo de Diodo

$$J(V) = J_{ph} - J_0 \\left[\\exp\\left(\\frac{q(V + JR_s)}{nk_BT}\\right) - 1\\right] - \\frac{V + JR_s}{R_{sh}}$$

### Fator de Preenchimento

$$FF = \\frac{V_{mp} \\times J_{mp}}{V_{oc} \\times J_{sc}}$$

### Eficiência

$$\\eta = \\frac{P_{max}}{P_{inc}} = \\frac{V_{mp} \\times J_{mp}}{1000 \\text{ W/m}^2}$$

## 📦 Dependências

- Python 3.7+
- NumPy
- Matplotlib
- SciPy

## 📄 Licença

Este projeto foi desenvolvido para fins educacionais e de pesquisa.

## 👤 Autor

**Luiz Tiago Wilcke (LT)**

---

*Simulador Fotovoltaico Modular - Física Quântica aplicada a Dispositivos Semicondutores*
