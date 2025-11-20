# LT-Torch (mini PyTorch em português)

Uma biblioteca **do zero** de redes neurais com **autodiff** estilo PyTorch,
usando **NumPy** como backend.  
Objetivo: ser didática, enxuta, mas com recursos avançados para portfólio.

> Autor: **Luiz Tiago Wilcke (LT)**

---

## Recursos

- `Tensor` com grafo dinâmico e `backward()` automático
- Operações vetorizadas com broadcasting + suporte a gradiente
- Módulos estilo `nn.Module` (`Modulo`, `Parametro`, `Sequential`)
- Camadas:
  - `Linear`, `Conv2D` (im2col), `MaxPool2D`, `BatchNorm1d`
  - Ativações: `ReLU`, `Sigmoid`, `Tanh`, `Softmax`
  - `Dropout`, `Flatten`
- Perdas:
  - `mse`, `entropia_cruzada`, `nll`
- Otimizadores:
  - `SGD` (momentum, nesterov, weight decay)
  - `Adam` (weight decay)
- `Dataset` + `DataLoader` simples

---

## Instalação

Clone/baixe este repositório e instale em modo editável:

```bash
pip install -e .
```

Ou simplesmente adicione a pasta ao seu `PYTHONPATH`.

---

## Exemplo rápido (MLP)

```python
import numpy as np
from lt_torch import Tensor, Sequential, Linear, ReLU, entropia_cruzada, Adam

# dados
X = np.random.randn(256, 10).astype(np.float32)
y = np.random.randint(0, 3, size=(256,))

# modelo
modelo = Sequential(
    Linear(10, 64), ReLU(),
    Linear(64, 3)
)

otim = Adam(modelo.parametros(), lr=1e-2)

for epoca in range(100):
    logits = modelo(Tensor(X))
    perda = entropia_cruzada(logits, y)
    modelo.zero_grad()
    perda.backward()
    otim.passo()
```

---

## Estrutura

```
lt_torch/
  __init__.py
  tensor.py         # Tensor + autodiff
  modulo.py         # Sistema de módulos/parametros
  camadas.py        # Camadas/ativações avançadas
  perdas.py         # Funções de perda
  otimizadores.py   # SGD / Adam
  dados.py          # Dataset / DataLoader
  utils.py          # utilidades
examples/
  treino_mlp_sintetico.py
```

---

## Observações

- Esta biblioteca não usa GPU e não implementa todos os recursos do PyTorch.
- A ideia é ter uma base avançada o suficiente para estudos, TCC e portfólio.

Se você quiser, posso:
- adicionar `RNN/LSTM/Transformer`,
- salvar/recuperar pesos (`state_dict`),
- exportar gráficos, etc.
