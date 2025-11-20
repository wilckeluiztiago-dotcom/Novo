\
# ============================================================
# examples/treino_mlp_sintetico.py
# Demonstração de uso LT-Torch
# ============================================================

import numpy as np
from lt_torch import Tensor, Sequential, Linear, ReLU, Dropout, entropia_cruzada, Adam
from lt_torch.utils import fixar_seed, acuracia

fixar_seed(0)

# dados sintéticos: 3 classes em 2D
N = 600
X = np.random.randn(N, 2).astype(np.float32)
y = (X[:,0] + 0.5*X[:,1] > 0).astype(int)
y += (X[:,0] - 0.3*X[:,1] > 1).astype(int)  # vira 0/1/2
y = np.clip(y, 0, 2)

# modelo MLP
modelo = Sequential(
    Linear(2, 64),
    ReLU(),
    Dropout(0.1),
    Linear(64, 64),
    ReLU(),
    Linear(64, 3)
)

otim = Adam(modelo.parametros(), lr=1e-2)

# treino
for epoca in range(200):
    logits = modelo(Tensor(X))
    perda = entropia_cruzada(logits, y)
    modelo.zero_grad()
    perda.backward()
    otim.passo()

    if (epoca+1) % 25 == 0:
        pred = logits.softmax(1).dados.argmax(1)
        acc = acuracia(pred, y)
        print(f"Época {epoca+1:03d} | perda={perda.item():.4f} | acc={acc:.3f}")

print("Treino finalizado.")
