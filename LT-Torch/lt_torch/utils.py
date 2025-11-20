\
# ============================================================
# lt_torch/utils.py
# Utilidades: seed, acurácia etc.
# ============================================================
import numpy as np
import random

def fixar_seed(seed: int=42):
    random.seed(seed)
    np.random.seed(seed)

def acuracia(pred_indices, alvo_indices):
    pred_indices = np.array(pred_indices)
    alvo_indices = np.array(alvo_indices)
    return (pred_indices == alvo_indices).mean()
