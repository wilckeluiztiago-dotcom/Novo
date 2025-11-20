\
# ============================================================
# lt_torch/perdas.py
# Funções de perda
# ============================================================

from __future__ import annotations
import numpy as np
from .tensor import Tensor

def mse(pred: Tensor, alvo: Tensor) -> Tensor:
    \"\"\"Erro quadrático médio.\"\"\"
    diff = pred - alvo
    return (diff*diff).mean()

def nll(log_probs: Tensor, alvo_indices: np.ndarray) -> Tensor:
    \"\"\"Negative log-likelihood. alvo_indices: (N,)\"\"\"
    N = log_probs.shape[0]
    idx = (np.arange(N), alvo_indices)
    return -(log_probs[idx]).mean()

def entropia_cruzada(logits: Tensor, alvo_indices: np.ndarray) -> Tensor:
    \"\"\"Cross-entropy para classificação multi-classe.\"\"\"
    # softmax estável + log
    probs = logits.softmax(eixo=1)
    log_probs = probs.log()
    return nll(log_probs, alvo_indices)
