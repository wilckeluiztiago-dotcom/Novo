\
# ============================================================
# lt_torch/otimizadores.py
# Otimizadores SGD e Adam
# ============================================================

from __future__ import annotations
import numpy as np
from typing import Iterable, List
from .tensor import Tensor
from .modulo import Parametro

class Otimizador:
    def __init__(self, parametros: Iterable[Parametro], lr: float=1e-3):
        self.parametros: List[Parametro] = list(parametros)
        self.lr = lr

    def zero_grad(self):
        for p in self.parametros:
            p.zero_grad()

    def passo(self):
        raise NotImplementedError

class SGD(Otimizador):
    def __init__(self, parametros, lr=1e-2, momentum=0.0, nesterov: bool=False, peso_decay: float=0.0):
        super().__init__(parametros, lr)
        self.momentum = momentum
        self.nesterov = nesterov
        self.peso_decay = peso_decay
        self.v = [np.zeros_like(p.dados) for p in self.parametros]

    def passo(self):
        for i, p in enumerate(self.parametros):
            if p.grad is None: continue
            g = p.grad
            if self.peso_decay != 0.0:
                g = g + self.peso_decay*p.dados
            if self.momentum != 0.0:
                self.v[i] = self.momentum*self.v[i] + g
                g_eff = g + self.momentum*self.v[i] if self.nesterov else self.v[i]
            else:
                g_eff = g
            p.dados = p.dados - self.lr*g_eff

class Adam(Otimizador):
    def __init__(self, parametros, lr=1e-3, betas=(0.9,0.999), eps=1e-8, peso_decay: float=0.0):
        super().__init__(parametros, lr)
        self.b1, self.b2 = betas
        self.eps = eps
        self.peso_decay = peso_decay
        self.m = [np.zeros_like(p.dados) for p in self.parametros]
        self.v = [np.zeros_like(p.dados) for p in self.parametros]
        self.t = 0

    def passo(self):
        self.t += 1
        for i, p in enumerate(self.parametros):
            if p.grad is None: continue
            g = p.grad
            if self.peso_decay != 0.0:
                g = g + self.peso_decay*p.dados
            self.m[i] = self.b1*self.m[i] + (1-self.b1)*g
            self.v[i] = self.b2*self.v[i] + (1-self.b2)*(g*g)
            mhat = self.m[i] / (1 - self.b1**self.t)
            vhat = self.v[i] / (1 - self.b2**self.t)
            p.dados = p.dados - self.lr*mhat/(np.sqrt(vhat)+self.eps)
