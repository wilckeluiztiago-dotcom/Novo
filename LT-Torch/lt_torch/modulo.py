\
# ============================================================
# lt_torch/modulo.py
# Sistema de módulos/parametros (estilo nn.Module)
# ============================================================

from __future__ import annotations
import numpy as np
from typing import Dict, Iterable, List, Optional, Tuple, Any, Union
from .tensor import Tensor

class Parametro(Tensor):
    \"\"\"Tensor treinável por padrão.\"\"\"
    def __init__(self, dados, nome=\"\"):
        super().__init__(dados, requer_grad=True, nome=nome)

class Modulo:
    def __init__(self):
        self._parametros: Dict[str, Parametro] = {}
        self._submodulos: Dict[str, Modulo] = {}
        self.treinando: bool = True

    def parametros(self) -> List[Parametro]:
        ps = list(self._parametros.values())
        for m in self._submodulos.values():
            ps.extend(m.parametros())
        return ps

    def zero_grad(self):
        for p in self.parametros():
            p.zero_grad()

    def treino(self, modo: bool=True):
        self.treinando = modo
        for m in self._submodulos.values():
            m.treino(modo)
        return self

    def eval(self):
        return self.treino(False)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    # sobrescrever
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    # registro automático
    def __setattr__(self, chave, valor):
        if isinstance(valor, Parametro):
            self.__dict__.setdefault("_parametros", {})[chave] = valor
        elif isinstance(valor, Modulo):
            self.__dict__.setdefault("_submodulos", {})[chave] = valor
        super().__setattr__(chave, valor)

class Sequential(Modulo):
    def __init__(self, *camadas: Modulo):
        super().__init__()
        self.camadas = list(camadas)
        for i, c in enumerate(self.camadas):
            setattr(self, f"camada_{i}", c)

    def forward(self, x: Tensor) -> Tensor:
        for c in self.camadas:
            x = c(x)
        return x
