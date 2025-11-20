\
# ============================================================
# lt_torch/tensor.py
# Núcleo Tensor + Autodiff (estilo PyTorch)
# - backend NumPy
# - autodiff por grafo dinâmico
# - operações vetorizadas
# ============================================================

from __future__ import annotations
import numpy as np
from contextlib import contextmanager
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union, List, Set

ArrayLike = Union[float, int, list, tuple, np.ndarray]

def _arr(x: ArrayLike, dtype=np.float32):
    return x if isinstance(x, np.ndarray) else np.array(x, dtype=dtype)

def _unbroadcast(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    \"\"\"Soma gradientes em eixos que foram broadcastados.\"\"\"
    if grad.shape == shape:
        return grad
    # remove dimensões extras à esquerda
    while len(grad.shape) > len(shape):
        grad = grad.sum(axis=0)
    # soma eixos onde shape tinha 1
    for i, (g, s) in enumerate(zip(grad.shape, shape)):
        if s == 1 and g != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad

class Tensor:
    \"\"\"Tensor com autodiferenciação.\"\"\"
    def __init__(self, dados: ArrayLike, requer_grad: bool=False, nome: str=\"\"):
        self.dados = _arr(dados)
        self.requer_grad = requer_grad
        self.grad: Optional[np.ndarray] = None
        self._anteriores: Set[Tensor] = set()
        self._op: str = \"\"
        self._backward: Callable[[], None] = lambda: None
        self.nome = nome

    # --------- utilidades ----------
    @property
    def shape(self): return self.dados.shape
    @property
    def ndim(self): return self.dados.ndim
    def numpy(self): return self.dados
    def item(self): return float(self.dados)

    def zero_grad(self):
        self.grad = None

    # --------- criação ----------
    @staticmethod
    def zeros(*shape, requer_grad=False):
        return Tensor(np.zeros(shape, dtype=np.float32), requer_grad=requer_grad)

    @staticmethod
    def ones(*shape, requer_grad=False):
        return Tensor(np.ones(shape, dtype=np.float32), requer_grad=requer_grad)

    @staticmethod
    def randn(*shape, requer_grad=False, escala=1.0):
        return Tensor(np.random.randn(*shape).astype(np.float32)*escala, requer_grad=requer_grad)

    @staticmethod
    def arange(*args, requer_grad=False):
        return Tensor(np.arange(*args, dtype=np.float32), requer_grad=requer_grad)

    # --------- grafo / backward ----------
    def backward(self, grad: Optional[np.ndarray]=None):
        if not self.requer_grad:
            return
        if grad is None:
            if self.dados.size != 1:
                raise RuntimeError("backward() requer grad explícito para tensores não-escalares.")
            grad = np.ones_like(self.dados, dtype=np.float32)

        # ordem topológica
        topo: List[Tensor] = []
        visitado: Set[Tensor] = set()
        def build(v: Tensor):
            if v not in visitado:
                visitado.add(v)
                for child in v._anteriores:
                    build(child)
                topo.append(v)
        build(self)

        self.grad = grad.astype(np.float32)
        for v in reversed(topo):
            v._backward()

    # --------- representação ----------
    def __repr__(self):
        return f"Tensor(dados={self.dados}, requer_grad={self.requer_grad}, grad_shape={None if self.grad is None else self.grad.shape})"

    # --------- operações base ----------
    def __add__(self, outro: ArrayLike):
        outro = outro if isinstance(outro, Tensor) else Tensor(outro)
        out = Tensor(self.dados + outro.dados, requer_grad=self.requer_grad or outro.requer_grad)
        out._anteriores = {self, outro}
        out._op = "add"
        def _backward():
            if self.requer_grad:
                g = _unbroadcast(out.grad, self.shape)
                self.grad = g if self.grad is None else self.grad + g
            if outro.requer_grad:
                g = _unbroadcast(out.grad, outro.shape)
                outro.grad = g if outro.grad is None else outro.grad + g
        out._backward = _backward
        return out
    __radd__ = __add__

    def __sub__(self, outro: ArrayLike):
        return self + (-outro)
    def __rsub__(self, outro: ArrayLike):
        return outro + (-self)

    def __neg__(self):
        out = Tensor(-self.dados, requer_grad=self.requer_grad)
        out._anteriores = {self}
        out._op = "neg"
        def _backward():
            if self.requer_grad:
                g = -out.grad
                self.grad = g if self.grad is None else self.grad + g
        out._backward = _backward
        return out

    def __mul__(self, outro: ArrayLike):
        outro = outro if isinstance(outro, Tensor) else Tensor(outro)
        out = Tensor(self.dados * outro.dados, requer_grad=self.requer_grad or outro.requer_grad)
        out._anteriores = {self, outro}
        out._op = "mul"
        def _backward():
            if self.requer_grad:
                g = _unbroadcast(out.grad * outro.dados, self.shape)
                self.grad = g if self.grad is None else self.grad + g
            if outro.requer_grad:
                g = _unbroadcast(out.grad * self.dados, outro.shape)
                outro.grad = g if outro.grad is None else outro.grad + g
        out._backward = _backward
        return out
    __rmul__ = __mul__

    def __truediv__(self, outro: ArrayLike):
        outro = outro if isinstance(outro, Tensor) else Tensor(outro)
        return self * outro.pow(-1.0)
    def __rtruediv__(self, outro: ArrayLike):
        outro = outro if isinstance(outro, Tensor) else Tensor(outro)
        return outro / self

    def pow(self, p: float):
        out = Tensor(self.dados ** p, requer_grad=self.requer_grad)
        out._anteriores = {self}
        out._op = "pow"
        def _backward():
            if self.requer_grad:
                g = out.grad * (p * (self.dados ** (p-1.0)))
                self.grad = g if self.grad is None else self.grad + g
        out._backward = _backward
        return out

    def matmul(self, outro: ArrayLike):
        outro = outro if isinstance(outro, Tensor) else Tensor(outro)
        out = Tensor(self.dados @ outro.dados, requer_grad=self.requer_grad or outro.requer_grad)
        out._anteriores = {self, outro}
        out._op = "matmul"
        def _backward():
            if self.requer_grad:
                g = out.grad @ np.swapaxes(outro.dados, -1, -2)
                self.grad = g if self.grad is None else self.grad + g
            if outro.requer_grad:
                g = np.swapaxes(self.dados, -1, -2) @ out.grad
                outro.grad = g if outro.grad is None else outro.grad + g
        out._backward = _backward
        return out
    def __matmul__(self, outro): return self.matmul(outro)

    # --------- reduções ----------
    def sum(self, eixo=None, manter_dim=False):
        out = Tensor(self.dados.sum(axis=eixo, keepdims=manter_dim), requer_grad=self.requer_grad)
        out._anteriores = {self}
        out._op = "sum"
        def _backward():
            if self.requer_grad:
                g = out.grad
                if eixo is None:
                    g = np.broadcast_to(g, self.shape)
                else:
                    if not manter_dim:
                        g = np.expand_dims(g, axis=eixo)
                    g = np.broadcast_to(g, self.shape)
                self.grad = g if self.grad is None else self.grad + g
        out._backward = _backward
        return out

    def mean(self, eixo=None, manter_dim=False):
        div = self.dados.size if eixo is None else np.prod(np.array(self.dados.shape)[eixo])
        return self.sum(eixo=eixo, manter_dim=manter_dim) / div

    # --------- view/transpose ----------
    def reshape(self, *shape):
        out = Tensor(self.dados.reshape(*shape), requer_grad=self.requer_grad)
        out._anteriores = {self}
        out._op = "reshape"
        def _backward():
            if self.requer_grad:
                g = out.grad.reshape(self.shape)
                self.grad = g if self.grad is None else self.grad + g
        out._backward = _backward
        return out

    def transpose(self, *eixos):
        if len(eixos)==0:
            eixos = tuple(reversed(range(self.ndim)))
        out = Tensor(self.dados.transpose(*eixos), requer_grad=self.requer_grad)
        out._anteriores = {self}
        out._op = "transpose"
        inv = np.argsort(eixos)
        def _backward():
            if self.requer_grad:
                g = out.grad.transpose(*inv)
                self.grad = g if self.grad is None else self.grad + g
        out._backward = _backward
        return out
    T = property(lambda self: self.transpose())

    # --------- indexação ----------
    def __getitem__(self, idx):
        out = Tensor(self.dados[idx], requer_grad=self.requer_grad)
        out._anteriores = {self}
        out._op = "slice"
        def _backward():
            if self.requer_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.dados, dtype=np.float32)
                np.add.at(self.grad, idx, out.grad)
        out._backward = _backward
        return out

    # --------- funções não-lineares ----------
    def relu(self):
        out = Tensor(np.maximum(0, self.dados), requer_grad=self.requer_grad)
        out._anteriores = {self}
        out._op = "relu"
        mask = (self.dados > 0).astype(np.float32)
        def _backward():
            if self.requer_grad:
                g = out.grad * mask
                self.grad = g if self.grad is None else self.grad + g
        out._backward = _backward
        return out

    def sigmoid(self):
        sig = 1/(1+np.exp(-self.dados))
        out = Tensor(sig, requer_grad=self.requer_grad)
        out._anteriores = {self}
        out._op = "sigmoid"
        def _backward():
            if self.requer_grad:
                g = out.grad * sig*(1-sig)
                self.grad = g if self.grad is None else self.grad + g
        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.dados)
        out = Tensor(t, requer_grad=self.requer_grad)
        out._anteriores = {self}
        out._op = "tanh"
        def _backward():
            if self.requer_grad:
                g = out.grad * (1 - t**2)
                self.grad = g if self.grad is None else self.grad + g
        out._backward = _backward
        return out

    def exp(self):
        e = np.exp(self.dados)
        out = Tensor(e, requer_grad=self.requer_grad)
        out._anteriores = {self}
        out._op = "exp"
        def _backward():
            if self.requer_grad:
                g = out.grad * e
                self.grad = g if self.grad is None else self.grad + g
        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.dados + 1e-12), requer_grad=self.requer_grad)
        out._anteriores = {self}
        out._op = "log"
        def _backward():
            if self.requer_grad:
                g = out.grad / (self.dados + 1e-12)
                self.grad = g if self.grad is None else self.grad + g
        out._backward = _backward
        return out

    def softmax(self, eixo=-1):
        x = self.dados - self.dados.max(axis=eixo, keepdims=True)
        ex = np.exp(x)
        sm = ex / ex.sum(axis=eixo, keepdims=True)
        out = Tensor(sm, requer_grad=self.requer_grad)
        out._anteriores = {self}
        out._op = "softmax"
        def _backward():
            if self.requer_grad:
                g = out.grad
                # jacobiano * g, forma vetorizada
                s = sm
                dot = (g * s).sum(axis=eixo, keepdims=True)
                gx = s * (g - dot)
                self.grad = gx if self.grad is None else self.grad + gx
        out._backward = _backward
        return out

    # --------- conveniências ----------
    def detach(self):
        return Tensor(self.dados.copy(), requer_grad=False)

    def __len__(self): return len(self.dados)

@contextmanager
def no_grad():
    \"\"\"Contexto para desabilitar gradientes em blocos.\"\"\"
    antigo = Tensor.__init__
    def novo_init(self, dados, requer_grad=False, nome=""):
        antigo(self, dados, requer_grad=False, nome=nome)
    Tensor.__init__ = novo_init
    try:
        yield
    finally:
        Tensor.__init__ = antigo
