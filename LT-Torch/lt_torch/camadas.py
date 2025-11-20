\
# ============================================================
# lt_torch/camadas.py
# Camadas e ativações
# ============================================================

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple
from .tensor import Tensor
from .modulo import Modulo, Parametro

# -------- ativações simples como módulos --------
class ReLU(Modulo):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()

class Sigmoid(Modulo):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()

class Tanh(Modulo):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()

class Softmax(Modulo):
    def __init__(self, eixo=-1):
        super().__init__()
        self.eixo = eixo
    def forward(self, x: Tensor) -> Tensor:
        return x.softmax(eixo=self.eixo)

class Flatten(Modulo):
    def forward(self, x: Tensor) -> Tensor:
        return x.reshape(x.shape[0], -1)

# -------- linear --------
class Linear(Modulo):
    def __init__(self, entradas: int, saidas: int, bias: bool=True):
        super().__init__()
        # inicialização Xavier/Glorot
        limite = np.sqrt(6/(entradas+saidas))
        self.pesos = Parametro(np.random.uniform(-limite, limite, (entradas, saidas)).astype(np.float32), nome="pesos")
        self.usar_bias = bias
        if bias:
            self.bias = Parametro(np.zeros((saidas,), dtype=np.float32), nome="bias")

    def forward(self, x: Tensor) -> Tensor:
        y = x.matmul(self.pesos)
        if self.usar_bias:
            y = y + self.bias
        return y

# -------- dropout --------
class Dropout(Modulo):
    def __init__(self, p: float=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.treinando or self.p == 0.0:
            return x
        mask = (np.random.rand(*x.shape) > self.p).astype(np.float32) / (1.0 - self.p)
        return x * Tensor(mask)

# -------- batchnorm --------
class BatchNorm1d(Modulo):
    def __init__(self, num_features: int, eps: float=1e-5, momentum: float=0.1):
        super().__init__()
        self.gamma = Parametro(np.ones((num_features,), dtype=np.float32), nome="gamma")
        self.beta  = Parametro(np.zeros((num_features,), dtype=np.float32), nome="beta")
        self.eps = eps
        self.momentum = momentum
        self.media_exec = np.zeros((num_features,), dtype=np.float32)
        self.var_exec   = np.ones((num_features,), dtype=np.float32)

    def forward(self, x: Tensor) -> Tensor:
        if self.treinando:
            media = x.dados.mean(axis=0)
            var   = x.dados.var(axis=0)
            self.media_exec = (1-self.momentum)*self.media_exec + self.momentum*media
            self.var_exec   = (1-self.momentum)*self.var_exec   + self.momentum*var
        else:
            media, var = self.media_exec, self.var_exec

        x_norm = (x - media) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta

# -------- conv2d (ingênua, im2col) --------
def _im2col(x, kH, kW, stride=1, padding=0):
    N, C, H, W = x.shape
    H_p, W_p = H + 2*padding, W + 2*padding
    x_p = np.pad(x, ((0,0),(0,0),(padding,padding),(padding,padding)))
    out_h = (H_p - kH)//stride + 1
    out_w = (W_p - kW)//stride + 1
    col = np.zeros((N, C, kH, kW, out_h, out_w), dtype=x.dtype)
    for y in range(kH):
        y_max = y + stride*out_h
        for z in range(kW):
            z_max = z + stride*out_w
            col[:, :, y, z, :, :] = x_p[:, :, y:y_max:stride, z:z_max:stride]
    col = col.transpose(0,4,5,1,2,3).reshape(N*out_h*out_w, -1)
    return col, out_h, out_w

def _col2im(col, x_shape, kH, kW, stride=1, padding=0, out_h=None, out_w=None):
    N, C, H, W = x_shape
    H_p, W_p = H + 2*padding, W + 2*padding
    if out_h is None:
        out_h = (H_p - kH)//stride + 1
        out_w = (W_p - kW)//stride + 1
    col = col.reshape(N, out_h, out_w, C, kH, kW).transpose(0,3,4,5,1,2)
    img = np.zeros((N, C, H_p, W_p), dtype=col.dtype)
    for y in range(kH):
        y_max = y + stride*out_h
        for z in range(kW):
            z_max = z + stride*out_w
            img[:, :, y:y_max:stride, z:z_max:stride] += col[:, :, y, z, :, :]
    return img[:, :, padding:H+padding, padding:W+padding]

class Conv2D(Modulo):
    def __init__(self, canais_in: int, canais_out: int, kernel: Tuple[int,int]=(3,3), stride: int=1, padding: int=1, bias: bool=True):
        super().__init__()
        kH, kW = kernel
        escala = np.sqrt(2/(canais_in*kH*kW))
        self.pesos = Parametro(np.random.randn(canais_out, canais_in, kH, kW).astype(np.float32)*escala, nome="pesos")
        self.usar_bias = bias
        if bias:
            self.bias = Parametro(np.zeros((canais_out,), dtype=np.float32), nome="bias")
        self.kernel = kernel
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        kH, kW = self.kernel
        col, out_h, out_w = _im2col(x.dados, kH, kW, self.stride, self.padding)
        W_col = self.pesos.dados.reshape(self.pesos.shape[0], -1).T  # (C*kH*kW, Cout)
        saida = col @ W_col
        if self.usar_bias:
            saida += self.bias.dados
        saida = saida.reshape(x.shape[0], out_h, out_w, -1).transpose(0,3,1,2)
        out = Tensor(saida, requer_grad=x.requer_grad or self.pesos.requer_grad)

        out._anteriores = {x, self.pesos}
        if self.usar_bias: out._anteriores.add(self.bias)
        out._op = "conv2d"

        def _backward():
            grad_out = out.grad.transpose(0,2,3,1).reshape(-1, self.pesos.shape[0])  # (N*out_h*out_w, Cout)

            if self.pesos.requer_grad:
                dW = col.T @ grad_out  # (C*kH*kW, Cout)
                dW = dW.T.reshape(self.pesos.shape)
                self.pesos.grad = dW if self.pesos.grad is None else self.pesos.grad + dW

            if self.usar_bias and self.bias.requer_grad:
                db = grad_out.sum(axis=0)
                self.bias.grad = db if self.bias.grad is None else self.bias.grad + db

            if x.requer_grad:
                W_col_T = self.pesos.dados.reshape(self.pesos.shape[0], -1)  # (Cout, C*kH*kW)
                dcol = grad_out @ W_col_T  # (N*out_h*out_w, C*kH*kW)
                dx = _col2im(dcol, x.shape, kH, kW, self.stride, self.padding, out_h, out_w)
                x.grad = dx if x.grad is None else x.grad + dx

        out._backward = _backward
        return out

# -------- maxpool2d --------
class MaxPool2D(Modulo):
    def __init__(self, kernel: Tuple[int,int]=(2,2), stride: Optional[int]=None):
        super().__init__()
        self.kernel = kernel
        self.stride = stride if stride is not None else kernel[0]

    def forward(self, x: Tensor) -> Tensor:
        kH, kW = self.kernel
        col, out_h, out_w = _im2col(x.dados, kH, kW, self.stride, 0)
        col = col.reshape(-1, kH*kW)
        idx_max = col.argmax(axis=1)
        saida = col[np.arange(len(col)), idx_max]
        saida = saida.reshape(x.shape[0], out_h, out_w, x.shape[1]).transpose(0,3,1,2)

        out = Tensor(saida, requer_grad=x.requer_grad)
        out._anteriores = {x}
        out._op = "maxpool2d"

        def _backward():
            if not x.requer_grad: return
            dcol = np.zeros_like(col, dtype=np.float32)
            dcol[np.arange(len(col)), idx_max] = out.grad.transpose(0,2,3,1).reshape(-1)
            dcol = dcol.reshape(x.shape[0]*out_h*out_w, -1)
            dx = _col2im(dcol, x.shape, kH, kW, self.stride, 0, out_h, out_w)
            x.grad = dx if x.grad is None else x.grad + dx

        out._backward = _backward
        return out
