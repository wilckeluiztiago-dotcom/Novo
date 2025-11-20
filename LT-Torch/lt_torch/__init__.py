"""LT-Torch: mini biblioteca de redes neurais com autodiff (NumPy backend).
Autor: Luiz Tiago Wilcke (LT)
"""

from .tensor import Tensor, no_grad
from .modulo import Modulo, Sequential, Parametro
from .camadas import Linear, ReLU, Sigmoid, Tanh, Softmax, Dropout, Flatten, Conv2D, MaxPool2D, BatchNorm1d
from .perdas import mse, entropia_cruzada, nll
from .otimizadores import SGD, Adam
from .dados import Dataset, DataLoader
