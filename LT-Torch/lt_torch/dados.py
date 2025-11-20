\
# ============================================================
# lt_torch/dados.py
# Dataset + DataLoader simples
# ============================================================

from __future__ import annotations
import numpy as np
from typing import Iterator, List, Tuple, Optional, Callable, Any
from .tensor import Tensor

class Dataset:
    def __len__(self):
        raise NotImplementedError
    def __getitem__(self, idx):
        raise NotImplementedError

class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int=32, embaralhar: bool=True, drop_last: bool=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.embaralhar = embaralhar
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[Tuple[Any, Any]]:
        idxs = np.arange(len(self.dataset))
        if self.embaralhar:
            np.random.shuffle(idxs)
        for i in range(0, len(idxs), self.batch_size):
            lote = idxs[i:i+self.batch_size]
            if len(lote) < self.batch_size and self.drop_last:
                continue
            xs, ys = [], []
            for j in lote:
                x, y = self.dataset[j]
                xs.append(x); ys.append(y)
            yield Tensor(np.stack(xs)), np.array(ys)

    def __len__(self):
        n = len(self.dataset)//self.batch_size
        if not self.drop_last and len(self.dataset)%self.batch_size != 0:
            n += 1
        return n
