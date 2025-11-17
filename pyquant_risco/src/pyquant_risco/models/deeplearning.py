from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..logging_utils import get_logger

logger = get_logger()

try:
    import torch
    from torch import nn
except ImportError:  # torch opcional
    torch = None
    nn = None


@dataclass
class ConfigRedeLSTM:
    tamanho_janela: int = 20
    tamanho_oculto: int = 32
    camadas: int = 1
    epocas: int = 30
    taxa_aprendizado: float = 1e-3
    batch_size: int = 64
    usar_gpu: bool = False


class RedeLSTM(nn.Module):  # type: ignore[misc]
    def __init__(self, config: ConfigRedeLSTM) -> None:
        super().__init__()
        self.config = config
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=config.tamanho_oculto,
            num_layers=config.camadas,
            batch_first=True,
        )
        self.fc = nn.Linear(config.tamanho_oculto, 1)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        saida, _ = self.lstm(x)
        out = self.fc(saida[:, -1, :])
        return out


def criar_dataloader_retornos(
    retornos: np.ndarray,
    tamanho_janela: int,
    batch_size: int,
) -> "torch.utils.data.DataLoader":
    assert torch is not None

    X, y = [], []
    for i in range(len(retornos) - tamanho_janela):
        X.append(retornos[i : i + tamanho_janela])
        y.append(retornos[i + tamanho_janela])

    X_arr = np.array(X).reshape(-1, tamanho_janela, 1)
    y_arr = np.array(y).reshape(-1, 1)

    X_tensor = torch.tensor(X_arr, dtype=torch.float32)
    y_tensor = torch.tensor(y_arr, dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader


def treinar_lstm_retornos(
    retornos: np.ndarray,
    config: ConfigRedeLSTM,
) -> Tuple[Optional[RedeLSTM], Optional[float]]:
    if torch is None:
        logger.warning("PyTorch não está instalado. Parte de deep learning será pulada.")
        return None, None

    device = torch.device("cuda" if config.usar_gpu and torch.cuda.is_available() else "cpu")
    logger.info("Treinando LSTM em device: %s", device)

    loader = criar_dataloader_retornos(retornos, config.tamanho_janela, config.batch_size)
    modelo = RedeLSTM(config).to(device)

    criterio = nn.MSELoss()
    otimizador = torch.optim.Adam(modelo.parameters(), lr=config.taxa_aprendizado)

    modelo.train()
    for epoca in range(1, config.epocas + 1):
        perdas = []
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            otimizador.zero_grad()
            previsao = modelo(X_batch)
            perda = criterio(previsao, y_batch)
            perda.backward()
            otimizador.step()
            perdas.append(perda.item())

        logger.info("Época %d/%d - perda média: %.6f", epoca, config.epocas, float(np.mean(perdas)))

    return modelo, float(np.mean(perdas))


def prever_proximo_retorno(
    modelo: RedeLSTM,
    retornos_recentes: np.ndarray,
    config: ConfigRedeLSTM,
) -> float:
    assert torch is not None
    modelo.eval()

    if len(retornos_recentes) < config.tamanho_janela:
        raise ValueError("Poucos pontos para prever; aumente o histórico.")

    janela = retornos_recentes[-config.tamanho_janela :]
    X = janela.reshape(1, config.tamanho_janela, 1)
    X_tensor = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        previsao = modelo(X_tensor).cpu().numpy().ravel()[0]

    return float(previsao)
