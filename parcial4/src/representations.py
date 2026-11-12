"""Representaciones visuales de un vector tabular x en [0, 1]^N como imagen RGB.

Ambas se calculan en PyTorch por lotes, para generar las imágenes al vuelo en GPU.

1. ``feature_matrix``: las N variables se ordenan por clustering de correlación y se
   colocan en una cuadrícula s x s (s = ceil(sqrt(N))); cada celda se colorea con el
   mapa de color *viridis*. Variables correlacionadas quedan vecinas, lo que da a la
   CNN patrones locales.
2. ``gaf``: Gramian Angular Field. Cada valor se codifica como ángulo
   phi = arccos(2x - 1) y se forma una imagen N x N con tres canales:
   R = GASF  cos(phi_i + phi_j),  G = GADF  sin(phi_i - phi_j),  B = x_i * x_j.
   Captura las interacciones por pares entre variables.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

IMG_SIZE = 32
REPRESENTATIONS = {"feature_matrix": "Matriz de características", "gaf": "Gramian Angular Field"}


def _viridis_lut() -> torch.Tensor:
    import matplotlib

    return torch.tensor(matplotlib.colormaps["viridis"](np.linspace(0, 1, 256))[:, :3], dtype=torch.float32)


_LUT = None


def feature_matrix(x: torch.Tensor, order: np.ndarray, size: int = IMG_SIZE) -> torch.Tensor:
    global _LUT
    if _LUT is None:
        _LUT = _viridis_lut()
    b, n = x.shape
    side = math.ceil(math.sqrt(n))
    grid = torch.zeros(b, side * side, device=x.device)
    grid[:, :n] = x[:, torch.as_tensor(order, device=x.device)]
    idx = (grid.clamp(0, 1) * 255).round().long()
    rgb = _LUT.to(x.device)[idx]  # B, side*side, 3
    rgb[:, n:] = 0.0  # celdas de relleno en negro
    img = rgb.view(b, side, side, 3).permute(0, 3, 1, 2)
    return F.interpolate(img, size=(size, size), mode="nearest")


def gaf(x: torch.Tensor, order: np.ndarray, size: int = IMG_SIZE) -> torch.Tensor:
    x = x[:, torch.as_tensor(order, device=x.device)].clamp(0, 1)
    phi = torch.arccos((2 * x - 1).clamp(-1, 1))
    gasf = torch.cos(phi[:, :, None] + phi[:, None, :])
    gadf = torch.sin(phi[:, :, None] - phi[:, None, :])
    prod = x[:, :, None] * x[:, None, :]
    img = torch.stack([(gasf + 1) / 2, (gadf + 1) / 2, prod], dim=1)
    return F.interpolate(img, size=(size, size), mode="nearest")


def make_images(x: torch.Tensor, kind: str, order: np.ndarray, size: int = IMG_SIZE) -> torch.Tensor:
    """Devuelve imágenes B x 3 x size x size con valores en [0, 1]."""
    if kind == "feature_matrix":
        return feature_matrix(x, order, size)
    if kind == "gaf":
        return gaf(x, order, size)
    raise ValueError(kind)
