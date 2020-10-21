"""
Modules and network utilities
"""

import torch
from typing import Union
from enum import Enum
from torch.nn import Identity
import torch.nn.modules.activation as Activations


class Direction(Enum):
    Forward = 1
    Backward = -1

    def __str__(self):
        if self.name == "Forward":
            return "Fwd"
        else:
            return "Bwd"


def genActivation(actType: Union[str, None], params: dict = None):
    if actType is None:
        layer = Identity()
    else:
        if params is None:
            layer = getattr(Activations, actType)()
        else:
            layer = getattr(Activations, actType)(**params)
    return layer


def genUnitBaseName(code: Union[int, str]) -> str:
    if isinstance(code, int):
        unitBaseName = "{:03d}".format(code)
    else:
        unitBaseName = code
    return unitBaseName


def genSubUnitBaseName(code: Union[int, str], direction: Direction) -> str:
    unitBaseName = genUnitBaseName(code)
    baseName = "{}_{}".format(direction, unitBaseName)
    return baseName


def normalize(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """
    This "normalization" absorbs large amplitudes in x at the cost of greater
    variance
    See "Feature Hashing for Large scale Multi-task learning"
    Args:
      x: (torch.Tensor) Input tensor

    Returns: Normalized tensor

    """

    d = (x != 0).type(torch.float32).sum(dim, True)
    u = x / torch.sqrt(d)

    return u
