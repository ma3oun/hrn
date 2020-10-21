"""
Decoder module
"""

import torch
import torch.nn as nn
from hrn.routing.common import genActivation
from hrn.config_flags import *


class Decoder(nn.Module):
    def __init__(
        self,
        architecture: dict,
        embeddingSize: int,
        fullBasisSize: int,
        outputShape: torch.Size,
        device: torch.device = torch.device("cpu"),
        autoOutputSize: bool = True,
    ):
        super().__init__()
        self.name = "Decoder"
        self.device = device
        self.architecture = architecture  # type: dict
        self.embeddingSize = embeddingSize
        self.fullBasisSize = fullBasisSize
        self.outputShape = outputShape
        self.autoOutputSize = autoOutputSize

        denseParams = architecture["denseParams"]  # type: dict
        if self.autoOutputSize:
            lastParams = architecture["lastParams"]  # type: dict
            lastParams["out_features"] = None

        self.nLayers = len(denseParams)

        # input is composed of the concatenation of the summed
        # projections (embeddingSize), the projections coeffs and the unit codes
        # fullBasisSize = unitBasisSize * NUnits
        # The input size is therefore: embeddingSize + fullBasisSize + depth
        # lastOutputSize = 2*self.embeddingSize + self.fullBasisSize # + self.depth
        if USE_PROJ:
            lastOutputSize = 2 * self.embeddingSize
        else:
            lastOutputSize = self.embeddingSize

        for idx, params in enumerate(denseParams):
            currentParams = params
            linearLayer = nn.Linear(lastOutputSize, currentParams["out_features"])
            actType = currentParams["act"]["type"]
            actParams = currentParams["act"]["params"]
            actLayer = genActivation(actType, actParams)
            self.__setattr__(f"dense_{idx}", linearLayer)
            self.__setattr__(f"act_{idx}", actLayer)
            if "bn" in currentParams.keys():
                self.__setattr__(
                    f"bn_{idx}", nn.BatchNorm1d(currentParams["out_features"])
                )
            else:
                self.__setattr__(f"bn_{idx}", nn.Identity())
            if "drpt" in currentParams.keys():
                p = currentParams["drpt"]
                self.__setattr__(f"drpt_{idx}", nn.Dropout(p=p))
            else:
                self.__setattr__(f"drpt_{idx}", nn.Identity())
            lastOutputSize = currentParams["out_features"]

        if self.autoOutputSize:
            # remove batch dimension
            if len(self.outputShape) > 3:
                finalOutputSize = 1
                for s in self.outputShape[1:]:
                    finalOutputSize *= s
            else:
                finalOutputSize = self.outputShape.numel()
            self.lastLayer = nn.Linear(lastOutputSize, finalOutputSize)
            self.lastAct = genActivation(lastParams["type"], lastParams["params"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        try:
            for idx in range(self.nLayers):
                y = self.__getattr__(f"dense_{idx}")(y)
                y = self.__getattr__(f"bn_{idx}")(y)
                y = self.__getattr__(f"act_{idx}")(y)
                y = self.__getattr__(f"drpt_{idx}")(y)

            if self.autoOutputSize:
                y = self.lastLayer(y)
                y = self.lastAct(y)
                y = y.reshape(self.outputShape)
        except RuntimeError as e:
            print(
                f"Encountered runtime error, input data shape might help:"
                f"\n{x.shape}. Error: {e}"
            )  # for debug
            raise

        return y
