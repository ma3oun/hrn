"""
Routing unit wrapper
"""

from typing import Union
import torch
import torch.nn as nn
import logging
import numpy as np
from enum import Enum
from queue import SimpleQueue
from torch.nn.functional import normalize
from hrn.routing.common import Direction, genUnitBaseName
from hrn.routing.codecs import Codec
from hrn.hash.fhashing import generateHasher, Hash1d
from hrn.config_flags import *

logger = logging.getLogger("hrn.unit")


class UnitStatus(Enum):
    empty = -1
    initiated = 0
    full = 1

    def isEmpty(self):
        return self.value == UnitStatus.empty.value

    def isFull(self):
        return self.value == UnitStatus.full.value


class Unit(nn.Module):
    @classmethod
    def basisToTensor(cls, basis: list) -> torch.Tensor:
        return torch.stack(basis, -1).squeeze(dim=1)

    @classmethod
    def tensorTobasis(
        cls, basisTensor: torch.Tensor, removeZeroVectors: bool = False
    ) -> list:
        vectorList = []
        for v in list(torch.unbind(basisTensor, 1)):
            if not removeZeroVectors or torch.norm(v) > 0:
                # each vector is of shape (embeddingSize,1)
                vectorList.append(v.unsqueeze(-1))

        return vectorList

    def __init__(
        self,
        code: Union[int, str],
        architecture: dict,
        basisSize: int,
        embeddingSize: int,
        dataChannels: int,
        hashing: Union[int, Hash1d],
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        unitName = "U_" + genUnitBaseName(code)
        self.status = UnitStatus.empty
        self.name = unitName
        self.device = device
        self.code = code

        self.architecture = architecture
        self.expansionThr = architecture["expansionThr"]
        self.enableBasisUpdate = architecture["enableBasisUpdate"]
        self.gradUpdate = architecture["gradUpdate"]
        self.basisSize = basisSize
        self.embeddingSize = embeddingSize
        self.dataChannels = dataChannels
        coderOutputShape = (1, dataChannels, -1)
        self.coder = Codec(
            self.code, Direction.Forward, coderOutputShape, architecture["coder"]
        )

        if isinstance(hashing, Hash1d):
            self.hasher = hashing
        else:
            self.hasher = generateHasher(hashing, self.embeddingSize, self.device)

        self.currentBasis = nn.Parameter(
            torch.zeros((self.embeddingSize, self.basisSize), dtype=torch.float32),
            False,
        )

        self.basis = []
        counterInit = architecture["counterInit"]
        counterAgeFactor = architecture["counterAgeFactor"]
        assert counterInit > 0
        assert counterAgeFactor > 1
        self.counterInit = counterInit
        self.counterInitTensor = nn.Parameter(
            torch.zeros((1,), dtype=torch.float32), False
        )
        self.counterAgeFactor = counterAgeFactor
        self.updateCounters = [self.counterInit] * self.basisSize
        self.updateCountersTensor = nn.Parameter(
            torch.zeros(self.basisSize, dtype=torch.float32), False
        )
        self.lastResidus = SimpleQueue()
        self.lastProjections = SimpleQueue()
        self.lastProjCoeffs = SimpleQueue()

        self.gradResidus = []

        self.usage = 0
        if self.gradUpdate:
            self.register_backward_hook(bwHook)
        logger.debug(f"Created {unitName}")

    def synthesisView(self) -> tuple:
        """
        Print the status of the Unit
        Returns: (str) Unit view

        """
        if self.status == UnitStatus.empty:
            status = f"\033[31m"  # red
        elif self.status == UnitStatus.initiated:
            status = f"\033[33m"  # yellow
        elif self.status == UnitStatus.full:
            status = f"\033[32m"  # green
        else:
            status = None
        s = f"\033[37m{self.name}: {status}{len(self.basis)}/{self.basisSize}"

        return s, self.status.value

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        super(Unit, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

        # TODO: see if register_buffer does the trick instead of using this hack

        self.basis = self.tensorTobasis(self.currentBasis, True)
        # logger.debug(f'{self.name} loaded external basis: {self.currentBasis}')
        logger.debug(f"{self.name} loaded external basis")
        self.counterInit = float(self.counterInitTensor.cpu().numpy())
        self.updateCounters = self.updateCountersTensor.cpu().numpy().tolist()
        logger.debug(f"{self.name} age counter: {self.counterInit}")
        if len(self.basis) == self.basisSize:
            self.status = UnitStatus.full
            logger.debug(f"{self.name} is already full.")
        elif len(self.basis) > 0:
            self.status = UnitStatus.initiated
            logger.debug(f"{self.name} initiated.")
        else:
            self.status = UnitStatus.empty
            logger.debug(f"{self.name} is empty.")
        return

    @property
    def currentBasisSize(self) -> int:
        return len(self.basis)

    @property
    def fullBasis(self) -> list:
        emptyVector = torch.zeros(
            (self.embeddingSize, 1), dtype=torch.float32, device=self.device
        )
        if self.status.isEmpty():
            _fullBasis = [emptyVector] * self.basisSize
        else:
            _fullBasis = self.basis[:]
            if not self.status.isFull():
                _fullBasis.extend([emptyVector] * (self.basisSize - len(self.basis)))
        return _fullBasis

    def getProjection(self, z: torch.Tensor, forcedMode: bool) -> tuple:
        """
        This is called only when the unit is the best unit.
        Args:
          z: (torch.Tensor) Hashed and normalized data tensor
          forcedMode: (bool) Forced routing flag. Avoids storing data.
        Returns: (tuple) Projection [1,embeddingSize],
                 Projection coefficients [1,basisSize]

        """
        R = self.basisToTensor(self.fullBasis)
        if not self.status.isEmpty():
            # logger.debug(f'{self.name} basis:\n {R} ({R.shape})')
            # logger.debug(f'z to project shape: {z.shape}')

            projCoeffs = z @ R
            proj = projCoeffs @ R.transpose(-1, -2)
            residu = z - proj
        else:
            proj = z  # initialization for empty basis
            projCoeffs = torch.ones(
                (1, self.basisSize), dtype=torch.float32, device=self.device
            )
            residu = torch.zeros_like(z)
        if not forcedMode and self.training:
            # logger.debug(f'{self.name} putting p,r,coeffs...')
            self.lastResidus.put(residu)
            self.lastProjections.put(proj)
            self.lastProjCoeffs.put(projCoeffs)
        return projCoeffs, proj

    def getGradResidu(self) -> torch.Tensor:
        """
        Compute residus average for use in gradient update. This clears the
        gradient residus buffer.

        Returns: (torch.Tensor) Average residu

        """
        nResidus = len(self.gradResidus)
        if nResidus > 0:
            gradResidus = torch.cat(self.gradResidus).reshape((nResidus, 1, -1))
            gradResidu = gradResidus.mean(dim=0)
            self.gradResidus = []
        else:
            gradResidu = torch.ones(
                (1, self.embeddingSize), dtype=torch.float32, device=self.device
            )
        return gradResidu

    def forward(
        self, x: torch.Tensor, expansionFlag: bool = True, forcedMode: bool = False
    ) -> torch.Tensor:
        """
        PyTorch forward pass
        Args:
          x: (torch.Tensor) Data tensor
          expansionFlag: (bool) Enable basis expansion. Can be disabled for
                                  inference
          forcedMode: (bool) Indicated if routing is being forced.
        Returns:

        """
        if not forcedMode and self.training:
            self.updateBasis(expansionFlag)

        # channels = x.chunk(x.shape[1]//self.dataChannels,dim=1)
        # out = [self.coder(channel) for channel in channels]
        # codedData = torch.cat(out,dim=1)

        channels = torch.chunk(x, x.shape[1] // self.dataChannels, dim=1)
        out = [self.coder(channel) for channel in channels]
        codedData = torch.stack(out, dim=1).sum(dim=1)

        self.usage += 1
        return codedData

    def _basisExpansion(
        self, expansionFlag: bool, projection: torch.Tensor, residu: torch.Tensor
    ):
        """
        Basis expansion uses incoming hashed samples to expand the unit's basis
        during initialization.

        Args:
          expansionFlag: (bool) Expansion flag. Expansion can be disabled to
                          avoid changing the unit's basis during inference.
          projection: (torch.Tensor) Latest projection
          residu: (torch.Tensor) Latest residu

        Returns:

        """
        if expansionFlag and self.status != UnitStatus.full:
            if self.status.isEmpty():
                self.basis.append(projection.reshape((-1, 1)))  # column vector
                if self.currentBasisSize < self.basisSize:
                    self.status = UnitStatus.initiated
                    logger.debug(f"{self.name} initiated.")
                else:
                    self.status = UnitStatus.full
                    logger.debug(f"{self.name} has been initiated and is full.")

            else:
                if torch.norm(projection) < self.expansionThr:
                    # expand only if the projection has little energy
                    newBasisElement = normalize(residu)
                    self.basis.append(newBasisElement.reshape((-1, 1)))
                    logger.debug(f"{self.name} has expanded to {self.currentBasisSize}")
                    if len(self.basis) == self.basisSize:
                        self.status = UnitStatus.full
                        logger.debug(f"{self.name} is full")
            self.currentBasis.data = self.basisToTensor(self.fullBasis)
        return

    def updateBasis(self, expansionFlag: bool):
        """
        Basis update using new hashed examples. Their projection residu is used
        to update the unit's basis. It is also used to update the gradResidus
        buffer to manage gradient update (if enabled).

        Args:
          expansionFlag: (bool) This is passed to _basisExpansion. It is unused
                          in this method.

        Returns:

        """
        # logger.debug(f'{self.name} getting p,r,coeffs...')

        # cloning and detaching avoids "in-place" bugs
        p = self.lastProjections.get().clone().detach().squeeze(dim=1)
        r = self.lastResidus.get().clone().detach().squeeze(dim=1)
        coeffs = self.lastProjCoeffs.get().clone().detach()

        if self.gradUpdate:
            if torch.nonzero(r, as_tuple=False).nelement() != 0:
                self.gradResidus.append(r)
            else:
                # avoid zero residus
                self.gradResidus.append(torch.ones_like(r))

        # Basis expansion
        self._basisExpansion(expansionFlag, p, r)

        # logger.debug(f'New basis:\n {self.currentBasis} '
        #              f'({self.currentBasis.shape})')

        # avoid basis update when the basis contains a single vector
        if self.enableBasisUpdate and self.currentBasisSize > 1:
            # Basis counters update
            # only consider used basis vectors
            relevantCoeffs = coeffs.squeeze()[: self.currentBasisSize]
            smallestCoeffIdx = torch.argmin(torch.abs(relevantCoeffs)).cpu().numpy()
            # logger.debug(f'Worst index for {self.name}: {smallestCoeffIdx}')
            self.updateCounters[smallestCoeffIdx] -= 1
            if self.updateCounters[smallestCoeffIdx] == 0:
                # logger.debug(f'Basis update for {self.name} ({smallestCoeffIdx})')
                self.counterInit = self.counterAgeFactor * self.counterInit
                self.counterInitTensor.data = torch.Tensor([self.counterInit]).to(
                    self.device
                )
                self.updateCounters[smallestCoeffIdx] = int(np.floor(self.counterInit))

                z = p + r
                if self.currentBasisSize > 1:
                    reducedBasis = (
                        self.basis[0:smallestCoeffIdx]
                        + self.basis[smallestCoeffIdx + 1 :]
                    )

                    reducedBasisTensor = self.basisToTensor(reducedBasis)

                    projCoeffs = z @ reducedBasisTensor
                    proj = projCoeffs @ reducedBasisTensor.transpose(-1, -2)
                    newBasisVector = normalize(z - proj)
                    # basis contains vectors of shape [embeddingSize,1]
                    self.basis[smallestCoeffIdx] = newBasisVector.transpose(-1, -2)
                else:
                    logger.warning("Updating single element basis!")
                    # basis contains only a single element of shape [embeddingSize,1]
                    self.basis[0] = z.transpose(-1, -2)
                self.currentBasis.data = self.basisToTensor(self.fullBasis)
            self.updateCountersTensor.data = torch.Tensor(self.updateCounters).to(
                self.device
            )
        return


def bwHook(
    module: Unit, grad_input: torch.Tensor, grad_output: torch.Tensor
) -> Union[torch.Tensor, None]:
    """
    This hook runs when loss.backward() is called. It updates the gradient
    using the last residu's norm.
    Args:
      module: (Unit) Unit
      grad_input: (torch.Tensor) Unused
      grad_output: (torch.Tensor) Unused

    Returns: (torch.Tensor) Unused

    """
    resNorm = torch.norm(module.getGradResidu()).reshape(1)
    # logger.debug(f'{module.name} - ResNorm: {resNorm} ({resNorm.shape}')
    # Take the minimum of residu norm and 1. If residu is low, no model update
    # is deemed necessary. If residu is two big, just use the gradient norm.
    # This should help with lifelong learning
    gradCoeff = torch.min(torch.cat([resNorm, torch.ones_like(resNorm)]))
    for p in module.parameters(recurse=True):
        if not p.grad is None:
            p.grad = gradCoeff * p.grad
    return
