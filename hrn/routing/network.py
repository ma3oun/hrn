"""
Network handling
"""

from __future__ import annotations

import logging
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from torch.nn.functional import normalize
from hrn.hash.fhashing import generateHasher
from hrn.routing.unit import Unit, UnitStatus
from hrn.config_flags import *

logger = logging.getLogger("hrn.network")


def getResiduLoss(r: torch.Tensor) -> tuple:
    # r is of shape (batchSize, embeddingSize, maxDepth)
    # When maxDepth is not reached, residu is zero

    # logger.debug(f'Residu: {r} ({r.shape})')

    # sum the L1 norm of each residu
    l1_sum = torch.sum(torch.norm(r, p=1, dim=-2), -1, True)

    # yields shape [batchSize,maxDepth]
    l2_norms = torch.norm(r, dim=-2)

    # logger.debug(f'l2_norms shape: {l2_norms.shape}')

    nonzeroMask = l2_norms > 0

    # obtain indices of elements to extract
    nonzeroIdx = torch.sum(nonzeroMask, dim=-1, keepdim=True) - 1

    # logger.debug(f'NZ mask:\n {nonzeroMask} ({nonzeroMask.shape})')
    # logger.debug(f'NZ idx:\n {nonzeroIdx} ({nonzeroIdx.shape})')

    # extract l2 norms of residus along dim=1 from l2_norms at extraction indices
    l2_residus = l2_norms.gather(1, nonzeroIdx.type(torch.long))

    # logger.debug(f'L1 residu: {l1_sum}\nL2 residus: {l2_residus}')

    # output shapes are [batchSize, 1]

    return l1_sum, l2_residus


class HashRoutedNetwork(nn.Module):
    def __init__(
        self,
        name: str,
        architecture: dict,
        device: torch.device = torch.device("cpu"),
        enableUnitExpansion: bool = True,
        reshapeOutput: bool = False,
    ):
        super().__init__()
        self.name = name
        self.device = device
        self.architecture = architecture
        self.reshapeOutput = reshapeOutput
        self.maxDepth = self.architecture["routing"]["depth"]
        self.depthThr = self.architecture["routing"]["depthThr"]
        self.energyThr = self.architecture["routing"]["energyThr"]
        self.batchSize = self.architecture["training"]["batchSize"]
        self.inputShape = self.architecture["training"]["inputShape"]
        self.unitBasisSize = architecture["routing"]["basisSize"]
        self.embeddingSize = architecture["routing"]["embeddingSize"]
        self.primeOffset = architecture["routing"]["primeOffset"]
        self.perUnitHashing = architecture["routing"]["perUnitHashing"]
        self.dataChannels = self.inputShape[1]
        self.unitExpansionFlag = enableUnitExpansion
        self.allUnitsInitiated = False
        logger.debug(f"Found a total of {self.NUnits} units")
        try:
            assert self.NUnits >= self.maxDepth
        except AssertionError:
            logger.exception(
                f"Insufficient units ({self.NUnits}) "
                f"for maximum depth of {self.maxDepth}"
            )
        self.hasher = generateHasher(self.primeOffset, self.embeddingSize, self.device)

        unitGroups = self.architecture["units"]
        lastCode = 0
        unitsNames = []
        for groupIdx, unitGroup in enumerate(unitGroups):
            groupSize = unitGroup["N"]
            logger.debug(f"Found a group of {groupSize} units")
            for unitIdx in range(lastCode, lastCode + groupSize):
                if self.perUnitHashing:
                    # prime offset to use for the unit
                    unitHash = self.primeOffset + unitIdx + 1
                else:
                    unitHash = self.hasher
                newUnit = Unit(
                    unitIdx,
                    unitGroup,
                    self.unitBasisSize,
                    self.embeddingSize,
                    self.dataChannels,
                    unitHash,
                    self.device,
                )
                self.__setattr__(newUnit.name, newUnit)
                unitsNames.append(newUnit.name)
            lastCode += groupSize
        self.unitsNames = unitsNames

        self.nCalls = 0

    def synthesisView(self) -> tuple:
        """
        Print the status of the network
        Returns: (str) Network view

        """
        netStats = {}
        total = 0
        s = f"\n\n========== {self.name} ==========\n\n|| "
        rowLength = int(np.sqrt(self.NUnits))
        for unitIdx, unit in enumerate(self.iterateUnits(True)):
            total += len(unit.basis)
            unitSynthView, unitStatus = unit.synthesisView()
            netStats.update({unit.name + "_status": unitStatus})
            netStats.update({unit.name + "_age": unit.counterInit})
            s += f"{unitSynthView}"
            if (unitIdx + 1) % rowLength == 0:
                s += "\033[37m ||\n"
                if unitIdx != self.NUnits - 1:
                    s += "\033[37m|| "
            else:
                s += "\t"
        maxTotal = self.NUnits * self.unitBasisSize
        if total == maxTotal:
            color = "\033[32m"
        elif total == 0:
            color = "\033[31m"
        else:
            color = "\033[33m"
        s += f"\nNetwork: {color}{total}/{maxTotal}\033[37m\n"
        return s, netStats

    def resetNetworkUsage(self):
        """
        Set each unit usage to 0
        Returns:

        """
        for unit in self.iterateUnits(True):
            unit.usage = 0
        return

    def getNetworkUsage(self, reset: bool = True) -> tuple:
        """
        Get the units usage rate
        Args:
          reset: (bool) Reset unit stats

        Returns: (str) Network usage rate

        """
        total = 0
        stats = []
        statsDict = {}
        s = f"\n\n========== {self.name} ==========\n\n|| "
        rowLength = int(np.sqrt(self.NUnits))

        for unitIdx, unit in enumerate(self.iterateUnits(True)):
            currentUsage = deepcopy(unit.usage)
            stats.append((unit.name + "_usage", currentUsage))
            total += currentUsage
            if reset:
                unit.usage = 0
        averageLoad = 100 * total / self.NUnits
        for unitIdx, unitData in enumerate(stats):
            name, stat = unitData
            currentRate = 100 * stat / total
            statsDict[name] = currentRate
            if currentRate * total >= 1.2 * averageLoad:
                color = "\033[31m"
            elif currentRate * total < 0.8 * averageLoad:
                color = "\033[33m"
            else:
                color = "\033[37m"

            s += f"{color}{name}: {currentRate:02.0f}%\033[37m"
            if (unitIdx + 1) % rowLength == 0:
                s += " ||\n"
                if unitIdx != self.NUnits - 1:
                    s += "|| "
            else:
                s += "\t"
        return s, statsDict

    def getUnit(self, code: int) -> Unit:
        """
        Get unit by index
        Args:
          code: (int) Unit index

        Returns:

        """
        unitName = self.unitsNames[code]
        return self.__getattr__(unitName)

    def getUnitProjection(self, code: int, u: torch.Tensor, forcedMode: bool) -> tuple:
        """
        Get the projection on a unit's basis, ONLY after that unit was selected
        as the best. This affects the unit's internal state.
        Args:
          code: Unit's code
          u: Data tensor
          forcedMode: Indicate if routing is forced
        Returns: Projection

        """
        unit = self.getUnit(code)
        return unit.getProjection(u, forcedMode)

    @property
    def NUnits(self) -> int:
        """
        Get the total number of units
        Returns:

        """
        units = self.architecture["units"]
        nUnits = 0
        for unit in units:
            nUnits += unit["N"]

        return nUnits

    def iterateUnits(self, includeEmptyBasis=False):
        """
        Units iterator. Can include empty units.
        Args:
          includeEmptyBasis: (bool) Includes units with empty basis.

        Returns: (iter) Unit iterator

        """
        for unitIdx in range(self.NUnits):
            unit = self.getUnit(unitIdx)
            if (
                includeEmptyBasis
                or unit.status == UnitStatus.initiated
                or unit.status == UnitStatus.full
            ):
                yield unit

    @property
    def currentBasis(self) -> torch.Tensor:
        basis = []
        for unit in self.iterateUnits(True):
            # logger.debug(f'{unit.name} basis size: {len(unit.basis)}')
            basis.extend(unit.fullBasis)
        tensorBasis = Unit.basisToTensor(basis)
        return tensorBasis

    def _maskLastCodes(self, lastCodes: torch.Tensor) -> torch.Tensor:
        # lastCodes is of shape (batchSize, something)
        # obtain vectorIndices
        codes = torch.cat(
            [self.unitBasisSize * lastCodes + idx for idx in range(self.unitBasisSize)],
            -1,
        )

        # logger.debug(f'last_codes: {codes} ({codes.shape})')

        basisMask = torch.ones(
            (1, self.embeddingSize, self.unitBasisSize * self.NUnits),
            dtype=torch.float32,
            device=self.device,
        )

        # invalidate indices of visited codes
        basisMask[:, :, codes.type(torch.long)] = 0.0

        return basisMask

    def _getNonEmptyBasisMask(self) -> torch.Tensor:
        nonEmptyUnitCodes = [
            unit.code
            for unit in self.iterateUnits(includeEmptyBasis=False)
            if not unit.status.isEmpty()
        ]
        if len(nonEmptyUnitCodes) == self.NUnits:
            self.allUnitsInitiated = True
            # basisMask = torch.ones((self.batchSize,
            #                         self.embeddingSize,
            #                         self.unitBasisSize*self.NUnits),
            #                        dtype=torch.float32,device=self.device)

            basisMask = torch.ones(
                (1, self.embeddingSize, self.unitBasisSize * self.NUnits),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            vectorIndices = [
                idx
                for unitCode in nonEmptyUnitCodes
                for idx in range(
                    unitCode * self.unitBasisSize, (unitCode + 1) * self.unitBasisSize
                )
            ]
            vectorIndices = torch.from_numpy(np.array(vectorIndices, dtype=np.int))

            basisElementaryMask = torch.zeros(
                (1, self.embeddingSize, self.unitBasisSize * self.NUnits),
                dtype=torch.float32,
                device=self.device,
            )

            basisElementaryMask[:, :, vectorIndices] = 1.0

            basisMask = basisElementaryMask
            # basisMask = torch.repeat_interleave(basisElementaryMask,self.batchSize,0)
        return basisMask

    def _getRandCodes(self, lastCodes: torch.Tensor = None) -> torch.Tensor:

        emptyUnitCodes = [
            unit.code
            for unit in self.iterateUnits(includeEmptyBasis=True)
            if unit.status.isEmpty()
        ]

        # remove used codes and rand choice from remaining units
        if not lastCodes is None and len(emptyUnitCodes) > 0:
            emptyUnitCodesTensor = torch.LongTensor(emptyUnitCodes).to(self.device)
            # logger.debug('Getting random unit from unused units')
            # allCodesTensor = torch.arange(self.NUnits, device=self.device)
            # randMask = torch.ones_like(allCodesTensor, device=self.device).type(
            #  torch.bool)
            randMask = torch.ones_like(emptyUnitCodesTensor, device=self.device).type(
                torch.bool
            )
            for previousCode in lastCodes.squeeze(dim=0):
                # randMask = randMask & (allCodesTensor != previousCode.to(
                #  self.device)).type(torch.bool)
                randMask = randMask & (
                    emptyUnitCodesTensor != previousCode.to(self.device)
                ).type(torch.bool)

            # unusedCodes = allCodesTensor[randMask]
            unusedCodes = emptyUnitCodesTensor[randMask]
            if unusedCodes.numel() > 0:
                # an equivalent of numpy.choice would have been nice !
                nUnusedCodes = len(unusedCodes)
                randUnusedIdx = torch.randint(nUnusedCodes, (1, 1), device="cpu")
                codesTensor = unusedCodes[randUnusedIdx].cpu()
            else:
                codesTensor = torch.randint(self.NUnits, (1, 1), device="cpu")
        else:
            # logger.debug('Cannot compute projections yet. Next code will be '
            #              'fully random')
            codesTensor = torch.randint(self.NUnits, (1, 1), device="cpu")
        return codesTensor

    def getBestProjections(
        self, u: torch.Tensor, lastCodes: torch.Tensor = None
    ) -> torch.Tensor:

        # note that batching is not necessary for this function
        # u is a batch of shape (batchSize, embeddingSize, 1)
        # lastCodes is of shape (batchSize, something)
        # return a tensor of batched codes of shape (batchSize, 1)

        fullBasis = self.currentBasis
        # logger.debug(f'Full basis shape: {fullBasis.shape}')
        nonEmptyMask = self._getNonEmptyBasisMask()
        # logger.debug(f'Non-empty mask shape {nonEmptyMask.shape}')
        if not lastCodes is None:
            codesMask = self._maskLastCodes(lastCodes)
            basis = nonEmptyMask * codesMask * fullBasis
            # logger.debug(f'nonEmptyMask:\n {nonEmptyMask} ({nonEmptyMask.shape})\n'
            #              f'Last codes mask:\n {codesMask} ({codesMask.shape})\n'
            #              f'basis:\n {basis} ({basis.shape})')

        else:
            basis = nonEmptyMask * fullBasis

        if torch.nonzero(basis, as_tuple=False).numel() == 0:
            # basis is empty
            newCodesTensor = self._getRandCodes(lastCodes).cpu()
        else:
            # projections are of shape:
            # (batchSize, fullBasisSize * NUnits, embeddingSize)
            # get a list of tensors of shape (batchSize,fullBasisSize, 1)
            elementaryProjections = list(
                (torch.transpose(basis, -1, -2) @ u.unsqueeze(-1)).chunk(
                    self.NUnits, -2
                )
            )

            # projection energies list
            energiesList = [
                torch.norm(proj, dim=1).reshape((-1, 1))
                for proj in elementaryProjections
            ]

            # logger.debug(f'Energies list: {energiesList}')

            # regroup energies into tensor. Shape is (batchSize,NUnits)
            energiesTensor = torch.cat(energiesList, -1)
            # logger.debug(f'Energies tensor shape: {energiesTensor.shape}')

            maxEnergy = torch.max(energiesTensor.squeeze())

            if not self.allUnitsInitiated and maxEnergy.cpu().item() < self.energyThr:
                # there are still empty units
                # logger.debug('Projection energy too low: using remaining empty units')
                newCodesTensor = self._getRandCodes(lastCodes).cpu()
            else:
                # get new codes batch. Shape is (batchSize,1)
                newCodesTensor = torch.argmax(
                    energiesTensor, dim=-1, keepdim=True
                ).cpu()

        return newCodesTensor

    def _shiftProjCoeffs(self, pCoeffs: torch.Tensor, unitIdx: int) -> torch.Tensor:
        """
        Add missing zeros to projection coefficients returned from a unit. The
        coefficients have to be shifted according to the unit index in the
        network, in order to have full network basis projection coefficients.

        Args:
          pCoeffs: Projection coefficients as returned by a unit ([1,
                   unitBasisSize])
          unitIdx: Unit index in the network

        Returns: Full network projection coefficients ([1,NUnits*unitBasisSize])

        """
        fullCoeffs = torch.zeros(
            (1, self.NUnits * self.unitBasisSize),
            dtype=torch.float32,
            device=self.device,
        )
        fullCoeffs[
            :, unitIdx * self.unitBasisSize : (unitIdx + 1) * self.unitBasisSize
        ] = pCoeffs
        return fullCoeffs

    def switchUnits(
        self,
        data: list,
        codes: torch.Tensor,
        procMask: list,
        lastCodes: torch.Tensor = None,
        ignoreDepthThr=False,
        forcedMode: bool = False,
    ) -> tuple:
        """
        Apply a batch of network units on a data tensor using a given code and
        direction.

        Args:
          data: (list) Input data to units (batch list of tensors of shape:
                [1,C,H, W]).
                A batch list is used instead of a tensor, to support various
                tensor shapes when the data goes through the network. It is
                indeed not possible to cat incompatible tensors with different
                shapes.
          codes: (torch.Tensor) Codes batch tensor (shape: [batchSize, 1])
          procMask: (list) Samples processing mask. This indicates if a sample
                    still needs to be processed. This is typically False when its
                    residu is lower than the depth threshold.
          lastCodes: (torch.Tensor) Last codes batch tensor (shape:
                      [batchSize, 1, ...])
          ignoreDepthThr: (bool) If True, ignores the depth threshold
          forcedMode: (bool) Set to True when using forced routing

        Returns: (tuple) processedData (batch list of shape: [1,C,H,W]),
                 new residus batch (shape: [batchSize, embeddingSize])
                 next codes batch (shape: [batchSize, 1],
                 projections batch (shape: [batchSize, embeddingSize])
                 projections coeffs batch (shape: [batchSize, unitBasisSize]),
                 new processing mask

        """
        tensor2code = lambda c: int(np.squeeze(c.cpu().numpy()))

        dataList = data  # torch.unbind(data, dim=0)
        codesList = torch.unbind(codes, dim=0)
        if not lastCodes is None:
            lastCodesList = torch.unbind(lastCodes, dim=0)
        else:
            lastCodesList = None

        # used for lastCodes masking to work properly
        epsilonResidu = (
            torch.ones((1, self.embeddingSize), dtype=torch.float32, device=self.device)
            * 1e-10
        )

        processedData = []
        newResidus = []
        nextCodes = []
        projections = []
        projCoeffs = []

        # logger.debug(f'Codes list: {codesList}')
        for idx, code in enumerate(codesList[: len(dataList)]):
            x = dataList[idx]
            # logger.debug(f'\tx: {x.shape}')
            # logger.debug(f'\tcode: {code} ({code.shape})')
            unitCode = tensor2code(code)
            # logger.debug(f'\tusing unit: {unitCode}')
            if unitCode >= 0:
                unit = self.getUnit(unitCode)
                y = unit(x, self.unitExpansionFlag, forcedMode)
                procData = y
            else:
                unit = None
                procData = x
            # logger.debug(f'proc data shape: {procData.shape}')
            if not lastCodesList is None:
                previousCodes = lastCodesList[idx]
                # the first "previous code" is obtained after the first operation ==>
                # maxDepth -1 = max number of previous codes
                maxDepthReached = previousCodes.shape[-1] == self.maxDepth - 1
                # logger.debug(f'Previous codes:\n{previousCodes} '
                #              f'({previousCodes.shape[-1]})')
            else:
                maxDepthReached = False
                previousCodes = None

            # if maxDepthReached:
            #   logger.debug('Max depth reached')

            if (
                not maxDepthReached
                and (ignoreDepthThr or unitCode != -2)
                and procMask[idx]
            ):
                if not self.perUnitHashing or unit is None:
                    hasher = self.hasher
                else:
                    hasher = unit.hasher
                if USE_INF_NORM:
                    h = hasher(
                        normalize(torch.flatten(procData, 1, -1), p=float("inf"))
                    )
                else:
                    h = hasher(torch.flatten(procData, 1, -1))

                z = normalize(h)  # normalized hash
                # logger.debug('\tz: {}'.format(z.shape))

                # Does a big matrix multilication and applies a mask to determine the
                # best projection. Does not call individual unit projection methods
                nextCode = self.getBestProjections(z, previousCodes)

                # logger.debug(f'Next codes tensor: {nextCode}')

                nextUnitCode = tensor2code(nextCode)
                # Calls the getProjection method of the next unit. The projection and
                # projectionCoeffs are saved by the unit
                pCoeffs, p = self.getUnitProjection(nextUnitCode, z, forcedMode)
                pCoeffs = self._shiftProjCoeffs(pCoeffs, nextUnitCode)
                newResidu = z - p + epsilonResidu
                nextUnit = self.getUnit(nextUnitCode)
                # Check residu norm only if unit is not empty
                if not ignoreDepthThr and not nextUnit.status.isEmpty():
                    newResiduNorm = torch.norm(newResidu.squeeze(), dim=0)
                    if newResiduNorm < self.depthThr:
                        # logger.debug('Low residu: max depth reached')
                        procMask[
                            idx
                        ] = False  # disables processing for the current sample
            else:
                nextCode = -2 * torch.ones(1, dtype=torch.long, device="cpu")
                p = torch.zeros(
                    (1, self.embeddingSize), dtype=torch.float32, device=self.device
                )
                pCoeffs = torch.zeros(
                    (1, self.NUnits * self.unitBasisSize),
                    dtype=torch.float32,
                    device=self.device,
                )
                newResidu = p

            # logger.debug(f'\ty: {y.shape}')
            # logger.debug(f'\tp: {p.shape}')
            # logger.debug(f'\tnextCode: {nextCode.shape} (value: {nextCode.numpy()})')

            processedData.append(procData)
            newResidus.append(newResidu)
            nextCodes.append(torch.squeeze(nextCode))
            projections.append(p)
            projCoeffs.append(pCoeffs)

        # processedDataTensor = torch.cat(processedData)
        newResidusTensor = torch.cat(newResidus)
        nextCodesTensor = torch.stack(nextCodes).reshape((self.batchSize, 1))
        projectionsTensor = torch.cat(projections)
        projCoeffsTensor = torch.cat(projCoeffs)

        # logger.debug(f'Residus: {newResidusTensor.shape}')
        # logger.debug(f'Projections:\n {projectionsTensor} ({projectionsTensor.shape})')
        # logger.debug(f'Proj Coeffs:\n {projCoeffsTensor} ({projCoeffsTensor.shape})')

        return (
            processedData,
            newResidusTensor,
            nextCodesTensor,
            projectionsTensor,
            projCoeffsTensor,
            procMask,
        )

    def autorouteTensor(self, inputData: torch.Tensor) -> tuple:
        """
        Autorouting using generated codes, limited by
        specified depth or residu. Batching is supported.

        Args:
          inputData: Input data (can be batched)

        Returns: lastCodes, data, projectionsTensor, projCoeffsTensor, residusTensor

        """
        codes = []
        projections = []
        projCoeffs = []
        residus = []
        lastCodes = None

        procMask = [True] * self.batchSize

        data = torch.unbind(inputData, dim=0)
        data = [torch.unsqueeze(x, 0) for x in data]
        currentCodes = -torch.ones((self.batchSize, 1), dtype=torch.long)
        for currentDepth in range(self.maxDepth):
            # logger.debug(f'Depth: {currentDepth} - Data: {data[0].shape}')
            data, residu, code, projection, projCoeff, procMask = self.switchUnits(
                data, currentCodes, procMask, lastCodes
            )

            residus.append(residu)
            projections.append(projection)
            projCoeffs.append(projCoeff)
            codes.append(code)
            lastCodes = torch.stack(codes, dim=-1)
            currentCodes = code

        projectionsTensor = torch.stack(projections, dim=-1)
        projCoeffsTensor = torch.stack(projCoeffs, dim=-1)
        residusTensor = torch.stack(residus, dim=-1)

        # Output shapes:
        # Codes: [batchSize,1,maxDepth]
        # Data: [batchSize,...]
        # Projections: [batchSize,embeddingSize*basisSize,maxDepth]
        # Projection Coefficients: [batchSize,embeddingSize*basisSize,maxDepth]
        # Residus: [batchSize,embeddingSize,maxDepth]

        return lastCodes, data, projectionsTensor, projCoeffsTensor, residusTensor

    def forcerouteTensor(self, inputData: torch.Tensor, codes: torch.Tensor) -> tuple:
        """
        Force routing using given codes. Batching is supported.
        Depth threshold is ignored in this case.

        Args:
          inputData: Input data (can be batched)
          codes: Forced routing codes tensor. Shape must be [batch,1,depth]

        Returns: generatedCodes,processedData, projections, residus

        """
        generatedCodes = []
        residus = []
        projections = []
        projCoeffs = []

        procMask = [True] * self.batchSize

        # codes tensor must be of shape [batchSize,1,depth]
        assert len(codes.shape) == 3
        assert codes.shape[1] == 1

        codesList = torch.unbind(codes, dim=-1)  # get tensors of shape [batchSize,1]
        data = inputData.reshape((self.batchSize, self.dataChannels, -1)).unbind(dim=0)
        data = [x.unsqueeze(0) for x in data]
        for currentDepth, currentCode in enumerate(codesList):
            # last codes are not considered when routing is forced
            data, residu, code, projection, projCoeff, procMask = self.switchUnits(
                data,
                currentCode,
                procMask=procMask,
                ignoreDepthThr=True,
                forcedMode=True,
            )
            generatedCodes.append(code)
            residus.append(residu)
            projections.append(projection)
            projCoeffs.append(projCoeff)
        generatedCodesTensor = torch.stack(generatedCodes, dim=-1)
        residusTensor = torch.stack(residus, dim=-1)
        projectionsTensor = torch.stack(projections, dim=-1)
        projCoeffsTensor = torch.stack(projCoeffs, dim=-1)

        # logger.debug(f'Proj tensor:\n{projectionsTensor} ('
        #              f'{projectionsTensor.shape})')
        #              f'{projCoeffsTensor.shape})')

        # Output shapes:
        # Codes: [batchSize,1,maxDepth]
        # Data: [batchSize,...]
        # Projections: [batchSize,embeddingSize*basisSize,maxDepth]
        # Projections coeffs: [batchSize,embeddingSize*basisSize,maxDepth]
        # Residus: [batchSize,embeddingSize,maxDepth]

        return (
            generatedCodesTensor,
            data,
            projectionsTensor,
            projCoeffsTensor,
            residusTensor,
        )

    def forward(
        self,
        x: torch.Tensor,
        fwdCodes: torch.Tensor = None,
        fastInference: bool = False,
    ) -> tuple:
        if fwdCodes is None:  # autoroute forward path
            codes, data, projections, projCoeffs, residus = self.autorouteTensor(x)
        else:  # force forward routing
            codes, data, projections, projCoeffs, residus = self.forcerouteTensor(
                x, fwdCodes
            )

        # shape: (batchSize,embeddingSize,1)
        summedProjections = torch.sum(projections, dim=-1)
        # logger.debug(f'projections:\n {projections} ({projections.shape})')
        # logger.debug(f'summed projections:\n{summedProjections} ('
        #              f'{summedProjections.shape})')
        # logger.debug(f'ProjCoeffs:\n{projCoeffs} ({projCoeffs.shape})')
        # logger.debug(f'current basis:\n{self.currentBasis} ('
        #              f'{self.currentBasis.shape})')

        # logger.debug(f'Coefs:\n{coefs} ({coefs.shape})')

        # sum the projection coefficients along the depth dimension
        # summedProjCoeffs = torch.sum(projCoeffs,-1)

        summedResidus = torch.sum(residus, -1)

        # output = torch.cat((summedProjections,summedResidus,
        #                          summedProjCoeffs), dim=-1)

        if USE_PROJ:
            output = torch.cat((summedProjections, summedResidus), dim=-1)
        else:
            output = summedResidus

        codes = torch.unsqueeze(codes, dim=1)  # [batchSize,1,depth]

        if fastInference:
            l1_residuNorm = 0
            l2_residuNorm = 0
        else:
            l1_residuNorm, l2_residuNorm = getResiduLoss(residus)

        self.nCalls += 1
        return output, codes, l1_residuNorm, l2_residuNorm

    def expandNetwork(self, newUnitsGroups: list) -> HashRoutedNetwork:
        """
        Add new units to a network by creating a new one from the current one

        Args:
          newUnitsGroups: (list) New unit groups to add

        Returns: (CoherentNetwork) New network

        """
        newArchitecture = deepcopy(self.architecture)
        newArchitecture["units"].extend(newUnitsGroups)
        newNetwork = HashRoutedNetwork(
            self.name,
            newArchitecture,
            self.device,
            self.unitExpansionFlag,
            self.reshapeOutput,
        ).to(self.device)

        newNetwork.load_state_dict(self.state_dict(), False)

        logger.debug(f"{newNetwork.name} has expanded to {newNetwork.NUnits} units")

        return newNetwork

    def doubleNetworkUnits(self) -> HashRoutedNetwork:
        """
        Double the network capacity by duplicating the unit groups

        Returns: (CoherentNetwork) New network

        """
        currentUnitGroups = self.architecture["units"]
        newUnitGroups = [deepcopy(group) for group in currentUnitGroups]
        return self.expandNetwork(newUnitGroups)

    def addRndNetworkUnits(self, nExtraUnits: int) -> HashRoutedNetwork:
        """
        Add units by randomly duplicating units from existing units
        Args:
          nExtraUnits: (int) Number of new units to add

        Returns: (CoherentNetwork) Extended network

        """
        currentUnitGroups = self.architecture["units"]
        newUnits = []
        for unitIdx in range(nExtraUnits):
            randomUnitGroup = deepcopy(np.random.choice(currentUnitGroups))
            randomUnitGroup["N"] = 1  # only one unit
            newUnits.append(randomUnitGroup)
        return self.expandNetwork(newUnits)
