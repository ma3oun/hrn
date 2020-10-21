"""
Feature hashing
"""

import pkg_resources
import torch
import torch.nn as nn
import numpy as np


class Hasher(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = nn.Parameter(torch.Tensor([dim]), False)
        return

    def hash(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hash(x)


class MSHash(Hasher):
    def __init__(self, m: int, p: int, a: int = None, b: int = None):
        super().__init__(m)
        assert p >= m
        self.m = nn.Parameter(torch.Tensor([m]), False)
        self.p = nn.Parameter(torch.Tensor([p]), False)
        if not a is None:
            assert 0 < a < p
            self.a = nn.Parameter(torch.Tensor([a]), False)
        else:
            self.a = nn.Parameter(torch.randint(1, p, (1,)), False)
        if not b is None:
            self.b = nn.Parameter(torch.Tensor([b]), False)
        else:
            self.b = nn.Parameter(torch.randint(1, 10000, (1,)) % self.p, False)
        return

    def hash(self, x: torch.Tensor) -> torch.Tensor:
        h = ((self.a * x + self.b) % self.p) % self.m
        return h


class BinaryHash(MSHash):
    def __init__(self, p, a: int = None, b: int = None):
        super(BinaryHash, self).__init__(2, p, a, b)
        return

    def hash(self, x: torch.Tensor) -> torch.Tensor:
        h = super(BinaryHash, self).hash(x)
        h = torch.where(h.type(torch.BoolTensor), torch.Tensor([1]), torch.Tensor([-1]))
        return h


class Hash1d(nn.Module):
    def __init__(
        self, h: Hasher, ksi: Hasher, device: torch.device = torch.device("cpu")
    ):
        super(Hash1d, self).__init__()
        self.hasherH = h
        self.ksi = ksi
        self.embSize = int(h.dim.cpu().numpy())
        self.device = device
        return

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
        # avoid rebuilding projection matrices
        for key in state_dict.keys():
            if key.startswith("hasher.hash_dim_"):
                hashDim = int(key.split("dim_")[-1])
                hashProj = torch.zeros((hashDim, self.embSize), dtype=torch.float32)
                self.register_parameter(
                    f"hash_dim_{hashDim}", nn.Parameter(hashProj, False)
                )
        super(Hash1d, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def getHashProjection(self, inputDim: int) -> torch.Tensor:
        hash_dim = f"hash_dim_{inputDim}"
        if hasattr(self, hash_dim):
            hashProj = self.__getattr__(hash_dim)
        else:  # hash projection must be built
            # hashProj = torch.zeros((inputDim,self.embSize),dtype=torch.float32)
            # for i in range(self.embSize):
            #   for j in range(inputDim):
            #     cond = (self.hasherH(j) == i).type(torch.BoolTensor)
            #     coeff = torch.where(cond,torch.Tensor([1.]),torch.Tensor([0.]))
            #     hashProj[j][i] = self.ksi(j) * coeff

            # optimisation of the algorithm above
            l1 = np.array(
                [
                    [j, self.hasherH(j).cpu().item(), self.ksi(j).cpu().item()]
                    for j in range(inputDim)
                ]
            )
            l2 = l1[l1[:, 1] < self.embSize]
            idx = torch.LongTensor(l2[:, :2].astype(np.long)).t()  # indices
            values = torch.FloatTensor(l2[:, 2].astype(np.float32))  # values
            hashProj = torch.sparse.FloatTensor(
                idx, values, torch.Size([inputDim, self.embSize])
            ).to_dense()
            hashProj = hashProj.to(device=self.device)
            self.register_parameter(
                f"hash_dim_{inputDim}", nn.Parameter(hashProj, False)
            )
        return hashProj

    def forward(self, x: torch.Tensor):
        assert len(x.shape) == 2
        inputDim = x.shape[1]
        try:
            hashProj = self.getHashProjection(inputDim)
            s = torch.matmul(x, hashProj)
        except RuntimeError:
            hashProj = hashProj.to(device=self.device, copy=True)
            s = torch.matmul(x, hashProj)

        return s


def generateHasher(
    primeOffset: int, embeddingSize: int, device: torch.device
) -> Hash1d:
    primesData = pkg_resources.resource_filename(__name__, "primes.npz")
    primes = np.load(primesData)["primes"]
    randint = np.random.randint
    nPrimes = len(primes)
    pHash = primes[primeOffset]
    aHash = randint(1, pHash)
    bHash = randint(1, 10000)

    pKsi = primes[randint(2, nPrimes)]
    aKsi = randint(2, pKsi)
    bKsi = 2 * randint(1, 10000) + 1

    h = MSHash(embeddingSize, pHash, aHash, bHash)
    ksi = BinaryHash(pKsi, aKsi, bKsi)
    hasher = Hash1d(h, ksi, device)
    return hasher
