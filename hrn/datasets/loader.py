"""
Datasets loader
"""

from typing import Union, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from hrn.datasets.tinyImageNet import TinyImageNet


def sizesAreEqual(
    newSize: Union[int, Iterable], originalSize: Union[int, Iterable]
) -> bool:
    def toTupleSize(size):
        if type(size) is int:
            tupleSize = (size, size)
        else:
            tupleSize = tuple(size)
        return tupleSize

    _newSize = toTupleSize(newSize)
    _originalSize = toTupleSize(originalSize)

    return _newSize == _originalSize


def getTinyImageNet(
    batchSize: int,
    taskID: int,
    nTasks: int = 10,
    size: Union[int, Iterable] = None,
    channels: int = None,
) -> tuple:
    originalSize = (64, 64)
    nClsPerTask = 200 // nTasks

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    if not size is None and sizesAreEqual(size, originalSize):
        resize = transforms.Resize(size)
        tfms = transforms.Compose([resize, transforms.ToTensor(), normalize])
    else:
        tfms = transforms.Compose([transforms.ToTensor(), normalize])

    if channels == 1:
        tfms = transforms.Compose([transforms.Grayscale(num_output_channels=1), tfms])

    train = TinyImageNet(train=True, transform=tfms)

    test = TinyImageNet(train=False, transform=tfms)

    targets_train = torch.tensor(train.targets)
    targets_train_idx = (targets_train >= taskID * nClsPerTask) & (
        targets_train < (taskID + 1) * nClsPerTask
    )

    targets_test = torch.tensor(test.targets)
    targets_test_idx = (targets_test >= taskID * nClsPerTask) & (
        targets_test < (taskID + 1) * nClsPerTask
    )

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.dataset.Subset(train, np.where(targets_train_idx == 1)[0]),
        batch_size=batchSize,
        shuffle=True,
        drop_last=True,
    )

    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.dataset.Subset(test, np.where(targets_test_idx == 1)[0]),
        batch_size=batchSize,
        shuffle=True,
        drop_last=True,
    )

    return train_loader, test_loader


def getPairwiseMNIST(
    batchSize: int,
    labels: tuple,
    size: Union[int, Iterable] = None,
    channels: int = None,
) -> tuple:
    originalSize = (28, 28)
    if not size is None and sizesAreEqual(size, originalSize):
        tfms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
    else:
        resize = transforms.Resize(size)
        tfms = transforms.Compose(
            [resize, transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    if channels == 3:
        tfms = transforms.Compose(
            [tfms, transforms.Lambda(lambda x: x.repeat(3, 1, 1))]
        )

    train = datasets.MNIST("../data", train=True, download=True, transform=tfms)

    test = datasets.MNIST("../data", train=False, transform=tfms)

    targets_train = train.targets.clone().detach()
    targets_train_idx = (targets_train == labels[0]) | (targets_train == labels[1])

    targets_test = test.targets.clone().detach()
    targets_test_idx = (targets_test == labels[0]) | (targets_test == labels[1])

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.dataset.Subset(train, np.where(targets_train_idx == 1)[0]),
        batch_size=batchSize,
        shuffle=True,
        drop_last=True,
    )

    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.dataset.Subset(test, np.where(targets_test_idx == 1)[0]),
        batch_size=batchSize,
        shuffle=True,
        drop_last=True,
    )

    return train_loader, test_loader


def getMNIST(
    batchSize: int, size: Union[int, Iterable] = None, channels: int = None
) -> tuple:
    originalSize = (28, 28)

    if not size is None and sizesAreEqual(size[-2:], originalSize[-2:]):
        tfms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    else:
        resize = transforms.Resize(size)
        tfms = transforms.Compose(
            [resize, transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    if channels == 3:
        tfms = transforms.Compose(
            [tfms, transforms.Lambda(lambda x: x.repeat(3, 1, 1))]
        )

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST("../data", train=True, download=True, transform=tfms),
        batch_size=batchSize,
        shuffle=True,
        drop_last=True,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST("../data", train=False, transform=tfms),
        batch_size=batchSize,
        shuffle=True,
        drop_last=True,
    )
    return train_loader, test_loader


def getFashion(
    batchSize: int, size: Union[int, Iterable] = None, channels: int = None
) -> tuple:
    originalSize = (28, 28)
    if not size is None and sizesAreEqual(size, originalSize):
        tfms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
    else:
        resize = transforms.Resize(size)
        tfms = transforms.Compose(
            [resize, transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

    if channels == 3:
        tfms = transforms.Compose(
            [tfms, transforms.Lambda(lambda x: x.repeat(3, 1, 1))]
        )

    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST("../data", train=True, download=True, transform=tfms),
        batch_size=batchSize,
        shuffle=True,
        drop_last=True,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST("../data", train=False, transform=tfms),
        batch_size=batchSize,
        shuffle=True,
        drop_last=True,
    )

    return train_loader, test_loader


def getCifar100(
    batchSize: int, size: Union[int, Iterable] = None, channels: int = None
) -> tuple:
    originalSize = (32, 32)
    if not size is None and sizesAreEqual(size, originalSize):
        tfms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                    (0.2673342858792401, 0.2564384629170883, 0.27615047132568404),
                ),
            ]
        )
    else:
        resize = transforms.Resize(size)
        tfms = transforms.Compose(
            [
                resize,
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                    (0.2673342858792401, 0.2564384629170883, 0.27615047132568404),
                ),
            ]
        )

    if channels == 1:
        tfms = transforms.Compose([tfms, transforms.Grayscale(num_output_channels=1)])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100("../data", train=True, download=True, transform=tfms),
        batch_size=batchSize,
        shuffle=True,
        drop_last=True,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100("../data", train=False, transform=tfms),
        batch_size=batchSize,
        shuffle=True,
        drop_last=True,
    )
    return train_loader, test_loader


def getIncrementalCifar100(
    batchSize: int,
    taskID: int,
    nTasks: int = 10,
    size: Union[int, Iterable] = None,
    channels: int = None,
) -> tuple:
    originalSize = (32, 32)
    if not size is None and sizesAreEqual(size, originalSize):
        tfms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                    (0.2673342858792401, 0.2564384629170883, 0.27615047132568404),
                ),
            ]
        )
    else:
        resize = transforms.Resize(size)
        tfms = transforms.Compose(
            [
                resize,
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                    (0.2673342858792401, 0.2564384629170883, 0.27615047132568404),
                ),
            ]
        )

    if channels == 1:
        tfms = transforms.Compose([tfms, transforms.Grayscale(num_output_channels=1)])

    nClsPerTask = 100 // nTasks

    train = datasets.CIFAR100("../data", train=True, download=True, transform=tfms)

    test = datasets.CIFAR100("../data", train=False, transform=tfms)

    targets_train = torch.tensor(train.targets)
    targets_train_idx = (targets_train >= taskID * nClsPerTask) & (
        targets_train < (taskID + 1) * nClsPerTask
    )

    targets_test = torch.tensor(test.targets)
    targets_test_idx = (targets_test >= taskID * nClsPerTask) & (
        targets_test < (taskID + 1) * nClsPerTask
    )

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.dataset.Subset(train, np.where(targets_train_idx == 1)[0]),
        batch_size=batchSize,
        shuffle=True,
        drop_last=True,
    )

    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.dataset.Subset(test, np.where(targets_test_idx == 1)[0]),
        batch_size=batchSize,
        shuffle=True,
        drop_last=True,
    )

    return train_loader, test_loader


def getCifar10(
    batchSize: int, size: Union[int, Iterable] = None, channels: int = None
) -> tuple:
    originalSize = (32, 32)
    if not size is None and sizesAreEqual(size, originalSize):
        tfms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    else:
        resize = transforms.Resize(size)
        tfms = transforms.Compose(
            [
                resize,
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    if channels == 1:
        tfms = transforms.Compose([tfms, transforms.Grayscale(num_output_channels=1)])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10("../data", train=True, download=True, transform=tfms),
        batch_size=batchSize,
        shuffle=True,
        drop_last=True,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10("../data", train=False, transform=tfms),
        batch_size=batchSize,
        shuffle=True,
        drop_last=True,
    )
    return train_loader, test_loader


def getSVHN(
    batchSize: int, size: Union[int, Iterable] = None, channels: int = None
) -> tuple:
    originalSize = (32, 32)
    if not size is None and sizesAreEqual(size, originalSize):
        tfms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    else:
        resize = transforms.Resize(size)
        tfms = transforms.Compose(
            [
                resize,
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    if channels == 1:
        tfms = transforms.Compose([tfms, transforms.Grayscale(num_output_channels=1)])

    train_loader = torch.utils.data.DataLoader(
        datasets.SVHN("../data", split="train", download=True, transform=tfms),
        batch_size=batchSize,
        shuffle=True,
        drop_last=True,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.SVHN("../data", split="test", download=True, transform=tfms),
        batch_size=batchSize,
        shuffle=True,
        drop_last=True,
    )
    return train_loader, test_loader


def getDatasets(
    names: Union[str, list],
    batchSize: int,
    size: Union[int, Iterable] = None,
    channels: int = None,
) -> Union[tuple, list]:
    datasetMap = {
        "mnist": lambda x: getMNIST(x, size, channels),
        "cifar10": lambda x: getCifar10(x, size, channels),
        "cifar100": lambda x: getCifar100(x, size, channels),
        "fashion": lambda x: getFashion(x, size, channels),
        "svhn": lambda x: getSVHN(x, size, channels),
    }

    datasetMap.update(
        {
            f"mnist_{i}{i+1}": lambda x, i=i: getPairwiseMNIST(
                x, (i, i + 1), size=size, channels=channels
            )
            for i in range(9)
        }
    )

    datasetMap.update(
        {
            f"cifar100_{i}": lambda x, i=i: getIncrementalCifar100(
                x, i, size=size, channels=channels
            )
            for i in range(10)
        }
    )

    datasetMap.update(
        {
            f"TIN_{i}": lambda x, i=i: getTinyImageNet(
                x, i, size=size, channels=channels
            )
            for i in range(10)
        }
    )

    if type(names) is str or len(names) == 1:
        if type(names) is str:
            loaders = datasetMap[names](batchSize)
        else:
            loaders = datasetMap[names[0]](batchSize)
    else:
        loaders = [datasetMap[name](batchSize) for name in names]
    return loaders
