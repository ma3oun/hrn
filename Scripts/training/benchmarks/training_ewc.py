"""
Supervised lifelong training - EWC
"""

from typing import Union, Tuple, List
from copy import deepcopy
from itertools import chain
from tqdm import tqdm

import argparse, gc, os
import logging, coloredlogs, mlflow
import ruamel.yaml as yaml
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard.writer import SummaryWriter
from hrn.datasets.loader import getDatasets
from hrn.routing.decoder import Decoder
from hrn.routing.codecs import Codec, Direction
from hrn.config_flags import *

logger = logging.getLogger("ewc")
logger.addHandler(logging.StreamHandler())

coloredlogs.install(
    logger=logger,
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.DEBUG,
)


def log_scalars(scalars, writer, step):
    """Log a scalar value to both MLflow and TensorBoard"""
    for scalar, name in scalars:
        value = scalar.item()
        writer.add_scalar(name, value, step)
        mlflow.log_metric(name, value, step)


def getTrainingSamples(
    datasets: Union[str, List],
    sampleSize: int,
    trainingDataSize: tuple,
    trainingDataChannels: int,
) -> List:
    """
    This is used for Fisher matrix estimation
    Args:
      trainingDataChannels: (tuple) Training data channels
      trainingDataSize: (int) Training data size
      datasets: (str or list) List of datasets to sample from
      sampleSize: (int) Total sample size

    Returns: (list) Samples list

    """

    if type(datasets) is str:
        datasetSampleSize = int(np.ceil(sampleSize / len(datasets)))
    else:
        datasetSampleSize = sampleSize
    fullDatasets = getDatasets(datasets, 1, trainingDataSize, trainingDataChannels)
    if not type(fullDatasets) is list:
        fullDatasets = [fullDatasets]
    samples = []
    for dataset in fullDatasets:
        train_loader, _ = dataset
        samplesIdx = np.random.randint(0, len(train_loader.dataset), datasetSampleSize)
        currentSamples = [train_loader.dataset[i][0] for i in samplesIdx]
        samples.extend(currentSamples)
    return samples


class EWC(object):
    def __init__(self, model: torch.nn.Module, sampleSet: list, device: torch.device):
        self.device = device
        self.model = deepcopy(model)
        self.sampleSet = sampleSet

        self.params = {
            n: p for n, p in self.model.named_parameters() if p.requires_grad
        }
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            self._means[n] = p.clone().detach().requires_grad_(True)

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = p.clone().detach().requires_grad_(True)

        self.model.eval()
        for x in self.sampleSet:
            self.model.zero_grad()
            # x = torch.tensor(x)
            output = self.model(x.unsqueeze(dim=0).to(self.device)).view(1, -1)
            label = output.max(1)[1].view(-1)
            loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            loss.backward()

            for n, p in self.model.named_parameters():
                # if torch.norm(p.grad.data) > 1e-2:
                precision_matrices[n].data += p.grad.data ** 2
        # avoid nan
        for n, p in self.model.named_parameters():
            precision_matrices[n].data /= len(self.sampleSet)

        # precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: torch.nn.Module):
        loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            totalLoss = _loss.sum()

            if not torch.isnan(totalLoss):  # and totalLoss > 1e-2:
                loss += totalLoss
            # else:
            #   logger.warning('loss is accumulating nan')
        return loss


def train(
    model: torch.nn.Module,
    decoder: torch.nn.Module,
    ewc: EWC,
    importance: np.float32,
    device: torch.device,
    train_loader: torch.utils.data.DataLoader,
    optimizer: Union[optim.Adam, optim.SGD],
    epoch: int,
    reconCoeff: np.float32,
    writer: SummaryWriter = None,
):

    torch.cuda.empty_cache()

    model.train()
    decoder.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        decoderData = model(data).flatten(start_dim=1)
        outputData = decoder(decoderData)

        output = F.log_softmax(outputData, dim=1)
        reconLoss = reconCoeff * F.nll_loss(output, target)

        if not ewc is None:
            penalty = importance * ewc.penalty(model)
        else:
            penalty = torch.tensor(0)

        loss = reconLoss + penalty
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            logger.info(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100. * batch_idx / len(train_loader):.0f}%)]\n"
                f"\tLoss: {loss.item():.4f}"
                f"\tXentropy: {reconLoss.item():.4f}"
            )
            fullStep = len(train_loader) * (epoch - 1) + batch_idx
            scalars = [loss, reconLoss]
            names = ["train/Loss", "train/Xentropy"]
            log_scalars(zip(scalars, names), writer, fullStep)

        # For testing purposes only
        # if batch_idx == 400:
        #   break

    return


def test(
    model: torch.nn.Module,
    decoder: torch.nn.Module,
    device: torch.device,
    test_loader: torch.utils.data.DataLoader,
    train_loader_length: int,
    epoch: int,
    reconCoeff: np.float32,
    writer: SummaryWriter = None,
) -> Tuple[float, int]:

    torch.cuda.empty_cache()

    model.eval()
    decoder.eval()
    test_loss = 0
    correct = 0

    trainStep = train_loader_length * epoch
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.to(device)
            target = target.to(device)
            decoderData = model(data).flatten(start_dim=1)
            outputData = decoder(decoderData)

            output = F.log_softmax(outputData, dim=1)
            predictedLabel = outputData.argmax(dim=1, keepdim=True)
            reconLoss = reconCoeff * F.nll_loss(output, target, reduction="sum")

            test_loss += reconLoss
            correct += predictedLabel.eq(target.view_as(predictedLabel)).sum()

    testLoss = test_loss / len(test_loader.dataset)
    accuracyPercent = 100.0 * correct / len(test_loader.dataset)
    logger.info(
        f"\nTest set: Average loss: {testLoss.item():.4f}\n"
        f"Accuracy: {correct.item()}/{len(test_loader.dataset)} "
        f"({accuracyPercent.item():.0f}%)"
    )

    scalars = [testLoss, accuracyPercent]
    names = ["test/loss", "test/accuracy"]
    log_scalars(zip(scalars, names), writer, trainStep)

    return accuracyPercent.item(), correct.item()


def encodeDataset(
    model: torch.nn.Module,
    device: torch.device,
    dataLoader: torch.utils.data.DataLoader,
) -> list:
    dataBuffer = []
    model.eval()

    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(
            enumerate(dataLoader), total=len(dataLoader)
        ):
            data = data.to(device)
            target = target.to(device)
            decoderData = model(data)
            dataBuffer.append((decoderData, target))

    return dataBuffer


def testClassifier(
    classifier: torch.nn.Module,
    testDataBuffer: List[Tuple],
    datasetSize: int,
    epoch: int,
    testName: str = "supervised",
) -> Tuple[float, int]:
    """
    Test a trained classifier using a given test dataset

    Args:
      classifier: (torch.nn.Module) Trained classifier
      testDataBuffer: (list) List of (embeddings,labels)
      datasetSize: (int) Dataset size (used to compute accuracy)
      epoch: (int) Epoch
      testName: (str) Test name

    Returns: (np.float32) Accuracy

    """
    logger.info("\n\t==== Testing classifier ====\n")
    correct = 0
    classifier.eval()

    for batch_idx, (data, target) in tqdm(
        enumerate(testDataBuffer), total=len(testDataBuffer)
    ):
        with torch.no_grad():
            outputData = classifier(data.flatten(start_dim=1))
            predictedLabel = outputData.argmax(dim=1, keepdim=True)
            correct += predictedLabel.eq(target.view_as(predictedLabel)).sum().item()
    acc = 100.0 * correct / datasetSize
    logger.info(
        f"\nAccuracy at epoch {epoch + 1}:"
        f"\n\t{acc:.0f}% ({correct}/{datasetSize})\n"
    )
    mlflow.log_metric(f"{testName}/accuracy", acc, epoch)

    return acc, correct


def saveModel(model: torch.nn.Module, filename: str):
    try:
        torch.save(model.state_dict(), filename)
        mlflow.log_artifact(filename)
    except:
        logger.warning("Save model failed! Continuing...")
    return


def saveData(dataBuffer: list, filename: str):
    """
    Save generated codes and embedding vectors

    Args:
      dataBuffer: (list) Each list element contains (embedding,label)
      filename: (str) Filename to use for saving data

    Returns:

    """

    dataArray = np.array([x[0].cpu().numpy() for x in dataBuffer], dtype=np.float32)
    targetArray = np.array([x[1].cpu().numpy() for x in dataBuffer], dtype=np.float32)
    np.savez_compressed(filename, data=dataArray, targets=targetArray)
    mlflow.log_artifact(filename)
    return


def _asList(x, length: int = 1) -> list:
    """
    Get a list no matter what. If the input is already a list, does nothing. If
    it is not a list, it replicates it.
    Args:
      x: (Any) Any parameter
      length: List length

    Returns: (list) Parameter as list

    """
    if not isinstance(x, list):
        y = [x] * length
    else:
        y = x
    return y


def _genExpName(dataList: list) -> str:
    """
    Generate MLFlow experiment name
    Args:
      dataList: (list) Datasets list

    Returns: (str) Experiment name (supervised, unsupervised, mixed)

    """
    if len(dataList) > 1:
        expName = "ewc_"
    else:
        expName = dataList[0] + "_"

    expName += "s"

    return expName


def run(runParams: dict) -> float:
    if not runParams["paramsFile"] is None:
        with open(runParams["paramsFile"], "r") as f:
            params = yaml.safe_load(f)
            params["device"] = runParams["device"]
            params["subDeviceIdx"] = runParams["subDeviceIdx"]
    else:
        params = runParams

    if params["device"] != "cuda":
        use_cuda = False
    else:
        use_cuda = torch.cuda.is_available()

    if params["subDeviceIdx"] is None:
        subDeviceIdx = 0
    else:
        subDeviceIdx = params["subDeviceIdx"]

    device = torch.device("cuda:{}".format(subDeviceIdx) if use_cuda else "cpu")
    seed = params["training"]["seed"]
    if seed is None:
        seed = np.random.randint(10000)
        logger.debug("Using random seed")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)

    dataList = _asList(params["training"]["datasets"])
    logger.info(f"Datasets: {dataList}")
    nDatasets = len(dataList)

    batchSize = params["training"]["batchSize"]
    epochs = _asList(params["training"]["epochs"], nDatasets)

    dataShape = params["training"]["inputShape"]
    dataSize = dataShape[-2:]
    dataChannels = dataShape[1]

    fullLossFactor = _asList(params["training"]["fullLossCoeff"], nDatasets)
    learningRate = _asList(params["training"]["learningRate"], nDatasets)
    ewcCoeff = params["training"]["importance"]
    ewcSampleSize = params["training"]["sampleSize"]

    expName = _genExpName(dataList)
    experiment = mlflow.get_experiment_by_name(expName)
    if experiment is None:
        logger.info("Creating new experiment")
        expID = mlflow.create_experiment(expName)
    else:
        logger.info(f"Using existing experiment")
        expID = experiment.experiment_id

    with mlflow.start_run(experiment_id=expID):
        modelName = params["name"]
        mlflow.log_param("params", params)
        mlflow.log_param("name", modelName)

        coder = Codec(
            0, Direction.Forward, params["training"]["inputShape"], params["coder"]
        ).to(device)
        logDir = "../data/logs/" + modelName
        tbWriter = SummaryWriter(logDir)

        previousTestData = []
        previousClassifiers = []
        for datasetIdx, dataset in enumerate(dataList):
            embeddingSize = params["training"]["embeddingSize"]

            # one decoder per dataset
            classifier = Decoder(
                _asList(params["decoder"], nDatasets)[datasetIdx],
                embeddingSize,
                0,
                params["training"]["inputShape"],
                device,
                False,
            ).to(device)

            logger.info(f"\n\t==== {dataset}: TRAINING ====\n")
            train_loader, test_loader = getDatasets(
                dataset, batchSize, dataSize, dataChannels
            )

            optimParams = chain(coder.parameters(), classifier.parameters())
            optimizer = optim.Adam(optimParams, lr=learningRate[datasetIdx])
            if datasetIdx > 0:
                ewc = EWC(
                    coder,
                    getTrainingSamples(
                        dataList[:datasetIdx], ewcSampleSize, dataSize, dataChannels
                    ),
                    device,
                )
            else:
                ewc = None

            currentAcc = 0
            currentCorrect = 0
            currentTotalSize = len(test_loader.dataset)
            for epoch in range(1, epochs[datasetIdx] + 1):
                train(
                    coder,
                    classifier,
                    ewc,
                    ewcCoeff,
                    device,
                    train_loader,
                    optimizer,
                    epoch,
                    fullLossFactor[datasetIdx],
                    tbWriter,
                )
                logger.info(f"\n\t==== {dataset}: TEST ({dataset}) ====\n")
                currentAcc, currentCorrect = test(
                    coder,
                    classifier,
                    device,
                    test_loader,
                    len(train_loader),
                    epoch,
                    fullLossFactor[datasetIdx],
                    tbWriter,
                )

            # saveModel(coder, f'../data/{modelName}_{dataset}.pt')
            # saveModel(classifier, f'../data/{modelName}_{dataset}_classifier.pt')
            #
            # trainBuffer = encodeDataset(coder, device, train_loader)
            # testBuffer = encodeDataset(coder, device, test_loader)
            #
            # saveData(trainBuffer,f'../data/{modelName}_{dataset}_trainEmbeddings.npz')
            # saveData(testBuffer,f'../data/{modelName}_{dataset}_testEmbeddings.npz')

            totalCorrect = currentCorrect
            totalDatasetSize = currentTotalSize
            for pIdx, pTestData in enumerate(previousTestData):
                previousDataName = dataList[pIdx]
                logger.info(
                    f"\n\t==== {dataset}: Lifelong TEST  ({previousDataName}) "
                    f"====\n"
                )
                testDatasetSize = len(pTestData.dataset)
                totalDatasetSize += testDatasetSize
                logger.info("Re-encoding test data")
                newEncodedTestData = encodeDataset(coder, device, pTestData)
                # saveData(newEncodedTestData,
                #          f'../data/{modelName}_lifelong_{previousDataName}'
                #          f'_testEmbeddings.npz')
                pClassifier = previousClassifiers[pIdx].to(device)

                previousAcc, previousCorrect = testClassifier(
                    pClassifier,
                    newEncodedTestData,
                    testDatasetSize,
                    epochs[datasetIdx],
                    f"lifelong/{previousDataName}",
                )

                totalCorrect += previousCorrect

                globalAcc = 100.0 * totalCorrect / totalDatasetSize
                mlflow.log_metric(
                    "lifelong/globalAccuracy", globalAcc, epochs[datasetIdx]
                )
                logger.info(
                    f"Global accuracy at task {pIdx}: {globalAcc:.0f}% "
                    f"({totalCorrect}/{totalDatasetSize})"
                )

            previousTestData.append(test_loader)
            previousClassifiers.append(classifier.cpu())

        mlflow.log_artifacts(logDir, artifact_path="events")

    return currentAcc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generic trainer")
    parser.add_argument("configuration", type=str, help="Configuration file")
    parser.add_argument(
        "device", type=str, help="Device to use", default="cpu", choices=["cpu", "cuda"]
    )
    parser.add_argument("-d", type=int, help="Cuda device index", default=0)

    args = parser.parse_args()
    paramsFile = args.configuration
    devParam = args.device
    subDeviceIdx = args.d

    run({"paramsFile": paramsFile, "device": devParam, "subDeviceIdx": subDeviceIdx})
