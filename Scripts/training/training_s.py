"""
Supervised training
"""

from typing import Union, Tuple, List
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
from hrn.routing.network import HashRoutedNetwork
from hrn.config_flags import *

logger = logging.getLogger("hrn")
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


def train(
    model: torch.nn.Module,
    decoder: torch.nn.Module,
    device: torch.device,
    train_loader: torch.utils.data.DataLoader,
    optimizer: Union[optim.Adam, optim.SGD],
    epoch: int,
    reconCoeff: np.float32,
    l1ResiduCoeff: np.float32,
    l2ResiduCoeff: np.float32,
    writer: SummaryWriter = None,
):

    torch.cuda.empty_cache()

    model.train()
    model.resetNetworkUsage()
    decoder.train()

    for batch_idx, (data, target) in enumerate(train_loader):

        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        decoderData, outputCodes, l1_residu, l2_residu = model(data)
        outputData = decoder(decoderData)

        output = F.log_softmax(outputData, dim=1)
        reconLoss = reconCoeff * F.nll_loss(output, target)
        l1_residu = torch.mean(l1_residu)
        l2_residu = torch.mean(l2_residu)
        residuLoss = l1ResiduCoeff * l1_residu + l2ResiduCoeff * l2_residu
        loss = reconLoss + residuLoss
        loss.backward(retain_graph=True)
        optimizer.step()

        if batch_idx % 100 == 0:
            logger.info(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100. * batch_idx / len(train_loader):.0f}%)]\n"
                f"\tLoss: {loss.item():.4f}"
                f"\tXentropy: {reconLoss.item():.4f}"
                f"\tl1_residu: {l1_residu.item():.4f}"
                f"\tl2_residu: {l2_residu.item():.4f}"
            )
            fullStep = len(train_loader) * (epoch - 1) + batch_idx
            scalars = [loss, reconLoss, residuLoss, l1_residu, l2_residu]
            names = [
                "train/Loss",
                "train/Xentropy",
                "train/residuLoss",
                "train/l1_residuLoss",
                "train/l2_residuLoss",
            ]
            log_scalars(zip(scalars, names), writer, fullStep)
        # del loss
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
    l1ResiduCoeff: np.float32,
    l2ResiduCoeff: np.float32,
    writer: SummaryWriter = None,
) -> Tuple[float, int]:

    gc.collect()
    torch.cuda.empty_cache()

    model.eval()
    decoder.eval()
    test_loss = 0
    correct = 0

    synthesis, synthesisMetrics = model.synthesisView()
    usage, usageMetrics = model.getNetworkUsage(True)
    logger.info(synthesis)
    logger.info(usage)
    trainStep = train_loader_length * epoch
    mlflow.log_metrics(synthesisMetrics, trainStep)
    mlflow.log_metrics(usageMetrics, trainStep)
    with torch.no_grad():
        # def processData(testLoaderElement):
        #   data,target = testLoaderElement
        #   data = data.to(device)
        #   target = target.to(device)
        #   decoderData, outputCodes, l1_residu, l2_residu = model(data)
        #   outputData = decoder(decoderData)
        #
        #   output = F.log_softmax(outputData, dim=1)
        #   predictedLabel = outputData.argmax(dim=1,
        #                                      keepdim=True)
        #   reconLoss = reconCoeff * F.nll_loss(output, target,
        #                                       reduction='sum')
        #
        #   residuLoss = torch.mean(l1ResiduCoeff * l1_residu +
        #                           l2ResiduCoeff * l2_residu)
        #   _test_loss = reconLoss + residuLoss
        #   _correct = predictedLabel.eq(target.view_as(predictedLabel)).sum()
        #   return _test_loss,_correct
        # pool = mp.Pool()
        # results = pool.map(processData,test_loader)
        # pool.close()
        # pool.join()
        # losses = [x[0] for x in results]
        # corrects = [x[1] for x in results]
        # test_loss = torch.stack(losses).sum()
        # correct = torch.stack(corrects).sum()
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.to(device)
            target = target.to(device)
            decoderData, outputCodes, l1_residu, l2_residu = model(data)
            outputData = decoder(decoderData)

            output = F.log_softmax(outputData, dim=1)
            predictedLabel = outputData.argmax(dim=1, keepdim=True)
            reconLoss = reconCoeff * F.nll_loss(output, target, reduction="sum")

            residuLoss = torch.mean(
                l1ResiduCoeff * l1_residu + l2ResiduCoeff * l2_residu
            )
            test_loss += reconLoss + residuLoss
            correct += predictedLabel.eq(target.view_as(predictedLabel)).sum()
            del data, target, decoderData, outputData
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
    decoderDataSize: int = None,
) -> list:
    dataBuffer = []
    model.eval()
    if decoderDataSize is None:
        resizeData = lambda x: x
    else:
        resizeData = lambda x: x[:, :decoderDataSize]
    with torch.no_grad():
        # def processData(dataLoaderElement):
        #   data, target = dataLoaderElement
        #   data = data.to(device)
        #   target = target.to(device)
        #   decoderData, outputCodes, l1_residu, l2_residu = model(data)
        #
        #   return (outputCodes,resizeData(decoderData),target)

        # pool = mp.Pool()
        # dataBuffer = pool.map(processData,dataLoader)
        # pool.close()
        # pool.join()
        for batch_idx, (data, target) in tqdm(
            enumerate(dataLoader), total=len(dataLoader)
        ):
            data = data.to(device)
            target = target.to(device)
            decoderData, outputCodes, l1_residu, l2_residu = model(data)
            dataBuffer.append(
                (
                    outputCodes.cpu().detach(),
                    resizeData(decoderData.cpu().detach()),
                    target.cpu().detach(),
                )
            )
            del data
            del target

    return dataBuffer


def testClassifier(
    classifier: torch.nn.Module,
    testDataBuffer: List[Tuple],
    datasetSize: int,
    epoch: int,
    device: torch.device,
    testName: str = "supervised",
) -> Tuple[float, int]:
    """
    Test a trained classifier using a given test dataset

    Args:
      classifier: (torch.nn.Module) Trained classifier
      testDataBuffer: (list) List of (codes,embeddings,labels)
      datasetSize: (int) Dataset size (used to compute accuracy)
      epoch: (int) Epoch
      testName: (str) Test name

    Returns: (np.float32) Accuracy

    """
    logger.info("\n\t==== Testing classifier ====\n")
    correct = 0
    classifier.eval()

    with torch.no_grad():
        for batch_idx, (code, data, target) in tqdm(
            enumerate(testDataBuffer), total=len(testDataBuffer)
        ):
            outputData = classifier(data.to(device))
            predictedLabel = outputData.argmax(dim=1, keepdim=True)
            correct += (
                predictedLabel.eq(target.to(device).view_as(predictedLabel))
                .sum()
                .item()
            )

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
      dataBuffer: (list) Each list element contains (code,embedding,label)
      filename: (str) Filename to use for saving data

    Returns:

    """

    # remove batch dimension
    # data is of shape (...,batchSize,embeddingSize)
    dataShape = dataBuffer[0][1].shape
    newDataShape = (-1, dataShape[-1])  # (...,embeddingSize)

    codeShape = dataBuffer[0][0].shape
    newCodeShape = (-1, codeShape[-1])

    codesArray = np.array(
        [x[0].squeeze().cpu().numpy().reshape(newCodeShape) for x in dataBuffer],
        dtype=np.float32,
    )
    dataArray = np.array(
        [x[1].cpu().numpy().reshape(newDataShape) for x in dataBuffer], dtype=np.float32
    )
    targetArray = np.array(
        [x[2].cpu().numpy().reshape((-1, 1)) for x in dataBuffer], dtype=np.float32
    )
    np.savez_compressed(filename, codes=codesArray, data=dataArray, targets=targetArray)
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
        expName = "continual_"
    else:
        expName = dataList[0] + "_"

    expName += "s"

    return expName


def saveContext(
    name,
    koh,
    classifier,
    previousClassifiers,
    previousTestData,
    previousDecoderDataSize,
    datasetIdx,
    epoch,
):
    context = {
        "koh": koh.state_dict(),
        "classifier": classifier.state_dict(),
        "pClassifiers": [p.state_dict() for p in previousClassifiers],
        "pTestData": previousTestData,
        "pDecoderDataSize": previousDecoderDataSize,
        "datasetIdx": datasetIdx,
        "epoch": epoch,
        "runID": mlflow.active_run(),
    }
    torch.save(context, f"../data/{name}_context.pt")
    return


def loadContext(name):
    filename = f"../data/{name}_context.pth"
    context = None
    if os.path.exists(filename):
        logger.info("Found existing context file. Loading context and resuming...")
        context = torch.load(filename)
    return context


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
    extraUnits = _asList(params["training"]["extraUnits"], nDatasets)  # type: List[int]

    dataShape = params["training"]["inputShape"]
    dataSize = dataShape[-2:]
    dataChannels = dataShape[1]

    fullLossFactor = _asList(params["training"]["fullLossCoeff"], nDatasets)
    l1LossFactor = _asList(params["training"]["l1_residuCoeff"], nDatasets)
    l2LossFactor = _asList(params["training"]["l2_residuCoeff"], nDatasets)
    learningRate = _asList(params["training"]["learningRate"], nDatasets)

    expName = _genExpName(dataList)
    experiment = mlflow.get_experiment_by_name(expName)
    if experiment is None:
        logger.info("Creating new experiment")
        expID = mlflow.create_experiment(expName)
    else:
        logger.info(f"Using existing experiment")
        expID = experiment.experiment_id

    contextName = params["name"]
    context = loadContext(contextName)
    resuming = False
    if not context is None:
        resuming = True
        runID = context["runID"]
    else:
        runID = None

    with mlflow.start_run(run_id=runID, experiment_id=expID):
        modelName = params["name"]
        if not resuming:
            mlflow.log_param("params", params)
            mlflow.log_param("name", modelName)

        hrnet = HashRoutedNetwork(modelName, params, device).to(device)
        if resuming:
            hrnet.load_state_dict(context["hrnet"])

        logDir = "../data/logs/" + modelName
        tbWriter = SummaryWriter(logDir)

        if resuming:
            previousTestData = context["pTestData"]
            previousDecoderDataSize = context["pDecoderDataSize"]
            previousClassifiers = []
            for idx, classifierState in enumerate(context["pClassifiers"]):
                oldClassifier = Decoder(
                    _asList(params["routing"]["decoder"], nDatasets)[idx],
                    params["routing"]["embeddingSize"],
                    params["routing"]["basisSize"] * hrnet.NUnits,
                    params["training"]["inputShape"],
                    device,
                    False,
                ).to(device)
                oldClassifier.load_state_dict(classifierState)
                previousClassifiers.append(oldClassifier)
        else:
            previousTestData = []
            previousClassifiers = []
            previousDecoderDataSize = []

        for datasetIdx, dataset in enumerate(dataList):
            if resuming and datasetIdx != context["datasetIdx"]:
                continue

            fullBasisSize = params["routing"]["basisSize"] * hrnet.NUnits
            embeddingSize = params["routing"]["embeddingSize"]

            if USE_PROJ:
                decoderDataSize = 2 * embeddingSize  # + fullBasisSize
            else:
                decoderDataSize = embeddingSize  # 2*embeddingSize #+ fullBasisSize

            # one decoder per dataset
            classifier = Decoder(
                _asList(params["routing"]["decoder"], nDatasets)[datasetIdx],
                embeddingSize,
                fullBasisSize,
                params["training"]["inputShape"],
                device,
                False,
            ).to(device)
            if resuming:
                classifier.load_state_dict(context["classifier"])

            logger.info(f"\n\t==== {dataset}: TRAINING ====\n")
            train_loader, test_loader = getDatasets(
                dataset, batchSize, dataSize, dataChannels
            )

            optimParams = chain(hrnet.parameters(), classifier.parameters())
            optimizer = optim.Adam(optimParams, lr=learningRate[datasetIdx])

            currentAcc = 0
            currentCorrect = 0
            currentTotalSize = len(test_loader.dataset)
            for epoch in range(1, epochs[datasetIdx] + 1):
                if resuming and epoch != context["epoch"]:
                    continue
                try:
                    train(
                        hrnet,
                        classifier,
                        device,
                        train_loader,
                        optimizer,
                        epoch,
                        fullLossFactor[datasetIdx],
                        l1LossFactor[datasetIdx],
                        l2LossFactor[datasetIdx],
                        tbWriter,
                    )
                    logger.info(f"\n\t==== {dataset}: TEST ({dataset}) ====\n")
                    currentAcc, currentCorrect = test(
                        hrnet,
                        classifier,
                        device,
                        test_loader,
                        len(train_loader),
                        epoch,
                        fullLossFactor[datasetIdx],
                        l1LossFactor[datasetIdx],
                        l2LossFactor[datasetIdx],
                        tbWriter,
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.warning(
                            f"Runtime error:\n{e}\nSaving models, deleting context and "
                            f"retrying..."
                        )
                        saveContext(
                            contextName,
                            hrnet,
                            classifier,
                            previousClassifiers,
                            previousTestData,
                            previousDecoderDataSize,
                            datasetIdx,
                            epoch,
                        )
                    else:
                        raise
                    exit(-1)

            saveModel(hrnet, f"../data/{modelName}_{dataset}.pt")
            saveModel(classifier, f"../data/{modelName}_{dataset}_classifier.pt")

            # trainBuffer = encodeDataset(hrnet, device, train_loader)
            # testBuffer = encodeDataset(hrnet, device, test_loader)
            del train_loader
            # saveData(trainBuffer,f'../data/{modelName}
            # _{dataset}_trainEmbeddings.npz')
            # saveData(testBuffer,f'../data/{modelName}_{dataset}_testEmbeddings.npz')

            totalCorrect = currentCorrect
            totalDatasetSize = currentTotalSize
            for pIdx, pTestData in enumerate(previousTestData):
                previousDataName = dataList[pIdx]
                previousDecodDataSize = previousDecoderDataSize[pIdx]
                logger.info(
                    f"\n\t==== {dataset}: Lifelong TEST  ({previousDataName}) "
                    f"====\n"
                )
                testDatasetSize = len(pTestData.dataset)
                totalDatasetSize += testDatasetSize
                logger.info("Re-encoding test data")
                newEncodedTestData = encodeDataset(
                    hrnet, device, pTestData, previousDecodDataSize
                )
                # saveData(newEncodedTestData,
                #         f'../data/{modelName}_lifelong_{previousDataName}'
                #         f'_testEmbeddings.npz')
                pClassifier = previousClassifiers[pIdx].to(device)

                previousAcc, previousCorrect = testClassifier(
                    pClassifier,
                    newEncodedTestData,
                    testDatasetSize,
                    epochs[datasetIdx],
                    device,
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
            previousDecoderDataSize.append(decoderDataSize)

            if datasetIdx < len(dataList) - 1:
                # +1 because the extra units list starts with a 0
                nNewUnits = extraUnits[datasetIdx + 1]
                logger.info(f"Adding {nNewUnits} unit to network")
                previousKoh = hrnet.cpu()
                hrnet = previousKoh.addRndNetworkUnits(nNewUnits)
                del previousKoh
                hrnet = hrnet.to(device)

        mlflow.log_artifacts(logDir, artifact_path="events")

    if resuming:
        os.remove(f"../data/{contextName}_context.pt")

    return currentAcc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generic trainer")
    parser.add_argument(
        "configuration", nargs="?", type=str, default=None, help="Configuration file"
    )
    parser.add_argument(
        "device", type=str, help="Device to use", default="cpu", choices=["cpu", "cuda"]
    )
    parser.add_argument("-d", type=int, help="Cuda device index", default=0)

    args = parser.parse_args()
    paramsFile = args.configuration
    devParam = args.device
    subDeviceIdx = args.d

    run({"paramsFile": paramsFile, "device": devParam, "subDeviceIdx": subDeviceIdx})
