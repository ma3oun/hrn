"""
Model inference statistical analysis tool
"""

import os, argparse
import os.path as path
import logging, coloredlogs
import ruamel.yaml as yaml

import cv2
import pandas as pd

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from hrn.routing.network import HashRoutedNetwork
from hrn.analysis.visualization import genTitle, getTensorList

logger = logging.root
if logger.hasHandlers():
    logger.handlers.clear()
    logger.addHandler(logging.StreamHandler())

coloredlogs.install(
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.DEBUG,
)


class StatViewer(object):
    @staticmethod
    def _getCLIParams() -> tuple:
        parser = argparse.ArgumentParser(
            description="Lifelong learning model " "evaluation CLI"
        )

        parser.add_argument(
            "config", type=str, help="Model configuration file (YAML)", nargs=1
        )
        parser.add_argument("model", type=str, help="Model file (.pt or .pth)", nargs=1)
        parser.add_argument(
            "dataset",
            type=str,
            help="Dataset name",
            nargs=1,
            choices=["MNIST", "CIFAR10"],
        )
        parser.add_argument(
            "-d", type=str, help="device", choices=["cuda", "cpu"], default="cpu"
        )
        parser.add_argument("-i", type=int, help="cuda device index", default=0)

        defaultDatasetCacheDir = "/tmp/datasetCache"
        parser.add_argument(
            "-dataDir",
            type=str,
            help="Dataset cache directory",
            default=defaultDatasetCacheDir,
        )

        args = parser.parse_args()
        config = args.config[0]
        assert path.exists(config)
        modelFile = args.model[0]
        assert path.exists(modelFile)
        datasetName = args.dataset[0]
        device = args.d
        if device == "cuda":
            if not torch.cuda.is_available():
                logger.warning("Cuda not available on this device, using CPU instead!")
                device = "cpu"
            else:
                cudaIdx = args.i
                device = f"cuda: {cudaIdx}"
        dataDir = args.dataDir
        if dataDir == defaultDatasetCacheDir:
            os.makedirs(dataDir, exist_ok=True)
        else:
            assert path.exists(dataDir)
            logger.info(f"Using {dataDir} for dataset")
        return config, modelFile, datasetName, device, dataDir

    def loadModel(self, modelFile: str, config: str) -> nn.Module:
        with open(config, "r") as f:
            params = yaml.safe_load(f)
        params["training"]["batchSize"] = 1
        logger.info(f"Loading trained model from {modelFile}")
        modelName = params["name"]
        self.name = modelName
        model = HashRoutedNetwork(modelName, params, self.device).to(self.device)
        model.load_state_dict(torch.load(modelFile, map_location=self.device), False)
        model.eval()
        return model

    def loadDataset(self, datasetName: str) -> torch.utils.data.DataLoader:
        if datasetName == "MNIST":
            reader = torch.utils.data.DataLoader(
                datasets.MNIST(
                    self.dataDir,
                    train=False,
                    download=True,
                    transform=transforms.Compose(
                        [
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,)),
                        ]
                    ),
                ),
                batch_size=1,
                shuffle=False,
            )
        elif datasetName == "CIFAR10":
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            reader = torch.utils.data.DataLoader(
                datasets.CIFAR10(
                    self.dataDir, train=False, download=True, transform=transform
                ),
                batch_size=1,
                shuffle=False,
            )
        else:
            reader = None
            logger.exception(f"Unknown dataset: {datasetName}")
        return reader

    def viewer(self):
        with torch.no_grad():
            for dataIdx, (data, target) in enumerate(self.datasetReader):
                inData = data.to(self.device)
                outData, codes, _, _ = self.model(inData)
                imgData = (
                    inData.squeeze(dim=0)
                    .permute(1, 2, 0)
                    .squeeze()
                    .cpu()
                    .numpy()
                    .clip(0, 1)
                )
                title = f"{dataIdx}\n{genTitle(codes.squeeze(),target.squeeze())}"
                cv2.namedWindow(title, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(title, 500, 500)
                cv2.imshow(title, imgData)
                while True:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("n"):
                        cv2.destroyAllWindows()
                        break
                    if key == ord("e"):
                        cv2.destroyAllWindows()
                        return

        return

    def detailedStats(self) -> pd.DataFrame:
        df = pd.DataFrame()
        with torch.no_grad():
            for dataIdx, (data, target) in enumerate(self.datasetReader):
                logger.debug(f"{dataIdx:03d}")
                inData = data.to(self.device)
                outData, decoderData, codes, _, _ = self.model(
                    inData, fastInference=True
                )
                _codes = codes.squeeze().numpy().tolist()
                _target = target.numpy()[0]
                df = df.append([{_target: _codes}], ignore_index=True)
        logger.info(f"Dataframe extract:\n{df.head(30)}")
        df.to_csv(self.rawStatsPath)
        return df

    def codeStats(self):
        detailedDf = pd.read_csv(self.rawStatsPath)
        df = pd.DataFrame()
        for c in detailedDf.columns[1:]:
            df[c] = detailedDf[c].value_counts().sort_index()
        df.to_csv(self.codeStatsPath)
        return

    def __init__(self):
        config, modelFile, datasetName, device, dataDir = self._getCLIParams()
        self.name = None
        self.device = device
        self.dataDir = dataDir
        self.model = self.loadModel(modelFile, config)
        self.datasetReader = self.loadDataset(datasetName)
        self.rawStatsPath = path.join(self.dataDir, self.name + "_raw.csv")
        self.codeStatsPath = path.join(self.dataDir, self.name + "_codes.csv")

    def run(self):
        # self.viewer()
        self.detailedStats()
        self.codeStats()


if __name__ == "__main__":
    StatViewer().run()
