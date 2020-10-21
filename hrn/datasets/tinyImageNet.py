"""
TinyImageNet dataset
"""

import os
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets.folder import ImageFolder
from PIL import Image


class TinyImagenetVal(Dataset):
    def __init__(self, rootDir, transform=None):
        super(TinyImagenetVal, self).__init__()
        self.rootDir = rootDir
        self.transform = transform
        self.valDir = os.path.join(self.rootDir, "val")
        self.valImages = os.path.join(self.valDir, "images")
        self.nImages = len(
            [
                name
                for name in os.listdir(self.valImages)
                if os.path.isfile(os.path.join(self.valImages, name))
            ]
        )
        csv_file = os.path.join(self.valDir, "val_annotations.txt")
        self.annotations = pd.read_csv(csv_file, sep="\t", header=None)
        self.annotations.columns = [
            "image_name",
            "label",
            "col1",
            "col2",
            "col3",
            "col4",
        ]

        # SAME AS DatasetFolder._find_classes()
        self.classes = self.annotations["label"].unique()
        self.classes.sort()
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.targets = [
            self.class_to_idx[self.annotations.iloc[i, 1]] for i in range(self.nImages)
        ]
        pass

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, imgIdx):
        if torch.is_tensor(imgIdx):
            imgIdx = imgIdx.tolist()

        img_name = os.path.join(self.valImages, self.annotations.iloc[imgIdx, 0])
        image = Image.open(img_name).convert("RGB")
        # label_name = self.annotations.iloc[imgIdx, 1]
        # label = self.class_to_idx[label_name]
        label = self.targets[imgIdx]
        if self.transform:
            image = self.transform(image)
        return image, label


class TinyImageNet(Dataset):
    def __init__(self, train=True, rootDir=None, transform=None, test=False):
        super(TinyImageNet, self).__init__()
        datasetURL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        if rootDir is None:
            rootDir = os.path.abspath(os.path.join(os.getcwd(), "../data"))
        if not os.path.exists(os.path.join(rootDir, "../data/tiny-imagenet-200")):
            print(f"Downloading TinyImageNet data to {rootDir}")
            download_and_extract_archive(datasetURL, rootDir)
            print("...done")
        self.rootDir = os.path.abspath(os.path.join(rootDir, "tiny-imagenet-200"))
        self.train = train
        self.test = test
        self.transforms = transforms
        trainDataset = ImageFolder(os.path.join(self.rootDir, "train"), transform)
        testDataset = ImageFolder(os.path.join(self.rootDir, "test"), transform)
        validDataset = TinyImagenetVal(self.rootDir, transform)

        if not self.test:
            if self.train:
                self._dataset = trainDataset
            else:
                self._dataset = validDataset
            self.targets = self._dataset.targets
        else:
            self._dataset = testDataset
            self.targets = None

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, imgIdx):
        if torch.is_tensor(imgIdx):
            imgIdx = imgIdx.tolist()

        image, label = self._dataset.__getitem__(imgIdx)
        return image, label


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    print("Testing TinyImageNet loader")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    # normalization is not great for visualization
    # tfms = transforms.Compose([transforms.ToTensor(), normalize])
    tfms = transforms.Compose([transforms.ToTensor()])
    trainLoader = DataLoader(TinyImageNet(transform=tfms), batch_size=1, shuffle=True)
    for idx, data in enumerate(trainLoader):
        img, lbl = data
        plt.title(f"Class: {lbl.squeeze().item()}")
        plt.imshow(img.squeeze().permute(1, 2, 0))
        plt.show()

        if idx > 10:
            break
