"""
Visual analysis tools
"""

from typing import Union
import os
import torch
import mlflow
import tempfile
import matplotlib.pyplot as plt
from numpy import sqrt, ceil
from torchvision.utils import make_grid, save_image
from torch.utils.tensorboard.writer import SummaryWriter


def addEmbedding(
    writer: SummaryWriter,
    codes: torch.Tensor,
    tag: str,
    step: int,
    label: torch.Tensor = None,
    labelImg: torch.Tensor = None,
):

    writer.add_embedding(
        codes,
        metadata=label.cpu().numpy(),
        label_img=labelImg,
        tag=tag,
        global_step=step,
    )
    return


def addGrayscaleGrid(
    inputImages: torch.Tensor,
    tag: str,
    step: int,
    writer: SummaryWriter = None,
    mlflowFile: str = None,
):
    images = list(inputImages.cpu().unbind(dim=0))
    displayImages = [torch.cat((img, img, img)) for img in images]
    nImages = len(images)
    rows = int(ceil(sqrt(nImages)))

    if not writer is None:
        grid = make_grid(displayImages, scale_each=True, normalize=True, nrow=rows)
        writer.add_image(tag, grid, step)
    if not mlflowFile is None:
        with tempfile.NamedTemporaryFile(prefix=mlflowFile + "_", suffix=".png") as f:

            save_image(
                displayImages, f.name, scale_each=True, normalize=True, nrow=rows
            )
            mlflow.log_artifact(f.name)

    return


def addColorGrid(
    inputImages: torch.Tensor,
    tag: str,
    step: int,
    writer: SummaryWriter = None,
    mlflowFile=None,
):
    images = list(inputImages.cpu().unbind(dim=0))
    nImages = len(images)
    rows = int(ceil(sqrt(nImages)))
    if not writer is None:
        grid = make_grid(images, scale_each=True, normalize=True, nrow=rows)
        writer.add_image(tag, grid, step)
    if not mlflowFile is None:
        with tempfile.NamedTemporaryFile(prefix=mlflowFile + "_", suffix=".png") as f:

            save_image(images, f.name, scale_each=True, normalize=True, nrow=rows)
            mlflow.log_artifact(f.name)

    return


def getTensorList(t: torch.Tensor) -> list:
    return t.cpu().unbind()


def genTitle(code, label=None):
    if label is None:
        s = f"C: {code.numpy().tolist()}"
    else:
        s = f"L: {label.numpy()}\nC: {code.numpy().tolist()}"
    return s


def addCodesGrid(
    inputImages: Union[torch.Tensor, list],
    codes: Union[torch.Tensor, list],
    tag: str,
    step: int,
    labels: Union[torch.Tensor, list] = None,
    writer: SummaryWriter = None,
    mlflowFile: str = None,
):

    images = getTensorList(inputImages)
    fCodes = getTensorList(codes)
    nImages = len(images)
    rows = int(ceil(sqrt(nImages)))
    if not labels is None:
        _labels = getTensorList(labels.cpu())
    else:
        _labels = None

    gridSize = rows ** 2
    figure = plt.figure(figsize=(10, 10))
    for imgIdx, img in enumerate(images):
        fCode = fCodes[imgIdx]
        if not _labels is None:
            label = _labels[imgIdx]
        else:
            label = None
        title = genTitle(fCode, label)
        plt.subplots_adjust(wspace=1.0, hspace=1.5)
        plt.subplot(rows, rows, imgIdx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        pltImg = img.permute(1, 2, 0).squeeze().numpy().clip(0, 1)
        if img.shape[0] == 3:
            plt.imshow(pltImg)
        else:
            plt.imshow(pltImg, cmap=plt.cm.binary)
        if imgIdx == gridSize:
            break

    # mlflow takes precendence over TB writer. Both cannot be used
    if not writer is None and mlflowFile is None:
        writer.add_figure(tag, figure, step)

    if not mlflowFile is None:
        with tempfile.NamedTemporaryFile(prefix=mlflowFile + "_", suffix=".png") as f:
            plt.savefig(f.name, bbox_inches="tight", format="png")
            os.sync()
            mlflow.log_artifact(f.name)
    plt.close(figure)
    return
