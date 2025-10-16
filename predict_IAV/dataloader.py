import random
import os
from pathlib import Path
from PIL import Image
import pandas as pd
import torch
from torchvision.transforms import InterpolationMode
import torchvision.transforms as trF
import torchvision.transforms.functional as trF_func
import numpy as np
from typing import Union
from loguru import logger


class RandomRotate90(object):
    """
    Transform to rotate an image to a desired angle of rotation as specified.
    Inspired by: https://github.com/SaoYan/IPMI2019-AttnMel/blob/master/transforms.py
    """

    def __init__(self, expand=False, center=None):
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params():
        idx = random.randint(0, 3)
        angle = idx * 90
        return angle

    def __call__(self, image):
        angle = self.get_params()
        img = trF_func.rotate(
            img=image,
            angle=angle,
            expand=self.expand,
            center=self.center,
            interpolation=InterpolationMode.BILINEAR,
        )
        return img


class ISIC_MultiAnnot_pp(torch.utils.data.Dataset):
    """
    Dataset object to fetch image names, images, and agreement metrics from
    the ISIC_MultiAnnot_pp dataset.
    """

    def __init__(
        self,
        img_dir: os.PathLike,
        file_list: Union[os.PathLike, pd.DataFrame],
        metric: str,
        img_size: tuple[int, int] = (224, 224),
        transform=None,
    ):
        super(ISIC_MultiAnnot_pp, self).__init__()
        self.img_dir = Path(img_dir)
        self.metric = metric
        self.img_size = img_size
        self.transform = transform

        if isinstance(file_list, pd.DataFrame):
            self.metadata = file_list
        elif isinstance(file_list, os.PathLike):
            self.metadata = pd.read_csv(
                file_list, header="infer", sep=",", low_memory=False
            )
        else:
            raise ValueError(f"Invalid file list type: {type(file_list)}")

        ALLOWED_METRICS = [
            "dice_score_mean",
            "jaccard_score_mean",
            "hd_score_mean",
            "hd95_score_mean",
            "assd_score_mean",
            "nsd_score_mean",
            "bf1_score_mean",
        ]
        if metric not in ALLOWED_METRICS:
            raise ValueError(f"Metric {metric} not in {ALLOWED_METRICS}")

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_name, target_metric = row["image"], row[self.metric]

        img = Image.open(self.img_dir / img_name)

        if self.transform:
            img = self.transform(img)

        target_metric = torch.from_numpy(np.asarray(target_metric)).float()

        return img_name, img, target_metric

    def __len__(self):
        return len(self.metadata)


def get_transforms(eval: bool = True, img_size: tuple[int, int] = (224, 224)):
    # Fetch the mean and standard deviation intensities calculated from the
    # ISIC_MultiAnnot_pp dataset.
    mean, std = (0.7079, 0.5688, 0.5130), (0.1304, 0.1248, 0.1429)
    normalize = trF.Normalize(mean, std)

    if eval:
        transform = trF.Compose(
            [
                trF.Resize(img_size, interpolation=InterpolationMode.BICUBIC),
                trF.ToTensor(),
                normalize,
            ]
        )
    else:
        transform = trF.Compose(
            [
                trF.Resize(img_size, interpolation=InterpolationMode.BICUBIC),
                RandomRotate90(),
                trF.RandomHorizontalFlip(),
                trF.RandomVerticalFlip(),
                trF.RandomAutocontrast(),
                trF.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 0.1)),
                trF.RandomAdjustSharpness(sharpness_factor=2),
                trF.ToTensor(),
                normalize,
            ]
        )

    return transform


def _worker_init_fn(worker_id):
    """
    Sets the worker functions to use a seed for reproducibility.
    """
    torch_seed = torch.initial_seed()
    np_seed = torch_seed & 2**32 - 1
    random.seed(torch_seed)
    np.random.seed(np_seed)


def get_train_dloaders(
    img_dir: os.PathLike,
    file_list: Union[os.PathLike, pd.DataFrame],
    metric: str,
    batch_size: int,
    num_workers: int,
    img_size: tuple[int, int] = (224, 224),
):
    transforms = get_transforms(eval=False, img_size=img_size)

    trainset = ISIC_MultiAnnot_pp(
        img_dir=img_dir,
        file_list=file_list,
        metric=metric,
        transform=transforms,
    )
    # print(f"Train set size: {len(trainset)}")
    logger.info(f"Train set size: {len(trainset)}")

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=_worker_init_fn,
        drop_last=True,
    )

    return trainloader


def get_eval_dloaders(
    img_dir: os.PathLike,
    file_list: Union[os.PathLike, pd.DataFrame],
    metric: str,
    batch_size: int,
    num_workers: int,
    img_size: tuple[int, int] = (224, 224),
):
    transforms = get_transforms(eval=True, img_size=img_size)

    evalset = ISIC_MultiAnnot_pp(
        img_dir=img_dir,
        file_list=file_list,
        metric=metric,
        transform=transforms,
    )

    evalloader = torch.utils.data.DataLoader(
        evalset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    return evalloader
