import os
import random
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as trF
from loguru import logger
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms import InterpolationMode
from utils.custom_transforms import RandomRotate90


class ISIC_MultiAnnot_pp_MT(torch.utils.data.Dataset):
    """
    Dataset object to fetch image names, images, agreement metrics, and
    malignancy from the ISIC_MultiAnnot_pp dataset.
    The agreement metric is "dice_score_mean" by default. The dataset is
    used for multitask learning (MT).
    """

    def __init__(
        self,
        img_dir: os.PathLike,
        file_list: Union[os.PathLike, pd.DataFrame],
        transform=None,
    ):
        super(ISIC_MultiAnnot_pp_MT, self).__init__()
        self.img_dir = Path(img_dir)
        self.metric = "dice_score_mean"
        self.transform = transform

        if not isinstance(file_list, (os.PathLike, pd.DataFrame)):
            raise ValueError(f"Invalid file list type: {type(file_list)}")

        if isinstance(file_list, os.PathLike):
            self.metadata = pd.read_csv(
                file_list, header="infer", sep=",", low_memory=False
            )
        else:  # It's a DataFrame
            self.metadata = file_list

        # Filter metadata to only include rows where malignancy is "benign"
        # or "malignant".
        self.metadata = self.metadata[
            self.metadata["malignancy"].isin(["benign", "malignant"])
        ]

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_name, malignancy, target_metric = (
            row["image"],
            row["malignancy"],
            row[self.metric],
        )

        img = Image.open(self.img_dir / img_name)

        if self.transform:
            img = self.transform(img)

        malignancy = int(1) if malignancy == "malignant" else int(0)

        return img_name, img, malignancy, target_metric

    def __len__(self):
        return len(self.metadata)


class ISIC_2018_2019(torch.utils.data.Dataset):
    """
    Dataset object to fetch image names, images, agreement metrics, and
    diagnosis labels from the ISIC_2018 and ISIC_2019 datasets.
    """

    def __init__(
        self,
        img_dir: os.PathLike,
        file_list: Union[os.PathLike, pd.DataFrame],
        transform=None,
    ):
        super(ISIC_2018_2019, self).__init__()
        self.img_dir = Path(img_dir)
        self.transform = transform

        if not isinstance(file_list, (os.PathLike, pd.DataFrame)):
            raise ValueError(f"Invalid file list type: {type(file_list)}")

        if isinstance(file_list, os.PathLike):
            # If the file list is a path, read it as a CSV file.
            self.metadata = pd.read_csv(
                file_list, header="infer", sep=",", low_memory=False
            )
        else:  # It's a DataFrame
            # If the file list is a DataFrame, use it directly.
            self.metadata = file_list

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        # The column name for the diagnosis label is `diag_label`.
        img_name, diagnosis = row["image"], row["diag_label"]

        img = Image.open(self.img_dir / f"{img_name}.jpg")

        if self.transform:
            img = self.transform(img)

        # Map "MEL" to 1 and "NON_MEL" to 0.
        diagnosis = int(1) if diagnosis == "MEL" else int(0)

        return img_name, img, diagnosis

    def __len__(self):
        return len(self.metadata)


class derm7pt_PH2(torch.utils.data.Dataset):
    """
    Dataset object to fetch image names, images, and diagnosis labels from the
    derm7pt and the PH2 datasets.
    """

    def __init__(
        self,
        img_dir: os.PathLike,
        file_list: Union[os.PathLike, pd.DataFrame],
        dataset_name: str,
        modality: str,
        transform=None,
    ):
        super(derm7pt_PH2, self).__init__()
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.dataset_name = dataset_name

        # Check if the dataset name is valid.
        assert dataset_name in ["derm7pt", "PH2"], (
            f"Invalid dataset name: {dataset_name}"
        )

        # If derm7pt, check if modality is specified and is valid.
        if dataset_name == "derm7pt":
            assert modality in ["derm", "clinic"], (
                f"Invalid modality: {modality}"
            )

        if dataset_name == "PH2":
            self.image_column_name = "image_path"
        else:
            self.image_column_name = "derm" if modality == "derm" else "clinic"

        if not isinstance(file_list, (os.PathLike, pd.DataFrame)):
            raise ValueError(f"Invalid file list type: {type(file_list)}")

        if isinstance(file_list, os.PathLike):
            self.metadata = pd.read_csv(
                file_list, header="infer", sep=",", low_memory=False
            )
        else:
            self.metadata = file_list

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_name, diagnosis = (
            row[self.image_column_name],
            row["diag_label_binary"],
        )

        img = Image.open(self.img_dir / f"{img_name}")

        if self.transform:
            img = self.transform(img)

        diagnosis = int(1) if diagnosis == "MEL" else int(0)

        return img_name, img, diagnosis

    def __len__(self):
        return len(self.metadata)


def get_transforms(
    eval: bool = True,
    custom_mean_std: Optional[
        tuple[tuple[float, float, float], tuple[float, float, float]]
    ] = None,
    img_size: tuple[int, int] = (224, 224),
):
    # Fetch the mean and standard deviation intensities calculated from the
    # ISIC_MultiAnnot_pp dataset.
    if custom_mean_std is None:
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    else:
        mean, std = custom_mean_std
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


def get_dloaders(config: OmegaConf, img_size: tuple[int, int] = (224, 224)):
    # Check if the dataset name is valid.
    assert config.dset_name in [
        "ISIC2018",
        "ISIC2019",
        "PH2",
        "derm7pt",
        "IMApp",
    ], f"Invalid dataset name: {config.dset_name}"

    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    if config.dset_name == "ISIC2018":
        mean, std = imagenet_mean, imagenet_std
        num_aug = 1
        focal_loss_weights = torch.tensor([0.5625, 4.4991])
    elif config.dset_name == "ISIC2019":
        mean, std = imagenet_mean, imagenet_std
        num_aug = 1
        focal_loss_weights = torch.tensor([0.6087, 2.8009])
    elif config.dset_name == "PH2":
        mean, std = imagenet_mean, imagenet_std
        num_aug = 25
        focal_loss_weights = torch.tensor([0.6250, 2.5000])
    elif config.dset_name == "derm7pt":
        mean, std = imagenet_mean, imagenet_std
        num_aug = 3
        focal_loss_weights = torch.tensor([0.6120, 2.7315])
    elif config.dset_name == "IMApp":
        mean, std = (0.7079, 0.5688, 0.5130), (0.1304, 0.1248, 0.1429)
        num_aug = 3
        focal_loss_weights = torch.tensor([0.6340, 2.3663])

    train_transform = get_transforms(
        eval=False,
        custom_mean_std=(mean, std),
        img_size=img_size,
    )

    eval_transform = get_transforms(
        eval=True,
        custom_mean_std=(mean, std),
        img_size=img_size,
    )

    if config.dset_name == "IMApp":
        trainset = ISIC_MultiAnnot_pp_MT(
            img_dir=config.train_img_dir,
            file_list=Path(config.train_file_list),
            transform=train_transform,
        )
        logger.info(f"Train set size: {len(trainset)}")

        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=config.train_batch_size,
            shuffle=True,
            num_workers=config.train_num_workers,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
        )

        valset = ISIC_MultiAnnot_pp_MT(
            img_dir=config.val_img_dir,
            file_list=Path(config.val_file_list),
            transform=eval_transform,
        )
        logger.info(f"Val set size: {len(valset)}")

        valloader = torch.utils.data.DataLoader(
            valset,
            batch_size=config.eval_batch_size,
            shuffle=False,
            num_workers=config.eval_num_workers,
            drop_last=False,
        )

        testset = ISIC_MultiAnnot_pp_MT(
            img_dir=config.test_img_dir,
            file_list=Path(config.test_file_list),
            transform=eval_transform,
        )
        logger.info(f"Test set size: {len(testset)}")

        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=config.eval_batch_size,
            shuffle=False,
            num_workers=config.eval_num_workers,
            drop_last=False,
        )

        return num_aug, focal_loss_weights, trainloader, valloader, testloader

    elif config.dset_name in ["ISIC2018", "ISIC2019"]:
        trainset = ISIC_2018_2019(
            img_dir=config.train_img_dir,
            file_list=Path(config.train_file_list),
            transform=train_transform,
        )
        logger.info(f"Train set size: {len(trainset)}")

        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=config.train_batch_size,
            shuffle=True,
            num_workers=config.train_num_workers,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
        )

        valset = ISIC_2018_2019(
            img_dir=config.val_img_dir,
            file_list=Path(config.val_file_list),
            transform=eval_transform,
        )
        logger.info(f"Val set size: {len(valset)}")

        valloader = torch.utils.data.DataLoader(
            valset,
            batch_size=config.eval_batch_size,
            shuffle=False,
            num_workers=config.eval_num_workers,
            drop_last=False,
        )

        testset = ISIC_2018_2019(
            img_dir=config.test_img_dir,
            file_list=Path(config.test_file_list),
            transform=eval_transform,
        )
        logger.info(f"Test set size: {len(testset)}")

        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=config.eval_batch_size,
            shuffle=False,
            num_workers=config.eval_num_workers,
            drop_last=False,
        )

        return num_aug, focal_loss_weights, trainloader, valloader, testloader

    elif config.dset_name in ["derm7pt", "PH2"]:
        trainset = derm7pt_PH2(
            img_dir=config.train_img_dir,
            file_list=Path(config.train_file_list),
            dataset_name=config.dset_name,
            modality="derm",
            transform=train_transform,
        )
        logger.info(f"Train set size: {len(trainset)}")

        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=config.train_batch_size,
            shuffle=True,
            num_workers=config.train_num_workers,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
        )

        valset = derm7pt_PH2(
            img_dir=config.val_img_dir,
            file_list=Path(config.val_file_list),
            dataset_name=config.dset_name,
            modality="derm",
            transform=eval_transform,
        )
        logger.info(f"Val set size: {len(valset)}")

        valloader = torch.utils.data.DataLoader(
            valset,
            batch_size=config.eval_batch_size,
            shuffle=False,
            num_workers=config.eval_num_workers,
            drop_last=False,
        )

        testset = derm7pt_PH2(
            img_dir=config.test_img_dir,
            file_list=Path(config.test_file_list),
            dataset_name=config.dset_name,
            modality="derm",
            transform=eval_transform,
        )
        logger.info(f"Test set size: {len(testset)}")

        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=config.eval_batch_size,
            shuffle=False,
            num_workers=config.eval_num_workers,
            drop_last=False,
        )

        return num_aug, focal_loss_weights, trainloader, valloader, testloader
