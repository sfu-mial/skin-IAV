"""
Prepares datasets for training and testing. To be used with
`dataset_configs.json`.
"""

import argparse
import json
from pathlib import Path
from shutil import copy2

import cv2
import pandas as pd
from utils import resize_image_set


def _get_dataset_config(dset_name, target_img_size):
    """Returns the configuration for a given dataset by reading it from a JSON file."""
    with open("./dataset_configs.json") as f:
        configs = json.load(f)

    cfg_template = configs[dset_name]

    # Create a format mapping for placeholder substitution.
    subs = {"target_img_size": str(target_img_size)}
    for key in ["base_dir", "processed_dir", "processed_metadata_dir"]:
        if key in cfg_template:
            subs[key] = cfg_template[key].format(**subs)

    # Recursively format strings and convert to Path objects.
    def format_and_pathify(data):
        if isinstance(data, dict):
            return {k: format_and_pathify(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [format_and_pathify(i) for i in data]
        elif isinstance(data, str):
            formatted_str = data.format(**subs)
            if "/" in formatted_str or "\\" in formatted_str:
                return Path(formatted_str)
            return formatted_str
        return data

    config = format_and_pathify(cfg_template)
    config["name"] = dset_name
    return config


def prepare_metadata(config):
    """
    Prepares metadata for a given dataset.
    """

    print(f"Preparing metadata for {config['name']}...")

    for split, split_config in config["splits"].items():
        if config["name"] == "PH2":
            # For PH2, we use the `diag_class_label` column to create the
            # diagnosis label.
            df = pd.read_csv(
                split_config["metadata_path"],
                header=None,
                sep=",",
                names=["image_path", "diag_class_label"],
                low_memory=False,
            )
            df["diag_label_binary"] = df["diag_class_label"].apply(
                lambda x: "MEL" if x == 1 else "NON_MEL"
            )
        else:
            # For other datasets, we read the metadata file directly.
            df = pd.read_csv(
                split_config["metadata_path"],
                header=0,
                sep=",",
                low_memory=False,
            )
            # For derm7pt, we check if the `diag_class_label` column contains
            # the expected labels.
            if config["name"] == "derm7pt":
                all_labels = config["non_mel_labels"] + ["MEL"]
                if set(all_labels) != set(df["diag_class_label"].unique()):
                    raise ValueError(
                        "diag_class_label column contains unexpected labels."
                    )
                # Set all "non-MEL" labels to "NON_MEL".
                df["diag_label_binary"] = df["diag_class_label"].apply(
                    lambda x: "MEL" if x == "MEL" else "NON_MEL"
                )
            elif config["name"] in ["ISIC2018", "ISIC2019"]:
                # For ISIC2018 and ISIC2019, we check if there are any images
                # with multiple labels, i.e., an image with both "MEL" and
                # "NON_MEL" labels.
                multi_diag_mel = df[
                    (df["MEL"] == 1.0)
                    & (df[config["non_mel_labels"]].sum(axis=1) > 0)
                ]
                if not multi_diag_mel.empty:
                    raise ValueError(
                        f"Found melanoma images with multiple diagnoses in {split} set."
                    )
                # Set all "non-MEL" labels to "NON_MEL".
                df["diag_label"] = df["MEL"].apply(
                    lambda x: "MEL" if x == 1.0 else "NON_MEL"
                )

        df.to_csv(config["processed_metadata_paths"][split], index=False)
        print(
            f"  - Saved {split} metadata to {config['processed_metadata_paths'][split]}"
        )


def prepare_images(config, target_size, interpolation):
    """
    Resizes and saves images for a given dataset.
    """

    print(f"Resizing images for {config['name']}...")

    # Get the image columns to process.
    image_cols = (
        config["image_col"]
        if isinstance(config["image_col"], list)
        else [config["image_col"]]
    )

    for col in image_cols:
        print(f"  - Processing column: {col}")
        for split, path in config["processed_metadata_paths"].items():
            source_dir = (
                config["splits"][split]["image_dir"]
                if "image_dir" in config["splits"][split]
                else config["source_image_dir"]
            )
            dest_dir = (
                config["splits"][split]["dest_image_dir"]
                if "dest_image_dir" in config["splits"][split]
                else config["dest_image_dir"]
            )
            resize_image_set(
                metadata_path=path,
                image_col=col,
                source_dir=source_dir,
                dest_dir=dest_dir,
                target_size=target_size,
                interpolation=interpolation,
                set_name=split,
                dset_name=config["name"],
            )


def main(args):
    """
    Main function to prepare a dataset.
    """

    # Get the dataset configuration.
    config = _get_dataset_config(args.dset_name, args.target_img_size)

    # Create all the directories.
    # Create the processed directory.
    config["processed_dir"].mkdir(parents=True, exist_ok=True)
    # Create the processed metadata directory.
    if "processed_metadata_dir" in config:
        config["processed_metadata_dir"].mkdir(exist_ok=True)
    # Create the destination image directory.
    if "dest_image_dir" in config:
        config["dest_image_dir"].mkdir(exist_ok=True)
    for split_config in config["splits"].values():
        # Create the destination image directory for each split.
        if "dest_image_dir" in split_config:
            split_config["dest_image_dir"].mkdir(exist_ok=True)
    # Create the local partitions directory.
    config["local_partitions_dir"].mkdir(parents=True, exist_ok=True)
    # Prepare metadata.
    prepare_metadata(config)
    # Prepare images.
    # Set the interpolation mode.
    if args.interpolation_mode == "bilinear":
        interpolation = cv2.INTER_LINEAR
    elif args.interpolation_mode == "bicubic":
        interpolation = cv2.INTER_CUBIC
    else:
        raise ValueError(
            f"Invalid interpolation mode: {args.interpolation_mode}"
        )

    # Prepare images.
    prepare_images(config, args.target_img_size, interpolation)

    # Copy metadata to local partitions.
    for split, processed_path in config["processed_metadata_paths"].items():
        copy2(processed_path, config["local_partitions_dir"] / f"{split}.csv")
    print(f"Copied metadata to {config['local_partitions_dir']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare skin lesion datasets for training and testing."
    )
    parser.add_argument(
        "dset_name",
        type=str,
        choices=["ISIC2018", "ISIC2019", "derm7pt", "PH2"],
        help="Name of the dataset to prepare.",
    )
    parser.add_argument(
        "--target_img_size",
        type=int,
        default=224,
        help="Target image size.",
    )
    parser.add_argument(
        "--interpolation_mode",
        type=str,
        default="bicubic",
        choices=["bilinear", "bicubic"],
        help="Interpolation mode for resizing.",
    )
    args = parser.parse_args()
    main(args)
