"""
Script to visualize the predictions of the model on the test partition using
the specified CAM algorithm.
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

# Import the necessary pytorch-grad-cam components
from pytorch_grad_cam import (  # EigenCAM,; EigenGradCAM,; GradCAMElementWise,; HiResCAM,
    GradCAMPlusPlus,
)
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image
from tqdm import tqdm

# Assume FlexibleMultiTaskModel is defined in this file or imported from networks.py
sys.path.append("../")
from networks import FlexibleMultiTaskModel

# Define constants or absolute paths.

# Model weights.
# Note: We only visualize the predictions for the best-performing model, i.e.,
# resnet18.
# Change this to the model you want to visualize the predictions for.
MODEL_WEIGHTS_DIR = Path("./saved_models/resnet18/")
MODEL_WEIGHTS_FILE = MODEL_WEIGHTS_DIR / "MODEL_WEIGHTS_FILE.pth"
PREDICTIONS_CSV_PATH = MODEL_WEIGHTS_DIR / "test_results.csv"

# Image directory. This is the directory where the images are stored.
IMAGE_DIR = Path("/path/to/images/")

# Image metadata file. This is for the malignancy label.
IMAGE_METADATA_FILE = Path("/path/to/test_partition_metadata.csv")

# Specify the CAM algorithm.
CAM_ALGORITHM = GradCAMPlusPlus

# Visualization output directory.
VISUALIZATION_OUTPUT_DIR = f"./vis_output/{CAM_ALGORITHM.__name__}/"


if __name__ == "__main__":
    device = torch.device("cuda")

    # 1. Instantiate the Model and CAM Algorithm Once
    print("Loading model...")
    model = FlexibleMultiTaskModel(
        base="resnet18",
        num_classes=1,
        mode="regression",
        pretrained=True,
    )

    # Load the model weights.
    model.load_state_dict(torch.load(MODEL_WEIGHTS_FILE), strict=True)
    model.to(device)
    model.eval()

    target_layers = [model.backbone.layer4]

    # Ensuure that all the files are in place.
    if not MODEL_WEIGHTS_FILE.exists():
        raise FileNotFoundError(
            f"Model weights file not found at {MODEL_WEIGHTS_FILE}"
        )
    if not PREDICTIONS_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Predictions CSV file not found at {PREDICTIONS_CSV_PATH}"
        )
    if not IMAGE_METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Image metadata file not found at {IMAGE_METADATA_FILE}"
        )

    # Ensure the output directory exists. If not, create it.
    os.makedirs(VISUALIZATION_OUTPUT_DIR, exist_ok=True)

    # Combine the malignancy label from the image metadata file with the
    # predictions.
    PREDICTIONS_DF = pd.read_csv(
        PREDICTIONS_CSV_PATH, header="infer", sep=",", low_memory=False
    )
    IMAGE_METADATA_DF = pd.read_csv(
        IMAGE_METADATA_FILE, header="infer", sep=",", low_memory=False
    )
    PREDICTIONS_DF = PREDICTIONS_DF.merge(
        IMAGE_METADATA_DF, left_on="image_id", right_on="image", how="left"
    )

    for i, row in tqdm(PREDICTIONS_DF.iterrows(), total=len(PREDICTIONS_DF)):
        try:
            image_id = row["image_id"]
            true_metric = float(row["true_metric"])
            pred_metric = float(row["pred_metric"])
            malignant_label = row["malignancy"].lower()

            image_path = os.path.join(IMAGE_DIR, image_id)

            if not os.path.exists(image_path):
                print(f"Warning: Image not found at {image_path}. Skipping.")
                continue

            # Load and preprocess the image for the model
            rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
            rgb_img = np.float32(rgb_img) / 255
            input_tensor = preprocess_image(
                rgb_img,
                # Fetch the mean and standard deviation intensities calculated from the
                # ISIC_MultiAnnot_pp dataset.
                # See dataloader.py for more details.
                mean=[0.7079, 0.5688, 0.5130],
                std=[0.1304, 0.1248, 0.1429],
            ).to(device)

            # Generate CAM for the current image.
            with CAM_ALGORITHM(
                model=model, target_layers=target_layers
            ) as cam:
                grayscale_cam = cam(input_tensor=input_tensor, targets=None)
                grayscale_cam = grayscale_cam[0, :]

            # Construct the Custom Output Filename.
            image_name_no_ext = os.path.splitext(image_id)[0]
            TM = true_metric
            PM = pred_metric
            M = "M" if malignant_label == "malignant" else "B"

            # Format TM and PM to 2 decimal places for the filename.
            output_filename = f"{M}_{image_name_no_ext}_{TM:.4f}_{PM:.4f}.png"
            output_path = os.path.join(
                VISUALIZATION_OUTPUT_DIR, output_filename
            )

            # Create and save the final annotated image.
            cam_image = show_cam_on_image(
                rgb_img, grayscale_cam, use_rgb=True, image_weight=0.5
            )
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, cam_image)

        except Exception as e:
            print(
                f"Error processing row {i + 1} ({row.get('image_id', 'N/A')}): {e}"
            )

    print("Processing complete.")
