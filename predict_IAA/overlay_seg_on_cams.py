"""
Script to overlay multiple segmentation masks as contours with different
colors on CAM images.

The script expects the following:

- The CAM images are stored in the VISUALIZATION_OUTPUT_DIR directory (see
  cam_visualization.py for more details).
- The CAM images have the format "M_ISIC_0000112_0.88_0.89.png" (see
  cam_visualization.py for more details).
- The segmentation masks have the format "ISIC_0000112_*.png".
- The segmentation masks are all stored in the SEGS_DIR directory, without any
  subdirectories.
"""

from pathlib import Path

import cv2
import numpy as np
from cam_visualization import CAM_ALGORITHM, VISUALIZATION_OUTPUT_DIR
from skimage import segmentation
from tqdm import tqdm

# Import the necessary pytorch-grad-cam components
# Note: These are not directly used in this script but are kept for context
# from pytorch_grad_cam import (  # EigenCAM,; EigenGradCAM,; GradCAMElementWise,; HiResCAM,
#     GradCAMPlusPlus,
# )

SEGS_DIR = Path("/path/to/segmentation_masks/")

# The CAM_ALGORITHM and VISUALIZATION_OUTPUT_DIR are used as defined in the
# cam_visualization.py script.
# To get the overlay output directory, we append the CAM_ALGORITHM name with
# "_overlay" to the VISUALIZATION_OUTPUT_DIR path.
OVERLAY_OUTPUT_DIR = VISUALIZATION_OUTPUT_DIR.replace(
    CAM_ALGORITHM.__name__, f"{CAM_ALGORITHM.__name__}_overlay"
)


def parse_cam_filename(cam_path: Path) -> tuple[str, str, str]:
    """Parses a CAM filename to extract key components.

    The filename format is expected to be "M_ISIC_0000112_0.88_0.89.png".

    Args:
        cam_path: The path to the CAM image file.

    Returns:
        A tuple containing:
        - The ISIC ID (e.g., "ISIC_0000112").
        - The base filename part for reconstruction.
        - The prefix for the new filename (e.g., "M").
    """
    parts = cam_path.stem.split("_")
    isic_id = f"{parts[1]}_{parts[2]}"
    base_filename = "_".join(parts[3:])
    prefix = parts[0]
    return isic_id, base_filename, prefix


def overlay_segmentations(
    image: np.ndarray, seg_paths: list[Path], colors: list[tuple]
) -> np.ndarray:
    """Overlays multiple segmentation mask boundaries onto a base image.

    To ensure that contours along the image edges are visible, the image and
    masks are padded before marking boundaries and then cropped back.

    Args:
        image: The base image (in RGB float32 format).
        seg_paths: A list of paths to the binary segmentation masks.
        colors: A list of RGB tuples for the contour colors.

    Returns:
        The image with segmentation boundaries overlaid.
    """
    # Pad image to make edge contours visible. 'edge' mode avoids black borders.
    padded_image = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode="edge")
    result_image = padded_image

    for i, seg_path in enumerate(seg_paths):
        mask = cv2.imread(str(seg_path), cv2.IMREAD_GRAYSCALE)
        binary_mask = (mask > 127).astype(int)  # Ensure binary {0, 1}

        # Pad the mask to match the padded image dimensions.
        padded_mask = np.pad(
            binary_mask, 1, mode="constant", constant_values=0
        )

        color = colors[i % len(colors)]
        result_image = segmentation.mark_boundaries(
            result_image, padded_mask, color=color, mode="thick"
        )

    # Crop the image back to its original size to remove the padding.
    cropped_result = result_image[1:-1, 1:-1, :]
    return cropped_result


def process_image(cam_path: Path, colors: list[tuple]):
    """
    Processes a single CAM image: finds segmentations, overlays them, and saves.

    Args:
        cam_path: Path to the input CAM image.
        colors: List of colors for segmentation contours.
    """
    try:
        isic_id, base_filename, prefix = parse_cam_filename(cam_path)

        seg_paths = sorted(list(SEGS_DIR.glob(f"{isic_id}*.png")))
        num_segmentations = len(seg_paths)

        if num_segmentations < 2:
            print(
                f"Warning: Found {num_segmentations} masks for {isic_id}. Skipping."
            )
            return

        # Load the CAM image and convert to float RGB for scikit-image.
        cam_image_bgr = cv2.imread(str(cam_path))
        cam_image_rgb = cv2.cvtColor(cam_image_bgr, cv2.COLOR_BGR2RGB)
        cam_image_float = cam_image_rgb.astype(np.float32) / 255.0

        # Overlay all segmentations.
        result_image_float = overlay_segmentations(
            cam_image_float, seg_paths, colors
        )

        # Construct the new filename and save the result.
        new_filename = f"{prefix}_{num_segmentations}_{base_filename}.png"
        output_path = OVERLAY_OUTPUT_DIR / new_filename

        # Convert back to uint8 BGR for saving with OpenCV.
        result_image_uint8 = (result_image_float * 255).astype(np.uint8)
        result_image_bgr = cv2.cvtColor(result_image_uint8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), result_image_bgr)

    except Exception as e:
        print(f"Error processing {cam_path.name}: {e}")


def main():
    """
    Main function to orchestrate the overlay process.
    It finds all CAM images, and for each, it overlays available segmentation
    masks if they meet the criteria (>=2 masks).
    """
    # Create the output directory.
    OVERLAY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Define a list of 5 unique colors for the contours.
    COLORS = [
        (0, 0, 0),  # Black
        (1, 0, 1),  # Magenta
        (0, 0, 1),  # Blue
        (0, 1, 1),  # Cyan
        (1, 0.5, 0),  # Orange
    ]

    CAM_FILE_PATHS = list(VISUALIZATION_OUTPUT_DIR.glob("*.png"))
    print(
        f"Found {len(CAM_FILE_PATHS)} CAM images in {VISUALIZATION_OUTPUT_DIR}"
    )

    for cam_path in tqdm(CAM_FILE_PATHS, desc="Processing CAM images"):
        process_image(cam_path, COLORS)

    print("Processing complete.")


if __name__ == "__main__":
    main()
