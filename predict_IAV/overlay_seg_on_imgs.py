"""
Script to overlay multiple segmentation masks as contours with different
colors on skin lesion images.

The script expects the following:

- The images have the format "ISIC_0000112.JPG" or "ISIC_0000112.jpg".
- The segmentation masks have the format "ISIC_0000112_*.png".
- The segmentation masks are all stored in the SEGS_DIR directory, without any
  subdirectories.
- The images are stored in the IMGS_DIR directory.
- The output directory is the OVERLAY_OUTPUT_DIR directory.
"""

from pathlib import Path

import cv2
import numpy as np
from skimage import segmentation
from tqdm import tqdm

IMGS_DIR = Path("/path/to/images/")
SEGS_DIR = Path("/path/to/segmentation_masks/")

OVERLAY_OUTPUT_DIR = Path("/path/to/segmentation_masks_on_images_overlay/")


def parse_img_filename(img_path: Path) -> str:
    """Parses a skin lesion image filename to extract key components.

    The filename format is expected to be "ISIC_0000112.png".

    Args:
        img_path: The path to the image file.

    Returns:
        The ISIC ID (e.g., "ISIC_0000112").
    """
    return img_path.stem


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
        binary_mask = (mask > 127).astype(int)  # Ensure binary {0, 1}.

        # Pad the mask to match the padded image dimensions.
        padded_mask = np.pad(
            binary_mask, 1, mode="constant", constant_values=0
        )

        color = colors[i % len(colors)]
        result_image = segmentation.mark_boundaries(
            result_image, padded_mask, color=color, mode="thick"
        )

    # Crop the image back to its original size to remove the padding
    cropped_result = result_image[1:-1, 1:-1, :]
    return cropped_result


def process_image(img_path: Path, colors: list[tuple]):
    """
    Processes a single image: finds segmentations, overlays them, and saves.

    Args:
        img_path: Path to the input image.
        colors: List of colors for segmentation contours.
    """
    try:
        isic_id = parse_img_filename(img_path)

        seg_paths = sorted(list(SEGS_DIR.glob(f"{isic_id}*.png")))
        num_segmentations = len(seg_paths)

        if num_segmentations < 2:
            print(
                f"Warning: Found {num_segmentations} masks for {isic_id}. Skipping."
            )
            return

        # Load the image and convert to float RGB for scikit-image
        image_bgr = cv2.imread(str(img_path))
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_float = image_rgb.astype(np.float32) / 255.0

        # Overlay all segmentations
        result_image_float = overlay_segmentations(
            image_float, seg_paths, colors
        )

        # Construct the new filename and save the result
        new_filename = f"{isic_id}_{num_segmentations}.png"
        output_path = OVERLAY_OUTPUT_DIR / new_filename

        # Convert back to uint8 BGR for saving with OpenCV
        result_image_uint8 = (result_image_float * 255).astype(np.uint8)
        result_image_bgr = cv2.cvtColor(result_image_uint8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), result_image_bgr)

    except Exception as e:
        print(f"Error processing {img_path.name}: {e}")


def main():
    """
    Main function to orchestrate the overlay process.
    It finds all image files, and for each, it overlays available segmentation
    masks if they meet the criteria (>=2 masks).
    """
    # Create the output directory.
    OVERLAY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Define a list of 5 unique colors for the contours.
    COLORS = [
        # (1, 1, 0),  # Yellow
        # (1, 1, 1),  # White
        (0, 0, 0),  # Black
        (1, 0, 1),  # Magenta
        (0, 0, 1),  # Blue
        # (1, 0, 0),  # Red
        # (0, 1, 0),  # Green
        (0, 1, 1),  # Cyan
        (1, 0.5, 0),  # Orange
    ]

    # Find all images with .JPG or .jpg extension.
    IMG_FILE_PATHS = list(IMGS_DIR.glob("*.JPG")) + list(
        IMGS_DIR.glob("*.jpg")
    )
    print(f"Found {len(IMG_FILE_PATHS)} IMG images in {IMGS_DIR}")

    for img_path in tqdm(IMG_FILE_PATHS, desc="Processing images"):
        process_image(img_path, COLORS)

    print("Processing complete.")


if __name__ == "__main__":
    main()
