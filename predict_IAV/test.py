import argparse
import csv
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from comet_ml import Experiment
from loguru import logger
from monai.metrics import MAEMetric, MSEMetric
from omegaconf import OmegaConf

sys.path.append("../")
sys.path.append("../../")

from dataloader import get_eval_dloaders

from networks import FlexibleMultiTaskModel


def main(config):
    # Set random seeds.
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    os.environ["PYTHONHASHSEED"] = str(config.seed)

    # Set image size based on the model architecture.
    # All models except Swin-V2-T use 224x224 images.
    # Swin-V2-T uses 256x256 images.
    if config.base == "swin_v2_t":
        img_size = (256, 256)
    else:
        img_size = (224, 224)

    # Load test dataloader
    dloaders_test = get_eval_dloaders(
        img_dir=config.img_dir,
        file_list=Path(config.test_file_list),
        metric=config.metric,
        batch_size=config.eval_batch_size,
        num_workers=config.eval_num_workers,
        img_size=img_size,
    )

    # Initialize the model.
    model = FlexibleMultiTaskModel(
        base=config.base,
        num_classes=config.num_classes,
        mode="regression",
        pretrained=config.pretrained,
        dropout_rate=config.dropout_rate,
    )

    # Create the experiment name.
    experiment_name = (
        f"{config.base}_{config.loss}_{config.train_batch_size}_"
        f"{config.epochs}_{config.lr}_{config.seed}"
    )

    # Configure logging.
    logger.remove()
    logger.add(sys.stdout, level="INFO")

    # Create the logs directory.
    logs_dir = Path(config.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"{experiment_name}_test.log"
    logger.add(log_file)

    # Create the experiment directory.
    expt_dir = Path(config.expt_dir) / experiment_name
    expt_dir.mkdir(parents=True, exist_ok=True)

    # Initialize the accelerator.
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        rng_types=["torch", "cuda", "generator"],
    )

    # Prepare the model and dataloader.
    model, dloaders_test = accelerator.prepare(model, dloaders_test)

    # Load the best model weights
    best_model_path = expt_dir / "best_model.pth"
    if not best_model_path.exists():
        raise FileNotFoundError(
            f"Best model not found at {best_model_path}. "
            "Please run training first to generate the best model."
        )

    logger.info(f"Loading best model from {best_model_path}")
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.load_state_dict(
        torch.load(best_model_path, map_location="cpu")
    )
    model.eval()

    # Initialize the experiment.
    accelerator.init_trackers(
        project_name="skin-iav-prediction",
        config=config,
    )

    experiment = accelerator.get_tracker("comet_ml").tracker
    experiment.set_name(f"{experiment_name}_test")
    experiment.log_parameters(config)

    # Perform testing.
    with experiment.test():
        with torch.no_grad():
            test_mae_meter, test_mse_meter = MAEMetric(), MSEMetric()

            csv_path = expt_dir / "test_results.csv"
            with open(csv_path, "wt", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["image_id", "true_metric", "pred_metric"])

                for i, data in enumerate(dloaders_test, 0):
                    names_test, inputs_test, labels_test = data
                    labels_test = labels_test.unsqueeze(1)

                    # Forward pass.
                    pred_test = model(inputs_test)

                    # Update metrics.
                    test_mae_meter(pred_test, labels_test)
                    test_mse_meter(pred_test, labels_test)

                    # Write predictions to CSV.
                    responses = [
                        [
                            f"{names_test[idx]}",
                            f"{labels_test[idx].item()}",
                            # Clamp the predictions to the range [0, 1] to avoid
                            # any out-of-bounds values.
                            f"{torch.clamp(pred_test[idx], 0, 1).item()}",
                        ]
                        for idx in range(pred_test.shape[0])
                    ]
                    writer.writerows(responses)

            test_mae = test_mae_meter.aggregate().item()
            test_mse = test_mse_meter.aggregate().item()

            test_mae_meter.reset()
            test_mse_meter.reset()

            experiment.log_metrics(
                dic={
                    "test_mae": test_mae,
                    "test_mse": test_mse,
                },
            )

    logger.info("Testing completed.")
    logger.info(f"Test MAE: {test_mae:.4f}")
    logger.info(f"Test MSE: {test_mse:.4f}")
    logger.info(f"Results saved to: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="experiment configuration file",
    )
    parser.add_argument(
        "--base", type=str, required=True, help="base model name"
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    config.base = args.base

    main(config)
