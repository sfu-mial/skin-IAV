import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from accelerate import Accelerator
from comet_ml import Experiment
from loguru import logger
from monai.metrics import MAEMetric, MSEMetric
from omegaconf import OmegaConf

sys.path.append("../")
sys.path.append("../../")

from dataloader import get_eval_dloaders, get_train_dloaders

from networks import FlexibleMultiTaskModel


def main(config):
    # Set random seeds.
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    os.environ["PYTHONHASHSEED"] = str(config.seed)

    # Specify the augmentation order. num_aug = 3 means the model is trained
    # 5 times on the dataset in each epoch.
    num_aug = 3

    # Set image size based on the model architecture.
    # All models except Swin-V2-T use 224x224 images.
    # Swin-V2-T uses 256x256 images.
    # https://docs.pytorch.org/vision/main/models/generated/torchvision.models.swin_v2_t.html
    if config.base == "swin_v2_t":
        img_size = (256, 256)
    else:
        img_size = (224, 224)

    dloaders_train, dloaders_val = (
        get_train_dloaders(
            img_dir=config.img_dir,
            file_list=Path(config.train_file_list),
            metric=config.metric,
            batch_size=config.train_batch_size,
            num_workers=config.train_num_workers,
            img_size=img_size,
        ),
        get_eval_dloaders(
            img_dir=config.img_dir,
            file_list=Path(config.val_file_list),
            metric=config.metric,
            batch_size=config.eval_batch_size,
            num_workers=config.eval_num_workers,
            img_size=img_size,
        ),
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
    log_file = logs_dir / f"{experiment_name}.log"
    logger.add(log_file)

    # Create the experiment directory.
    expt_dir = Path(config.expt_dir) / experiment_name
    expt_dir.mkdir(parents=True, exist_ok=True)

    # Initialize the accelerator.
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        rng_types=["torch", "cuda", "generator"],
        log_with="comet_ml",
    )

    accelerator.init_trackers(
        project_name="skin-iav-prediction",
        config=config,
    )

    experiment = accelerator.get_tracker("comet_ml").tracker
    experiment.set_name(experiment_name)
    experiment.log_parameters(config)

    if config.loss == "smooth_l1":
        criterion = nn.SmoothL1Loss()
    elif config.loss == "mse":
        criterion = nn.MSELoss()
    elif config.loss == "mae":
        criterion = nn.L1Loss()
    else:
        raise ValueError(f"Invalid loss function: {config.loss}")

    # Initialize the optimizer.
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True,
    )

    # Initialize the scheduler.
    lr_lambda = lambda epoch: np.power(0.1, epoch // 10)

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Prepare the model, optimizer, and dataloaders.
    model, optimizer, dloaders_train, dloaders_val = accelerator.prepare(
        model,
        optimizer,
        dloaders_train,
        dloaders_val,
    )

    # Start model training.
    step, best_valid_MAE, best_valid_epoch = 0, float("inf"), 0

    # Iterate over the number of epochs.
    for epoch in range(config.epochs):
        # Keep track of learning rate.
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch {epoch + 1}/{config.epochs} - Learning rate: {current_lr}"
        )

        with experiment.train():
            train_mae_meter, train_mse_meter = MAEMetric(), MSEMetric()
            epoch_train_loss = 0.0
            num_train_batches = 0
            # One training epoch.
            for aug in range(num_aug):
                # Fetch data from the training dataloader.
                for i, data in enumerate(dloaders_train, 0):
                    model.train()
                    model.zero_grad()
                    optimizer.zero_grad()

                    _, inputs, labels = data
                    labels = labels.unsqueeze(1)

                    # Forward pass.
                    pred = model(inputs)

                    # Compute the loss.
                    loss = criterion(pred, labels)
                    epoch_train_loss += loss.item()
                    num_train_batches += 1

                    # Backward pass.
                    with accelerator.autocast():
                        accelerator.backward(loss)
                    optimizer.step()

                    # Update metrics
                    train_mae_meter(pred, labels)
                    train_mse_meter(pred, labels)

                    # Update the step.
                    step += 1

            train_mae = train_mae_meter.aggregate().item()
            train_mse = train_mse_meter.aggregate().item()

            train_mae_meter.reset()
            train_mse_meter.reset()

            experiment.log_metrics(
                dic={
                    "train_loss": epoch_train_loss / num_train_batches,
                    "train_mae": train_mae,
                    "train_mse": train_mse,
                    "epoch": epoch,
                    "lr": current_lr,
                },
                epoch=epoch,
                step=step,
            )

            scheduler.step()

        # Perform validation.
        model.eval()

        with experiment.validate():
            with torch.no_grad():
                val_mae_meter, val_mse_meter = MAEMetric(), MSEMetric()

                for i, data in enumerate(dloaders_val, 0):
                    _, inputs_eval, labels_eval = data
                    labels_eval = labels_eval.unsqueeze(1)

                    # Forward pass.
                    pred_eval = model(inputs_eval)

                    val_mae_meter(pred_eval, labels_eval)
                    val_mse_meter(pred_eval, labels_eval)

                epoch_valid_mae = val_mae_meter.aggregate().item()
                epoch_valid_mse = val_mse_meter.aggregate().item()

                val_mae_meter.reset()
                val_mse_meter.reset()

                experiment.log_metrics(
                    dic={
                        "val_mae": epoch_valid_mae,
                        "val_mse": epoch_valid_mse,
                    },
                    epoch=epoch,
                )

                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)

                if epoch_valid_mae < best_valid_MAE:
                    best_valid_MAE = epoch_valid_mae
                    best_valid_epoch = epoch
                    # Save the best model
                    accelerator.save(
                        unwrapped_model.state_dict(),
                        expt_dir / "best_model.pth",
                    )

        logger.info(f"[epoch {epoch + 1}/{config.epochs}]")
        logger.info(f"MAE: {epoch_valid_mae:.4f}")
        logger.info(
            f"Best MAE: {best_valid_MAE:.4f} at epoch: {best_valid_epoch}"
        )

    logger.info(
        f"Training completed. Best model saved with MAE: {best_valid_MAE:.4f}"
    )


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
