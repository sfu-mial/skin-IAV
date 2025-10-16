import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import sklearn.metrics as sk_m
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from accelerate import Accelerator
from loguru import logger
from monai.metrics import compute_roc_auc
from omegaconf import OmegaConf

sys.path.append("../")
sys.path.append("../../")

from dataloader import get_dloaders
from utils.loss import FocalLoss

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
    # https://docs.pytorch.org/vision/main/models/generated/torchvision.models.swin_v2_t.html
    if config.base == "swin_v2_t":
        img_size = (256, 256)
    else:
        img_size = (224, 224)

    # Get the dataloaders and any other dataset-specific details.
    dloader_payload = get_dloaders(config, img_size)
    if dloader_payload is None:
        logger.error(
            f"Could not get dataloaders for {config.dset_name}. "
            "The get_dloaders function in dataloader.py might only be "
            "implemented for certain datasets (e.g., 'IMApp')."
        )
        sys.exit(1)

    # Unpack the dataloaders and any other dataset-specific details.
    (
        num_aug,
        focal_loss_weights,
        dloaders_train,
        dloaders_val,
        dloaders_test,
    ) = dloader_payload

    # Specify the weights file.
    if config.base == "resnet18":
        weight_file = config.weight_file_resnet18
    elif config.base == "mobilenetv2":
        weight_file = config.weight_file_mobilenetv2
    elif config.base == "efficientnetb1":
        weight_file = config.weight_file_efficientnetb1
    else:
        raise ValueError(f"Invalid base model: {config.base}")

    # Initialize the model.
    model = FlexibleMultiTaskModel(
        base=config.base,
        num_classes=config.num_classes,
        mode="classification",
        pretrained=config.pretrained,
        dropout_rate=config.dropout_rate,
    )
    model.load_state_dict(torch.load(weight_file))

    # Create the experiment name.
    experiment_name = (
        f"FT-D-{config.base}_{config.dset_name}_{config.train_batch_size}_"
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

    # Set the device based on the accelerator.
    device = accelerator.device

    # Initialize the trackers from within the accelerator.
    accelerator.init_trackers(
        project_name="YOUR_PROJECT_NAME",
        config=config,
    )

    # Get the tracker from the accelerator.
    experiment = accelerator.get_tracker("comet_ml").tracker
    experiment.set_name(experiment_name)
    experiment.log_parameters(config)

    # Initialize the criterion.
    assert config.loss == "focal"
    if focal_loss_weights is None:
        logger.error(
            "Focal loss is selected, but no weights are provided for "
            f"dataset {config.dset_name} in dataloader.py."
        )
        sys.exit(1)
    criterion = FocalLoss(weight=focal_loss_weights.to(device))

    # Initialize the optimizer.
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True,
    )

    # Initialize the scheduler.
    lr_lambda = lambda epoch: np.power(0.1, epoch // 3)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Prepare the model, optimizer, and dataloaders.
    model, optimizer, dloaders_train, dloaders_val, dloaders_test = (
        accelerator.prepare(
            model,
            optimizer,
            dloaders_train,
            dloaders_val,
            dloaders_test,
        )
    )

    # Initialize the step, best validation balanced accuracy, and best validation epoch.
    step, best_valid_balacc, best_valid_epoch = 0, 0.0, 0

    # Start model training.
    # Iterate over the number of epochs.
    for epoch in range(config.epochs):
        # Keep track of learning rate.
        current_lr = optimizer.param_groups[0]["lr"]

        # Log the current epoch and learning rate.
        logger.info(
            f"Epoch {epoch + 1}/{config.epochs} - Learning rate: {current_lr}"
        )

        # Train one epoch.
        with experiment.train():
            # Initialize the training loss and number of training batches.
            epoch_train_loss = 0.0
            num_train_batches = 0
            for aug in range(num_aug):
                # Fetch data from the training dataloader.
                for i, data in enumerate(dloaders_train, 0):
                    model.train()
                    model.zero_grad()
                    optimizer.zero_grad()

                    _, inputs, labels = data

                    # Forward pass.
                    pred = model(inputs)

                    # Compute the loss.
                    loss = criterion(pred, labels)
                    # Update the training loss.
                    epoch_train_loss += loss.item()
                    # Update the number of training batches.
                    num_train_batches += 1

                    # Backward pass.
                    with accelerator.autocast():
                        accelerator.backward(loss)
                    optimizer.step()
                    # Update the step.
                    step += 1

            # Log the training metrics.
            experiment.log_metrics(
                dic={
                    "train_loss": epoch_train_loss / max(1, num_train_batches),
                    "epoch": epoch,
                    "lr": current_lr,
                },
                epoch=epoch,
                step=step,
            )

            # Update the scheduler.
            scheduler.step()

        # Perform validation (skip for ISIC2019 since it has no validation
        # set).
        model.eval()
        if config.dset_name != "ISIC2019":
            # Perform validation.
            with experiment.validate():
                # Disable gradient computation.
                with torch.no_grad():
                    # Initialize the validation ground truth, prediction, and logits.
                    all_val_GTs = torch.Tensor([]).long().to(device)
                    all_val_preds = torch.Tensor([]).long().to(device)
                    all_val_logits = torch.Tensor([]).float().to(device)

                    # Iterate over the validation dataloader.
                    for i, data in enumerate(dloaders_val, 0):
                        _, inputs_eval, labels_eval = data

                        # Forward pass.
                        pred_eval = model(inputs_eval)
                        # Convert the predicted logits to predicted labels.
                        pred_eval_labels = torch.argmax(pred_eval, dim=1)

                        # Update the validation ground truth, prediction, and logits.
                        all_val_GTs = torch.cat((all_val_GTs, labels_eval), 0)
                        all_val_preds = torch.cat(
                            (all_val_preds, pred_eval_labels), 0
                        )
                        all_val_logits = torch.cat(
                            (all_val_logits, pred_eval), 0
                        )

                    # Apply softmax and one-hot encoding before calculating AUROC.
                    y_pred_softmax = F.softmax(all_val_logits, dim=1)
                    y_one_hot = F.one_hot(
                        all_val_GTs, num_classes=config.num_classes
                    ).float()
                    auroc = compute_roc_auc(y_pred=y_pred_softmax, y=y_one_hot)

                    # Convert the validation ground truth, prediction, and logits to numpy arrays.
                    all_val_GTs = all_val_GTs.detach().cpu().numpy().flatten()
                    all_val_preds = (
                        all_val_preds.detach().cpu().numpy().flatten()
                    )

                    # Calculate the accuracy, balanced accuracy, F1 score, precision, and recall.
                    acc = sk_m.accuracy_score(
                        y_true=all_val_GTs, y_pred=all_val_preds
                    )
                    balacc = sk_m.balanced_accuracy_score(
                        y_true=all_val_GTs, y_pred=all_val_preds
                    )
                    f1_mi = sk_m.f1_score(
                        y_true=all_val_GTs,
                        y_pred=all_val_preds,
                        average="micro",
                    )
                    f1_ma = sk_m.f1_score(
                        y_true=all_val_GTs,
                        y_pred=all_val_preds,
                        average="macro",
                    )
                    prec_mi = sk_m.precision_score(
                        y_true=all_val_GTs,
                        y_pred=all_val_preds,
                        average="micro",
                    )
                    prec_ma = sk_m.precision_score(
                        y_true=all_val_GTs,
                        y_pred=all_val_preds,
                        average="macro",
                    )
                    rec_mi = sk_m.recall_score(
                        y_true=all_val_GTs,
                        y_pred=all_val_preds,
                        average="micro",
                    )
                    rec_ma = sk_m.recall_score(
                        y_true=all_val_GTs,
                        y_pred=all_val_preds,
                        average="macro",
                    )

                    experiment.log_metrics(
                        dic={
                            "val_acc": acc,
                            "val_balacc": balacc,
                            "val_f1_mi": f1_mi,
                            "val_f1_ma": f1_ma,
                            "val_prec_mi": prec_mi,
                            "val_prec_ma": prec_ma,
                            "val_rec_mi": rec_mi,
                            "val_rec_ma": rec_ma,
                            "val_auroc": auroc,
                        },
                        epoch=epoch,
                    )

                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)

                    if balacc > best_valid_balacc:
                        best_valid_balacc = balacc
                        best_valid_epoch = epoch
                        accelerator.save(
                            unwrapped_model.state_dict(),
                            expt_dir / "best_model.pth",
                        )

        logger.info(f"[epoch {epoch + 1}/{config.epochs}]")
        logger.info(f"Balanced accuracy: {balacc:.4f}")
        logger.info(
            f"Best balanced accuracy: {best_valid_balacc:.4f} at epoch: {best_valid_epoch}"
        )

    # Save last epoch weights for all datasets.
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    accelerator.save(unwrapped_model.state_dict(), expt_dir / "last_model.pth")

    logger.info("Training completed.")
    logger.info(
        f"Best balanced accuracy: {best_valid_balacc:.4f} at epoch: {best_valid_epoch}"
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
    parser.add_argument(
        "--dset_name", type=str, required=True, help="dataset name"
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    config.base = args.base
    config.dset_name = args.dset_name

    main(config)
