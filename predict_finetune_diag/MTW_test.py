import argparse
import csv
import os
import random
import sys
from pathlib import Path

import numpy as np
import sklearn.metrics as sk_m
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from loguru import logger
from monai.metrics import compute_roc_auc
from omegaconf import OmegaConf

sys.path.append("../")
sys.path.append("../../")

from dataloader import get_dloaders

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

    # Initialize the model.
    model = FlexibleMultiTaskModel(
        base=config.base,
        num_classes=config.num_classes,
        mode="multitask",
        pretrained=config.pretrained,
        dropout_rate=config.dropout_rate,
    )

    # Create the experiment name.
    experiment_name = (
        f"MTW-{config.cls_loss_weight}-{config.base}_{config.train_batch_size}_"
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

    # Set the device based on the accelerator.
    device = accelerator.device

    # Prepare the model and dataloader.
    model, dloaders_test = accelerator.prepare(model, dloaders_test)

    # Load the best model weights.
    best_model_path = expt_dir / "best_model.pth"
    if not best_model_path.exists():
        raise FileNotFoundError(f"Best model not found at {best_model_path}.")

    logger.info(f"Loading best model from {best_model_path}.")
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.load_state_dict(
        torch.load(best_model_path, map_location=device)
    )
    model.eval()

    # Initialize the experiment.
    accelerator.init_trackers(
        project_name="skin-iav-diag-iampp",
        config=config,
    )

    experiment = accelerator.get_tracker("comet_ml").tracker
    experiment.set_name(f"{experiment_name}_test")
    experiment.log_parameters(config)

    # Perform testing.
    with experiment.test():
        with torch.no_grad():
            all_test_GTs = torch.Tensor([]).long().to(device)
            all_test_preds = torch.Tensor([]).long().to(device)
            all_test_logits = torch.Tensor([]).float().to(device)

            csv_path = expt_dir / "test_results.csv"
            with open(csv_path, "wt", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["image_id", "true_label", "pred_prob_malignant"]
                )

                for i, data in enumerate(dloaders_test, 0):
                    (
                        names_test,
                        inputs_test,
                        diag_labels_test,
                        agreement_labels_test,
                    ) = data

                    # Forward pass.
                    cls_pred_test, reg_pred_test = model(inputs_test)
                    pred_test_labels = torch.argmax(cls_pred_test, dim=1)

                    all_test_GTs = torch.cat(
                        (all_test_GTs, diag_labels_test), 0
                    )
                    all_test_preds = torch.cat(
                        (all_test_preds, pred_test_labels), 0
                    )
                    all_test_logits = torch.cat(
                        (all_test_logits, cls_pred_test), 0
                    )

                    # Write predictions to CSV.
                    probs_test = F.softmax(cls_pred_test, dim=1)
                    responses = [
                        [
                            f"{names_test[idx]}",
                            f"{diag_labels_test[idx].item()}",
                            f"{probs_test[idx, 1].item()}",
                        ]
                        for idx in range(cls_pred_test.shape[0])
                    ]
                    writer.writerows(responses)

            # Apply softmax and one-hot encoding before calculating AUROC.
            y_pred_softmax = F.softmax(all_test_logits, dim=1)
            y_one_hot = F.one_hot(
                all_test_GTs, num_classes=config.num_classes
            ).float()
            auroc = compute_roc_auc(y_pred=y_pred_softmax, y=y_one_hot)

            all_test_GTs = all_test_GTs.detach().cpu().numpy().flatten()
            all_test_preds = all_test_preds.detach().cpu().numpy().flatten()

            acc = sk_m.accuracy_score(
                y_true=all_test_GTs, y_pred=all_test_preds
            )
            balacc = sk_m.balanced_accuracy_score(
                y_true=all_test_GTs, y_pred=all_test_preds
            )
            f1_mi = sk_m.f1_score(
                y_true=all_test_GTs, y_pred=all_test_preds, average="micro"
            )
            f1_ma = sk_m.f1_score(
                y_true=all_test_GTs, y_pred=all_test_preds, average="macro"
            )
            prec_mi = sk_m.precision_score(
                y_true=all_test_GTs, y_pred=all_test_preds, average="micro"
            )
            prec_ma = sk_m.precision_score(
                y_true=all_test_GTs, y_pred=all_test_preds, average="macro"
            )
            rec_mi = sk_m.recall_score(
                y_true=all_test_GTs, y_pred=all_test_preds, average="micro"
            )
            rec_ma = sk_m.recall_score(
                y_true=all_test_GTs, y_pred=all_test_preds, average="macro"
            )

            experiment.log_metrics(
                dic={
                    "test_acc": acc,
                    "test_balacc": balacc,
                    "test_f1_mi": f1_mi,
                    "test_f1_ma": f1_ma,
                    "test_prec_mi": prec_mi,
                    "test_prec_ma": prec_ma,
                    "test_rec_mi": rec_mi,
                    "test_rec_ma": rec_ma,
                    "test_auroc": auroc,
                },
            )

    logger.info("Testing completed.")
    logger.info(f"Test Accuracy: {acc:.4f}")
    logger.info(f"Test Balanced Accuracy: {balacc:.4f}")
    logger.info(f"Test AUROC: {auroc:.4f}")
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
    parser.add_argument(
        "--dset_name",
        type=str,
        required=True,
        help="dataset name",
    )
    parser.add_argument(
        "--cls_loss_weight",
        type=float,
        required=True,
        help="classification loss weight",
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    config.base = args.base
    config.dset_name = args.dset_name
    config.cls_loss_weight = args.cls_loss_weight

    main(config)
