"""
This script calculates the class weights for the focal loss to be used in a
binary classification task using the inverse frequency formula.
"""

import argparse

import torch


def calculate_class_weights(*partitions):
    """
    Calculates class weights for an imbalanced dataset across multiple
    partitions.

    Args:
        *partitions (str): A variable number of strings, where each string
                           represents the class counts in a partition,
                           formatted as 'class0_count,class1_count'.

    Returns:
        torch.Tensor: A tensor containing the calculated class weights for
                      each class.
                      Returns None if the input is invalid.
    """
    total_class0 = 0
    total_class1 = 0
    num_classes = 2  # Binary classification.

    if not partitions:
        print("Error: No partition data provided.")
        return None

    print("Processing partition data...")
    for i, part in enumerate(partitions):
        try:
            counts = part.split(",")
            if len(counts) != num_classes:
                print(
                    f"Error: Partition {i + 1} ('{part}') is not formatted correctly. Expected {num_classes} counts."
                )
                return None

            class0_count = int(counts[0])
            class1_count = int(counts[1])

            print(
                f"  - Partition {i + 1}: Class 0 = {class0_count}, Class 1 = {class1_count}"
            )

            total_class0 += class0_count
            total_class1 += class1_count
        except (ValueError, IndexError) as e:
            print(
                f"Error processing partition '{part}': {e}. "
                "Please ensure the format is 'class0_count,class1_count'."
            )
            return None

    total_samples = total_class0 + total_class1

    if total_class0 == 0 or total_class1 == 0:
        print(
            "Error: One of the classes has zero samples across all partitions."
        )
        return None

    print("\n--- Aggregated Counts ---")
    print(f"Total samples for Class 0: {total_class0}")
    print(f"Total samples for Class 1: {total_class1}")
    print(f"Total samples overall: {total_samples}")
    print("-------------------------")

    # Calculate weights using the inverse frequency formula
    weight_class0 = total_samples / (num_classes * total_class0)
    weight_class1 = total_samples / (num_classes * total_class1)

    weights = torch.tensor([weight_class0, weight_class1], dtype=torch.float32)

    return weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dset_name",
        type=str,
        required=True,
        choices=["IMApp", "ISIC2018", "ISIC2019", "derm7pt", "PH2"],
        help="Dataset name",
    )
    args = parser.parse_args()

    if args.dset_name == "IMApp":
        # IMApp (benign, malignant)
        test_partition = "1315,353"
        val_partition = "188,51"
        train_partition = "377,101"
    elif args.dset_name == "ISIC2018":
        # ISIC2018 (NON_MEL, MEL)
        test_partition = "1341,171"
        val_partition = "172,21"
        train_partition = "8902,1113"
    elif args.dset_name == "ISIC2019":
        # ISIC2019 (NON_MEL, MEL)
        test_partition = "6911,1327"
        train_partition = "20809,4522"
    elif args.dset_name == "derm7pt":
        # derm7pt (NON_MEL, MEL)
        test_partition = "152,50"
        val_partition = "76,26"
        train_partition = "1571,352"
    elif args.dset_name == "PH2":
        # PH2 (NON_MEL, MEL)
        test_partition = "32,8"
        val_partition = "16,4"
        train_partition = "112,28"
    else:
        raise ValueError(f"Invalid dataset name: {args.dset_name}")

    # Calculate the weights over the train set.
    class_weights = calculate_class_weights(train_partition)

    print(f"Class weights for {args.dset_name}: {class_weights}")
