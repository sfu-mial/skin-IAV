import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import (
    combine_pvalues,
    ks_2samp,
    mannwhitneyu,
    pearsonr,
    spearmanr,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Set your parent directory
MODEL_WEIGHTS_DIR = Path("./saved_models/")

# Set the file path to the test partition metadata
TEST_PARTITION_METADATA = Path("/path/to/test_partition_metadata.csv")

# Read the test partition metadata
test_partition_metadata = pd.read_csv(
    TEST_PARTITION_METADATA, header="infer", sep=","
)

# Get the list of images in the test partition that are malignant
# and benign.
test_partition_malignant_list = test_partition_metadata[
    test_partition_metadata["malignancy"] == "malignant"
]["image"].tolist()

test_partition_benign_list = test_partition_metadata[
    test_partition_metadata["malignancy"] != "malignant"
]["image"].tolist()

results = []

# Traverse each subdirectory in the parent directory
for model_key in os.listdir(MODEL_WEIGHTS_DIR):
    model_dir = os.path.join(MODEL_WEIGHTS_DIR, model_key)
    if not os.path.isdir(model_dir):
        continue  # Skip non-directories

    # Extract model architecture from model_key.
    # In this case, the model architecture is the same as the model key.
    model_arch = model_key

    mae_list, mse_list, pcc_list, pval_list_ks, pval_list_mw = (
        [],
        [],
        [],
        [],
        [],
    )
    true_metric_all, pred_metric_all = [], []
    true_metric_malignant_all, pred_metric_malignant_all = [], []
    true_metric_benign_all, pred_metric_benign_all = [], []

    # Read all CSV files in the model_dir
    csv_files = glob.glob(os.path.join(model_dir, "*.csv"))
    csv_files = sorted(csv_files)  # Ensure consistent ordering

    for i, csv_file in enumerate(csv_files):
        df = pd.read_csv(csv_file, header="infer", sep=",")
        y_true = df["true_metric"].values
        y_pred = df["pred_metric"].values

        # Filter the df to only include images that are malignant
        # or benign in the test set.
        y_true_malignant = df[
            df["image_id"].isin(test_partition_malignant_list)
        ]["true_metric"].values
        y_true_benign = df[df["image_id"].isin(test_partition_benign_list)][
            "true_metric"
        ].values
        y_pred_malignant = df[
            df["image_id"].isin(test_partition_malignant_list)
        ]["pred_metric"].values
        y_pred_benign = df[df["image_id"].isin(test_partition_benign_list)][
            "pred_metric"
        ].values

        pval = ks_2samp(y_true, y_pred).pvalue

        pval_list_ks.append(pval)
        pval_list_mw.append(mannwhitneyu(y_true, y_pred).pvalue)

        true_metric_all.extend(y_true)
        pred_metric_all.extend(y_pred)

        true_metric_malignant_all.extend(y_true_malignant)
        pred_metric_malignant_all.extend(y_pred_malignant)
        true_metric_benign_all.extend(y_true_benign)
        pred_metric_benign_all.extend(y_pred_benign)

    # Run KS test on concatenated metrics
    ks_pval_overall = ks_2samp(pred_metric_all, true_metric_all).pvalue
    ks_pval_combined = combine_pvalues(
        [pval_list_ks[0], pval_list_ks[1], pval_list_ks[2]]
    ).pvalue

    # Run MW test on concatenated metrics
    mw_pval_overall = mannwhitneyu(pred_metric_all, true_metric_all).pvalue
    mw_pval_combined = combine_pvalues(
        [pval_list_mw[0], pval_list_mw[1], pval_list_mw[2]]
    ).pvalue

    # Run Mann-Whitney U test on concatenated metrics
    # This is to check if we can say something about the annotation agreement
    # for malignant versus benign images.
    mann_pval_malignant_benign = mannwhitneyu(
        pred_metric_malignant_all, pred_metric_benign_all
    ).pvalue

    # Run the same Mann-Whitney U test on the true metrics.
    mann_pval_malignant_benign_true = mannwhitneyu(
        true_metric_malignant_all, true_metric_benign_all
    ).pvalue

    # Prepare the result row
    result = {
        "model_arch": model_arch,
        "MAE_mean": mean_absolute_error(true_metric_all, pred_metric_all),
        "MAE_std": np.std(
            np.array(true_metric_all) - np.array(pred_metric_all), ddof=1
        ),
        "MSE_mean": mean_squared_error(true_metric_all, pred_metric_all),
        "MSE_std": np.std(
            (np.array(true_metric_all) - np.array(pred_metric_all)) ** 2,
            ddof=1,
        ),
        "MAE_malignant_mean": mean_absolute_error(
            true_metric_malignant_all, pred_metric_malignant_all
        ),
        "MAE_malignant_std": np.std(
            np.array(true_metric_malignant_all)
            - np.array(pred_metric_malignant_all),
            ddof=1,
        ),
        "MAE_benign_mean": mean_absolute_error(
            true_metric_benign_all, pred_metric_benign_all
        ),
        "MAE_benign_std": np.std(
            np.array(true_metric_benign_all)
            - np.array(pred_metric_benign_all),
            ddof=1,
        ),
        "MSE_malignant_mean": mean_squared_error(
            true_metric_malignant_all, pred_metric_malignant_all
        ),
        "MSE_malignant_std": np.std(
            (
                np.array(true_metric_malignant_all)
                - np.array(pred_metric_malignant_all)
            )
            ** 2,
            ddof=1,
        ),
        "MSE_benign_mean": mean_squared_error(
            true_metric_benign_all, pred_metric_benign_all
        ),
        "MSE_benign_std": np.std(
            (
                np.array(true_metric_benign_all)
                - np.array(pred_metric_benign_all)
            )
            ** 2,
            ddof=1,
        ),
        "PCC_overall": pearsonr(true_metric_all, pred_metric_all)[0],
        "PCC_pval": pearsonr(true_metric_all, pred_metric_all).pvalue,
        "SCC_overall": spearmanr(true_metric_all, pred_metric_all)[0],
        "SCC_pval": spearmanr(true_metric_all, pred_metric_all).pvalue,
        "mann_pval_malignant_benign": mann_pval_malignant_benign,
        "mann_pval_malignant_benign_true": mann_pval_malignant_benign_true,
        "pval_overall_ks": ks_pval_overall,
        "pval_combined_ks": ks_pval_combined,
        "pval_overall_mw": mw_pval_overall,
        "pval_combined_mw": mw_pval_combined,
    }

    results.append(result)

# Final dataframe
results_df = pd.DataFrame(results)

# Save if needed
results_df.to_csv("metrics_summary-stratified.csv", index=False)

# Display the result. Show decimal places up to 4 digits after the decimal
# point, except for the p-values, which should be shown as scientific notation.
# print(results_df.round(4).style.format({"pval_overall": "{:.2e}", "pval_combined": "{:.2e}"}))
results_df_rounded = results_df.round(4)
results_df_rounded["PCC_pval"] = results_df["PCC_pval"].apply(
    lambda x: f"{x:.2e}"
)
results_df_rounded["SCC_pval"] = results_df["SCC_pval"].apply(
    lambda x: f"{x:.2e}"
)
results_df_rounded["pval_overall_ks"] = results_df["pval_overall_ks"].apply(
    lambda x: f"{x:.2e}"
)
results_df_rounded["pval_combined_ks"] = results_df["pval_combined_ks"].apply(
    lambda x: f"{x:.2e}"
)
results_df_rounded["pval_overall_mw"] = results_df["pval_overall_mw"].apply(
    lambda x: f"{x:.2e}"
)
results_df_rounded["pval_combined_mw"] = results_df["pval_combined_mw"].apply(
    lambda x: f"{x:.2e}"
)
results_df_rounded["mann_pval_malignant_benign"] = results_df[
    "mann_pval_malignant_benign"
].apply(lambda x: f"{x:.2e}")
results_df_rounded["mann_pval_malignant_benign_true"] = results_df[
    "mann_pval_malignant_benign_true"
].apply(lambda x: f"{x:.2e}")

# Display the result
print(results_df_rounded)

# Print the mean ± std of the true metrics for malignant and benign images.
print(
    f"Malignant agreement: {np.asarray(true_metric_malignant_all).mean():.4f} ± "
    f"{np.asarray(true_metric_malignant_all).std():.4f}"
)
print(
    f"Benign agreement: {np.asarray(true_metric_benign_all).mean():.4f} ± "
    f"{np.asarray(true_metric_benign_all).std():.4f}"
)
print(f"Mann-Whitney U test p-value: {mann_pval_malignant_benign_true:.2e}")
