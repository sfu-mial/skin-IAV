# Inter-Annotator Variability in Skin Lesion Segmentation
<!-- Ruff badge -->
<!-- arXiv badge -->
<!-- DOI badge -->
<!-- Citation badge - commented out for now -->
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) [![arXiv](https://img.shields.io/badge/arXiv-2508.09381-b31b1b.svg)](https://arxiv.org/abs/2508.09381) [![DOI](https://zenodo.org/badge/DOI/10.1007/978-3-032-05825-6_3.svg)](https://doi.org/10.1007/978-3-032-05825-6_3) 
<!-- [![Citation](https://api.juleskreuer.eu/citation-badge.php?doi=10.1007/978-3-032-05825-6_3)](https://scholar.google.com/citations?view_op=view_citation&hl=en&citation_for_view=HagMdKMAAAAJ:J_g5lzvAfSwC) -->

## 🏆 Best Paper Award at the ISIC Skin Image Analysis Workshop 2025 🏆

This repository contains the official PyTorch implementation for the paper: "**What Can We Learn from Inter-Annotator Variability in Skin Lesion Segmentation?**", accepted for presentation at the [10th ISIC Skin Image Analysis Workshop](https://workshop.isic-archive.com/2025/), [MICCAI 2025](https://conferences.miccai.org/2025/en/).

> _What Can We Learn from Inter-Annotator Variability in Skin Lesion Segmentation?_<br>
> Kumar Abhishek, Jeremy Kawahara, Ghassan Hamarneh<br>
Medical Image Analysis Lab, School of Computing Science, Simon Fraser University, Canada<br>
> [[DOI]](https://doi.org/10.1007/978-3-032-05825-6_3) [[PDF]](http://www.cs.sfu.ca/~hamarneh/ecopy/miccai_isic2025.pdf) [[Oral Presentation Slides]](https://workshop.isic-archive.com/2025/slides_abhishek.pdf)

![IMA++ Overview](OverviewFigure.png)


In this work, we explore the relationship between inter-annotator agreement (IAA) in skin lesion segmentation and lesion malignancy. We find that malignant lesions are associated with significantly lower agreement among annotators. Leveraging this insight, we show that IAA can be accurately predicted directly from dermoscopic images. Finally, we introduce a multi-task learning model that jointly predicts diagnosis and IAA, achieving a notable improvement in diagnostic accuracy across five datasets.

## Key Contributions

- **A New Dataset**: We introduce **IMA++**, the largest publicly available multi-annotator skin lesion segmentation dataset, with 5,111 segmentations from 15 unique annotators across 2,394 images (**coming soon**).

- **IAA-Malignacy Association**: We provide a formal investigation demonstrating a statistically significant association between lower inter-annotator agreement (IAA), as measured by the average pairwise Dice similarity coefficient, and lesion malignancy.

- **Direct IAA Prediction**: We show that IAA scores can be predicted directly from image content with low error (MAE = 0.108)

- **Improved Diagnosis Prediction with MTL**: We demonstrate that using IAA prediction as an auxiliary task in a multi-task learning (MTL) framework consistently improves diagnostic performance over single-task models, including on single-annotator datasets: PH2, derm7pt, ISIC 2018, and ISIC 2019.


# Code

There are two primary folders in this repository:
- [`predict_IAA`](predict_IAA/): This folder contains the code for predicting IAA from images. This is done on the IMA++ dataset.
- [`predict_finetune_diag`](predict_finetune_diag/): This folder contains the code for predicting diagnosis and IAA from images in a multi-task learning framework. Since only the IMA++ dataset contains IAA scores, this folder contains code to train diagnosis prediction models and multi-task learningmodels on the IMA++ dataset. Moreover, this folder also contains code to finetune both these kinds of models on four other dermoscopic datasets: PH2, derm7pt, ISIC 2018, and ISIC 2019.

<details>
<summary>Repository Structure</summary>

### [`predict_IAA`](predict_IAA/):

This folder contains the code for predicting IAA from images. This is done on the IMA++ dataset.
- `dataloader.py`: The dataloader for IAA prediction on the IMA++ dataset.
- `train.py`: Training script.
- `test.py`: Testing script.
- `config.yaml`: Configuration file for the experiments.
- `compute_test_metrics.py`: Script to compute the test metrics (MAE, MSE, PCC, KS test p-value, Mann-Whitney U test p-value) for the IAA prediction model.
- `run.sh`: Bash script to train and test all models.
- `cam_visualization.py`: Script to visualize the predictions of the model on the test partition using the specified CAM algorithm.
- `overlay_seg_on_cams.py`: Script to overlay the segmentation masks on the CAM images.
- `overlay_seg_on_imgs.py`: Script to overlay the segmentation masks on the images.
- `saved_models/`: Folder containing saved models for the IAA prediction model (empty on GitHub, see below).


### [`predict_finetune_diag`](predict_finetune_diag/):

This folder contains the code for diagnosis prediction (`diag`) and multi-task learning (`MTW`) predicting diagnosis and IAA from images in a multi-task learning framework. 

The following files are present in this folder:
* `configs/`: Configuration files for the training and finetuning (`FT`) of the diagnosis prediction (`diag`) and multi-task learning (`MTW`) models.
  * `IMApp/`: Configuration files for the training of the diagnosis prediction (`diag`) and multi-task learning (`MTW`) models on the IMA++ dataset.
  * `FT/`: Configuration files for the finetuning of the diagnosis prediction (`diag`) and multi-task learning (`MTW`) models on the four other dermoscopic datasets: PH2, derm7pt, ISIC 2018, and ISIC 2019.
* `data_preparation/`: Code to prepare datasets for training and testing.
  * `prepare_datasets.py`: Script to prepare the datasets for training and testing.
  * `dataset_configs.json`: Configuration file for all the five datasets.
  * `partitions/`: Folder containing the processed datasets (empty).
* `utils/`: Utility functions for the diagnosis prediction (`diag`) and multi-task learning (`MTW`) models.
  * `loss.py`: Contains the Focal Loss implementation.
  * `calculate_weights_focalloss.py`: Script to calculate the weights for the Focal Loss based on the class distribution.
  * `custom_transforms.py`: Implements the RandomRotate90 image transform.
* `dataloader.py`: The dataloader definitions.
* {`diag_train.py`, `diag_test.py`}: Training and testing scripts for the diagnosis prediction model (`diag`) on the IMA++ dataset.
* {`MTW_train.py`, `MTW_test.py`}: Training and testing scripts for the multi-task learning model (`MTW`) on the IMA++ dataset.
* {FT_diag_train.py, FT_diag_test.py}: Finetuning scripts (training and testing) for the diagnosis prediction model (`diag`) on the four other dermoscopic datasets: PH2, derm7pt, ISIC 2018, and ISIC 2019.
* {FT_MTW_train.py, FT_MTW_test.py}: Finetuning scripts (training and testing) for the multi-task learning model (`MTW`) on the four other dermoscopic datasets: PH2, derm7pt, ISIC 2018, and ISIC 2019.
* `saved_models/`: Folder containing saved models for the diagnosis prediction (`diag`) and multi-task learning (`MTW`) models (empty on GitHub, see below).

`networks.py`: Contains the model architecture definition for a flexible multi-task model that can be configured for:
  - Diagnosis only ('classification' mode)
  - IAA prediction only ('regression' mode)
  - Both tasks simultaneously ('multitask' mode)

</details>


# Saved Models

The pre-trained models are hosted on 🤗 Hugging Face for easy access and reproducibility: 🤗 **[skin-IAV](https://huggingface.co/kabhishe/skin-IAV)**, which contains all the models in the following directory structure:

<details>
<summary>Models' Directory Structure</summary>

* [`predict_IAA/saved_models/`](https://huggingface.co/kabhishe/skin-IAV/tree/main/predict_IAA/saved_models): Folder containing saved models for the IAA prediction model with the top 3 performing backbones. Each subfolder contains the best model from 3 runs.
  * [`efficientnetb1/`](https://huggingface.co/kabhishe/skin-IAV/tree/main/predict_IAA/saved_models/efficientnetb1)
  * [`mobilenetv2/`](https://huggingface.co/kabhishe/skin-IAV/tree/main/predict_IAA/saved_models/mobilenetv2)
  * [`resnet18/`](https://huggingface.co/kabhishe/skin-IAV/tree/main/predict_IAA/saved_models/resnet18)

* [`predict_finetune_diag/saved_models/`](https://huggingface.co/kabhishe/skin-IAV/tree/main/predict_finetune_diag/saved_models): Folder containing saved models for the diagnosis prediction (`diag`) and multi-task learning (`MTW`) models with the top 3 performing backbones. Each subfolder contains the best model from 3 runs.
  * [`diag/`](https://huggingface.co/kabhishe/skin-IAV/tree/main/predict_finetune_diag/saved_models/diag): Diagnosis-only prediction models.
    * [`IMApp/`](https://huggingface.co/kabhishe/skin-IAV/tree/main/predict_finetune_diag/saved_models/diag/IMApp): Contains the best model from 3 runs on the top 3 performing backbones for the diagnosis prediction model (`diag`) **trained on the IMA++ dataset**.
      * [`efficientnetb1/`](https://huggingface.co/kabhishe/skin-IAV/tree/main/predict_finetune_diag/saved_models/diag/IMApp/efficientnetb1)
      * [`mobilenetv2/`](https://huggingface.co/kabhishe/skin-IAV/tree/main/predict_finetune_diag/saved_models/diag/IMApp/mobilenetv2)
      * [`resnet18/`](https://huggingface.co/kabhishe/skin-IAV/tree/main/predict_finetune_diag/saved_models/diag/IMApp/resnet18)
    * [`FT/`](https://huggingface.co/kabhishe/skin-IAV/tree/main/predict_finetune_diag/saved_models/diag/FT): Each of its subfolders contains the best model from 3 runs on the top 3 performing backbones for the diagnosis prediction model (`diag`) that has been **finetuned on the four other dermoscopic datasets**: PH2, derm7pt, ISIC 2018, and ISIC 2019.
      * [`PH2/`](https://huggingface.co/kabhishe/skin-IAV/tree/main/predict_finetune_diag/saved_models/diag/FT/PH2)
      * [`derm7pt/`](https://huggingface.co/kabhishe/skin-IAV/tree/main/predict_finetune_diag/saved_models/diag/FT/derm7pt)
      * [`ISIC2018/`](https://huggingface.co/kabhishe/skin-IAV/tree/main/predict_finetune_diag/saved_models/diag/FT/ISIC2018)
      * [`ISIC2019/`](https://huggingface.co/kabhishe/skin-IAV/tree/main/predict_finetune_diag/saved_models/diag/FT/ISIC2019)
  * [`MTW/`](https://huggingface.co/kabhishe/skin-IAV/tree/main/predict_finetune_diag/saved_models/MTW): (Weighted) multi-task learning models.
    * [`IMApp/`](https://huggingface.co/kabhishe/skin-IAV/tree/main/predict_finetune_diag/saved_models/MTW/IMApp): Contains the best model from 3 runs on the top 3 performing backbones for the (weighted) multi-task learning model (`MTW`) **trained on the IMA++ dataset**, including the ablation study results on varying the value of α (Eqn. 3 in the paper). So, each of the following subfolders contains 5 directories each: MT_{0.1, 0.2, 0.5, 0.8, 0.9}, each of which then in turn contains the best model from 3 runs.
      * [`efficientnetb1/`](https://huggingface.co/kabhishe/skin-IAV/tree/main/predict_finetune_diag/saved_models/MTW/IMApp/efficientnetb1)
      * [`mobilenetv2/`](https://huggingface.co/kabhishe/skin-IAV/tree/main/predict_finetune_diag/saved_models/MTW/IMApp/mobilenetv2)
      * [`resnet18/`](https://huggingface.co/kabhishe/skin-IAV/tree/main/predict_finetune_diag/saved_models/MTW/IMApp/resnet18)
    * [`FT/`](https://huggingface.co/kabhishe/skin-IAV/tree/main/predict_finetune_diag/saved_models/MTW/FT): Each of its subfolders contains the best model from 3 runs on the top 3 performing backbones for the (weighted) multi-task learning model (`MTW`) that has been **finetuned on the four other dermoscopic datasets**: PH2, derm7pt, ISIC 2018, and ISIC 2019.
      * [`PH2/`](https://huggingface.co/kabhishe/skin-IAV/tree/main/predict_finetune_diag/saved_models/MTW/FT/PH2)
      * [`derm7pt/`](https://huggingface.co/kabhishe/skin-IAV/tree/main/predict_finetune_diag/saved_models/MTW/FT/derm7pt)
      * [`ISIC2018/`](https://huggingface.co/kabhishe/skin-IAV/tree/main/predict_finetune_diag/saved_models/MTW/FT/ISIC2018)
      * [`ISIC2019/`](https://huggingface.co/kabhishe/skin-IAV/tree/main/predict_finetune_diag/saved_models/MTW/FT/ISIC2019)
</details>



# IMA++ Dataset

Coming soon.

# Citation

If you find this work useful, please cite our paper:

Kumar Abhishek, Jeremy Kawahara, Ghassan Hamarneh, "[What Can We Learn from Inter-Annotator Variability in Skin Lesion Segmentation?](http://www.cs.sfu.ca/~hamarneh/ecopy/miccai_isic2025.pdf)", Medical Image Computing and Computer-Assisted Intervention (MICCAI) ISIC Skin Image Analysis Workshop (ISIC), pp. 23-33, 2025, DOI: [10.1007/978-3-032-05825-6_3](https://doi.org/10.1007/978-3-032-05825-6_3).

The corresponding BibTeX entry is:

```bibtex
@InProceedings{abhishek2025what,
  title = {What Can We Learn from Inter-Annotator Variability in Skin Lesion Segmentation?},
  author = {Abhishek, Kumar and Kawahara, Jeremy and Hamarneh, Ghassan},
  booktitle = {Proceedings of the International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI) ISIC Skin Image Analysis Workshop},
  month = {September},
  volume = {16149},
  pages = {23-33},
  year = {2025},
  doi = {10.1007/978-3-032-05825-6_3},
  url = {https://link.springer.com/chapter/10.1007/978-3-032-05825-6_3},
  year = {2025}
}
```