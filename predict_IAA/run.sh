#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate ENV_NAME

models=(
    vgg16
    resnet18
    resnet50
    densenet121
    mobilenetv2
    mobilenetv3l
    efficientnetb0
    efficientnetb1
    convnext_tiny
    swin_t
    swin_v2_t
    vit_b_16
    vit_b_32
)

# Train and test all models.
for model in "${models[@]}"; do
    echo "Training and testing $model..."
    python train.py --config config.yaml --base "$model" && python test.py --config config.yaml --base "$model"
    echo "Training and testing $model completed."
done
