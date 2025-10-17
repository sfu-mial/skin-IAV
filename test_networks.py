"""
Test all the architectures and modes in `networks.py` with random weights.
"""

import pytest
import torch

from networks import FlexibleMultiTaskModel

BATCH_SIZE = 4
NUM_CLASSES = 3

ALL_BASE_MODELS = [
    "vgg16",
    "resnet18",
    "resnet50",
    "densenet121",
    "mobilenetv2",
    "mobilenetv3l",
    "efficientnetb0",
    "efficientnetb1",
    "convnext_tiny",
    "swin_t",
    "swin_v2_t",
    "vit_b_16",
    "vit_b_32",
]

ALL_MODES = [
    "classification",
    "regression",
    "multitask",
]


@pytest.mark.parametrize("base", ALL_BASE_MODELS)
@pytest.mark.parametrize("mode", ALL_MODES)
def test_networks(base, mode):
    """
    Tests that FlexibleMultiTaskModel can be instantiated and run for all
    architectures and in all modes, producing the correct output shapes.
    """
    # Instantiate the model.
    model = FlexibleMultiTaskModel(base, NUM_CLASSES, mode, False)
    # Set the model to evaluation mode.
    model.eval()

    # Generate random input tensor.
    x = torch.randn(BATCH_SIZE, 3, 224, 224)

    # Perform a forward pass.
    with torch.inference_mode():
        out = model(x)

    # Check the output shape based on the mode.
    if mode == "classification":
        assert isinstance(out, torch.Tensor), "Output should be a tensor."
        assert out.shape == (BATCH_SIZE, NUM_CLASSES), (
            "Classification logits shape should be (BATCH_SIZE, NUM_CLASSES)."
        )
    elif mode == "regression":
        assert isinstance(out, torch.Tensor), "Output should be a tensor."
        assert out.shape == (BATCH_SIZE, 1), (
            "Regression logits shape should be (BATCH_SIZE, 1)."
        )
    elif mode == "multitask":
        assert isinstance(out, tuple), "Output should be a tuple."
        assert len(out) == 2, "Output should contain two tensors."
        assert isinstance(out[0], torch.Tensor), (
            "First output should be a tensor."
        )
        assert isinstance(out[1], torch.Tensor), (
            "Second output should be a tensor."
        )
        assert out[0].shape == (BATCH_SIZE, NUM_CLASSES), (
            "Classification logits shape should be (BATCH_SIZE, NUM_CLASSES)."
        )
        assert out[1].shape == (BATCH_SIZE, 1), (
            "Regression logits shape should be (BATCH_SIZE, 1)."
        )
