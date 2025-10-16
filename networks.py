import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ViewFlatten(nn.Module):
    """
    Flattens the output for all dimensions except the first, using `.view()`.
    Similar to `torch.flatten(x, start_dim=1)`.
    Taken from: https://stackoverflow.com/a/61039752
    """

    def __init__(self):
        super(ViewFlatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class FlexibleMultiTaskModel(nn.Module):
    """
    A flexible multi-task model that can be configured for:
    1. Diagnosis only ('classification' mode)
    2. IAA prediction only ('regression' mode)
    3. Both tasks simultaneously ('multitask' mode)
    """

    def __init__(
        self,
        base: str,
        num_classes: int,
        mode: str,
        pretrained: bool = True,
        dropout_rate: float = 0.5,
    ):
        super(FlexibleMultiTaskModel, self).__init__()

        # Mode validation and storage.
        self.mode = mode.lower()
        if self.mode not in ["classification", "regression", "multitask"]:
            raise ValueError(
                f"Invalid mode: {self.mode}. Must be one of: 'classification', \
                'regression', 'multitask'."
            )

        self.base = base.lower()
        if self.base not in [
            "vgg16",
            "resnet50",
            "resnet18",
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
        ]:
            raise ValueError(
                f"Invalid base model: {self.base}. Must be one of: 'vgg16', \
                'resnet50', 'resnet18', 'densenet121', 'mobilenetv2', \
                'mobilenetv3l', 'efficientnetb0', 'efficientnetb1', \
                'convnext_tiny', 'swin_t', 'swin_v2_t', 'vit_b_16', \
                'vit_b_32'."
            )

        feature_dim = 0

        # Load the backbone.
        if self.base == "vgg16":
            # Use the same architecture as in `TransLearnModel` but as a backbone.
            net = models.vgg16_bn(pretrained=pretrained)
            self.conv_block1 = nn.Sequential(
                *list(net.features.children())[0:6]
            )
            self.conv_block2 = nn.Sequential(
                *list(net.features.children())[7:13]
            )
            self.conv_block3 = nn.Sequential(
                *list(net.features.children())[14:23]
            )
            self.conv_block4 = nn.Sequential(
                *list(net.features.children())[24:33]
            )
            self.conv_block5 = nn.Sequential(
                *list(net.features.children())[34:43]
            )
            self.pool = nn.AvgPool2d(7, stride=1)
            feature_dim = 512

        elif self.base in ["resnet18", "resnet50"]:
            if self.base == "resnet18":
                self.backbone = models.resnet18(pretrained=pretrained)
            else:
                self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        elif self.base == "densenet121":
            self.backbone = models.densenet121(pretrained=pretrained)
            feature_dim = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()

        elif self.base in [
            "mobilenetv2",
            "efficientnetb0",
            "efficientnetb1",
            "convnext_tiny",
        ]:
            if self.base == "mobilenetv2":
                self.backbone = models.mobilenet_v2(pretrained=pretrained)
            elif self.base == "mobilenetv3l":
                self.backbone = models.mobilenet_v3_large(
                    pretrained=pretrained
                )
            elif self.base == "efficientnetb0":
                self.backbone = models.efficientnet_b0(pretrained=pretrained)
            elif self.base == "efficientnetb1":
                self.backbone = models.efficientnet_b1(pretrained=pretrained)
            elif self.base == "convnext_tiny":
                self.backbone = models.convnext_tiny(pretrained=pretrained)
            feature_dim = self.backbone.classifier[-1].in_features
            self.backbone.classifier = nn.Identity()

        elif self.base == "mobilenetv3l":
            self.backbone = models.mobilenet_v3_large(pretrained=pretrained)
            # `mobilenet_v3_large` has a more complex, two-stage classifier head.
            # Its classifier is a `nn.Sequential` with two layers. The first layer
            # expands the features from 960 to 1280, before the final classification
            # (linear) layer. So, we need to use the in_features from the FIRST
            # layer of the classifier.
            feature_dim = self.backbone.classifier[0].in_features
            self.backbone.classifier = nn.Identity()

        elif self.base in ["swin_t", "swin_v2_t"]:
            if self.base == "swin_t":
                self.backbone = models.swin_t(pretrained=pretrained)
            else:
                self.backbone = models.swin_v2_t(pretrained=pretrained)
            feature_dim = self.backbone.head.in_features
            self.backbone.head = nn.Identity()

        elif self.base in ["vit_b_16", "vit_b_32"]:
            if self.base == "vit_b_16":
                self.backbone = models.vit_b_16(pretrained=pretrained)
            else:
                self.backbone = models.vit_b_32(pretrained=pretrained)
            feature_dim = self.backbone.heads.head.in_features
            self.backbone.heads.head = nn.Identity()

        # Conditional initialization of classification and regression heads.
        hidden_dim = 256

        if self.mode in ["classification", "multitask"]:
            self.classification_head = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, num_classes),
            )
        if self.mode in ["regression", "multitask"]:
            self.regression_head = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, 1),
            )

        self._init_head_weights()

    def _init_head_weights(self):
        # Initialize the weights only for the heads that have been initialized.
        if hasattr(self, "classification_head"):
            for m in self.classification_head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_in", nonlinearity="relu"
                    )
                    nn.init.constant_(m.bias, 0.0)
        if hasattr(self, "regression_head"):
            for m in self.regression_head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_in", nonlinearity="relu"
                    )
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        if self.base == "vgg16":
            block1 = self.conv_block1(x)
            pool1 = F.max_pool2d(block1, 2, 2)
            block2 = self.conv_block2(pool1)
            pool2 = F.max_pool2d(block2, 2, 2)
            block3 = self.conv_block3(pool2)
            pool3 = F.max_pool2d(block3, 2, 2)
            block4 = self.conv_block4(pool3)
            pool4 = F.max_pool2d(block4, 2, 2)
            block5 = self.conv_block5(pool4)
            pool5 = F.max_pool2d(block5, 2, 2)
            # N, _, _, _ = pool5.size()
            pooled_features = self.pool(pool5)  # .view(N, -1)
            features = torch.flatten(pooled_features, 1)
        else:
            features = self.backbone(x)
            features = torch.flatten(features, 1)

        # Conditional forward pass based on mode.
        if self.mode == "classification":
            cls_logits = self.classification_head(features)
            return cls_logits

        elif self.mode == "regression":
            reg_logits = self.regression_head(features)
            return reg_logits

        elif self.mode == "multitask":
            cls_logits = self.classification_head(features)
            reg_logits = self.regression_head(features)
            return cls_logits, reg_logits
