# ===========================================================================
# Project:      On the Byzantine-Resilience of Distillation-Based Federated Learning - IOL Lab @ ZIB
# Paper:        arxiv.org/abs/2402.12265
# File:         models/clothing1m.py
# Description:  Clothing1M Models
# ===========================================================================

import torchvision


def ResNet50():
    return torchvision.models.resnet50(pretrained=False, num_classes=14)
