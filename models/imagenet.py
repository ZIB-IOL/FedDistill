# ===========================================================================
# Project:      On the Byzantine-Resilience of Distillation-Based Federated Learning - IOL Lab @ ZIB
# Paper:        arxiv.org/abs/2402.12265
# File:         models/imagenet.py
# Description:  ImageNet Models
# ===========================================================================

import torchvision


def ResNet50():
    return torchvision.models.resnet50(pretrained=False)
