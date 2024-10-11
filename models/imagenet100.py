# ===========================================================================
# Project:      On the Byzantine-Resilience of Distillation-Based Federated Learning - IOL Lab @ ZIB
# Paper:        arxiv.org/abs/2402.12265
# File:         models/imagenet100.py
# Description:  ImageNet-100 Models
# ===========================================================================
import torchvision

def ResNet50():
    return torchvision.models.resnet50(pretrained=False, num_classes=100)
