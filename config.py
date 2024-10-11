# ===========================================================================
# Project:      On the Byzantine-Resilience of Distillation-Based Federated Learning - IOL Lab @ ZIB
# Paper:        arxiv.org/abs/2402.12265
# File:         config.py
# Description:  Datasets, Normalization and Transforms
# ===========================================================================

import torch
import torchvision
from torchvision import transforms

means = {
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4867, 0.4408),
    'imagenet': (0.485, 0.456, 0.406),
    'imagenet100': (0.485, 0.456, 0.406),
    'tinyimagenet': (0.485, 0.456, 0.406),
    'cinic10': (0.47889522, 0.47227842, 0.43047404),
    'clothing1m': (0.6959, 0.6537, 0.6371), # from https://github.com/puar-playground/LRA-diffusion/blob/main/utils/cloth_data_utils.py#L11
}
stds = {
    'cifar10': (0.2023, 0.1994, 0.2010),
    'cifar100': (0.2675, 0.2565, 0.2761),
    'imagenet': (0.229, 0.224, 0.225),
    'imagenet100': (0.229, 0.224, 0.225),
    'tinyimagenet': (0.229, 0.224, 0.225),
    'cinic10': (0.24205776, 0.23828046, 0.25874835),
    'clothing1m': (0.3113, 0.3192, 0.3214),
}

n_classesDict = {
    'mnist': 10,
    'cifar10': 10,
    'cifar100': 100,
    'imagenet': 1000,
    'imagenet100': 100,
    'tinyimagenet': 200,
    'cinic10': 10,
    'clothing1m': 14
}

num_workersDict = {
    'mnist': 2 if torch.cuda.is_available() else 0,
    'cifar10': 2 if torch.cuda.is_available() else 0,
    'cifar100': 4 * torch.cuda.device_count() if torch.cuda.is_available() else 0,
    'imagenet': 4 * torch.cuda.device_count() if torch.cuda.is_available() else 0,
    'imagenet100': 4 * torch.cuda.device_count() if torch.cuda.is_available() else 0,
    'tinyimagenet': 4 * torch.cuda.device_count() if torch.cuda.is_available() else 0,
    'cinic10': 2 if torch.cuda.is_available() else 0,
    'clothing1m': 4 if torch.cuda.is_available() else 0,
}

datasetDict = {  # Links dataset names to actual torch datasets
    'mnist': getattr(torchvision.datasets, 'MNIST'),
    'cifar10': getattr(torchvision.datasets, 'CIFAR10'),
    'fashionMNIST': getattr(torchvision.datasets, 'FashionMNIST'),
    'SVHN': getattr(torchvision.datasets, 'SVHN'),  # This needs scipy
    'STL10': getattr(torchvision.datasets, 'STL10'),
    'cifar100': getattr(torchvision.datasets, 'CIFAR100'),
    'imagenet': getattr(torchvision.datasets, 'ImageNet'),
    'imagenet100': getattr(torchvision.datasets, 'ImageFolder'),
    'tinyimagenet': getattr(torchvision.datasets, 'ImageFolder'),
    'cinic10': getattr(torchvision.datasets, 'ImageFolder'),
    'clothing1m': getattr(torchvision.datasets, 'ImageFolder'),
}

trainTransformDict = {  # Links dataset names to train dataset transformers
    'mnist': transforms.Compose([transforms.ToTensor()]),
    'cifar10': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=means['cifar10'], std=stds['cifar10']), ]),
    'cifar100': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=means['cifar100'], std=stds['cifar100']), ]),
    'imagenet': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=means['imagenet'], std=stds['imagenet']), ]),
    'imagenet100': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=means['imagenet'], std=stds['imagenet']), ]),
    'tinyimagenet': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=means['tinyimagenet'], std=stds['tinyimagenet']), ]),
    'cinic10': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=means['cinic10'], std=stds['cinic10']), ]),
    'clothing1m': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=means['clothing1m'], std=stds['clothing1m']), ]),
}
testTransformDict = {  # Links dataset names to test dataset transformers
    'mnist': transforms.Compose([transforms.ToTensor()]),
    'cifar10': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=means['cifar10'], std=stds['cifar10']), ]),
    'cifar100': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=means['cifar100'], std=stds['cifar100']), ]),
    'imagenet': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=means['imagenet'], std=stds['imagenet']), ]),
    'imagenet100': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=means['imagenet'], std=stds['imagenet']), ]),
    'tinyimagenet': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=means['tinyimagenet'], std=stds['tinyimagenet']), ]),
    'cinic10': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=means['cinic10'], std=stds['cinic10']), ]),
    'clothing1m': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=means['clothing1m'], std=stds['clothing1m']), ]),
}
