# ===========================================================================
# Project:      On the Byzantine-Resilience of Distillation-Based Federated Learning - IOL Lab @ ZIB
# Paper:        arxiv.org/abs/2402.12265
# File:         public_config.py
# Description:  Public datasets
# ===========================================================================

import torchvision
from torchvision import transforms

from utilities import ImageNetDownSample
from utilities import Utilities as Utils

means = {
    'stl10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4867, 0.4408),
    'cifar100-ext': (0.5071, 0.4867, 0.4408),
    'imagenet32': (0.485, 0.456, 0.406),
    'imagenet100': (0.485, 0.456, 0.406),
    'cinic10': (0.47889522, 0.47227842, 0.43047404),
    'clothing1m': (0.6959, 0.6537, 0.6371),
}

stds = {
    'stl10': (0.2471, 0.2435, 0.2616),
    'cifar100': (0.2675, 0.2565, 0.2761),
    'cifar100-ext': (0.2675, 0.2565, 0.2761),
    'imagenet32': (0.229, 0.224, 0.225),
    'imagenet100': (0.229, 0.224, 0.225),
    'cinic10': (0.24205776, 0.23828046, 0.25874835),
    'clothing1m': (0.3113, 0.3192, 0.3214),
}

public_trainTransform_dict = {
    # Links dataset names to train dataset transforms, which are used for training the server on public datasets
    'stl10': transforms.Compose([
        transforms.Pad(4),
        transforms.RandomCrop(96),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(size=32),
        transforms.ToTensor(),
        transforms.Normalize(mean=means['stl10'], std=stds['stl10']),
    ]),
    'cifar100': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=means['cifar100'], std=stds['cifar100']), ]),
    'cifar100-ext': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=means['cifar100'], std=stds['cifar100']), ]),
    'imagenet100': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=means['imagenet100'], std=stds['imagenet100']), ]),
    'imagenet32': transforms.Compose([  # Actually downsampled to 32x32
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=means['imagenet32'], std=stds['imagenet32']), ]),
    'cinic10': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=means['cinic10'], std=stds['cinic10']), ]),
    'clothing1m': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=means['clothing1m'], std=stds['clothing1m']), ]),
}

public_testTransform_dict = {
    # Links dataset names to train dataset transforms, which are used for inference on public datasets
    'stl10': transforms.Compose([
        transforms.Resize(size=32),
        transforms.ToTensor(),
        transforms.Normalize(mean=means['stl10'], std=stds['stl10']),
    ]),
    'cifar100': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=means['cifar100'], std=stds['cifar100']), ]),
    'cifar100-ext': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=means['cifar100-ext'], std=stds['cifar100-ext']), ]),
    'imagenet100': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=means['imagenet100'], std=stds['imagenet32']), ]),
    'imagenet32': transforms.Compose([  # Actually downsampled to 32x32
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=means['imagenet32'], std=stds['imagenet32']), ]),
    'cinic10': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=means['cinic10'], std=stds['cinic10']), ]),
    'clothing1m': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=means['clothing1m'], std=stds['clothing1m']), ]),
}

public_datasetAssignmentDict = {  # Assigns to each training dataset a public dataset,
    'mnist': 'mnist',
    'cifar10': 'stl10',
    'cifar100': 'stl10',
    'cinic10': 'cinic10',
    'imagenet100':'imagenet100',
    'clothing1m': 'clothing1m',
}

public_trainDataset_dict = {
    'mnist': Utils.get_preinitialized_dataset(OriginalDataset=torchvision.datasets.MNIST, download=True,
                                              transform=transforms.Compose([transforms.ToTensor()])),
    'cifar100': Utils.get_preinitialized_dataset(OriginalDataset=torchvision.datasets.CIFAR100, download=True,
                                                 transform=public_trainTransform_dict['cifar100']),
    'stl10': Utils.get_preinitialized_dataset(OriginalDataset=torchvision.datasets.STL10, split='unlabeled',
                                              download=True,
                                              transform=public_trainTransform_dict['stl10']),
    # C100-ext dataset taken from https://www.kaggle.com/datasets/dunky11/cifar100-100x100-images-extension
    'cifar100-ext': Utils.get_preinitialized_dataset(OriginalDataset=torchvision.datasets.ImageFolder,
                                                     transform=public_trainTransform_dict['cifar100-ext']),
    'imagenet32': Utils.get_preinitialized_dataset(OriginalDataset=ImageNetDownSample,
                                                   transform=public_trainTransform_dict['imagenet32']),
    'imagenet100': Utils.get_preinitialized_dataset(OriginalDataset=torchvision.datasets.ImageFolder,
                                                   transform=public_trainTransform_dict['imagenet100']),
    'cinic10': Utils.get_preinitialized_dataset(OriginalDataset=torchvision.datasets.ImageFolder,
                                                transform=public_trainTransform_dict['cinic10']),
    'clothing1m': Utils.get_preinitialized_dataset(OriginalDataset=torchvision.datasets.ImageFolder,
                                                transform=public_trainTransform_dict['clothing1m']),
}

public_testDataset_dict = {
    'mnist': Utils.get_preinitialized_dataset(OriginalDataset=torchvision.datasets.MNIST, download=True,
                                              transform=transforms.Compose([transforms.ToTensor()])),
    'cifar100': Utils.get_preinitialized_dataset(OriginalDataset=torchvision.datasets.CIFAR100, download=True,
                                                 transform=public_testTransform_dict['cifar100']),
    'stl10': Utils.get_preinitialized_dataset(OriginalDataset=torchvision.datasets.STL10, split='unlabeled',
                                              download=True,
                                              transform=public_testTransform_dict['stl10']),
    # C100-ext dataset taken from https://www.kaggle.com/datasets/dunky11/cifar100-100x100-images-extension
    'cifar100-ext': Utils.get_preinitialized_dataset(OriginalDataset=torchvision.datasets.ImageFolder,
                                                     transform=public_testTransform_dict['cifar100-ext']),
    'imagenet32': Utils.get_preinitialized_dataset(OriginalDataset=ImageNetDownSample,
                                                   transform=public_testTransform_dict['imagenet32']),
    'imagenet100': Utils.get_preinitialized_dataset(OriginalDataset=torchvision.datasets.ImageFolder,
                                                   transform=public_testTransform_dict['imagenet100']),
    'cinic10': Utils.get_preinitialized_dataset(OriginalDataset=torchvision.datasets.ImageFolder,
                                                transform=public_testTransform_dict['cinic10']),
    'clothing1m': Utils.get_preinitialized_dataset(OriginalDataset=torchvision.datasets.ImageFolder,
                                                transform=public_testTransform_dict['clothing1m']),
}
