import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from .svhn import SVHNMIX
from .cifar10 import CIFAR10

data_root = os.path.dirname(os.path.realpath(__file__)) + '/../data'
data_paths = {
    'cifar10': os.path.join(data_root, 'cifar10'),
    'cifar100': os.path.join(data_root, 'cifar100'),
    'mnist': os.path.join(data_root, 'mnist'),
    'svhn': os.path.join(data_root, 'svhn'),
    'imagenet': {
        'train': os.path.join(data_root, 'imagenet/train'),
        'val': os.path.join(data_root, 'imagenet/val')
    }
}

data_means = {
    'cifar10': [x / 255 for x in [125.3, 123.0, 113.9]],
    'cifar100': [x / 255 for x in [129.3, 124.1, 112.4]],
    'mnist': [0.1307],
    'svhn': [0.5, 0.5, 0.5],
    'imagenet': [0.485, 0.456, 0.406]
}

data_stds = {
    'cifar10': [x / 255 for x in [63.0, 62.1, 66.7]],
    'cifar100': [x / 255 for x in [68.2, 65.4, 70.4]],
    'mnist': [0.3081],
    'svhn': [0.5, 0.5, 0.5],
    'imagenet': [0.229, 0.224, 0.225]
}


def get_dataset(name, split='train', transform=None, target_transform=None, download=True):
    if name == 'cifar10':
        dataset = CIFAR10(root=data_paths['cifar10'],
                          split=split,
                          transform=transform,
                          target_transform=target_transform,
                          download=download)
        dataset.num_classes = 10
    elif name == 'cifar100':
        train = (split == 'train')
        dataset = datasets.CIFAR100(root=data_paths['cifar100'],
                                    train=train,
                                    transform=transform,
                                    target_transform=target_transform,
                                    download=download)
        dataset.num_classes = 100
    elif name == 'mnist':
        train = (split == 'train')
        dataset = datasets.MNIST(root=data_paths['mnist'],
                                 train=train,
                                 transform=transform,
                                 target_transform=target_transform,
                                 download=download)
        dataset.num_classes = 10
    elif name == 'svhn':
        if split == 'val': split = 'test'
        dataset = SVHNMIX(root=data_paths['svhn'],
                          split=split,
                          transform=transform,
                          target_transform=target_transform,
                          download=download)
        dataset.num_classes = 10
    elif name == 'imagenet':
        dataset = datasets.ImageFolder(root=data_paths[name][split],
                                       transform=transform,
                                       target_transform=target_transform)
        dataset.num_classes = 1000

    return dataset


def get_transform(dataset, split):
    mean = data_means[dataset]
    std = data_stds[dataset]
    if 'cifar10' in dataset:
        t = {
            'train':
            transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'val':
            transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        }
    elif 'cifar100' in dataset:
        t = {
            'train':
            transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'val':
            transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        }
    elif 'mnist' in dataset:
        t = {
            'train': transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)]),
            'val': transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        }
    elif 'svhn' in dataset:
        t = {
            'train':
            transforms.Compose([transforms.Resize(40),
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std)]),
            'train_extra':
            transforms.Compose([transforms.Resize(40),
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std)]),
            'val':
            transforms.Compose([transforms.Resize(40),
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std)])
        }
    elif 'imagenet' in dataset:
        t = {
            'train':
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'val':
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }
    return t[split]
