import torchvision
from torchvision.transforms import transforms

from common import DIR_CIFAR10
from .base import DataManager


CIFAR10_TRAIN_AVG = (0.4914, 0.4822, 0.4465)
CIFAR10_TRAIN_STD = (0.2023, 0.1994, 0.2010)
CIFAR10_TEST_AVG  = (0.485, 0.456, 0.406)
CIFAR10_TEST_STD  = (0.229, 0.224, 0.225)

cifar10_config = {
    'name' : 'cifar10',
    'nclasses' : 10,
    'height' : 32,
    'width' : 32,
    'channels' : 3
}

class CIFAR10DM(DataManager):
    def __init__(self, batch_size, resize):
        super().__init__('cifar10', batch_size, resize)

    def get_datasets(self):
        train_transform = transforms.Compose([
            transforms.Resize(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_TRAIN_AVG, CIFAR10_TRAIN_STD)
        ])

        test_transform = transforms.Compose([
            transforms.Resize(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_TEST_AVG, CIFAR10_TEST_STD)
        ])

        train_set = torchvision.datasets.CIFAR10(
            root=DIR_CIFAR10,
            train=True,
            # download=True,
            transform=train_transform
        )

        test_set = torchvision.datasets.CIFAR10(
            root=DIR_CIFAR10,
            train=False,
            # download=True,
            transform=test_transform
        )

        return train_set, test_set
