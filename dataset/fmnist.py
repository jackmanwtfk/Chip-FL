import torchvision
import torchvision.transforms as transforms

from common import DIR_FMNIST
from .base import DataManager



# ==================================================
#                Fashion MNIST
# ==================================================

FMNIST_AVG = (0.2860, )
FMNIST_STD_VARIANCE = (0.3540, )

fmnist_config = {
    'name' : 'fmnist',
    'nclasses' : 10,
    'height' : 28,
    'width' : 28,
    'channels' : 1
}


class FMnistDM(DataManager):
    def __init__(self, batch_size, resize):
        super().__init__('fmnist', batch_size, resize)

    def get_datasets(self):
        transform = transforms.Compose([
            transforms.Resize(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(FMNIST_AVG, FMNIST_STD_VARIANCE)
        ])
    
        train_set = torchvision.datasets.FashionMNIST(
            root = DIR_FMNIST,
            train=True,
            # download=True,
            transform=transform,
        )

        test_set = torchvision.datasets.FashionMNIST(
            root=DIR_FMNIST,
            train=False,
            # download=True,
            transform=transform
        )
        return train_set, test_set
