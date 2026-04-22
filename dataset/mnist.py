import torchvision
import torchvision.transforms as transforms

from common import DIR_MNIST
from .base import DataManager


# ==================================================
#                   MNIST
# ==================================================

MNIST_AVG = (0.1307, )
MNIST_STD_VARIANCE = (0.3081, )

mnist_config = {
    'name' : 'mnist',
    'nclasses' : 10,
    'height' : 28,
    'width' : 28,
    'channels' : 1,
}

class MnistDM(DataManager):
    def __init__(self, batch_size, resize):
        super().__init__('mnist', batch_size, resize)

    def get_datasets(self):
        transform = transforms.Compose([
            transforms.Resize(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(MNIST_AVG, MNIST_STD_VARIANCE)
        ])

        train_set = torchvision.datasets.MNIST(
            root=DIR_MNIST,
            train=True,
            # download=True,
            transform=transform
        )

        test_set = torchvision.datasets.MNIST(
            root=DIR_MNIST,
            train=False,
            # download=True,
            transform=transform
        )
        return train_set, test_set
