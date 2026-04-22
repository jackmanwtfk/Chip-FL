from .mnist import mnist_config
from .fmnist import fmnist_config
from .cifar10 import cifar10_config



DatasetConfig = {
    'mnist' : mnist_config,
    'fmnist' : fmnist_config,
    'cifar10' : cifar10_config,
}