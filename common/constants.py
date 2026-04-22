import os
import enum
import logging

# ==================================================
#                   Path constants
# ==================================================

ROOT_PATH = ""

DIR_DATA = os.path.join("", "data")

DIR_DATASETS = os.path.join(DIR_DATA, "datasets")
DIR_MNIST = os.path.join(DIR_DATASETS, "mnist")
DIR_FMNIST = os.path.join(DIR_DATASETS, "fashion-mnist")
DIR_CIFAR10 = os.path.join(DIR_DATASETS, 'cifar10')

DIR_RESULTS = os.path.join(ROOT_PATH, "results")

# ==================================================
#                     Log
# ==================================================

logging.basicConfig(
    filename="training.log",
    level=logging.DEBUG,
)

# ==================================================
#             Global training constants
# ==================================================

SEED = 73

MNIST_CLASSES = 10

# ==================================================
#             Global Enumerations
# ==================================================

class Optimizer(enum.Enum):
    SGD = 1

class Criterion(enum.Enum):
    CrossEntropyLoss = 1

class LRScheduler(enum.Enum):
    StepLR = 1
    ReduceLROnPlateau = 2


# ==================================================
#             Network
# ==================================================

HOST_IP = 'localhost'
HOST_PORT = 9000
