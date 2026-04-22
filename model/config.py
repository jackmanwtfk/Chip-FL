from .alexnet import alex_config
from .lenet import lenet_config
from .vgg import vgg16_config


ModelConfig = {
    'lenet' : lenet_config,
    'alexnet' : alex_config,
    'vgg16' : vgg16_config,
}