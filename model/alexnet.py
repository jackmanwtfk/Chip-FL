import torch
import torch.nn as nn

from .base import make_conv_layer, make_fc_layer


alex_config = {
    'input_size' : (227,227)
}


class AlexNet(nn.Module):
    
    def __init__(self, in_channels, nclasses):
        super(AlexNet, self).__init__()
        self.nclasses = nclasses

        self.layer1 = make_conv_layer(in_channels, 96, 11, 4, 0, [3,2])
        self.layer2 = make_conv_layer(96, 256, 5, 1, 2, [3,2])
        self.layer3 = make_conv_layer(256, 384, 3, 1, 1)
        self.layer4 = make_conv_layer(384, 384, 3, 1, 1)
        self.layer5 = make_conv_layer(384, 256, 3, 1, 1, [3,2])

        self.fc1 = make_fc_layer(9216, 4096)
        self.fc2 = make_fc_layer(4096, 4096)
        self.fc3 = make_fc_layer(4096, self.nclasses, True)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)

        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out