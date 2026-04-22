import torch
import torch.nn as nn

from .base import make_conv_layer, make_fc_layer

vgg16_config = {
    'input_size' : (227,227)
}


class VGG16(nn.Module):
    def __init__(self, in_channels, nclasses):
        super(VGG16, self).__init__()
        self.nclasses = nclasses

        # ----- features -----
        self.layer1  = make_conv_layer(in_channels, 64,  3, 1, 1)
        self.layer2  = make_conv_layer(64,  64,  3, 1, 1, [2,2])
        self.layer3  = make_conv_layer(64,  128, 3, 1, 1)
        self.layer4  = make_conv_layer(128, 128, 3, 1, 1, [2,2])
        self.layer5  = make_conv_layer(128, 256, 3, 1, 1)
        self.layer6  = make_conv_layer(256, 256, 3, 1, 1)
        self.layer7  = make_conv_layer(256, 256, 3, 1, 1, [2,2])
        self.layer8  = make_conv_layer(256, 512, 3, 1, 1)
        self.layer9  = make_conv_layer(512, 512, 3, 1, 1)
        self.layer10 = make_conv_layer(512, 512, 3, 1, 1, [2,2])
        self.layer11 = make_conv_layer(512, 512, 3, 1, 1)
        self.layer12 = make_conv_layer(512, 512, 3, 1, 1)
        self.layer13 = make_conv_layer(512, 512, 3, 1, 1, [2,2])

        # ----- classifiers -----
        self.fc1 = make_fc_layer(25088, 4096)
        self.fc2 = make_fc_layer(4096, 4096)
        self.fc3 = make_fc_layer(4096, self.nclasses, True)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)

        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

