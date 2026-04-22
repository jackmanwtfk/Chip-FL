import torch
import torch.nn as nn

from .base import make_conv_layer, make_fc_layer


lenet_config = {
    'input_size' : (28,28)
}

# ==================================================
#                    LeNet12
# ==================================================
class LeNet12(nn.Module):
    def __init__(self, in_channels, nclasses):
        super(LeNet12, self).__init__()
        self.nclasses = nclasses

        # ----- features -----
        self.conv1 = make_conv_layer(in_channels, 32, 3, 1, 1)
        self.conv2 = make_conv_layer(32, 32, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = make_conv_layer(32, 64, 3, 1, 1)
        self.conv4 = make_conv_layer(64, 64, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(kernel_size=torch.Size([2, 2]))
        self.conv5 = make_conv_layer(64, 128, 3, 1, 1)
        self.conv6 = make_conv_layer(128, 128, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(kernel_size=torch.Size([2, 2]))

        # ----- classifiers -----
        self.fc1 = make_fc_layer(128*3*3, 382)
        self.fc2 = make_fc_layer(382, 192)
        self.fc3 = make_fc_layer(192, self.nclasses, True)
        self.classifiers = [self.fc1, self.fc2, self.fc3]


        self.features = [
            self.conv1, self.conv2, self.pool1,
            self.conv3, self.conv4, self.pool2,
            self.conv5, self.conv6, self.pool3
        ]
        self.layers = self.features + self.classifiers

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.pool1(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.pool2(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.pool3(out)

        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

    def split_model(self, cut_layer=3):
        """
            Split model into client side and server side.

            @cut_layer: default to be 3 (after POOL1)
        """
        assert cut_layer > 0

        if cut_layer < len(self.features):
            dmodel = nn.Sequential(*self.features[:cut_layer])
            smodel = nn.Sequential(
                *self.features[cut_layer:],
                nn.Flatten(),
                *self.classifiers
            )
        elif cut_layer == len(self.features):
            dmodel = nn.Sequential(*self.features, nn.Flatten())
            smodel = nn.Sequential(*self.classifiers)
        else:
            dmodel = nn.Sequential(
                *self.features[:cut_layer],
                nn.Flatten(),
                *self.classifiers[:cut_layer-len(self.features)]
            )
            smodel = nn.Sequential(*self.classifiers[cut_layer-len(self.features):])

        return dmodel, smodel


# ==================================================
#                LeNet12 for Ampere
# ==================================================

class LeNet12Ampere(nn.Module):
    def __init__(self, in_channels, nclasses, is_client = False):
        super(LeNet12Ampere, self).__init__()
        self.nclasses = nclasses
        self.is_client = is_client
        self.is_activation = False

        if self.is_client:
            # ----- client -----
            self.conv1 = make_conv_layer(in_channels, 32, 3, 1, 1)
            self.conv2 = make_conv_layer(32, 32, 3, 1, 1, [2,2])
            self.fc1 = make_fc_layer(32*14*14, nclasses, True)

            self.features = [self.conv1, self.conv2]
            self.classifiers = [self.fc1]
        else:
            # ----- server -----
            self.conv2 = make_conv_layer(32, 32, 3, 1, 1)
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv3 = make_conv_layer(32, 64, 3, 1, 1)
            self.conv4 = make_conv_layer(64, 64, 3, 1, 1)
            self.pool2 = nn.MaxPool2d(kernel_size=torch.Size([2, 2]))
            self.conv5 = make_conv_layer(64, 128, 3, 1, 1)
            self.conv6 = make_conv_layer(128, 128, 3, 1, 1)
            self.pool3 = nn.MaxPool2d(kernel_size=torch.Size([2, 2]))
            self.fc1 = make_fc_layer(128*3*3, 382)
            self.fc2 = make_fc_layer(382, 192)
            self.fc3 = make_fc_layer(192, nclasses, True)

            self.features = [
                self.conv2, self.pool1,
                self.conv3, self.conv4, self.pool2,
                self.conv5, self.conv6, self.pool3,
            ]
            self.classifiers = [
                self.fc1, self.fc2, self.fc3
            ]

    def forward(self, x):
        if self.is_client:
            out = self.conv1(x)
            if not self.is_activation:
                out = self.conv2(out)
                out = torch.flatten(out, 1)
                out = self.fc1(out)
        else:
            out = self.conv2(x)
            out = self.pool1(out)
            out = self.conv3(out)
            out = self.conv4(out)
            out = self.pool2(out)
            out = self.conv5(out)
            out = self.conv6(out)
            out = self.pool3(out)

            out = torch.flatten(out, 1)
            out = self.fc1(out)
            out = self.fc2(out)
            out = self.fc3(out)
        return out


"""
class LeNet12AmpereClient(nn.Module):
    def __init__(self, num_classes):
        super(LeNet12AmpereClient, self).__init__()

        self.conv1 = make_conv_layer(1, 32, 3, 1, 1)
        self.conv2 = make_conv_layer(32, 32, 3, 1, 1, [2,2])
        self.fc1 = make_fc_layer(32*14*14, num_classes, True)

        self.send_to_server = False

    def forward(self, x):
        out = self.conv1(x)
        if self.send_to_server:
            return out
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        return out

class LeNet12AmpereServer(nn.Module):
    def __init__(self, num_classes):
        super(LeNet12AmpereServer, self).__init__()

        self.conv2 = make_conv_layer(32, 32, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = make_conv_layer(32, 64, 3, 1, 1)
        self.conv4 = make_conv_layer(64, 64, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(kernel_size=torch.Size([2, 2]))
        self.conv5 = make_conv_layer(64, 128, 3, 1, 1)
        self.conv6 = make_conv_layer(128, 128, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(kernel_size=torch.Size([2, 2]))

        self.fc1 = make_fc_layer(128*3*3, 382)
        self.fc2 = make_fc_layer(382, 192)
        self.fc3 = make_fc_layer(192, num_classes, True)

    def forward(self, x):
        out = self.conv2(x)
        out = self.pool1(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.pool2(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.pool3(out)

        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out
"""