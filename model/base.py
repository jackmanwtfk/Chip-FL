import torch.nn as nn


def make_conv_layer(in_channels, out_channels, kerner_size, stride, padding, pool=[]):
    if pool:
        conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kerner_size,
                stride=stride,
                padding=padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool[0], stride=pool[1])
        )
    else:
        conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kerner_size,
                stride=stride,
                padding=padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    return conv


def make_fc_layer(in_channels, out_channels, is_last=False):
    if is_last:
        fc = nn.Sequential(
            nn.Linear(in_channels, out_channels)
        )
    else:
        fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_channels, out_channels),
            nn.ReLU(inplace=True)
        )
    return fc