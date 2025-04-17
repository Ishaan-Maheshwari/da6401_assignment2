# models/cnn_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_activation(name):
    name = name.lower()
    if name == 'relu':
        return nn.ReLU()
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'leakyrelu':
        return nn.LeakyReLU()
    else:
        raise ValueError(f"Unsupported activation: {name}")

class FlexibleCNN(nn.Module):
    def __init__(self, 
                 conv_filters=[32, 64, 128, 256, 256],
                 kernel_sizes=[3, 3, 3, 3, 3],
                 activation='relu',
                 dense_neurons=128,
                 num_classes=10,
                 use_batchnorm=False,
                 dropout=0.25):
        super().__init__()

        assert len(conv_filters) == len(kernel_sizes), "filters and kernel_sizes must match"

        self.activation_name = activation
        self.use_batchnorm = use_batchnorm
        self.conv_layers = nn.ModuleList()
        in_channels = 3  # RGB images

        for out_channels, kernel_size in zip(conv_filters, kernel_sizes):
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
                get_activation(activation),
                nn.MaxPool2d(kernel_size=2)
            ))
            in_channels = out_channels

        self.flatten = nn.Flatten()

        # Dummy input to calculate flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            for layer in self.conv_layers:
                dummy = layer(dummy)
            self.feature_dim = dummy.numel()

        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, dense_neurons),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(dense_neurons, num_classes)
        )

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
