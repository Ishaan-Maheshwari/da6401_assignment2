import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(name):
    """Returns activation layer from string."""
    name = name.lower()
    if name == 'relu':
        return nn.ReLU()
    elif name == 'leakyrelu':
        return nn.LeakyReLU()
    elif name == 'gelu':
        return nn.GELU()
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise ValueError(f"Unsupported activation: {name}")


class FlexibleCNN(nn.Module):
    def __init__(
        self,
        input_channels=3,
        conv_filters=[16, 32, 64, 128, 256],
        kernel_sizes=[3, 3, 3, 3, 3],
        activation='relu',
        dense_neurons=128,
        num_classes=10,
        use_batchnorm=False,
        dropout=0
    ):
        super(FlexibleCNN, self).__init__()

        self.conv_blocks = nn.Sequential()
        in_channels = input_channels

        self.conv_filters = conv_filters
        self.kernel_sizes = kernel_sizes

        for i, (filters, kernel_size) in enumerate(zip(conv_filters, kernel_sizes)):
            self.conv_blocks.add_module(f'conv{i+1}', nn.Conv2d(in_channels, filters, kernel_size=kernel_size, padding=kernel_size//2))
            
            if use_batchnorm:
                self.conv_blocks.add_module(f'bn{i+1}', nn.BatchNorm2d(filters))
            
            self.conv_blocks.add_module(f'activation{i+1}', get_activation(activation))
            
            if dropout > 0:
                self.conv_blocks.add_module(f'dropout{i+1}', nn.Dropout2d(p=dropout))

            self.conv_blocks.add_module(f'maxpool{i+1}', nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = filters


        # Calculate output feature map size after convolutions
        self.feature_size = self._get_flattened_size(input_shape=(3, 224, 224))

        self.fc1 = nn.Linear(self.feature_size, dense_neurons)
        self.act_fc = get_activation(activation)
        self.fc_out = nn.Linear(dense_neurons, num_classes)

    def _get_flattened_size(self, input_shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            output = self.conv_blocks(dummy_input)
            return output.view(1, -1).size(1)

    def forward(self, x):
        x = self.conv_blocks(x)
        x = x.view(x.size(0), -1)
        x = self.act_fc(self.fc1(x))
        x = self.fc_out(x)
        return x
    
    def analyze_model_computation(self, input_shape=(3, 224, 224), dense_neurons=128):
        C, H, W = input_shape
        k_values = self.kernel_sizes
        m_values = self.conv_filters
        n = dense_neurons

        total_ops = 0
        total_params = 0

        print(f"{'Layer':<8} | {'Output Size':<15} | {'#Params':<15} | {'#Computations'}")
        print("-" * 70)

        in_channels = C
        for i, (out_channels, k) in enumerate(zip(m_values, k_values)):
            # Conv Layer
            params = out_channels * (in_channels * k * k + 1)  # +1 for bias
            output_H = H
            output_W = W
            ops = output_H * output_W * out_channels * (in_channels * k * k)

            total_params += params
            total_ops += ops

            print(f"Conv{i+1:<4}  | {out_channels}x{output_H}x{output_W:<10} | {params:<15,} | {ops:<15,}")

            # MaxPool halves H and W
            H //= 2
            W //= 2
            in_channels = out_channels

        # After 5 conv blocks: Flattened size
        flattened = in_channels * H * W

        # Dense Layer
        dense_params = flattened * n + n
        dense_ops = flattened * n
        total_params += dense_params
        total_ops += dense_ops
        print(f"{'Dense':<8}   | {n} neurons         | {dense_params:<15,} | {dense_ops:<15,}")

        # Output Layer
        out_params = n * 10 + 10
        out_ops = n * 10
        total_params += out_params
        total_ops += out_ops
        print(f"{'Output':<8}  | 10 classes         | {out_params:<15,} | {out_ops:<15,}")

        print("-" * 70)
        print(f"{'TOTAL':<8} |                     | {total_params:<15,} | {total_ops:<15,}")

        return {
            "total_params": total_params,
            "total_ops": total_ops
        }
