# Template for conditional CNN

import equinox as eqx


class CNN(eqx.Module):
    conv: eqx.nn.Conv2d
    pool: eqx.nn.Module
    flatten: eqx.nn.functional.Flatten
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int
    ):
        super().__init__()
        self.conv = eqx.nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.pool = eqx.nn.functional.max_pool2d(kernel_size=2, stride=2)
        self.flatten = eqx.nn.functional.flatten
        self.linear1 = eqx.nn.Linear(
            out_channels * 7 * 7, 64
        )  # Adjust output size based on input
        self.linear2 = eqx.nn.Linear(64, 10)  # 10 for 10 class classification

    def forward(self, x):
        x = self.conv(x)
        x = eqx.nn.functional.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = eqx.nn.functional.relu(x)
        x = self.linear2(x)
        return x


# Example Usage
model = CNN(
    3, 16, 3, 1
)  # 3 input channels (RGB), 16 filters, kernel size 3x3, stride 1

# Define your input data (x) and labels (y)
# ...

# Forward pass
output = model(x)

# Define loss function and optimizer (not included for brevity)
# ...

# Train the model
# ...
