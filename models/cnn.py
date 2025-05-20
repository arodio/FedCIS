import torch.nn as nn
from torch import Tensor
from typing import List

class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, apply_pooling: bool = True):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.apply_pooling = apply_pooling
        if apply_pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.relu(x)
        if self.apply_pooling:
            x = self.pool(x)
        return x

class Classifier(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))  # Adaptive pooling
        x = x.view(x.size(0), -1)  # Flatten the tensor
        out = self.fc(x)
        return out

class CNNEE(nn.Module):
    def __init__(self, model_config: List[int], nClassifiers : int, num_classes: int = 10, initial_channels: int = 64, increase_factor: float = 1.25):
        super(CNNEE, self).__init__()
        self.blocks = nn.ModuleList()
        self.classifiers = nn.ModuleDict()
        self.nClassifiers = nClassifiers
        self.layers_per_classifier = model_config

        in_channels = 3  # Start with 3 input channels for RGB images
        for block_number in range(max(model_config) + 1):
            out_channels = min(int(initial_channels * (increase_factor ** block_number)), initial_channels * 2**5)  # Limiting max channels to prevent too large feature maps
            apply_pooling = block_number < 4  # Apply pooling only for the first few blocks
            stride = 1 if block_number < 4 else 2  # Use stride 1 for the first few blocks

            self.blocks.append(BasicBlock(in_channels, out_channels, stride, apply_pooling))
            in_channels = out_channels  # Update in_channels for the next block

            if block_number in model_config:
                self.classifiers[str(block_number)] = Classifier(out_channels, num_classes)

    def forward(self, x: Tensor) -> List[Tensor]:
        res = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if str(i) in self.classifiers:
                res.append(self.classifiers[str(i)](x))
        return res
