from typing import List, Optional, Callable

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models.resnet import Bottleneck, conv1x1, conv3x3


class BasicBlockL(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlockL, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes, )
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x: Tensor) -> List[Tensor]:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return [out, x]


class BasicBlockU(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlockU, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, )
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x: List[Tensor]) -> Tensor:

        identity = x[1]
        out = x[0]
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x[1])
        out += identity
        out = self.relu(out)
        return out


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes, )
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:

        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ClassifierModule(nn.Module):
    def __init__(self, channel, num_classes):
        super(ClassifierModule, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channel, num_classes)

    def forward(self, x):
        if type(x) == list:
            out = x[0]
        else:
            out = x
        res = self.avgpool(out)
        res = res.view(res.size(0), -1)
        res = self.fc(res)
        return res


class ResNetEE(nn.Module):
    def __init__(
            self,
            layers: List[int],
            num_classes: int = 10,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            growth=2,
            nClassifiers=8,
            channels=64,
            init_res=False,
            layers_per_classifier = tuple(range(3, 16, 2))
    ) -> None:
        super(ResNetEE, self).__init__()
        self.layers_per_classifier = layers_per_classifier
        block = BasicBlock
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.nClassifiers = nClassifiers
        self.growth = growth
        self.inplanes = channels
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        # self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        # self.bn1 = norm_layer(self.inplanes, )
        # self.relu = nn.ReLU(inplace=True)
        self.blocks = nn.ModuleList()
        self.classifiers = nn.ModuleList()
        # self.firstblock = nn.Sequential()
        for i in range(len(layers)):
            cIn = int(channels * (growth) ** i * block.expansion)
            self.blocks.extend(self._make_layer(cIn, layers[i], stride=1 if i <= 1 else 2, firstblock=(i == 0)))

        for i in range(len(layers)):
            # if (i + 1) in self.layers_per_classifier:
            cIn = int(channels * (growth) ** i * block.expansion)
            if i == 0:
                for j in range(layers[i] * 2 - 1):
                    self.classifiers.append(ClassifierModule(cIn, num_classes))
            else:
                for j in range(layers[i] * 2):
                    self.classifiers.append(ClassifierModule(cIn, num_classes))
        
        self.set_nBlocks(nClassifiers)

    def set_nBlocks(self, nClassifiers):
        self.nClassifiers = nClassifiers
        try:
            self.nBlocks = self.layers_per_classifier[nClassifiers - 1]
        except:
            raise Exception('Make sure resnet_layers_per_classifier is configured so that each classifier is assigned with a layer (e.g., 2 classifiers --resnet_layers_per_classifier 1-3-4)')
        if hasattr(self, 'blocks'):  # remove extra blocks from memory
            nBlocks = len(self.blocks)
            for i in range(nBlocks - 1, self.nBlocks - 1, -1):
                del self.classifiers[i]
                del self.blocks[i]

    def _make_layer(self, planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False, firstblock=False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * BasicBlock.expansion, stride),
                norm_layer(planes * BasicBlock.expansion, ),
            )
        if firstblock:
            # CIFAR10: kernel_size 7 -> 3, stride 2 -> 1, padding 3->1
            layers = [nn.Sequential(nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                                    norm_layer(self.inplanes, ),
                                    nn.ReLU(inplace=True))]  # Cifar-10 / Cifar-100 first block
        else:
            layers = [BasicBlockL(self.inplanes, planes, stride, downsample, self.groups,
                                  self.base_width, previous_dilation, norm_layer),
                      BasicBlockU(self.inplanes, planes, stride, downsample, self.groups,
                                  self.base_width, previous_dilation, norm_layer)]

        self.inplanes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlockL(self.inplanes, planes, groups=self.groups,
                                      base_width=self.base_width, dilation=self.dilation,
                                      norm_layer=norm_layer))
            layers.append(BasicBlockU(self.inplanes, planes, groups=self.groups,
                                      base_width=self.base_width, dilation=self.dilation,
                                      norm_layer=norm_layer))

        return layers

    def _forward_impl(self, x: Tensor) -> List[Tensor]:
        res = []
        for i in range(self.nBlocks):
            layer = self.blocks[i]
            x = layer(x)
            if (i + 1) in self.layers_per_classifier:  # Skip some blocks
                res.append(self.classifiers[i](x))
        return res

    def forward(self, x: Tensor) -> List[Tensor]:
        return self._forward_impl(x)


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = (ResNetEE([2] * 4, nClassifiers=7, growth=2, init_res=False)).to(device)
    model.train()
    input = torch.autograd.Variable(torch.randn(1, 3, 32, 32))
    print(model(input))
