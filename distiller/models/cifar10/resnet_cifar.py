#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Resnet for CIFAR10

Resnet for CIFAR10, based on "Deep Residual Learning for Image Recognition".
This is based on TorchVision's implementation of ResNet for ImageNet, with appropriate
changes for the 10-class Cifar-10 dataset.
This ResNet also has layer gates, to be able to dynamically remove layers.

@inproceedings{DBLP:conf/cvpr/HeZRS16,
  author    = {Kaiming He and
               Xiangyu Zhang and
               Shaoqing Ren and
               Jian Sun},
  title     = {Deep Residual Learning for Image Recognition},
  booktitle = {{CVPR}},
  pages     = {770--778},
  publisher = {{IEEE} Computer Society},
  year      = {2016}
}

"""
import os
import sys
import torch
import torch.nn as nn
import math
from collections import OrderedDict
import torch.utils.model_zoo as model_zoo
from distiller.modules.special_conv import Conv2d_TrimPsum

__all__ = ['resnet10_cifar', 'resnet20_cifar', 'resnet32_cifar', 'resnet44_cifar', 'resnet56_cifar']

model_pretrained = {
    'resnet10_cifar': os.path.join('D:', os.sep, 'playground', 'distiller', 'examples', 'classifier_compression', 'checkpoint', 'checkpoint_fp32_retrain_0.pth'),
    'resnet20_cifar': os.path.join('D:', os.sep, 'playground', 'distiller', 'examples', 'classifier_compression', 'logs', '2019.10.08-110134', 'checkpoint_new.pth.tar')
}

NUM_CLASSES = 10

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class SlicingLinearBlock(nn.Module):
    def __init__(self, in_features, out_features, bias=False, ch_group=8):
        super(SlicingLinearBlock, self).__init__()
        self.num_of_slice = 1
        self.ch_group = ch_group
        try:
            if(in_features%ch_group == 0):
                self.num_of_slice = in_features//self.ch_group
        except ValueError as e:
            print("Exception: {0}() value error({1}): {2}".format(__name__, e.errno, e.strerror))
            raise ValueError from e
        except: ## handle other exceptions such as attribute errors
            print("Unexpected error:", sys.exc_info()[0])
            sys.exit()
        self.fc_0 = nn.Linear(self.ch_group, out_features, bias=bias)
        self.fc_1 = nn.Linear(self.ch_group, out_features, bias=bias)
        if (self.num_of_slice > 2):
            self.fc_2 = nn.Linear(self.ch_group, out_features, bias=bias)
            self.fc_3 = nn.Linear(self.ch_group, out_features, bias=bias)
        if (self.num_of_slice > 4):
            self.fc_4 = nn.Linear(self.ch_group, out_features, bias=bias)
            self.fc_5 = nn.Linear(self.ch_group, out_features, bias=bias)
            self.fc_6 = nn.Linear(self.ch_group, out_features, bias=bias)
            self.fc_7 = nn.Linear(self.ch_group, out_features, bias=bias)
        if (self.num_of_slice > 8):
            self.fc_8 = nn.Linear(self.ch_group, out_features, bias=bias)
            self.fc_9 = nn.Linear(self.ch_group, out_features, bias=bias)
            self.fc_10 = nn.Linear(self.ch_group, out_features, bias=bias)
            self.fc_11 = nn.Linear(self.ch_group, out_features, bias=bias)
            self.fc_12 = nn.Linear(self.ch_group, out_features, bias=bias)
            self.fc_13 = nn.Linear(self.ch_group, out_features, bias=bias)
            self.fc_14 = nn.Linear(self.ch_group, out_features, bias=bias)
            self.fc_15 = nn.Linear(self.ch_group, out_features, bias=bias)
        if (self.num_of_slice > 16):
            self.fc_16 = nn.Linear(self.ch_group, out_features, bias=bias)
            self.fc_17 = nn.Linear(self.ch_group, out_features, bias=bias)
            self.fc_18 = nn.Linear(self.ch_group, out_features, bias=bias)
            self.fc_19 = nn.Linear(self.ch_group, out_features, bias=bias)
            self.fc_20 = nn.Linear(self.ch_group, out_features, bias=bias)
            self.fc_21 = nn.Linear(self.ch_group, out_features, bias=bias)
            self.fc_22 = nn.Linear(self.ch_group, out_features, bias=bias)
            self.fc_23 = nn.Linear(self.ch_group, out_features, bias=bias)
            self.fc_24 = nn.Linear(self.ch_group, out_features, bias=bias)
            self.fc_25 = nn.Linear(self.ch_group, out_features, bias=bias)
            self.fc_26 = nn.Linear(self.ch_group, out_features, bias=bias)
            self.fc_27 = nn.Linear(self.ch_group, out_features, bias=bias)
            self.fc_28 = nn.Linear(self.ch_group, out_features, bias=bias)
            self.fc_29 = nn.Linear(self.ch_group, out_features, bias=bias)
            self.fc_30 = nn.Linear(self.ch_group, out_features, bias=bias)
            self.fc_31 = nn.Linear(self.ch_group, out_features, bias=bias)

    def forward(self, x):
        _x = []
        _start = 0
        for i in range(0, self.num_of_slice, 1):
            _x.append(torch.narrow(x, 1, _start, self.ch_group))
            _start += self.ch_group
        if (self.num_of_slice == 2):  # 16
            out = self.fc_0(_x[0]) + self.fc_1(_x[1])
        elif (self.num_of_slice == 4):  # 32
            out = self.fc_0(_x[0]) + self.fc_1(_x[1]) + self.fc_2(_x[2]) + self.fc_3(_x[3])
        elif (self.num_of_slice == 8):  # 64
            out = self.fc_0(_x[0]) + self.fc_1(_x[1]) + self.fc_2(_x[2]) + self.fc_3(_x[3]) \
                  + self.fc_4(_x[4]) + self.fc_5(_x[5]) + self.fc_6(_x[6]) + self.fc_7(_x[7])
        elif (self.num_of_slice == 16):  # 128
            out = self.fc_0(_x[0]) + self.fc_1(_x[1]) + self.fc_2(_x[2]) + self.fc_3(_x[3]) \
                  + self.fc_4(_x[4]) + self.fc_5(_x[5]) + self.fc_6(_x[6]) + self.fc_7(_x[7]) \
                  + self.fc_8(_x[8]) + self.fc_9(_x[9]) + self.fc_10(_x[10]) + self.fc_11(_x[11]) \
                  + self.fc_12(_x[12]) + self.fc_13(_x[13]) + self.fc_14(_x[14]) + self.fc_15(_x[15])
        elif (self.num_of_slice == 32):  # 256
            out = self.fc_0(_x[0]) + self.fc_1(_x[1]) + self.fc_2(_x[2]) + self.fc_3(_x[3]) \
                  + self.fc_4(_x[4]) + self.fc_5(_x[5]) + self.fc_6(_x[6]) + self.fc_7(_x[7]) \
                  + self.fc_8(_x[8]) + self.fc_9(_x[9]) + self.fc_10(_x[10]) + self.fc_11(_x[11]) \
                  + self.fc_12(_x[12]) + self.fc_13(_x[13]) + self.fc_14(_x[14]) + self.fc_15(_x[15]) \
                  + self.fc_16(_x[16]) + self.fc_17(_x[17]) + self.fc_18(_x[18]) + self.fc_19(_x[19]) \
                  + self.fc_20(_x[20]) + self.fc_21(_x[21]) + self.fc_22(_x[22]) + self.fc_23(_x[23]) \
                  + self.fc_24(_x[24]) + self.fc_25(_x[25]) + self.fc_26(_x[26]) + self.fc_27(_x[27]) \
                  + self.fc_28(_x[28]) + self.fc_29(_x[29]) + self.fc_30(_x[30]) + self.fc_31(_x[31])
        else:
            print('Warning: num_of_slice is {0}, should not use SlicingLinearBlock'.format(self.num_of_slice))
        return out

class SlicingBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False, ch_group=8):
        super(SlicingBlock, self).__init__()
        self.num_of_slice = 1
        self.ch_group = ch_group
        try:
            if(inplanes%ch_group == 0):
                self.num_of_slice = inplanes//self.ch_group
        except ValueError as e:
            print("Exception: {0}() value error({1}): {2}".format(__name__, e.errno, e.strerror))
            raise ValueError from e
        except: ## handle other exceptions such as attribute errors
            print("Unexpected error:", sys.exc_info()[0])
            sys.exit()
        self.conv_0 = nn.Conv2d(self.ch_group, \
                    planes, \
                    kernel_size=kernel_size, \
                    stride=stride, \
                    padding=padding, \
                    bias=bias)
        self.conv_1 = nn.Conv2d(self.ch_group, \
                    planes, \
                    kernel_size=kernel_size, \
                    stride=stride, \
                    padding=padding, \
                    bias=bias)
        if(self.num_of_slice > 2):
            self.conv_2 = nn.Conv2d(self.ch_group, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
            self.conv_3 = nn.Conv2d(self.ch_group, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        if(self.num_of_slice > 4):
            self.conv_4 = nn.Conv2d(self.ch_group, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
            self.conv_5 = nn.Conv2d(self.ch_group, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
            self.conv_6 = nn.Conv2d(self.ch_group, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
            self.conv_7 = nn.Conv2d(self.ch_group, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        if(self.num_of_slice > 8):
            self.conv_8 = nn.Conv2d(self.ch_group, planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias=bias)
            self.conv_9 = nn.Conv2d(self.ch_group, planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias=bias)
            self.conv_10 = nn.Conv2d(self.ch_group, planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias=bias)
            self.conv_11 = nn.Conv2d(self.ch_group, planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias=bias)
            self.conv_12 = nn.Conv2d(self.ch_group, planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias=bias)
            self.conv_13 = nn.Conv2d(self.ch_group, planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias=bias)
            self.conv_14 = nn.Conv2d(self.ch_group, planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias=bias)
            self.conv_15 = nn.Conv2d(self.ch_group, planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias=bias)
        if (self.num_of_slice > 16):
            self.conv_16 = nn.Conv2d(self.ch_group, planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias=bias)
            self.conv_17 = nn.Conv2d(self.ch_group, planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias=bias)
            self.conv_18 = nn.Conv2d(self.ch_group, planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias=bias)
            self.conv_19 = nn.Conv2d(self.ch_group, planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias=bias)
            self.conv_20 = nn.Conv2d(self.ch_group, planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias=bias)
            self.conv_21 = nn.Conv2d(self.ch_group, planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias=bias)
            self.conv_22 = nn.Conv2d(self.ch_group, planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias=bias)
            self.conv_23 = nn.Conv2d(self.ch_group, planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias=bias)
            self.conv_24 = nn.Conv2d(self.ch_group, planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias=bias)
            self.conv_25 = nn.Conv2d(self.ch_group, planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias=bias)
            self.conv_26 = nn.Conv2d(self.ch_group, planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias=bias)
            self.conv_27 = nn.Conv2d(self.ch_group, planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias=bias)
            self.conv_28 = nn.Conv2d(self.ch_group, planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias=bias)
            self.conv_29 = nn.Conv2d(self.ch_group, planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias=bias)
            self.conv_30 = nn.Conv2d(self.ch_group, planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias=bias)
            self.conv_31 = nn.Conv2d(self.ch_group, planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias=bias)

    def forward(self, x):
        _x = []
        _start = 0
        for i in range(0, self.num_of_slice, 1):
            _x.append(torch.narrow(x, 1, _start, self.ch_group))
            _start += self.ch_group
        if(self.num_of_slice == 2): # 16
            out = self.conv_0(_x[0])+self.conv_1(_x[1])
        elif(self.num_of_slice == 4): # 32
            out = self.conv_0(_x[0])+self.conv_1(_x[1])+self.conv_2(_x[2])+self.conv_3(_x[3])
        elif(self.num_of_slice == 8): # 64
            out = self.conv_0(_x[0])+self.conv_1(_x[1])+self.conv_2(_x[2])+self.conv_3(_x[3])\
                 +self.conv_4(_x[4])+self.conv_5(_x[5])+self.conv_6(_x[6])+self.conv_7(_x[7])
        elif (self.num_of_slice == 16):  # 128
            out = self.conv_0(_x[0]) + self.conv_1(_x[1]) + self.conv_2(_x[2]) + self.conv_3(_x[3]) \
                  + self.conv_4(_x[4]) + self.conv_5(_x[5]) + self.conv_6(_x[6]) + self.conv_7(_x[7]) \
                  + self.conv_8(_x[8]) + self.conv_9(_x[9]) + self.conv_10(_x[10]) + self.conv_11(_x[11]) \
                  + self.conv_12(_x[12]) + self.conv_13(_x[13]) + self.conv_14(_x[14]) + self.conv_15(_x[15])
        elif (self.num_of_slice == 32):  # 256
            out = self.conv_0(_x[0]) + self.conv_1(_x[1]) + self.conv_2(_x[2]) + self.conv_3(_x[3]) \
                  + self.conv_4(_x[4]) + self.conv_5(_x[5]) + self.conv_6(_x[6]) + self.conv_7(_x[7]) \
                  + self.conv_8(_x[8]) + self.conv_9(_x[9]) + self.conv_10(_x[10]) + self.conv_11(_x[11]) \
                  + self.conv_12(_x[12]) + self.conv_13(_x[13]) + self.conv_14(_x[14]) + self.conv_15(_x[15]) \
                  + self.conv_16(_x[16]) + self.conv_17(_x[17]) + self.conv_18(_x[18]) + self.conv_19(_x[19]) \
                  + self.conv_20(_x[20]) + self.conv_21(_x[21]) + self.conv_22(_x[22]) + self.conv_23(_x[23]) \
                  + self.conv_24(_x[24]) + self.conv_25(_x[25]) + self.conv_26(_x[26]) + self.conv_27(_x[27]) \
                  + self.conv_28(_x[28]) + self.conv_29(_x[29]) + self.conv_30(_x[30]) + self.conv_31(_x[31])
        else:
            print('Warning: num_of_slice is {0}, should not use SlicingBlock'.format(self.num_of_slice))
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, block_gates, inplanes, planes, stride=1, downsample=None, ch_group=None):
        super(BasicBlock, self).__init__()
        self.block_gates = block_gates
        if(ch_group == None):
            self.conv1 = conv3x3(inplanes, planes, stride)
        else:
            self.conv1 = SlicingBlock(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False, ch_group=8)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=False)  # To enable layer removal inplace must be False
        if (ch_group == None):
            self.conv2 = conv3x3(planes, planes)
        else:
            self.conv2 = SlicingBlock(planes, planes, stride=1, padding=1, bias=False, ch_group=8)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = out = x

        if self.block_gates[0]:
            # print('conv1 input {0}'.format(x.size()))
            out = self.conv1(x)
            # print('conv1 output {0}'.format(out.size()))
            out = self.bn1(out)
            out = self.relu1(out)

        if self.block_gates[1]:
            # print('conv2 input {0}'.format(out.size()))
            out = self.conv2(out)
            # print('conv2 output {0}'.format(out.size()))
            out = self.bn2(out)

        if self.downsample is not None:
            # print('downsample input {0}'.format(x.size()))
            residual = self.downsample(x)
            # print('downsample output {0}'.format(residual.size()))

        out += residual
        out = self.relu2(out)

        return out

class ResNetCifar(nn.Module):

    def __init__(self, block, layers, num_classes=NUM_CLASSES, ch_group=None):
        self.nlayers = 0
        self.ch_group = ch_group
        # Each layer manages its own gates
        self.layer_gates = []
        for layer in range(4):
            # For each of the 3 layers, create block gates: each block has two layers
            self.layer_gates.append([])  # [True, True] * layers[layer])
            for blk in range(layers[layer]):
                self.layer_gates[layer].append([True, True])

        self.inplanes = 32  # 64
        super(ResNetCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)
        if (self.ch_group == None):
            self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv2 = SlicingBlock(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU(inplace=True)
        if (self.ch_group == None):
            self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv3 = SlicingBlock(16, 32, kernel_size=3, stride=1, padding=1, bias=False, ch_group=8)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(self.layer_gates[0], block, 32, layers[0], stride=1, ch_group=ch_group)
        self.layer2 = self._make_layer(self.layer_gates[1], block, 64, layers[1], stride=2, ch_group=ch_group)
        self.layer3 = self._make_layer(self.layer_gates[2], block, 128, layers[2], stride=2, ch_group=ch_group)
        self.layer4 = self._make_layer(self.layer_gates[3], block, 256, layers[3], stride=2, ch_group=ch_group)
        self.avgpool = nn.AvgPool2d(8, stride=8) #nn.AdaptiveAvgPool2d(1)
        if (ch_group == None):
            self.fc = nn.Linear(256 * block.expansion, num_classes)
        else:
            self.fc = SlicingLinearBlock(256 * block.expansion, num_classes)
        self.dropout = nn.Dropout()

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, Conv2d_TrimPsum):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, layer_gates, block, planes, blocks, stride=1, ch_group=None):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            if (ch_group == None):
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    SlicingBlock(self.inplanes, planes * block.expansion, \
                              kernel_size=1, stride=stride, padding=0, bias=False, ch_group=8),
                    nn.BatchNorm2d(planes * block.expansion),
                )

        layers = []
        layers.append(block(layer_gates[0], self.inplanes, planes, \
                            stride=stride, downsample=downsample, ch_group=ch_group))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(layer_gates[i], self.inplanes, planes, \
                                ch_group=ch_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        # print('input {0}'.format(x.size()))

        x = self.conv1(x)
        # print('conv1 output {0}'.format(x.size()))
        x = self.bn1(x)
        x = self.relu1(x)

        # print('conv2 input {0}'.format(x.size()))
        x = self.conv2(x)
        # print('conv2 output {0}'.format(x.size()))
        x = self.bn2(x)
        x = self.relu2(x)

        # print('conv3 input {0}'.format(x.size()))
        x = self.conv3(x)
        # print('conv3 output {0}'.format(x.size()))
        x = self.bn3(x)
        x = self.relu3(x)

        # print('maxpool input {0}'.format(x.size()))
        x = self.maxpool(x)
        # print('maxpool output {0}'.format(x.size()))
        x = self.layer1(x)
        # print('layer1 output {0}'.format(x.size()))
        x = self.layer2(x)
        # print('layer2 output {0}'.format(x.size()))
        x = self.layer3(x)
        # print('layer3 output {0}'.format(x.size()))
        x = self.layer4(x)
        # print('layer4 output {0}'.format(x.size()))
        x = self.dropout(x)

        # print('avgpool input {0}'.format(x.size()))
        x = self.avgpool(x)
        # print('avgpool output {0}'.format(x.size()))
        x = x.view(-1, x.size(1))
        x = self.fc(x)
        # print('fc output {0}'.format(x.size()))
        return x

def resnet10_cifar(pretrained, ch_group, **kwargs):
    model = ResNetCifar(BasicBlock, [1, 1, 1, 1], **kwargs, ch_group=ch_group)
    if pretrained: # no module. prefix is allowed #
        state_dict = torch.load(model_pretrained['resnet10_cifar'])
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove module #
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=True)
    return model

def resnet20_cifar(pretrained, **kwargs):
    model = ResNetCifar(BasicBlock, [3, 3, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(model_pretrained['resnet20_cifar'])['new_state_dict'], strict=False)
    return model

def resnet32_cifar(pretrained, **kwargs):
    model = ResNetCifar(BasicBlock, [5, 5, 5], **kwargs)
    if pretrained:
        model.load_state_dict(model_pretrained['resnet32_cifar'])
    return model

def resnet44_cifar(pretrained, **kwargs):
    model = ResNetCifar(BasicBlock, [7, 7, 7], **kwargs)
    if pretrained:
        model.load_state_dict(model_pretrained['resnet44_cifar'])
    return model

def resnet56_cifar(pretrained, **kwargs):
    model = ResNetCifar(BasicBlock, [9, 9, 9], **kwargs)
    if pretrained:
        model.load_state_dict(model_pretrained['resnet56_cifar'])
    return model
