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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.\\')))
from resnet_slicing_block import *
from resnet_basic_block import *
from net_utils import *

__all__ = ['resnet10_cifar', 'resnet20_cifar', 'resnet32_cifar', 'resnet44_cifar', 'resnet56_cifar']


model_saved = {
    'resnet10_cifar': os.path.join('D:', os.sep, 'playground', 'MyDistiller', 'examples', 'classifier_compression', 'checkpoint', '20191103_resnet10_fp32_ch8_224x', 'checkpoint_224x_ch8_retrain_5_fuse.pth'),
}
model_pretrained = {
    'resnet10_cifar': os.path.join('D:', os.sep, 'playground', 'MyDistiller', 'examples', 'classifier_compression', 'checkpoint', '20191103_resnet10_fp32_ch8_224x', 'checkpoint_224x_ch8_fp32_fuse.pth'),
}

NUM_CLASSES = 10
printLayerInfo = False

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
            self.conv2 = SlicingBlockFused(16, 16, kernel_size=3, stride=1, padding=1, bias=False, ch_group=ch_group)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU(inplace=True)
        if (self.ch_group == None):
            self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv3 = SlicingBlockFused(16, 32, kernel_size=3, stride=1, padding=1, bias=False, ch_group=ch_group)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(self.layer_gates[0], block, 32, layers[0], stride=1, ch_group=ch_group)
        self.layer2 = self._make_layer(self.layer_gates[1], block, 64, layers[1], stride=2, ch_group=ch_group)
        self.layer3 = self._make_layer(self.layer_gates[2], block, 128, layers[2], stride=2, ch_group=ch_group)
        self.layer4 = self._make_layer(self.layer_gates[3], block, 256, layers[3], stride=2, ch_group=ch_group)
        self.avgpool = nn.AvgPool2d(7, stride=7) #nn.AdaptiveAvgPool2d(1)
        if (self.ch_group == None):
            self.fc = nn.Linear(256 * block.expansion, num_classes)
        else:
            self.fc = SlicingLinearBlock(256 * block.expansion, num_classes)
        self.dropout = nn.Dropout()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
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
                    SlicingBlockFused(self.inplanes, planes * block.expansion, \
                              kernel_size=1, stride=stride, padding=0, bias=False, ch_group=ch_group),
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

    def forward(self, x, dump_act=None):
        # print('input {0}'.format(x.size()))
        # print('ResNetCifar {0}'.format(fusion))
        if(dump_act != None):
            dump_to_npy(name=str(dump_act)+'.input', tensor=x)

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
        x = self.dropout(x)
        # print('maxpool output {0}'.format(x.size()))

        x = self.layer1(x)
        x = self.dropout(x)
        # print('layer1 output {0}'.format(x.size()))

        x = self.layer2(x)
        x = self.dropout(x)
        # print('layer2 output {0}'.format(x.size()))

        x = self.layer3(x)
        x = self.dropout(x)
        # print('layer3 output {0}'.format(x.size()))

        x = self.layer4(x)
        x = self.dropout(x)
        # print('layer4 output {0}'.format(x.size()))

        # print('avgpool input {0}'.format(x.size()))
        x = self.avgpool(x)
        # print('avgpool output {0}'.format(x.size()))

        x = x.view(-1, x.size(1))
        x = self.fc(x)
        # print('fc output {0}'.format(x.size()))

        return x

class ResNetCifarReshape(nn.Module):
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
        super(ResNetCifarReshape, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)
        if (self.ch_group == None):
            self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(16)
        else:
            self.conv2 = SlicingBlock(16, 16, kernel_size=3, stride=1, padding=1, bias=False, ch_group=ch_group)
        self.relu2 = nn.ReLU(inplace=True)
        if (self.ch_group == None):
            self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn3 = nn.BatchNorm2d(32)
        else:
            self.conv3 = SlicingBlock(16, 32, kernel_size=3, stride=1, padding=1, bias=False, ch_group=ch_group)
        self.relu3 = nn.ReLU(inplace=True)
        self.poolPadding = nn.ZeroPad2d((0, 1, 0, 1)) # left, right, top, bottom
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer1 = self._make_layer(self.layer_gates[0], block, 32, layers[0], stride=1, ch_group=ch_group)
        self.layer2 = self._make_layer(self.layer_gates[1], block, 64, layers[1], stride=2, ch_group=ch_group)
        self.layer3 = self._make_layer(self.layer_gates[2], block, 128, layers[2], stride=2, ch_group=ch_group)
        self.layer4 = self._make_layer(self.layer_gates[3], block, 256, layers[3], stride=2, ch_group=ch_group)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        if (ch_group == None):
            self.fc = nn.Linear(256 * block.expansion, num_classes)
            # self.fc1024 = nn.Linear(4 * 256 * block.expansion, num_classes)
        else:
            self.fc = SlicingLinearBlock(256 * block.expansion, num_classes)
            # self.fc1024 = nn.Linear(4 * 256 * block.expansion, num_classes)
        self.dropout = nn.Dropout()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
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
                              kernel_size=1, stride=stride, padding=0, bias=False, ch_group=ch_group),
                    # nn.BatchNorm2d(planes * block.expansion),
                )
        layers = []
        layers.append(block(layer_gates[0], self.inplanes, planes, \
                            stride=stride, downsample=downsample, ch_group=ch_group))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(layer_gates[i], self.inplanes, planes, \
                                ch_group=ch_group))

        return nn.Sequential(*layers)

    def forward(self, x, dump_act=None):
        # print('input {0}'.format(x.size()))

        x = self.conv1(x)
        # print('conv1 output {0}'.format(x.size()))
        x = self.bn1(x)
        x = self.relu1(x)

        # print('---------------------------------')
        # print('conv2 input {0}'.format(x.size()))
        x = self.conv2(x)
        # print('conv2 output {0}'.format(x.size()))
        x = self.bn2(x)
        x = self.relu2(x)

        # print('---------------------------------')
        # print('conv3 input {0}'.format(x.size()))
        x = self.conv3(x)
        # print('conv3 output {0}'.format(x.size()))
        x = self.bn3(x)
        x = self.relu3(x)


        x = self.poolPadding(x)
        # print('---------------------------------')
        # print('maxpool input after padding {0}'.format(x.size()))
        x = self.maxpool(x)
        x = self.dropout(x)
        # print('maxpool output {0}'.format(x.size()))

        # print('---------------------------------')
        x = self.layer1(x)
        # print('layer1 output {0}'.format(x.size()))

        # print('---------------------------------')
        x = self.layer2(x)
        x = self.dropout(x)
        # print('layer2 output {0}'.format(x.size()))

        # print('---------------------------------')
        x = self.layer3(x)
        # x = self.dropout(x)
        # print('layer3 output {0}'.format(x.size()))

        # print('---------------------------------')
        x = self.layer4(x)
        x = self.dropout(x)
        # print('layer4 output {0}'.format(x.size()))

        x = self.poolPadding(x)
        # print('---------------------------------')
        # print('avgpool input after padding {0}'.format(x.size()))
        x = self.avgpool(x)
        # print('avgpool output 1 {0}'.format(x.size()))
        x = self.avgpool(x)
        # print('avgpool output 2 {0}'.format(x.size()))
        x = self.avgpool(x)
        # print('avgpool output 3 {0}'.format(x.size()))

        x = x.view(-1, x.size(1))
        print('fc input {0}'.format(x.size()))
        x = self.fc(x)

        # x = x.view(-1, x.size(1)*x.size(2)*x.size(3))
        # print('fc input {0}'.format(x.size()))
        # x = self.fc1024(x)

        # print('---------------------------------')
        # print('fc output {0}'.format(x.size()))

        return x

class ResNetCifarFused(nn.Module):

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
        super(ResNetCifarFused, self).__init__()

        self.fused1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)

        if (self.ch_group == None):
            self.fused2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=True)
        else:
            self.fused2 = SlicingBlockFused(16, 16, kernel_size=3, stride=1, padding=1, bias=True, ch_group=ch_group)
        self.relu2 = nn.ReLU(inplace=True)

        if (self.ch_group == None):
            self.fused3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=True)
        else:
            self.fused3 = SlicingBlockFused(16, 32, kernel_size=3, stride=1, padding=1, bias=True, ch_group=ch_group)
        self.relu3 = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(self.layer_gates[0], block, 32, layers[0], stride=1, ch_group=ch_group)
        self.layer2 = self._make_layer(self.layer_gates[1], block, 64, layers[1], stride=2, ch_group=ch_group)
        self.layer3 = self._make_layer(self.layer_gates[2], block, 128, layers[2], stride=2, ch_group=ch_group)
        self.layer4 = self._make_layer(self.layer_gates[3], block, 256, layers[3], stride=2, ch_group=ch_group)

        self.avgpool = nn.AvgPool2d(7, stride=7) #nn.AdaptiveAvgPool2d(1)

        if (ch_group == None):
            self.fc = nn.Linear(256 * block.expansion, num_classes)
        else:
            self.fc = SlicingLinearBlock(256 * block.expansion, num_classes)
        self.dropout = nn.Dropout()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, layer_gates, block, planes, blocks, stride=1, ch_group=None):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            if (ch_group == None):
                downsample = nn.Conv2d(self.inplanes, planes * block.expansion, \
                                       kernel_size=1, stride=stride, bias=True)
            else:
                downsample = SlicingBlockFused(self.inplanes, planes * block.expansion, \
                                          kernel_size=1, stride=stride, padding=0, bias=True, ch_group=ch_group)
        layers = []
        layers.append(block(layer_gates[0], self.inplanes, planes, \
                            stride=stride, downsample=downsample, ch_group=ch_group))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(layer_gates[i], self.inplanes, planes, \
                                ch_group=ch_group))

        return nn.Sequential(*layers)

    def forward(self, x, dump_act=None):
        global printLayerInfo

        if (printLayerInfo):
            print('input {0}'.format(x.size()))
        if(dump_act != None):
            dump_to_npy(name=str(dump_act)+'.input.activation', tensor=x)

        if (printLayerInfo):
            print('fused1')
        x = self.fused1(x)
        if (dump_act != None):
            dump_to_npy(name=str(dump_act) + '.conv1.activation', tensor=x)
            dump_to_npy(name=str(dump_act) + '.conv1.weight', tensor=self.fused1.weight)
            dump_to_npy(name=str(dump_act) + '.conv1.bias', tensor=self.fused1.bias)
        x = self.relu1(x)
        if (dump_act != None):
            dump_to_npy(name=str(dump_act) + '.conv1_relu.activation', tensor=x)

        if (printLayerInfo):
            print('fused2')
        x = self.fused2(x)
        if (dump_act != None):
            dump_to_npy(name=str(dump_act) + '.conv2.activation', tensor=x)
            dump_to_npy(name=str(dump_act) + '.conv2.weight', tensor=self.fused2.weight)
            dump_to_npy(name=str(dump_act) + '.conv2.bias', tensor=self.fused2.bias)
        x = self.relu2(x)
        if (dump_act != None):
            dump_to_npy(name=str(dump_act) + '.conv2_relu.activation', tensor=x)

        if (printLayerInfo):
            print('fused3')
        x = self.fused3(x)
        if (dump_act != None):
            dump_to_npy(name=str(dump_act) + '.conv3.activation', tensor=x)
            dump_to_npy(name=str(dump_act) + '.conv3.weight', tensor=self.fused3.weight)
            dump_to_npy(name=str(dump_act) + '.conv3.bias', tensor=self.fused3.bias)
        x = self.relu3(x)
        if (dump_act != None):
            dump_to_npy(name=str(dump_act) + '.conv3_relu.activation', tensor=x)

        if (printLayerInfo):
            print('maxpool input {0}'.format(x.size()))
        x = self.maxpool(x)
        x = self.dropout(x)
        if (dump_act != None):
            dump_to_npy(name=str(dump_act) + '.maxpooling.activation', tensor=x)
        if (printLayerInfo):
            print('maxpool output {0}'.format(x.size()))

        if (printLayerInfo):
            print('layer1')
        x = self.layer1((x, 1, dump_act))
        if (printLayerInfo):
            print('layer1 output {0}'.format(x.size()))

        if (printLayerInfo):
            print('layer2')
        x = self.layer2((x, 2, dump_act))
        # x = self.dropout(x)
        if (printLayerInfo):
            print('layer2 output {0}'.format(x.size()))

        if (printLayerInfo):
            print('layer3')
        x = self.layer3((x, 3, dump_act))
        # x = self.dropout(x)
        if (printLayerInfo):
            print('layer3 output {0}'.format(x.size()))

        if (printLayerInfo):
            print('layer4')
        x = self.layer4((x, 4, dump_act))
        # x = self.dropout(x)
        if (printLayerInfo):
            print('layer4 output {0}'.format(x.size()))

        if (printLayerInfo):
            print('avgpool input {0}'.format(x.size()))
        x = self.avgpool(x)
        if (printLayerInfo):
            print('avgpool output {0}'.format(x.size()))

        if(printLayerInfo):
            print('fc')
        x = x.view(-1, x.size(1))
        x = self.fc(x)
        if (dump_act != None):
            dump_to_npy(name=str(dump_act) + '.fc.activation', tensor=x)
            dump_to_npy(name=str(dump_act) + '.fc.weight', tensor=self.fc.weight)
            dump_to_npy(name=str(dump_act) + '.fc.bias', tensor=self.fc.bias)
        if (printLayerInfo):
            print('fc output {0}'.format(x.size()))

        return x

class ResNetCifarReshapeFused(nn.Module):

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
        super(ResNetCifarReshapeFused, self).__init__()

        self.fused1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)

        if (self.ch_group == None):
            self.fused2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=True)
        else:
            self.fused2 = SlicingBlockFused(16, 16, kernel_size=3, stride=1, padding=1, bias=True, ch_group=ch_group)
        self.relu2 = nn.ReLU(inplace=True)

        if (self.ch_group == None):
            self.fused3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=True)
        else:
            self.fused3 = SlicingBlockFused(16, 32, kernel_size=3, stride=1, padding=1, bias=True, ch_group=ch_group)
        self.relu3 = nn.ReLU(inplace=True)

        self.poolPadding = nn.ZeroPad2d((0, 1, 0, 1))  # left, right, top, bottom
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.layer1 = self._make_layer(self.layer_gates[0], block, 32, layers[0], stride=1, ch_group=ch_group)
        self.layer2 = self._make_layer(self.layer_gates[1], block, 64, layers[1], stride=2, ch_group=ch_group)
        self.layer3 = self._make_layer(self.layer_gates[2], block, 128, layers[2], stride=2, ch_group=ch_group)
        self.layer4 = self._make_layer(self.layer_gates[3], block, 256, layers[3], stride=2, ch_group=ch_group)

        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        if (ch_group == None):
            self.fc = nn.Linear(256 * block.expansion, num_classes)
            # self.fc1024 = nn.Linear(4 * 256 * block.expansion, num_classes)
        else:
            self.fc = SlicingLinearBlock(256 * block.expansion, num_classes)
            # self.fc1024 = SlicingLinearBlock(4 * 256 * block.expansion, num_classes)
        self.dropout = nn.Dropout()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, layer_gates, block, planes, blocks, stride=1, ch_group=None):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            if (ch_group == None):
                downsample = nn.Conv2d(self.inplanes, planes * block.expansion, \
                                       kernel_size=1, stride=stride, bias=True)
            else:
                downsample = SlicingBlockFused(self.inplanes, planes * block.expansion, \
                                          kernel_size=1, stride=stride, padding=0, bias=True, ch_group=ch_group)
        layers = []
        layers.append(block(layer_gates[0], self.inplanes, planes, \
                            stride=stride, downsample=downsample, ch_group=ch_group))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(layer_gates[i], self.inplanes, planes, \
                                ch_group=ch_group))

        return nn.Sequential(*layers)

    def forward(self, x, dump_act=None):
        global testPhase

        # print('input {0}'.format(x.size()))
        # print('ResNetCifar {0}'.format(fusion))
        if(dump_act != None):
            dump_to_npy(name=str(dump_act)+'.input.activation', tensor=x)
        x = self.fused1(x)
        if (dump_act != None):
            dump_to_npy(name=str(dump_act) + '.conv1.activation', tensor=x)
            dump_to_npy(name=str(dump_act) + '.conv1.weight', tensor=self.fused1.weight)
            dump_to_npy(name=str(dump_act) + '.conv1.bias', tensor=self.fused1.bias)
        x = self.relu1(x)
        if (dump_act != None):
            dump_to_npy(name=str(dump_act) + '.conv1_relu.activation', tensor=x)

        x = self.fused2(x)
        if (dump_act != None):
            dump_to_npy(name=str(dump_act) + '.conv2.activation', tensor=x)
            dump_to_npy(name=str(dump_act) + '.conv2.weight', tensor=self.fused2.weight)
            dump_to_npy(name=str(dump_act) + '.conv2.bias', tensor=self.fused2.bias)
        x = self.relu2(x)
        if (dump_act != None):
            dump_to_npy(name=str(dump_act) + '.conv2_relu.activation', tensor=x)

        x = self.fused3(x)
        if (dump_act != None):
            dump_to_npy(name=str(dump_act) + '.conv3.activation', tensor=x)
            dump_to_npy(name=str(dump_act) + '.conv3.weight', tensor=self.fused3.weight)
            dump_to_npy(name=str(dump_act) + '.conv3.bias', tensor=self.fused3.bias)
        x = self.relu3(x)
        if (dump_act != None):
            dump_to_npy(name=str(dump_act) + '.conv3_relu.activation', tensor=x)

        x = self.poolPadding(x)
        # print('maxpool input {0}'.format(x.size()))
        x = self.maxpool(x)
        if (dump_act != None):
            dump_to_npy(name=str(dump_act) + '.maxpooling.activation', tensor=x)
        x = self.dropout(x)
        # print('maxpool output {0}'.format(x.size()))

        x = self.layer1((x, 1, dump_act))
        # print('layer1 output {0}'.format(x.size()))

        x = self.layer2((x, 2, dump_act))
        x = self.dropout(x)
        # print('layer2 output {0}'.format(x.size()))

        x = self.layer3((x, 3, dump_act))
        # x = self.dropout(x)
        # print('layer3 output {0}'.format(x.size()))

        x = self.layer4((x, 4, dump_act))
        x = self.dropout(x)
        # print('layer4 output {0}'.format(x.size()))

        x = self.poolPadding(x)
        # print('avgpool input {0}'.format(x.size()))
        x = self.avgpool(x)
        x = self.avgpool(x)
        x = self.avgpool(x)
        # print('avgpool output {0}'.format(x.size()))

        x = x.view(-1, x.size(1))
        x = self.fc(x)
        if (dump_act != None):
            dump_to_npy(name=str(dump_act) + '.fc.activation', tensor=x)
            dump_to_npy(name=str(dump_act) + '.fc.weight', tensor=self.fc.weight)
            dump_to_npy(name=str(dump_act) + '.fc.bias', tensor=self.fc.bias)
        # print('fc output {0}'.format(x.size()))

        return x

def resnet10_cifar(pretrained, ch_group, fusion, **kwargs):
    if(fusion == False):
        model = ResNetCifar(BasicBlock, [1, 1, 1, 1], **kwargs, ch_group=ch_group)
    else:
        model = ResNetCifarFused(BasicBlockFused, [1, 1, 1, 1], **kwargs, ch_group=ch_group)
    if pretrained: # no module. prefix is allowed #
        state_dict = torch.load(model_pretrained['resnet10_cifar'])
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if('module.' in k):
                name = k[7:]  # remove module #
            else:
                name = k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)
    # freeze some layers #
    # model.conv1.weight.requires_grad = False
    # model.conv2.weight.requires_grad = False
    # model.conv3.weight.requires_grad = False
    # model.layer1[0].conv1.weight.requires_grad = False
    # model.layer1[0].conv2.weight.requires_grad = False
    # model.layer2[0].conv1.weight.requires_grad = False
    # model.layer2[0].conv2.weight.requires_grad = False
    # model.layer2[0].downsample[0].weight.requires_grad = False
    # model.layer3[0].conv1.weight.requires_grad = False
    # model.layer3[0].conv2.weight.requires_grad = False
    # model.layer3[0].downsample[0].weight.requires_grad = False
    # model.fc.weight.requires_grad = False
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

def fuse_conv_and_bn(dict, conv_key, bn_key, fuse_key):
    w = dict[conv_key+'.weight']
    mean = dict[bn_key+'.running_mean']
    var_sqrt = torch.sqrt(dict[bn_key+'.running_var'] + 1E-7)
    gamma = dict[bn_key+'.weight']
    beta = dict[bn_key+'.bias']
    b = mean.new_zeros(mean.shape)
    w = w * (gamma / var_sqrt).reshape([dict[conv_key+'.weight'].size(0), 1, 1, 1])
    if('.' not in fuse_key or ('.' in fuse_key and 'conv_0' in fuse_key)):
        b = ((b - mean) / var_sqrt) * gamma + beta
    dict[fuse_key+'.weight'] = w
    dict[fuse_key+'.bias'] = b

if __name__ == '__main__':
    state_dict = torch.load(model_pretrained['resnet10_cifar'])
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if ('module.' in k):
            name = k[7:]  # remove module #
        else:
            name = k
        new_state_dict[name] = v

    key_map = { ('conv1', 'bn1'): 'fused1',
                ('conv2', 'bn2'): 'fused2',
                ('conv3', 'bn3'): 'fused3',
                ('layer1.0.conv1', 'layer1.0.bn1'): 'layer1.0.fused1',
                ('layer1.0.conv2', 'layer1.0.bn2'): 'layer1.0.fused2',
                ('layer2.0.conv1', 'layer2.0.bn1'): 'layer2.0.fused1',
                ('layer2.0.conv2', 'layer2.0.bn2'): 'layer2.0.fused2',
                ('layer2.0.downsample.0', 'layer2.0.downsample.1'): 'layer2.0.downsample',
                ('layer3.0.conv1', 'layer3.0.bn1'): 'layer3.0.fused1',
                ('layer3.0.conv2', 'layer3.0.bn2'): 'layer3.0.fused2',
                ('layer3.0.downsample.0', 'layer3.0.downsample.1'): 'layer3.0.downsample',
                ('layer4.0.conv1', 'layer4.0.bn1'): 'layer4.0.fused1',
                ('layer4.0.conv2', 'layer4.0.bn2'): 'layer4.0.fused2',
                ('layer4.0.downsample.0', 'layer4.0.downsample.1'): 'layer4.0.downsample',
                }

    key_ch8_map = {('conv1', 'bn1'): 'fused1',
                   ('conv2.conv_0', 'bn2'): 'fused2.conv_0',
                   ('conv2.conv_1', 'bn2'): 'fused2.conv_1',
                   ('conv3.conv_0', 'bn3'): 'fused3.conv_0',
                   ('conv3.conv_1', 'bn3'): 'fused3.conv_1',
                   ('layer1.0.conv1.conv_0', 'layer1.0.bn1'): 'layer1.0.fused1.conv_0',
                   ('layer1.0.conv1.conv_1', 'layer1.0.bn1'): 'layer1.0.fused1.conv_1',
                   ('layer1.0.conv1.conv_2', 'layer1.0.bn1'): 'layer1.0.fused1.conv_2',
                   ('layer1.0.conv1.conv_3', 'layer1.0.bn1'): 'layer1.0.fused1.conv_3',
                   ('layer1.0.conv2.conv_0', 'layer1.0.bn2'): 'layer1.0.fused2.conv_0',
                   ('layer1.0.conv2.conv_1', 'layer1.0.bn2'): 'layer1.0.fused2.conv_1',
                   ('layer1.0.conv2.conv_2', 'layer1.0.bn2'): 'layer1.0.fused2.conv_2',
                   ('layer1.0.conv2.conv_3', 'layer1.0.bn2'): 'layer1.0.fused2.conv_3',
                   ('layer2.0.conv1.conv_0', 'layer2.0.bn1'): 'layer2.0.fused1.conv_0',
                   ('layer2.0.conv1.conv_1', 'layer2.0.bn1'): 'layer2.0.fused1.conv_1',
                   ('layer2.0.conv1.conv_2', 'layer2.0.bn1'): 'layer2.0.fused1.conv_2',
                   ('layer2.0.conv1.conv_3', 'layer2.0.bn1'): 'layer2.0.fused1.conv_3',
                   ('layer2.0.conv2.conv_0', 'layer2.0.bn2'): 'layer2.0.fused2.conv_0',
                   ('layer2.0.conv2.conv_1', 'layer2.0.bn2'): 'layer2.0.fused2.conv_1',
                   ('layer2.0.conv2.conv_2', 'layer2.0.bn2'): 'layer2.0.fused2.conv_2',
                   ('layer2.0.conv2.conv_3', 'layer2.0.bn2'): 'layer2.0.fused2.conv_3',
                   ('layer2.0.conv2.conv_4', 'layer2.0.bn2'): 'layer2.0.fused2.conv_4',
                   ('layer2.0.conv2.conv_5', 'layer2.0.bn2'): 'layer2.0.fused2.conv_5',
                   ('layer2.0.conv2.conv_6', 'layer2.0.bn2'): 'layer2.0.fused2.conv_6',
                   ('layer2.0.conv2.conv_7', 'layer2.0.bn2'): 'layer2.0.fused2.conv_7',
                   ('layer2.0.downsample.0.conv_0', 'layer2.0.downsample.1'): 'layer2.0.downsample.conv_0',
                   ('layer2.0.downsample.0.conv_1', 'layer2.0.downsample.1'): 'layer2.0.downsample.conv_1',
                   ('layer2.0.downsample.0.conv_2', 'layer2.0.downsample.1'): 'layer2.0.downsample.conv_2',
                   ('layer2.0.downsample.0.conv_3', 'layer2.0.downsample.1'): 'layer2.0.downsample.conv_3',
                   ('layer3.0.conv1.conv_0', 'layer3.0.bn1'): 'layer3.0.fused1.conv_0',
                   ('layer3.0.conv1.conv_1', 'layer3.0.bn1'): 'layer3.0.fused1.conv_1',
                   ('layer3.0.conv1.conv_2', 'layer3.0.bn1'): 'layer3.0.fused1.conv_2',
                   ('layer3.0.conv1.conv_3', 'layer3.0.bn1'): 'layer3.0.fused1.conv_3',
                   ('layer3.0.conv1.conv_4', 'layer3.0.bn1'): 'layer3.0.fused1.conv_4',
                   ('layer3.0.conv1.conv_5', 'layer3.0.bn1'): 'layer3.0.fused1.conv_5',
                   ('layer3.0.conv1.conv_6', 'layer3.0.bn1'): 'layer3.0.fused1.conv_6',
                   ('layer3.0.conv1.conv_7', 'layer3.0.bn1'): 'layer3.0.fused1.conv_7',
                   ('layer3.0.conv2.conv_0', 'layer3.0.bn2'): 'layer3.0.fused2.conv_0',
                   ('layer3.0.conv2.conv_1', 'layer3.0.bn2'): 'layer3.0.fused2.conv_1',
                   ('layer3.0.conv2.conv_2', 'layer3.0.bn2'): 'layer3.0.fused2.conv_2',
                   ('layer3.0.conv2.conv_3', 'layer3.0.bn2'): 'layer3.0.fused2.conv_3',
                   ('layer3.0.conv2.conv_4', 'layer3.0.bn2'): 'layer3.0.fused2.conv_4',
                   ('layer3.0.conv2.conv_5', 'layer3.0.bn2'): 'layer3.0.fused2.conv_5',
                   ('layer3.0.conv2.conv_6', 'layer3.0.bn2'): 'layer3.0.fused2.conv_6',
                   ('layer3.0.conv2.conv_7', 'layer3.0.bn2'): 'layer3.0.fused2.conv_7',
                   ('layer3.0.conv2.conv_8', 'layer3.0.bn2'): 'layer3.0.fused2.conv_8',
                   ('layer3.0.conv2.conv_9', 'layer3.0.bn2'): 'layer3.0.fused2.conv_9',
                   ('layer3.0.conv2.conv_10', 'layer3.0.bn2'): 'layer3.0.fused2.conv_10',
                   ('layer3.0.conv2.conv_11', 'layer3.0.bn2'): 'layer3.0.fused2.conv_11',
                   ('layer3.0.conv2.conv_12', 'layer3.0.bn2'): 'layer3.0.fused2.conv_12',
                   ('layer3.0.conv2.conv_13', 'layer3.0.bn2'): 'layer3.0.fused2.conv_13',
                   ('layer3.0.conv2.conv_14', 'layer3.0.bn2'): 'layer3.0.fused2.conv_14',
                   ('layer3.0.conv2.conv_15', 'layer3.0.bn2'): 'layer3.0.fused2.conv_15',
                   ('layer3.0.downsample.0.conv_0', 'layer3.0.downsample.1'): 'layer3.0.downsample.conv_0',
                   ('layer3.0.downsample.0.conv_1', 'layer3.0.downsample.1'): 'layer3.0.downsample.conv_1',
                   ('layer3.0.downsample.0.conv_2', 'layer3.0.downsample.1'): 'layer3.0.downsample.conv_2',
                   ('layer3.0.downsample.0.conv_3', 'layer3.0.downsample.1'): 'layer3.0.downsample.conv_3',
                   ('layer3.0.downsample.0.conv_4', 'layer3.0.downsample.1'): 'layer3.0.downsample.conv_4',
                   ('layer3.0.downsample.0.conv_5', 'layer3.0.downsample.1'): 'layer3.0.downsample.conv_5',
                   ('layer3.0.downsample.0.conv_6', 'layer3.0.downsample.1'): 'layer3.0.downsample.conv_6',
                   ('layer3.0.downsample.0.conv_7', 'layer3.0.downsample.1'): 'layer3.0.downsample.conv_7',
                   ('layer4.0.conv1.conv_0', 'layer4.0.bn1'): 'layer4.0.fused1.conv_0',
                   ('layer4.0.conv1.conv_1', 'layer4.0.bn1'): 'layer4.0.fused1.conv_1',
                   ('layer4.0.conv1.conv_2', 'layer4.0.bn1'): 'layer4.0.fused1.conv_2',
                   ('layer4.0.conv1.conv_3', 'layer4.0.bn1'): 'layer4.0.fused1.conv_3',
                   ('layer4.0.conv1.conv_4', 'layer4.0.bn1'): 'layer4.0.fused1.conv_4',
                   ('layer4.0.conv1.conv_5', 'layer4.0.bn1'): 'layer4.0.fused1.conv_5',
                   ('layer4.0.conv1.conv_6', 'layer4.0.bn1'): 'layer4.0.fused1.conv_6',
                   ('layer4.0.conv1.conv_7', 'layer4.0.bn1'): 'layer4.0.fused1.conv_7',
                   ('layer4.0.conv1.conv_8', 'layer4.0.bn1'): 'layer4.0.fused1.conv_8',
                   ('layer4.0.conv1.conv_9', 'layer4.0.bn1'): 'layer4.0.fused1.conv_9',
                   ('layer4.0.conv1.conv_10', 'layer4.0.bn1'): 'layer4.0.fused1.conv_10',
                   ('layer4.0.conv1.conv_11', 'layer4.0.bn1'): 'layer4.0.fused1.conv_11',
                   ('layer4.0.conv1.conv_12', 'layer4.0.bn1'): 'layer4.0.fused1.conv_12',
                   ('layer4.0.conv1.conv_13', 'layer4.0.bn1'): 'layer4.0.fused1.conv_13',
                   ('layer4.0.conv1.conv_14', 'layer4.0.bn1'): 'layer4.0.fused1.conv_14',
                   ('layer4.0.conv1.conv_15', 'layer4.0.bn1'): 'layer4.0.fused1.conv_15',
                   ('layer4.0.conv2.conv_0', 'layer4.0.bn2'): 'layer4.0.fused2.conv_0',
                   ('layer4.0.conv2.conv_1', 'layer4.0.bn2'): 'layer4.0.fused2.conv_1',
                   ('layer4.0.conv2.conv_2', 'layer4.0.bn2'): 'layer4.0.fused2.conv_2',
                   ('layer4.0.conv2.conv_3', 'layer4.0.bn2'): 'layer4.0.fused2.conv_3',
                   ('layer4.0.conv2.conv_4', 'layer4.0.bn2'): 'layer4.0.fused2.conv_4',
                   ('layer4.0.conv2.conv_5', 'layer4.0.bn2'): 'layer4.0.fused2.conv_5',
                   ('layer4.0.conv2.conv_6', 'layer4.0.bn2'): 'layer4.0.fused2.conv_6',
                   ('layer4.0.conv2.conv_7', 'layer4.0.bn2'): 'layer4.0.fused2.conv_7',
                   ('layer4.0.conv2.conv_8', 'layer4.0.bn2'): 'layer4.0.fused2.conv_8',
                   ('layer4.0.conv2.conv_9', 'layer4.0.bn2'): 'layer4.0.fused2.conv_9',
                   ('layer4.0.conv2.conv_10', 'layer4.0.bn2'): 'layer4.0.fused2.conv_10',
                   ('layer4.0.conv2.conv_11', 'layer4.0.bn2'): 'layer4.0.fused2.conv_11',
                   ('layer4.0.conv2.conv_12', 'layer4.0.bn2'): 'layer4.0.fused2.conv_12',
                   ('layer4.0.conv2.conv_13', 'layer4.0.bn2'): 'layer4.0.fused2.conv_13',
                   ('layer4.0.conv2.conv_14', 'layer4.0.bn2'): 'layer4.0.fused2.conv_14',
                   ('layer4.0.conv2.conv_15', 'layer4.0.bn2'): 'layer4.0.fused2.conv_15',
                   ('layer4.0.conv2.conv_16', 'layer4.0.bn2'): 'layer4.0.fused2.conv_16',
                   ('layer4.0.conv2.conv_17', 'layer4.0.bn2'): 'layer4.0.fused2.conv_17',
                   ('layer4.0.conv2.conv_18', 'layer4.0.bn2'): 'layer4.0.fused2.conv_18',
                   ('layer4.0.conv2.conv_19', 'layer4.0.bn2'): 'layer4.0.fused2.conv_19',
                   ('layer4.0.conv2.conv_20', 'layer4.0.bn2'): 'layer4.0.fused2.conv_20',
                   ('layer4.0.conv2.conv_21', 'layer4.0.bn2'): 'layer4.0.fused2.conv_21',
                   ('layer4.0.conv2.conv_22', 'layer4.0.bn2'): 'layer4.0.fused2.conv_22',
                   ('layer4.0.conv2.conv_23', 'layer4.0.bn2'): 'layer4.0.fused2.conv_23',
                   ('layer4.0.conv2.conv_24', 'layer4.0.bn2'): 'layer4.0.fused2.conv_24',
                   ('layer4.0.conv2.conv_25', 'layer4.0.bn2'): 'layer4.0.fused2.conv_25',
                   ('layer4.0.conv2.conv_26', 'layer4.0.bn2'): 'layer4.0.fused2.conv_26',
                   ('layer4.0.conv2.conv_27', 'layer4.0.bn2'): 'layer4.0.fused2.conv_27',
                   ('layer4.0.conv2.conv_28', 'layer4.0.bn2'): 'layer4.0.fused2.conv_28',
                   ('layer4.0.conv2.conv_29', 'layer4.0.bn2'): 'layer4.0.fused2.conv_29',
                   ('layer4.0.conv2.conv_30', 'layer4.0.bn2'): 'layer4.0.fused2.conv_30',
                   ('layer4.0.conv2.conv_31', 'layer4.0.bn2'): 'layer4.0.fused2.conv_31',
                   ('layer4.0.downsample.0.conv_0', 'layer4.0.downsample.1'): 'layer4.0.downsample.conv_0',
                   ('layer4.0.downsample.0.conv_1', 'layer4.0.downsample.1'): 'layer4.0.downsample.conv_1',
                   ('layer4.0.downsample.0.conv_2', 'layer4.0.downsample.1'): 'layer4.0.downsample.conv_2',
                   ('layer4.0.downsample.0.conv_3', 'layer4.0.downsample.1'): 'layer4.0.downsample.conv_3',
                   ('layer4.0.downsample.0.conv_4', 'layer4.0.downsample.1'): 'layer4.0.downsample.conv_4',
                   ('layer4.0.downsample.0.conv_5', 'layer4.0.downsample.1'): 'layer4.0.downsample.conv_5',
                   ('layer4.0.downsample.0.conv_6', 'layer4.0.downsample.1'): 'layer4.0.downsample.conv_6',
                   ('layer4.0.downsample.0.conv_7', 'layer4.0.downsample.1'): 'layer4.0.downsample.conv_7',
                   ('layer4.0.downsample.0.conv_8', 'layer4.0.downsample.1'): 'layer4.0.downsample.conv_8',
                   ('layer4.0.downsample.0.conv_9', 'layer4.0.downsample.1'): 'layer4.0.downsample.conv_9',
                   ('layer4.0.downsample.0.conv_10', 'layer4.0.downsample.1'): 'layer4.0.downsample.conv_10',
                   ('layer4.0.downsample.0.conv_11', 'layer4.0.downsample.1'): 'layer4.0.downsample.conv_11',
                   ('layer4.0.downsample.0.conv_12', 'layer4.0.downsample.1'): 'layer4.0.downsample.conv_12',
                   ('layer4.0.downsample.0.conv_13', 'layer4.0.downsample.1'): 'layer4.0.downsample.conv_13',
                   ('layer4.0.downsample.0.conv_14', 'layer4.0.downsample.1'): 'layer4.0.downsample.conv_14',
                   ('layer4.0.downsample.0.conv_15', 'layer4.0.downsample.1'): 'layer4.0.downsample.conv_15',
                   }

    for key, data in key_ch8_map.items():
        fuse_conv_and_bn(new_state_dict, key[0], key[1], data)

    torch.save(new_state_dict, model_saved['resnet10_cifar'])





