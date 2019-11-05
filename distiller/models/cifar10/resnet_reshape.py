import os
import sys
import torch
import torch.nn as nn
import math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.\\')))
from resnet_slicing_block import *
from resnet_basic_block import *
from net_utils import *
from net_utils import NUM_CLASSES as NUM_CLASSES

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
        # print('fc input {0}'.format(x.size()))
        x = self.fc(x)

        # x = x.view(-1, x.size(1)*x.size(2)*x.size(3))
        # print('fc input {0}'.format(x.size()))
        # x = self.fc1024(x)

        # print('---------------------------------')
        # print('fc output {0}'.format(x.size()))

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

        if (dump_act != None and self.ch_group == None):
            x = self.fused2(x)
            dump_to_npy(name=str(dump_act) + '.conv2.activation', tensor=x)
            dump_to_npy(name=str(dump_act) + '.conv2.weight', tensor=self.fused2.weight)
            dump_to_npy(name=str(dump_act) + '.conv2.bias', tensor=self.fused2.bias)
        elif(dump_act != None and self.ch_group != None):
            x = self.fused2((x, 'conv2', dump_act))
        else:
            x = self.fused2(x)

        x = self.relu2(x)
        if (dump_act != None):
            dump_to_npy(name=str(dump_act) + '.conv2_relu.activation', tensor=x)


        if (dump_act != None and self.ch_group == None):
            x = self.fused3(x)
            dump_to_npy(name=str(dump_act) + '.conv3.activation', tensor=x)
            dump_to_npy(name=str(dump_act) + '.conv3.weight', tensor=self.fused3.weight)
            dump_to_npy(name=str(dump_act) + '.conv3.bias', tensor=self.fused3.bias)
        elif (dump_act != None and self.ch_group != None):
            x = self.fused3((x, 'conv3', dump_act))
        else:
            x = self.fused3(x)

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

        if (dump_act != None and self.ch_group == None):
            x = self.fc(x)
            dump_to_npy(name=str(dump_act) + '.fc.activation', tensor=x)
            dump_to_npy(name=str(dump_act) + '.fc.weight', tensor=self.fc.weight)
            dump_to_npy(name=str(dump_act) + '.fc.bias', tensor=self.fc.bias)
        elif (dump_act != None and self.ch_group != None):
            x = self.fc((x, 'fc', dump_act))
        else:
            x = self.fc(x)
        # print('fc output {0}'.format(x.size()))

        return x
