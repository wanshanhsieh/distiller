import os
import sys
import torch
import torch.nn as nn
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.\\')))
from resnet_slicing_block import *
from net_utils import *

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, block_gates, inplanes, planes, stride=1, downsample=None, ch_group=None):
        super(BasicBlock, self).__init__()
        self.block_gates = block_gates
        self.ch_group = ch_group
        if(self.ch_group == None):
            self.conv1 = conv3x3(inplanes, planes, stride)
        else:
            self.conv1 = SlicingBlockFused(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False, ch_group=ch_group)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=False)  # To enable layer removal inplace must be False
        if (self.ch_group == None):
            self.conv2 = conv3x3(planes, planes)
        else:
            self.conv2 = SlicingBlockFused(planes, planes, stride=1, padding=1, bias=False, ch_group=ch_group)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        if type(x) is tuple:
            input = x[0]
            dump_act = x[1]
            residual = out = input
        else:
            dump_act = None
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
        if (dump_act != None):
            dump_to_npy(name=str(dump_act) + '.res2_adder', tensor=out)

        out = self.relu2(out)

        return out

class BasicBlockFused(nn.Module):
    expansion = 1

    def __init__(self, block_gates, inplanes, planes, stride=1, downsample=None, ch_group=None):
        super(BasicBlockFused, self).__init__()
        self.block_gates = block_gates
        self.ch_group = ch_group
        if (self.ch_group == None):
            self.fused1 = conv3x3_bias(inplanes, planes, stride)
        else:
            self.fused1 = SlicingBlockFused(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False, ch_group=ch_group)
        self.relu1 = nn.ReLU(inplace=False)  # To enable layer removal inplace must be False
        if (self.ch_group == None):
            self.fused2 = conv3x3_bias(planes, planes)
        else:
            self.fused2 = SlicingBlockFused(planes, planes, stride=1, padding=1, bias=False, ch_group=ch_group)
        self.relu2 = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        global testPhase
        if type(x) is tuple:
            _input = x[0]
            layerId = x[1]
            dump_act = x[2]
            residual = out = _input
        else:
            layerId = 0
            dump_act = None
            residual = out = x

        if self.block_gates[0]:
            # if (dump_act != None):
            #     dump_to_npy(name=str(dump_act) + '.res'+str(layerId)+'_input.activation', tensor=_input)

            if (dump_act != None and self.ch_group == None):
                out = self.fused1(_input)
                dump_to_npy(name=str(dump_act) + '.res'+str(layerId)+'_conv1.activation', tensor=out)
                dump_to_npy(name=str(dump_act) + '.res' + str(layerId) + '_conv1.weight', tensor=self.fused1.weight)
                dump_to_npy(name=str(dump_act) + '.res' + str(layerId) + '_conv1.bias', tensor=self.fused1.bias)
            elif(dump_act != None and self.ch_group != None):
                out = self.fused1((_input, 'res'+str(layerId)+'_conv1', dump_act))
                dump_to_npy(name=str(dump_act) + '.res' + str(layerId) + '_conv1.activation', tensor=out)
            else:
                out = self.fused1(_input)

            out = self.relu1(out)
            if (dump_act != None):
                dump_to_npy(name=str(dump_act) + '.res'+str(layerId)+'_conv1_relu.activation', tensor=out)

        if self.block_gates[1]:
            if (dump_act != None and self.ch_group == None):
                out = self.fused2(out)
                dump_to_npy(name=str(dump_act) + '.res'+str(layerId)+'_conv2.activation', tensor=out)
                dump_to_npy(name=str(dump_act) + '.res' + str(layerId) + '_conv2.weight', tensor=self.fused2.weight)
                dump_to_npy(name=str(dump_act) + '.res' + str(layerId) + '_conv2.bias', tensor=self.fused2.bias)
            elif (dump_act != None and self.ch_group != None):
                out = self.fused2((out, 'res' + str(layerId) + '_conv2', dump_act))
                dump_to_npy(name=str(dump_act) + '.res' + str(layerId) + '_conv2.activation', tensor=out)
            else:
                out = self.fused2(out)

        if self.downsample is not None:
            if (dump_act != None and self.ch_group == None):
                residual = self.downsample(_input)
                dump_to_npy(name=str(dump_act) + '.res'+str(layerId)+'_match.activation', tensor=residual)
                dump_to_npy(name=str(dump_act) + '.res' + str(layerId) + '_match.weight', tensor=self.downsample.weight)
                dump_to_npy(name=str(dump_act) + '.res' + str(layerId) + '_match.bias', tensor=self.downsample.bias)
            elif(dump_act != None and self.ch_group != None):
                residual = self.downsample((_input, 'res'+str(layerId)+'_match', dump_act))
                dump_to_npy(name=str(dump_act) + '.res' + str(layerId) + '_match.activation', tensor=residual)
            else:
                residual = self.downsample(_input)

        out += residual
        if (dump_act != None):
            dump_to_npy(name=str(dump_act) + '.res'+str(layerId)+'_adder.activation', tensor=out)

        out = self.relu2(out)
        if self.downsample is not None:
            if (dump_act != None):
                dump_to_npy(name=str(dump_act) + '.res'+str(layerId)+'_adder_relu.activation', tensor=out)
        else: ## no identity
            if (dump_act != None):
                dump_to_npy(name=str(dump_act) + '.res'+str(layerId)+'_conv2_relu.activation', tensor=out)
        return out