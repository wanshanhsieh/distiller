import os
import sys
import torch
import torch.nn as nn
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.\\')))
from net_utils import *
from net_utils import NUM_CLASSES as NUM_CLASSES

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
        if (type(x) is tuple):
            for i in range(0, self.num_of_slice, 1):
                _x.append(torch.narrow(x[0], 1, _start, self.ch_group))
                _start += self.ch_group
            layerPrefix = x[1]
            dump_act = x[2]
        else:
            for i in range(0, self.num_of_slice, 1):
                _x.append(torch.narrow(x, 1, _start, self.ch_group))
                _start += self.ch_group
            layerPrefix = None
            dump_act = None

        if (self.num_of_slice == 2):  # 16
            out_0 = self.fc_0(_x[0])
            out_1 = self.fc_1(_x[1])
            out = out_0 + out_1
            if (dump_act != None):
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_0.activation', tensor=out_0)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_0.weight', tensor=self.fc_0.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_0.bias',   tensor=self.fc_0.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_1.activation', tensor=out_1)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_1.weight', tensor=self.fc_1.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_1.bias',   tensor=self.fc_1.bias)
        elif (self.num_of_slice == 4):  # 32
            out_0 = self.fc_0(_x[0])
            out_1 = self.fc_1(_x[1])
            out_2 = self.fc_2(_x[2])
            out_3 = self.fc_3(_x[3])
            out = out_0 + out_1 + out_2 + out_3
            if (dump_act != None):
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_0.activation', tensor=out_0)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_0.weight', tensor=self.fc_0.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_0.bias',   tensor=self.fc_0.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_1.activation', tensor=out_1)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_1.weight', tensor=self.fc_1.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_1.bias',   tensor=self.fc_1.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_2.activation', tensor=out_2)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_2.weight', tensor=self.fc_2.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_2.bias',   tensor=self.fc_2.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_3.activation', tensor=out_3)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_3.weight', tensor=self.fc_3.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_3.bias',   tensor=self.fc_3.bias)
        elif (self.num_of_slice == 8):  # 64
            out_0 = self.fc_0(_x[0])
            out_1 = self.fc_1(_x[1])
            out_2 = self.fc_2(_x[2])
            out_3 = self.fc_3(_x[3])
            out_4 = self.fc_4(_x[4])
            out_5 = self.fc_5(_x[5])
            out_6 = self.fc_6(_x[6])
            out_7 = self.fc_7(_x[7])
            out = out_0 + out_1 + out_2 + out_3 + out_4 + out_5 + out_6 + out_7
            if (dump_act != None):
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_0.activation', tensor=out_0)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_0.weight', tensor=self.fc_0.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_0.bias',   tensor=self.fc_0.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_1.activation', tensor=out_1)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_1.weight', tensor=self.fc_1.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_1.bias',   tensor=self.fc_1.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_2.activation', tensor=out_2)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_2.weight', tensor=self.fc_2.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_2.bias',   tensor=self.fc_2.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_3.activation', tensor=out_3)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_3.weight', tensor=self.fc_3.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_3.bias',   tensor=self.fc_3.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_4.activation', tensor=out_4)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_4.weight', tensor=self.fc_4.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_4.bias',   tensor=self.fc_4.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_5.activation', tensor=out_5)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_5.weight', tensor=self.fc_5.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_5.bias',   tensor=self.fc_5.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_6.activation', tensor=out_6)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_6.weight', tensor=self.fc_6.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_6.bias',   tensor=self.fc_6.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_7.activation', tensor=out_7)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_7.weight', tensor=self.fc_7.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_7.bias',   tensor=self.fc_7.bias)
        elif (self.num_of_slice == 16):  # 128
            out_0 = self.fc_0(_x[0])
            out_1 = self.fc_1(_x[1])
            out_2 = self.fc_2(_x[2])
            out_3 = self.fc_3(_x[3])
            out_4 = self.fc_4(_x[4])
            out_5 = self.fc_5(_x[5])
            out_6 = self.fc_6(_x[6])
            out_7 = self.fc_7(_x[7])
            out_8 = self.fc_8(_x[8])
            out_9 = self.fc_9(_x[9])
            out_10 = self.fc_10(_x[10])
            out_11 = self.fc_11(_x[11])
            out_12 = self.fc_12(_x[12])
            out_13 = self.fc_13(_x[13])
            out_14 = self.fc_14(_x[14])
            out_15 = self.fc_15(_x[15])
            out = out_0 + out_1 + out_2 + out_3 + out_4 + out_5 + out_6 + out_7 \
                  + out_8 + out_9 + out_10 + out_11 + out_12 + out_13 + out_14 + out_15
            if (dump_act != None):
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_0.activation', tensor=out_0)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_0.weight', tensor=self.fc_0.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_0.bias',   tensor=self.fc_0.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_1.activation', tensor=out_1)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_1.weight', tensor=self.fc_1.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_1.bias',   tensor=self.fc_1.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_2.activation', tensor=out_2)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_2.weight', tensor=self.fc_2.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_2.bias',   tensor=self.fc_2.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_3.activation', tensor=out_3)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_3.weight', tensor=self.fc_3.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_3.bias',   tensor=self.fc_3.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_4.activation', tensor=out_4)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_4.weight', tensor=self.fc_4.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_4.bias',   tensor=self.fc_4.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_5.activation', tensor=out_5)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_5.weight', tensor=self.fc_5.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_5.bias',   tensor=self.fc_5.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_6.activation', tensor=out_6)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_6.weight', tensor=self.fc_6.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_6.bias',   tensor=self.fc_6.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_7.activation', tensor=out_7)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_7.weight', tensor=self.fc_7.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_7.bias',   tensor=self.fc_7.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_8.activation', tensor=out_8)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_8.weight', tensor=self.fc_8.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_8.bias',   tensor=self.fc_8.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_9.activation', tensor=out_9)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_9.weight', tensor=self.fc_9.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_9.bias',   tensor=self.fc_9.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_10.activation', tensor=out_10)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_10.weight', tensor=self.fc_10.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_10.bias',   tensor=self.fc_10.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_11.activation', tensor=out_11)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_11.weight', tensor=self.fc_11.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_11.bias',   tensor=self.fc_11.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_12.activation', tensor=out_12)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_12.weight', tensor=self.fc_12.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_12.bias',   tensor=self.fc_12.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_13.activation', tensor=out_13)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_13.weight', tensor=self.fc_13.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_13.bias',   tensor=self.fc_13.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_14.activation', tensor=out_14)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_14.weight', tensor=self.fc_14.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_14.bias',   tensor=self.fc_14.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_15.activation', tensor=out_15)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_15.weight', tensor=self.fc_15.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_15.bias',   tensor=self.fc_15.bias)
        elif (self.num_of_slice == 32):  # 256
            out_0 = self.fc_0(_x[0])
            out_1 = self.fc_1(_x[1])
            out_2 = self.fc_2(_x[2])
            out_3 = self.fc_3(_x[3])
            out_4 = self.fc_4(_x[4])
            out_5 = self.fc_5(_x[5])
            out_6 = self.fc_6(_x[6])
            out_7 = self.fc_7(_x[7])
            out_8 = self.fc_8(_x[8])
            out_9 = self.fc_9(_x[9])
            out_10 = self.fc_10(_x[10])
            out_11 = self.fc_11(_x[11])
            out_12 = self.fc_12(_x[12])
            out_13 = self.fc_13(_x[13])
            out_14 = self.fc_14(_x[14])
            out_15 = self.fc_15(_x[15])
            out_16 = self.fc_16(_x[16])
            out_17 = self.fc_17(_x[17])
            out_18 = self.fc_18(_x[18])
            out_19 = self.fc_19(_x[19])
            out_20 = self.fc_20(_x[20])
            out_21 = self.fc_21(_x[21])
            out_22 = self.fc_22(_x[22])
            out_23 = self.fc_23(_x[23])
            out_24 = self.fc_24(_x[24])
            out_25 = self.fc_25(_x[25])
            out_26 = self.fc_26(_x[26])
            out_27 = self.fc_27(_x[27])
            out_28 = self.fc_28(_x[28])
            out_29 = self.fc_29(_x[29])
            out_30 = self.fc_30(_x[30])
            out_31 = self.fc_31(_x[31])
            out = out_0 + out_1 + out_2 + out_3 + out_4 + out_5 + out_6 + out_7 \
                  + out_8 + out_9 + out_10 + out_11 + out_12 + out_13 + out_14 + out_15 \
                  + out_16 + out_17 + out_18 + out_19 + out_20 + out_21 + out_22 + out_23 \
                  + out_24 + out_25 + out_26 + out_27 + out_28 + out_29 + out_30 + out_31
            if (dump_act != None):
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_0.activation', tensor=out_0)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_0.weight', tensor=self.fc_0.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_0.bias',   tensor=self.fc_0.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_1.activation', tensor=out_1)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_1.weight', tensor=self.fc_1.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_1.bias',   tensor=self.fc_1.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_2.activation', tensor=out_2)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_2.weight', tensor=self.fc_2.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_2.bias',   tensor=self.fc_2.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_3.activation', tensor=out_3)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_3.weight', tensor=self.fc_3.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_3.bias',   tensor=self.fc_3.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_4.activation', tensor=out_4)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_4.weight', tensor=self.fc_4.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_4.bias',   tensor=self.fc_4.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_5.activation', tensor=out_5)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_5.weight', tensor=self.fc_5.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_5.bias',   tensor=self.fc_5.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_6.activation', tensor=out_6)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_6.weight', tensor=self.fc_6.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_6.bias',   tensor=self.fc_6.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_7.activation', tensor=out_7)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_7.weight', tensor=self.fc_7.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_7.bias',   tensor=self.fc_7.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_8.activation', tensor=out_8)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_8.weight', tensor=self.fc_8.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_8.bias',   tensor=self.fc_8.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_9.activation', tensor=out_9)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_9.weight', tensor=self.fc_9.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_9.bias',   tensor=self.fc_9.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_10.activation', tensor=out_10)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_10.weight', tensor=self.fc_10.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_10.bias',   tensor=self.fc_10.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_11.activation', tensor=out_11)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_11.weight', tensor=self.fc_11.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_11.bias',   tensor=self.fc_11.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_12.activation', tensor=out_12)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_12.weight', tensor=self.fc_12.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_12.bias',   tensor=self.fc_12.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_13.activation', tensor=out_13)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_13.weight', tensor=self.fc_13.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_13.bias',   tensor=self.fc_13.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_14.activation', tensor=out_14)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_14.weight', tensor=self.fc_14.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_14.bias',   tensor=self.fc_14.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_15.activation', tensor=out_15)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_15.weight', tensor=self.fc_15.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_15.bias',   tensor=self.fc_15.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_16.activation', tensor=out_16)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_16.weight', tensor=self.fc_16.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_16.bias',   tensor=self.fc_16.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_17.activation', tensor=out_17)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_17.weight', tensor=self.fc_17.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_17.bias',   tensor=self.fc_17.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_18.activation', tensor=out_18)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_18.weight', tensor=self.fc_18.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_18.bias',   tensor=self.fc_18.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_19.activation', tensor=out_19)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_19.weight', tensor=self.fc_19.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_19.bias',   tensor=self.fc_19.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_20.activation', tensor=out_20)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_20.weight', tensor=self.fc_20.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_20.bias',   tensor=self.fc_20.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_21.activation', tensor=out_21)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_21.weight', tensor=self.fc_21.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_21.bias',   tensor=self.fc_21.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_22.activation', tensor=out_22)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_22.weight', tensor=self.fc_22.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_22.bias',   tensor=self.fc_22.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_23.activation', tensor=out_23)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_23.weight', tensor=self.fc_23.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_23.bias',   tensor=self.fc_23.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_24.activation', tensor=out_24)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_24.weight', tensor=self.fc_24.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_24.bias',   tensor=self.fc_24.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_25.activation', tensor=out_25)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_25.weight', tensor=self.fc_25.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_25.bias',   tensor=self.fc_25.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_26.activation', tensor=out_26)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_26.weight', tensor=self.fc_26.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_26.bias',   tensor=self.fc_26.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_27.activation', tensor=out_27)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_27.weight', tensor=self.fc_27.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_27.bias',   tensor=self.fc_27.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_28.activation', tensor=out_28)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_28.weight', tensor=self.fc_28.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_28.bias',   tensor=self.fc_28.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_29.activation', tensor=out_29)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_29.weight', tensor=self.fc_29.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_29.bias',   tensor=self.fc_29.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_30.activation', tensor=out_30)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_30.weight', tensor=self.fc_30.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_30.bias',   tensor=self.fc_30.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_31.activation', tensor=out_31)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_31.weight', tensor=self.fc_31.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.fc_31.bias',   tensor=self.fc_31.bias)
        else:
            print('Warning: num_of_slice is {0}, should not use SlicingLinearBlock'.format(self.num_of_slice))
        return out

class SlicingBlockFused(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, bias=True, ch_group=8):
        super(SlicingBlockFused, self).__init__()
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
                    bias=True)
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
        if(type(x) is tuple):
            for i in range(0, self.num_of_slice, 1):
                _x.append(torch.narrow(x[0], 1, _start, self.ch_group))
                _start += self.ch_group
            layerPrefix = x[1]
            dump_act = x[2]
        else:
            for i in range(0, self.num_of_slice, 1):
                _x.append(torch.narrow(x, 1, _start, self.ch_group))
                _start += self.ch_group
            layerPrefix = None
            dump_act = None

        if(self.num_of_slice == 2): # 16
            out_0 = self.conv_0(_x[0])
            out_1 = self.conv_1(_x[1])
            out = out_0+out_1
            if (dump_act != None):
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_0.activation', tensor=out_0)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_0.weight', tensor=self.conv_0.weight)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_0.bias', tensor=self.conv_0.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_1.activation', tensor=out_1)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_1.weight', tensor=self.conv_1.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_1.bias', tensor=self.conv_1.bias)
        elif(self.num_of_slice == 4): # 32
            out = self.conv_0(_x[0])+self.conv_1(_x[1])+self.conv_2(_x[2])+self.conv_3(_x[3])
            if (dump_act != None):
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_0.weight', tensor=self.conv_0.weight)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_0.bias', tensor=self.conv_0.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_1.weight', tensor=self.conv_1.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_1.bias', tensor=self.conv_1.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_2.weight', tensor=self.conv_2.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_2.bias', tensor=self.conv_2.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_3.weight', tensor=self.conv_3.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_3.bias', tensor=self.conv_3.bias)
        elif(self.num_of_slice == 8): # 64
            out = self.conv_0(_x[0])+self.conv_1(_x[1])+self.conv_2(_x[2])+self.conv_3(_x[3])\
                 +self.conv_4(_x[4])+self.conv_5(_x[5])+self.conv_6(_x[6])+self.conv_7(_x[7])
            if (dump_act != None):
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_0.weight', tensor=self.conv_0.weight)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_0.bias', tensor=self.conv_0.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_1.weight', tensor=self.conv_1.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_1.bias', tensor=self.conv_1.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_2.weight', tensor=self.conv_2.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_2.bias', tensor=self.conv_2.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_3.weight', tensor=self.conv_3.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_3.bias', tensor=self.conv_3.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_4.weight', tensor=self.conv_4.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_4.bias', tensor=self.conv_4.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_5.weight', tensor=self.conv_5.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_5.bias', tensor=self.conv_5.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_6.weight', tensor=self.conv_6.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_6.bias', tensor=self.conv_6.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_7.weight', tensor=self.conv_7.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_7.bias', tensor=self.conv_7.bias)
        elif (self.num_of_slice == 16):  # 128
            out = self.conv_0(_x[0]) + self.conv_1(_x[1]) + self.conv_2(_x[2]) + self.conv_3(_x[3]) \
                  + self.conv_4(_x[4]) + self.conv_5(_x[5]) + self.conv_6(_x[6]) + self.conv_7(_x[7]) \
                  + self.conv_8(_x[8]) + self.conv_9(_x[9]) + self.conv_10(_x[10]) + self.conv_11(_x[11]) \
                  + self.conv_12(_x[12]) + self.conv_13(_x[13]) + self.conv_14(_x[14]) + self.conv_15(_x[15])
            if (dump_act != None):
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_0.weight', tensor=self.conv_0.weight)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_0.bias', tensor=self.conv_0.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_1.weight', tensor=self.conv_1.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_1.bias', tensor=self.conv_1.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_2.weight', tensor=self.conv_2.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_2.bias', tensor=self.conv_2.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_3.weight', tensor=self.conv_3.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_3.bias', tensor=self.conv_3.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_4.weight', tensor=self.conv_4.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_4.bias', tensor=self.conv_4.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_5.weight', tensor=self.conv_5.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_5.bias', tensor=self.conv_5.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_6.weight', tensor=self.conv_6.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_6.bias', tensor=self.conv_6.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_7.weight', tensor=self.conv_7.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_7.bias', tensor=self.conv_7.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_8.weight', tensor=self.conv_8.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_8.bias', tensor=self.conv_8.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_9.weight', tensor=self.conv_9.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_9.bias', tensor=self.conv_9.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_10.weight', tensor=self.conv_10.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_10.bias', tensor=self.conv_10.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_11.weight', tensor=self.conv_11.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_11.bias', tensor=self.conv_11.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_12.weight', tensor=self.conv_12.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_12.bias', tensor=self.conv_12.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_13.weight', tensor=self.conv_13.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_13.bias', tensor=self.conv_13.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_14.weight', tensor=self.conv_14.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_14.bias', tensor=self.conv_14.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_15.weight', tensor=self.conv_15.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_15.bias', tensor=self.conv_15.bias)
        elif (self.num_of_slice == 32):  # 256
            out = self.conv_0(_x[0]) + self.conv_1(_x[1]) + self.conv_2(_x[2]) + self.conv_3(_x[3]) \
                  + self.conv_4(_x[4]) + self.conv_5(_x[5]) + self.conv_6(_x[6]) + self.conv_7(_x[7]) \
                  + self.conv_8(_x[8]) + self.conv_9(_x[9]) + self.conv_10(_x[10]) + self.conv_11(_x[11]) \
                  + self.conv_12(_x[12]) + self.conv_13(_x[13]) + self.conv_14(_x[14]) + self.conv_15(_x[15]) \
                  + self.conv_16(_x[16]) + self.conv_17(_x[17]) + self.conv_18(_x[18]) + self.conv_19(_x[19]) \
                  + self.conv_20(_x[20]) + self.conv_21(_x[21]) + self.conv_22(_x[22]) + self.conv_23(_x[23]) \
                  + self.conv_24(_x[24]) + self.conv_25(_x[25]) + self.conv_26(_x[26]) + self.conv_27(_x[27]) \
                  + self.conv_28(_x[28]) + self.conv_29(_x[29]) + self.conv_30(_x[30]) + self.conv_31(_x[31])
            if (dump_act != None):
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_0.weight', tensor=self.conv_0.weight)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_0.bias', tensor=self.conv_0.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_1.weight', tensor=self.conv_1.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_1.bias', tensor=self.conv_1.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_2.weight', tensor=self.conv_2.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_2.bias', tensor=self.conv_2.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_3.weight', tensor=self.conv_3.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_3.bias', tensor=self.conv_3.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_4.weight', tensor=self.conv_4.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_4.bias', tensor=self.conv_4.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_5.weight', tensor=self.conv_5.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_5.bias', tensor=self.conv_5.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_6.weight', tensor=self.conv_6.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_6.bias', tensor=self.conv_6.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_7.weight', tensor=self.conv_7.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_7.bias', tensor=self.conv_7.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_8.weight', tensor=self.conv_8.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_8.bias', tensor=self.conv_8.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_9.weight', tensor=self.conv_9.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_9.bias', tensor=self.conv_9.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_10.weight', tensor=self.conv_10.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_10.bias', tensor=self.conv_10.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_11.weight', tensor=self.conv_11.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_11.bias', tensor=self.conv_11.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_12.weight', tensor=self.conv_12.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_12.bias', tensor=self.conv_12.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_13.weight', tensor=self.conv_13.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_13.bias', tensor=self.conv_13.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_14.weight', tensor=self.conv_14.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_14.bias', tensor=self.conv_14.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_15.weight', tensor=self.conv_15.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_15.bias', tensor=self.conv_15.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_16.weight', tensor=self.conv_16.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_16.bias', tensor=self.conv_16.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_17.weight', tensor=self.conv_17.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_17.bias', tensor=self.conv_17.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_18.weight', tensor=self.conv_18.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_18.bias', tensor=self.conv_18.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_19.weight', tensor=self.conv_19.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_19.bias', tensor=self.conv_19.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_20.weight', tensor=self.conv_20.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_20.bias', tensor=self.conv_20.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_21.weight', tensor=self.conv_21.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_21.bias', tensor=self.conv_21.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_22.weight', tensor=self.conv_22.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_22.bias', tensor=self.conv_22.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_23.weight', tensor=self.conv_23.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_23.bias', tensor=self.conv_23.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_24.weight', tensor=self.conv_24.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_24.bias', tensor=self.conv_24.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_25.weight', tensor=self.conv_25.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_25.bias', tensor=self.conv_25.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_26.weight', tensor=self.conv_26.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_26.bias', tensor=self.conv_26.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_27.weight', tensor=self.conv_27.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_27.bias', tensor=self.conv_27.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_28.weight', tensor=self.conv_28.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_28.bias', tensor=self.conv_28.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_29.weight', tensor=self.conv_29.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_29.bias', tensor=self.conv_29.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_30.weight', tensor=self.conv_30.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_30.bias', tensor=self.conv_30.bias)
                dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_31.weight', tensor=self.conv_31.weight)
                # dump_to_npy(name=str(dump_act) + '.' + str(layerPrefix) + '.conv_31.bias', tensor=self.conv_31.bias)
        else:
            print('Warning: num_of_slice is {0}, should not use SlicingBlock'.format(self.num_of_slice))
        return out