import torch
import torch.nn as nn

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
        self.bn_0 = nn.BatchNorm2d(planes)
        self.conv_1 = nn.Conv2d(self.ch_group, \
                    planes, \
                    kernel_size=kernel_size, \
                    stride=stride, \
                    padding=padding, \
                    bias=bias)
        self.bn_1 = nn.BatchNorm2d(planes)
        if(self.num_of_slice > 2):
            self.conv_2 = nn.Conv2d(self.ch_group, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
            self.bn_2 = nn.BatchNorm2d(planes)
            self.conv_3 = nn.Conv2d(self.ch_group, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
            self.bn_3 = nn.BatchNorm2d(planes)
        if(self.num_of_slice > 4):
            self.conv_4 = nn.Conv2d(self.ch_group, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
            self.bn_4 = nn.BatchNorm2d(planes)
            self.conv_5 = nn.Conv2d(self.ch_group, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
            self.bn_5 = nn.BatchNorm2d(planes)
            self.conv_6 = nn.Conv2d(self.ch_group, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
            self.bn_6 = nn.BatchNorm2d(planes)
            self.conv_7 = nn.Conv2d(self.ch_group, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
            self.bn_7 = nn.BatchNorm2d(planes)
        if(self.num_of_slice > 8):
            self.conv_8 = nn.Conv2d(self.ch_group, planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias=bias)
            self.bn_8 = nn.BatchNorm2d(planes)
            self.conv_9 = nn.Conv2d(self.ch_group, planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias=bias)
            self.bn_9 = nn.BatchNorm2d(planes)
            self.conv_10 = nn.Conv2d(self.ch_group, planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias=bias)
            self.bn_10 = nn.BatchNorm2d(planes)
            self.conv_11 = nn.Conv2d(self.ch_group, planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias=bias)
            self.bn_11 = nn.BatchNorm2d(planes)
            self.conv_12 = nn.Conv2d(self.ch_group, planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias=bias)
            self.bn_12 = nn.BatchNorm2d(planes)
            self.conv_13 = nn.Conv2d(self.ch_group, planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias=bias)
            self.bn_13 = nn.BatchNorm2d(planes)
            self.conv_14 = nn.Conv2d(self.ch_group, planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias=bias)
            self.bn_14 = nn.BatchNorm2d(planes)
            self.conv_15 = nn.Conv2d(self.ch_group, planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias=bias)
            self.bn_15 = nn.BatchNorm2d(planes)
        if (self.num_of_slice > 16):
            self.conv_16 = nn.Conv2d(self.ch_group, planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias=bias)
            self.bn_16 = nn.BatchNorm2d(planes)
            self.conv_17 = nn.Conv2d(self.ch_group, planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias=bias)
            self.bn_17 = nn.BatchNorm2d(planes)
            self.conv_18 = nn.Conv2d(self.ch_group, planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias=bias)
            self.bn_18 = nn.BatchNorm2d(planes)
            self.conv_19 = nn.Conv2d(self.ch_group, planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias=bias)
            self.bn_19 = nn.BatchNorm2d(planes)
            self.conv_20 = nn.Conv2d(self.ch_group, planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias=bias)
            self.bn_20 = nn.BatchNorm2d(planes)
            self.conv_21 = nn.Conv2d(self.ch_group, planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias=bias)
            self.bn_21 = nn.BatchNorm2d(planes)
            self.conv_22 = nn.Conv2d(self.ch_group, planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias=bias)
            self.bn_22 = nn.BatchNorm2d(planes)
            self.conv_23 = nn.Conv2d(self.ch_group, planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias=bias)
            self.bn_23 = nn.BatchNorm2d(planes)
            self.conv_24 = nn.Conv2d(self.ch_group, planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias=bias)
            self.bn_24 = nn.BatchNorm2d(planes)
            self.conv_25 = nn.Conv2d(self.ch_group, planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias=bias)
            self.bn_25 = nn.BatchNorm2d(planes)
            self.conv_26 = nn.Conv2d(self.ch_group, planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias=bias)
            self.bn_26 = nn.BatchNorm2d(planes)
            self.conv_27 = nn.Conv2d(self.ch_group, planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias=bias)
            self.bn_27 = nn.BatchNorm2d(planes)
            self.conv_28 = nn.Conv2d(self.ch_group, planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias=bias)
            self.bn_28 = nn.BatchNorm2d(planes)
            self.conv_29 = nn.Conv2d(self.ch_group, planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias=bias)
            self.bn_29 = nn.BatchNorm2d(planes)
            self.conv_30 = nn.Conv2d(self.ch_group, planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias=bias)
            self.bn_30 = nn.BatchNorm2d(planes)
            self.conv_31 = nn.Conv2d(self.ch_group, planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias=bias)
            self.bn_31 = nn.BatchNorm2d(planes)

    def forward(self, x):
        _x = []
        _start = 0
        for i in range(0, self.num_of_slice, 1):
            _x.append(torch.narrow(x, 1, _start, self.ch_group))
            _start += self.ch_group
        if(self.num_of_slice == 2): # 16
            _out_0 = self.conv_0(_x[0])
            _out_0 = self.bn_0(_out_0)
            _out_1 = self.conv_1(_x[1])
            _out_1 = self.bn_1(_out_1)
            out = _out_0+_out_1
        elif(self.num_of_slice == 4): # 32
            _out_0 = self.conv_0(_x[0])
            _out_0 = self.bn_0(_out_0)
            _out_1 = self.conv_1(_x[1])
            _out_1 = self.bn_1(_out_1)
            _out_2 = self.conv_2(_x[2])
            _out_2 = self.bn_2(_out_2)
            _out_3 = self.conv_3(_x[3])
            _out_3 = self.bn_3(_out_3)
            out = _out_0+_out_1+_out_2+_out_3
        elif(self.num_of_slice == 8): # 64
            _out_0 = self.conv_0(_x[0])
            _out_0 = self.bn_0(_out_0)
            _out_1 = self.conv_1(_x[1])
            _out_1 = self.bn_1(_out_1)
            _out_2 = self.conv_2(_x[2])
            _out_2 = self.bn_2(_out_2)
            _out_3 = self.conv_3(_x[3])
            _out_3 = self.bn_3(_out_3)
            _out_4 = self.conv_4(_x[4])
            _out_4 = self.bn_4(_out_4)
            _out_5 = self.conv_5(_x[5])
            _out_5 = self.bn_5(_out_5)
            _out_6 = self.conv_6(_x[6])
            _out_6 = self.bn_6(_out_6)
            _out_7 = self.conv_7(_x[7])
            _out_7 = self.bn_7(_out_7)
            out = _out_0+_out_1+_out_2+_out_3+_out_4+_out_5+_out_6+_out_7
        elif (self.num_of_slice == 16):  # 128
            _out_0 = self.conv_0(_x[0])
            _out_0 = self.bn_0(_out_0)
            _out_1 = self.conv_1(_x[1])
            _out_1 = self.bn_1(_out_1)
            _out_2 = self.conv_2(_x[2])
            _out_2 = self.bn_2(_out_2)
            _out_3 = self.conv_3(_x[3])
            _out_3 = self.bn_3(_out_3)
            _out_4 = self.conv_4(_x[4])
            _out_4 = self.bn_4(_out_4)
            _out_5 = self.conv_5(_x[5])
            _out_5 = self.bn_5(_out_5)
            _out_6 = self.conv_6(_x[6])
            _out_6 = self.bn_6(_out_6)
            _out_7 = self.conv_7(_x[7])
            _out_7 = self.bn_7(_out_7)
            _out_8 = self.conv_8(_x[8])
            _out_8 = self.bn_8(_out_8)
            _out_9 = self.conv_9(_x[9])
            _out_9 = self.bn_9(_out_9)
            _out_10 = self.conv_10(_x[10])
            _out_10 = self.bn_10(_out_10)
            _out_11 = self.conv_11(_x[11])
            _out_11 = self.bn_11(_out_11)
            _out_12 = self.conv_12(_x[12])
            _out_12 = self.bn_12(_out_12)
            _out_13 = self.conv_13(_x[13])
            _out_13 = self.bn_13(_out_13)
            _out_14 = self.conv_14(_x[14])
            _out_14 = self.bn_14(_out_14)
            _out_15 = self.conv_15(_x[15])
            _out_15 = self.bn_15(_out_15)
            out = _out_0 + _out_1 + _out_2 + _out_3 + _out_4 + _out_5 + _out_6 + _out_7 \
                  + _out_8 + _out_9 + _out_10 + _out_11 + _out_12 + _out_13 + _out_14 + _out_15
        elif (self.num_of_slice == 32):  # 256
            _out_0 = self.conv_0(_x[0])
            _out_0 = self.bn_0(_out_0)
            _out_1 = self.conv_1(_x[1])
            _out_1 = self.bn_1(_out_1)
            _out_2 = self.conv_2(_x[2])
            _out_2 = self.bn_2(_out_2)
            _out_3 = self.conv_3(_x[3])
            _out_3 = self.bn_3(_out_3)
            _out_4 = self.conv_4(_x[4])
            _out_4 = self.bn_4(_out_4)
            _out_5 = self.conv_5(_x[5])
            _out_5 = self.bn_5(_out_5)
            _out_6 = self.conv_6(_x[6])
            _out_6 = self.bn_6(_out_6)
            _out_7 = self.conv_7(_x[7])
            _out_7 = self.bn_7(_out_7)
            _out_8 = self.conv_8(_x[8])
            _out_8 = self.bn_8(_out_8)
            _out_9 = self.conv_9(_x[9])
            _out_9 = self.bn_9(_out_9)
            _out_10 = self.conv_10(_x[10])
            _out_10 = self.bn_10(_out_10)
            _out_11 = self.conv_11(_x[11])
            _out_11 = self.bn_11(_out_11)
            _out_12 = self.conv_12(_x[12])
            _out_12 = self.bn_12(_out_12)
            _out_13 = self.conv_13(_x[13])
            _out_13 = self.bn_13(_out_13)
            _out_14 = self.conv_14(_x[14])
            _out_14 = self.bn_14(_out_14)
            _out_15 = self.conv_15(_x[15])
            _out_15 = self.bn_15(_out_15)
            _out_16 = self.conv_16(_x[16])
            _out_16 = self.bn_16(_out_16)
            _out_17 = self.conv_17(_x[17])
            _out_17 = self.bn_17(_out_17)
            _out_18 = self.conv_18(_x[18])
            _out_18 = self.bn_18(_out_18)
            _out_19 = self.conv_19(_x[19])
            _out_19 = self.bn_19(_out_19)
            _out_20 = self.conv_20(_x[20])
            _out_20 = self.bn_20(_out_20)
            _out_21 = self.conv_21(_x[21])
            _out_21 = self.bn_21(_out_21)
            _out_22 = self.conv_22(_x[22])
            _out_22 = self.bn_22(_out_22)
            _out_23 = self.conv_23(_x[23])
            _out_23 = self.bn_23(_out_23)
            _out_24 = self.conv_24(_x[24])
            _out_24 = self.bn_24(_out_24)
            _out_25 = self.conv_25(_x[25])
            _out_25 = self.bn_25(_out_25)
            _out_26 = self.conv_26(_x[26])
            _out_26 = self.bn_26(_out_26)
            _out_27 = self.conv_27(_x[27])
            _out_27 = self.bn_27(_out_27)
            _out_28 = self.conv_28(_x[28])
            _out_28 = self.bn_28(_out_28)
            _out_29 = self.conv_29(_x[29])
            _out_29 = self.bn_29(_out_29)
            _out_30 = self.conv_30(_x[30])
            _out_30 = self.bn_30(_out_30)
            _out_31 = self.conv_31(_x[31])
            _out_31 = self.bn_31(_out_31)
            out = _out_0 + _out_1 + _out_2 + _out_3 + _out_4 + _out_5 + _out_6 + _out_7 \
                  + _out_8 + _out_9 + _out_10 + _out_11 + _out_12 + _out_13 + _out_14 + _out_15 \
                  + _out_16 + _out_17 + _out_18 + _out_19 + _out_20 + _out_21 + _out_22 + _out_23 \
                  + _out_24 + _out_25 + _out_26 + _out_27 + _out_28 + _out_29 + _out_30 + _out_31
        else:
            print('Warning: num_of_slice is {0}, should not use SlicingBlock'.format(self.num_of_slice))
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