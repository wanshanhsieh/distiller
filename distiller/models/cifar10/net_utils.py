import os
import torch.nn as nn
import numpy as np

NUM_CLASSES = 10
fileDumpPath = os.path.join('D:', os.sep, 'playground', 'MyDistiller', 'examples', 'classifier_compression', 'checkpoint', \
                            '20191107_resnet10_fp32_ch8_200x', '20191113_pytorch_resnet10_ch8_200x_onebias_dummyrelu_batch1_num1')

def dump_to_npy(name, tensor):
    if('activation' in name):
        fileName = os.path.join(fileDumpPath, 'activation', name)
    else:
        fileName = os.path.join(fileDumpPath, 'weight_bias', name)
    tensorToNumpy = tensor.detach().cpu().numpy()
    np.save(fileName, tensorToNumpy)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv3x3_bias(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)