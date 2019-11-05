import os
import torch.nn as nn
import numpy as np

NUM_CLASSES = 10
fileDumpPath = os.path.join('D:', os.sep, 'playground', 'MyDistiller', 'examples', 'classifier_compression', 'checkpoint', '20191103_resnet10_fp32_200x')

def dump_to_npy(name, tensor):
    fileName = os.path.join(fileDumpPath, name)
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