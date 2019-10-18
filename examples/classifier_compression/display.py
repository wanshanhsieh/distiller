import sys

from matplotlib import pyplot as plt
import numpy as np
from itertools import cycle

import torch

class WeightsDistribution:
    def __init__(self):
        self.weightValue = []
        self.min = 0.0
        self.max = 0.0
        self.interval = []

    def getInterval(self, offset):
        minInteger = round(self.min, 1)
        maxInteger = round(self.max, 1)+offset
        dot = minInteger
        while(dot < maxInteger):
            self.interval.append(dot)
            dot += offset

    def getWeightValue(self, model):
        if(len(list(model.size())) == 4):
            for m in model:
                for c in m:
                    for S in c:
                        for R in S:
                            self.weightValue.append(R.item())
                            if(R > self.max):
                                self.max = R.item()
                            if(R <= self.min):
                                self.min = R.item()
        elif(len(list(model.size())) == 2):
            for S in model:
                for R in S:
                    self.weightValue.append(R.item())
                    if (R > self.max):
                        self.max = R.item()
                    if (R <= self.min):
                        self.min = R.item()

    def drawHistogram(self, color, label):
        a = np.array(self.weightValue)
        plt.hist(a, alpha=0.5, color=color, bins=self.interval, label=label)
        plt.legend()
        plt.savefig('./figure/' + label + '.png', dpi=None, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format=None,
                    transparent=False, bbox_inches=None, pad_inches=0.1,
                    frameon=None, metadata=None)
        plt.close('all')

def drawMutipleHistogram(valuesLists, intervalsLists, colors, labels):
    if(len(valuesLists) != len(intervalsLists)):
        print("Exception: {0}() unmatch numbers of data and interval".format(__name__))
        sys.exit()
    index = 0
    for values in valuesLists:
        a = np.array(values)
        plt.hist(a, alpha=0.5, color=colors[index], bins=intervalsLists[index], label=labels[index])
    plt.legend()
    plt.savefig('./figure/'+labels[index]+'.png', dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format=None,
            transparent=False, bbox_inches=None, pad_inches=0.1,
            frameon=None, metadata=None)
    #plt.show()



def main():
    cycol = cycle('bgrcmk')

    allWeightsLists = []
    allIntervalsLists = []
    allColors = []
    allNames = []

    checkpoint_name = 'D:/playground/MyDistiller/examples/classifier_compression/checkpoint/golden/quant8/2019.10.16-185349_resnet10_quant_ch8/checkpoint.pth'
    net = torch.load(checkpoint_name)
    for key in net['state_dict']:
        # print(key)
        value = net['state_dict'][key]
        if((('conv' in key or 'downsample' in key) and '.weight' in key and len(list(value.size())) == 5)\
                or ('fc' in key and '.weight' in key and len(list(value.size())) == 2)):
            allNames.append(key)

    for names in allNames:
        weightDis = WeightsDistribution()
        layer = torch.load(checkpoint_name)['state_dict'][names]
        weightDis.getWeightValue(layer)
        weightDis.getInterval(0.01)
        print('{0} done'.format(names))
        weightDis.drawHistogram(next(cycol), names)
        #allWeightsLists.append(weightDis.weightValue)
        #allIntervalsLists.append(weightDis.interval)
        #allColors.append(next(cycol))
        #drawMutipleHistogram(allWeightsLists, allIntervalsLists, allColors, allNames)

if __name__ == '__main__':
    main()
