'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

import sys
import os
import argparse
import numpy as np
from tifffile import imsave
from skimage import io

from utils import progress_bar
from distiller.models.cifar10.resnet_cifar import *

from torchviz import *
os.environ["PATH"] += os.pathsep + 'D:/2) install/graphviz-2.38/release/bin/'

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--net', '-n', help='NN model name', default='resnet10', dest='net')
parser.add_argument('--output', '-o', help='output file name', default='checkpoint', dest='output_file_name')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--ch', default=None, type=int, help='slicing input channels')
parser.add_argument('--epoch', '-e', default=1, type=int, help='training epoches')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--test', '-t', action='store_true', help='execute testing flow')
parser.add_argument('--fuse', '-f', action='store_true', help='fuse conv and bn')
parser.add_argument('--draw', '-d', action='store_true', help='draw the model graph')
parser.add_argument('--dump_act', '-dpa', default=None, type=int, help='dump img activation value')
parser.add_argument('--dump_img', '-dpi', action='store_true', help='dump resized image')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

def _mul(x):
    return x*255

def _sub(x):
    return x-128

def _round(x):
    return torch.round(x)

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Lambda(_mul),
    transforms.Lambda(_round),
    transforms.Lambda(_sub),
])

transform_test = transforms.Compose([
    transforms.RandomCrop(32, padding=0),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Lambda(_mul),
    transforms.Lambda(_round),
    transforms.Lambda(_sub),
])

if args.dump_img:
    transform_train_image = transforms.Compose([
        transforms.RandomCrop(32, padding=4), \
        transforms.Resize(224), \
        transforms.RandomHorizontalFlip(),
    ])

    transform_test_image = transforms.Compose([
        transforms.RandomCrop(32, padding=0), \
        transforms.Resize(224),
    ])
    train_image_set = torchvision.datasets.CIFAR10(root='../datasets/cifar10_resize_226x226', train=True, download=True, transform=transform_train_image)
    train_image_loader = torch.utils.data.DataLoader(train_image_set, batch_size=1, shuffle=True, num_workers=2)
    test_image_set = torchvision.datasets.CIFAR10(root='../datasets/cifar10_resize_226x226', train=False, download=True, transform=transform_test_image)
    test_image_loader = torch.utils.data.DataLoader(test_image_set, batch_size=1, shuffle=False, num_workers=2)

    filePath = os.path.join('D:', os.sep, 'playground', 'MyDistiller', 'examples', 'datasets', 'cifar10_img_train_224x224')
    f = open(os.path.join(filePath, 'cifar10_train.txt'), 'w')
    for i, (img, label) in enumerate(train_image_set):
        img_path = os.path.join(filePath, str(i)+'.jpg')
        img = np.asanyarray(img)
        io.imsave(img_path, img)
        f.write(str(i)+'.jpg' + ' ' + str(label) + '\n')
    f.close()
    sys.exit()

if not args.test:
    trainset = torchvision.datasets.CIFAR10(root='../datasets/cifar10_resize_226x226', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=400, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='../datasets/cifar10_resize_226x226', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
if args.fuse:
    fusion = True
else:
    fusion = False

if args.net == 'resnet10':
    if args.resume:
        net = resnet10_cifar(True, ch_group=args.ch, fusion=fusion)
    else:
        net = resnet10_cifar(False, ch_group=args.ch, fusion=fusion)
elif(args.net == 'resnet5'):
    net = Res5()
else:
    print('Error: unsupported net {0}'.format(args.net))

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
# print(net)
# sys.exit()

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer = optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
accRecord = []

def dump_NCHW(file, input):
    for n in range(0, input.size(0), 1):
        for c in range(0, input.size(1), 1):
            for h in range(0, input.size(2), 1):
                for w in range(0, input.size(3), 1):
                    file.write('{0}\t'.format(input[n][c][h][w]))
                file.write('\n')
            file.write('\n')
        file.write('\n')

# Training
def train(epoch):
    global input_train
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # if (batch_idx == 0):
        #    input_train = inputs
        optimizer.zero_grad()
        outputs = net(inputs, False, False)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch, fusion=False, dump_act=None):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if(dump_act == None or (dump_act != None and batch_idx == dump_act)):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs, fusion, dump_act)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    # print(input_dump.size(0), input_dump.size(1), input_dump.size(2), input_dump.size(3))

    # Save checkpoint.
    acc = 100.*correct/total
    if (acc > best_acc):
        best_acc = acc
    accRecord[epoch] = acc
    if not args.test or (args.test and fusion):
        print('Saving..')
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        innerFolder = '20191024_resnet10_quant8_fused_sym_-128_127_224x224_resize'
        if not os.path.isdir('checkpoint/'+str(innerFolder)):
            os.makedirs('checkpoint/'+str(innerFolder))
        if (acc > best_acc):
            torch.save(net.state_dict(), './checkpoint/' + str(innerFolder)+'/'+str(args.output_file_name)+'_best.pth')
        torch.save(net.state_dict(), './checkpoint/'+str(innerFolder)+'/'+str(args.output_file_name)+'.pth')
        # with open('./checkpoint/dump_train_input.txt', "w") as text_file0:
        #     dump_NCHW(text_file0, input_train)
        # text_file0.close()
        # with open('./checkpoint/dump_test_input.txt', "w") as text_file1:
        #     dump_NCHW(text_file1, input_test)
        # text_file1.close()

if __name__ == '__main__':
    if args.draw:
        dummy_inputs = torch.randn(1, 3, 224, 224)
        y = net(Variable(dummy_inputs), fusion)
        graph = make_dot(y, net.state_dict())
        graph.view()
        sys.exit()
    if args.test and not args.resume:
        print('Warning: no pretrained model is loaded, testing random data')
        times = 1
    elif args.test and args.resume:
        times = 1
    else:
        times = args.epoch
    if not args.test and args.dump_act != None:
        print('Error: cant dump img {0} activation values during training'.format(str(args.dump_act)))
        sys.exit()
    for epoch in range(start_epoch, start_epoch+times):
        accRecord.append(0.0)
        if not args.test:
            train(epoch)
        if (epoch == times-1):
            test(epoch, fusion, args.dump_act)
        else:
            test(epoch, False, args.dump_act)
    print('best_acc: {0}'.format(best_acc))
    for i in range(0, times):
        print(accRecord[i])
