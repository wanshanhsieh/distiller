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

from utils import progress_bar
from distiller.models.cifar10.resnet_cifar import *
from distiller.models.cifar10.net import *

from torchviz import *
os.environ["PATH"] += os.pathsep + 'D:/2) install/graphviz-2.38/release/bin/'

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--net', '-n', help='NN model name', dest='net')
parser.add_argument('--output', '-o', help='output file name', default='checkpoint.pth', dest='output_file_name')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--ch', default=None, type=int, help='slicing input channels')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--test', '-t', action='store_true', help='execute testing flow')
parser.add_argument('--draw', '-d', action='store_true', help='draw the model graph')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(226),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.RandomCrop(32, padding=0),
    transforms.Resize(226),
    transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.CIFAR10(root='../datasets/cifar10_resize_226x226', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=400, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='../datasets/cifar10_resize_226x226', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=400, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = resnet10_cifar(False)
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()

if args.net == 'resnet10':
    if args.resume:
        net = resnet10_cifar(True, ch_group=args.ch)
    else:
        net = resnet10_cifar(False, ch_group=args.ch)
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

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if not args.test:
        print('Saving..')
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(net.state_dict(), './checkpoint/'+str(args.output_file_name))
    if(acc > best_acc):
        best_acc = acc
    print('best_acc: {0}'.format(best_acc))

if __name__ == '__main__':
    if args.draw:
        dummy_inputs = torch.randn(1, 3, 226, 226)
        y = net(Variable(dummy_inputs))
        graph = make_dot(y, net.state_dict())
        graph.view()
        sys.exit()
    if args.test and not args.resume:
        print('Warning: no pretrained model is loaded, testing random data')
        times = 1
    elif args.test and args.resume:
        times = 1
    else:
        times = 50
    for epoch in range(start_epoch, start_epoch+times):
        if not args.test:
            train(epoch)
        test(epoch)
