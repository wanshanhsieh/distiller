#!/usr/bin/env python
# coding: utf-8
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torchnet.meter as tnt
import torch.onnx
import torch.optim as optim
from torchvision import datasets

import distiller
from distiller.quantization.range_linear import PostTrainLinearQuantizer, LinearQuantMode, RangeLinearQuantParamLayerWrapper
from distiller.utils import model_sparsity, model_params_stats
from distiller.apputils.data_loaders import __deterministic_worker_init_fn

import argparse

import math
import random
import time
import numpy as np

import glob, os, sys 

from importlib import reload
import logging
from distiller.model_summaries import weights_sparsity_summary

### Version check
print("python version : {}".format(sys.version))
print("Pytorch version : {}".format(torch.__version__))
print("Distiller version : {}".format(distiller.__version__))
print("Torchvision version : {}".format(torchvision.__version__))

print("The count of gpu: {}".format(torch.cuda.device_count()))
for idx in range(torch.cuda.device_count()):
    print("The gpu name: {}".format(torch.cuda.get_device_name(idx)))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(idx)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(idx)/1024**3,1), 'GB')
    print("============")

def _mul(x):
    return x*255

def _sub(x):
    return x-128

def _round(x):
    return torch.round(x)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(224), # 224 -> 220
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Lambda(_mul),
    transforms.Lambda(_round),
    transforms.Lambda(_sub),
])

transform_test = transforms.Compose([
    transforms.RandomCrop(32, padding=0),
    transforms.Resize(224), # 224 -> 220
    transforms.ToTensor(),
    transforms.Lambda(_mul),
    transforms.Lambda(_round),
    transforms.Lambda(_sub),
])

distiller.set_deterministic()
worker_init_fn = __deterministic_worker_init_fn

# trainset = torchvision.datasets.CIFAR10(root='../datasets/cifar10_resize_226x226', train=True, download=True, transform=transform_train)
# trainloader = torch.utils.data.DataLoader(trainset, \
#                                           batch_size=400, \
#                                           shuffle=True, \
#                                           num_workers=1, \
#                                           worker_init_fn=worker_init_fn)

testset = torchvision.datasets.CIFAR10(root='../datasets/cifar10_resize_226x226', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, \
                                         batch_size=100, \
                                         shuffle=False, \
                                         num_workers=1, \
                                         worker_init_fn=worker_init_fn)

def train(net, compression_scheduler, epochs, optimizer, criterion, args):
    start_t = time.time()
    net = net.cuda()
    optimizer = optimizer
    net.train()
    
    batch_size = trainloader.batch_size
    total_samples = len(trainloader.sampler)
    
    steps_per_epoch = math.ceil(total_samples / batch_size)

    print(batch_size, total_samples, steps_per_epoch)
        
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        if compression_scheduler:
                compression_scheduler.on_epoch_begin(epoch, )
        for step, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda() 
            if compression_scheduler:
                compression_scheduler.on_minibatch_begin(epoch, step, steps_per_epoch, optimizer)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            if args.kd_policy is None:
            # Revert to a "normal" forward-prop call if no knowledge distillation policy is present
                outputs = net(inputs)
            else:
                outputs = args.kd_policy.forward(inputs)
            
            loss = criterion(outputs, labels)
            
            if compression_scheduler:
            # Before running the backward phase, we allow the scheduler to modify the loss
            # (e.g. add regularization loss)
                agg_loss = compression_scheduler.before_backward_pass(epoch, step, steps_per_epoch, loss,
                                                                      optimizer=optimizer, return_loss_components=True)
                loss = agg_loss.overall_loss
                running_loss += loss.item()
                #losses[OVERALL_LOSS_KEY].add(loss.item())
            else:
                running_loss += loss.item()
                #losses[OVERALL_LOSS_KEY].add(loss.item())
            loss.backward()
            if compression_scheduler:
                pass
            optimizer.step()
            if compression_scheduler:
                compression_scheduler.on_minibatch_end(epoch, step, steps_per_epoch, optimizer)
            # print statistics
            if step+1 == steps_per_epoch: # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, step + 1, running_loss / steps_per_epoch))
                    
                running_loss = 0.0
            
        if epoch % 10 == 0:
            msglogger.info('[%d] loss: %.3f' %
                          (epoch + 1, running_loss / steps_per_epoch))
            msglogger.info("Sparsity: ")
            msglogger.info(model_params_stats(model))
            msglogger.info("Acc: ")
            msglogger.info(test(model, criterion))
            
        if compression_scheduler:
                compression_scheduler.on_epoch_end(epoch, optimizer)

    print('Finished Training')
    msglogger.info("Finished Train")
    end_t = time.time()
    print("Spent {} min".format((end_t - start_t)/60))
    msglogger.info("Spent {} min".format((end_t - start_t)/60))

def dump_to_npy(name, tensor):
    fileDumpPath = os.path.join('D:', os.sep, 'playground', 'MyDistiller', 'examples', 'classifier_compression', 'checkpoint', '20191031_resnet10_fp32_fused_220x220')
    fileName = os.path.join(fileDumpPath, name)
    tensorToNumpy = tensor.detach().cpu().numpy()
    np.save(fileName, tensorToNumpy)

def test(model, criterion):
    dump_act = None
    correct = 0
    total = 0
    classerr = tnt.ClassErrorMeter(accuracy=True, topk=(1, 5))
    losses = {'objective_loss': tnt.AverageValueMeter()}
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            if(dump_act == None or (dump_act != None and batch_idx == dump_act)):
                images, labels = images.cuda(), labels.cuda()
                # dump_to_npy(name= 'input.activation.int8.'+str(batch_idx), tensor=images)
                outputs = model(images)
                classerr.add(outputs.data, labels)

                loss = criterion(outputs, labels)
                losses['objective_loss'].add(loss.item())
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                if(total % 1000 == 0):
                    print('[{0}] accuracy {1}%'.format(total, str(correct/total*100)))

    acc = correct / total
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * acc))
    
    top1, top5 = classerr.value()[0], classerr.value()[1]
    print("Top1 = %.3f, Top5 = %.3f, loss = %.3f\n"%(top1, top5, losses["objective_loss"].mean))
    
    return top1, top5, losses['objective_loss'].mean


### Model training config
def configure(model, compress):
    model = model.cuda()
    compression_scheduler = distiller.CompressionScheduler(model)
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    if compress:
        source = compress
        compression_scheduler = distiller.CompressionScheduler(model)
        distiller.config.file_config(model, optimizer, compress, compression_scheduler,)
    parser = argparse.ArgumentParser()
    distiller.knowledge_distillation.add_distillation_args(parser)
    CONFIG_FILE = '.config_ipynb'
    if os.path.isfile(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            sys.argv = f.read().split()
    else:
        sys.argv = ['resnet.py','--kd-resume', 'net56_cifar.pth.tar','--kd-teacher',None,
                    '--kd-start-epoch', 0, 
                    '--kd-student-wt','0.5',
                    '--kd-teacher-wt', '0.0',
                    '--kd-distill-wt', '0.5',
                   ]
    args = parser.parse_args()
    args.kd_policy = None
    epochs = 30
    if args.kd_teacher:
        if args.kd_resume:
            teacher = torch.load(args.kd_resume)
        # Create policy and add to scheduler
        dlw = distiller.DistillationLossWeights(args.kd_distill_wt, args.kd_student_wt, args.kd_teacher_wt)
        args.kd_policy = distiller.KnowledgeDistillationPolicy(model, teacher, args.kd_temp, dlw)
        compression_scheduler.add_policy(args.kd_policy, starting_epoch=args.kd_start_epoch, ending_epoch=epochs,
                                         frequency=1)
    return model, compression_scheduler, epochs, optimizer, criterion, args


### Model Design
from distiller.models.cifar10 import resnet_cifar
model = resnet_cifar.resnet10_cifar(pretrained=True, ch_group=None, fusion=True).cuda()

compress="../quantization/quant_aware_train/quant_aware_train_linear_quant.yaml"
model, compression_scheduler, epochs, optimizer, criterion, args = configure(model, compress)
model.cuda()
model.eval()
filename = "1028" # log name
model_name = "1028" # model name
logName = "./logging/resnet10/"+filename+".log"
logging.basicConfig(level=logging.DEBUG,\
                    format='%(asctime)s - %(levelname)s : %(message)s',\
                    filename=logName, \
                    filemode='w')
msglogger = logging.getLogger(model_name)

if __name__ == '__main__':
    ### Training
    # train(model, compression_scheduler, epochs, optimizer, criterion, args)
    ### Testing
    test(model, criterion)
    # print(model)




