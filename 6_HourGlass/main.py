import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import models
import argparse
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
import sys
import csv
import shutil
import time
import model
import torch.nn.functional as F
import matplotlib.pyplot as plt
from IPython import embed
from torchsummary_1 import summary as modelsummary
#import torchsummary as modelsummary

#------ START added by Adeel ------
final_model_path = 'final_trained_model/200_epoch_squeezenet.pth'
best_prec1 = 0
start_epoch = 1
checkpoint_path = 'train_checkpoint/checkpoint.pth.tar'
#------ END added by Adeel ------ 

parser = argparse.ArgumentParser('Options for training SqueezeNet in pytorch')
parser.add_argument('--batch-size', type=int, default=200, metavar='N', help='batch size of train')
parser.add_argument('--epoch', type=int, default=500, metavar='N', help='number of epochs to train for')
parser.add_argument('--learning-rate', type=float, default=0.00001, metavar='LR', help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='percentage of past parameters to store')
parser.add_argument('--no-cuda', action='store_true', default=False, help='use cuda for training')
parser.add_argument('--log-schedule', type=int, default=1, metavar='N', help='number of epochs to save snapshot after')
parser.add_argument('--seed', type=int, default=10, help='set seed to some constant value to reproduce experiments')
parser.add_argument('--model_name', type=str, default=None, help='Use a pretrained model')
parser.add_argument('--want_to_test', type=bool, default=False, help='make true if you just want to test')
parser.add_argument('--epoch_55', action='store_true', help='would you like to use 55 epoch learning rule')
parser.add_argument('--num_classes', type=int, default=10, help="how many classes training for")
#------ START added by Adeel ------
parser.add_argument('--resume', default=checkpoint_path, type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
#------ END added by Adeel ------


#print(torch.cuda.get_device_name(0))

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.cuda.set_device(0)

# get the model and convert it into cuda for if necessary
#net  = model.SqueezeNet()
    
net  = model.gethgmodel()
torch.cuda.empty_cache()
print(torch.cuda.memory_allocated(device=None))

if args.model_name is not None:
    print("loading pre trained weights")
    pretrained_weights = torch.load(args.model_name)
    net.load_state_dict(pretrained_weights)

if args.cuda:
    print("GPU Working...\n")
    net.cuda()    

#------ Check Network parameters Size ----------------
#
#modelsummary(net.cuda(), input_size=(3, 32, 32))
#
#------ Check Network parameters Size ----------------    


#print(net)

avg_loss = list()
best_accuracy = 0.0
fig1, ax1 = plt.subplots()

# create a temporary optimizer
optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)

def adjustlrwd(params):
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = params['learning_rate']
        param_group['weight_decay'] = params['weight_decay']

#------ END added by Adeel ------

#==============================================================================================
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

#==============================================================================================



#====================================Main Method===============================================
if __name__ == '__main__':
    
    epoch_time = AverageMeter()
#    #------ Print and Save Network parameters ----------------
#    modelsummary.summary(net.cuda(), input_size=(3, 32, 32))
    
#    #---- Using Customized Model Summary Class ----------    
    modelsummary(net.cuda(), input_size=(3, 1024, 1024))
#    #---- Using Customized Model Summary Class ----------
#    #------ Print and Save Network parameters ----------------

#====================================Main Method===============================================        
#==============================================================================================
