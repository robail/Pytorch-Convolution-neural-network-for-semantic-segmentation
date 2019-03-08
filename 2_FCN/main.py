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

#kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
#train_loader = torch.utils.data.DataLoader(
#    datasets.CIFAR10('../', train=True, download=True,
#                   transform=transforms.Compose([
#                       transforms.RandomHorizontalFlip(),
#                       transforms.ToTensor(),
#                       transforms.Normalize((0.491399689874, 0.482158419622, 0.446530924224), (0.247032237587, 0.243485133253, 0.261587846975))
#                   ])),
#    batch_size=args.batch_size, shuffle=True, **kwargs)
#test_loader = torch.utils.data.DataLoader(
#    datasets.CIFAR10('../', train=False, transform=transforms.Compose([
#                       transforms.RandomHorizontalFlip(),
#                       transforms.ToTensor(),
#                       transforms.Normalize((0.491399689874, 0.482158419622, 0.446530924224), (0.247032237587, 0.243485133253, 0.261587846975))
#                   ])),
#    batch_size=args.batch_size, shuffle=False, **kwargs)

# get the model and convert it into cuda for if necessary
#net  = model.SqueezeNet()
    
#net  = models.vgg16(pretrained = True)
net  = model.FCN16s(n_class=21)
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

# create optimizer
# using the 55 epoch learning rule here
def paramsforepoch(epoch):
    p = dict()
    regimes = [[1, 18, 5e-3, 5e-4],
               [19, 29, 1e-3, 5e-4],
               [30, 43, 5e-4, 5e-4],
               [44, 52, 1e-4, 0],
               [53, 1e8, 1e-5, 0]]
    # regimes = [[1, 18, 1e-4, 5e-4],
    #            [19, 29, 5e-5, 5e-4],
    #            [30, 43, 1e-5, 5e-4],
    #            [44, 52, 5e-6, 0],
    #            [53, 1e8, 1e-6, 0]]
    for i, row in enumerate(regimes):
        if epoch >= row[0] and epoch <= row[1]:
            p['learning_rate'] = row[2]
            p['weight_decay'] = row[3]
    return p

avg_loss = list()
best_accuracy = 0.0
fig1, ax1 = plt.subplots()


# train the model
# TODO: Compute training accuracy and test accuracy

#print(list(net.parameters()))
#print(len(list(net.parameters())))
#
#print(list(net.parameters())[0].size())
#print(list(net.parameters())[1].size())
#print(list(net.parameters())[2].size())

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