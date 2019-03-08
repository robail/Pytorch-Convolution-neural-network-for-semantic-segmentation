import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.functional as F
import numpy as np
import torch.optim as optim
import math
import sys

class fire(nn.Module):
    def __init__(self, inplanes, squeeze_planes, expand_planes):
        super(fire, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(squeeze_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(expand_planes)
        self.conv3 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(expand_planes)
        self.relu2 = nn.ReLU()

        # using MSR initilization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2./n))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        out1 = self.conv2(x)
        out1 = self.bn2(out1)
        out2 = self.conv3(x)
        out2 = self.bn3(out2)
        out = torch.cat([out1, out2], 1)
        out = self.relu2(out)
        return out
    
class additional_layer(nn.Module):
    def __init__(self, add_input_channels, add_output_channels):
        super(additional_layer, self).__init__()
        self.conv1 = nn.Conv2d(add_input_channels, add_output_channels, kernel_size=3, stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(add_output_channels)
        self.relu1 = nn.ReLU(inplace=True)

        # using MSR initilization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2./n))

    def forward(self, x):
        out_add = self.conv1(x)
        out_add = self.bn1(x)
        out_add = self.relu1(x)
        return out_add  
    
    
class conv_layer(nn.Module):
    def __init__(self, add_input_channels, add_output_channels,kernel_size,stride,padding):
        super(conv_layer, self).__init__()
        self.conv1 = nn.Conv2d(add_input_channels, add_output_channels, kernel_size, stride,padding)
        self.bn1 = nn.BatchNorm2d(add_output_channels)
        self.relu1 = nn.ReLU(inplace=True)

        # using MSR initilization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2./n))

    def forward(self, x):
        out_add = self.conv1(x)
        out_add = self.bn1(out_add)
        out_add = self.relu1(out_add)
        return out_add      


class SqueezeNet(nn.Module):
    def __init__(self):
        super(SqueezeNet, self).__init__()
        self.conv1      = nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1) # 32
        self.bn1        = nn.BatchNorm2d(96)
        self.relu       = nn.ReLU()
        self.maxpool1   = nn.MaxPool2d(kernel_size=2, stride=2) # 16
        self.conv3x3_add_link_1 = conv_layer(96,96,kernel_size=3, stride=2, padding=1)
        self.fire2      = fire(96, 16, 64)
        self.dropout_1  = nn.Dropout2d(p=0.5,inplace=False)
        self.fire3      = fire(128, 16, 64)
        self.fire4      = fire(128, 32, 128)
        self.maxpool2   = nn.MaxPool2d(kernel_size=2, stride=2) # 8
#        self.conv3x3_add_link_2 = conv_layer(256,256,kernel_size=3, stride=2, padding=1)
        self.conv1x1_concat_1 = conv_layer(352,256,kernel_size=1, stride=1, padding=0)
        self.fire5      = fire(256, 32, 128)
        self.fire6      = fire(256, 48, 192)
        self.fire7      = fire(384, 48, 192)
        self.dropout_2  = nn.Dropout2d(p=0.5,inplace=False)
        self.fire8      = fire(384, 64, 256)
        self.maxpool3   = nn.MaxPool2d(kernel_size=2, stride=2) # 4
#        self.conv1x1_concat_2 = conv_layer(768,512,kernel_size=1, stride=1, padding=0)
        self.fire9      = fire(512, 64, 256)
        self.conv2      = nn.Conv2d(512, 10, kernel_size=1, stride=1)
        self.avg_pool   = nn.AvgPool2d(kernel_size=4, stride=4)
        self.softmax    = nn.LogSoftmax(dim=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
#        print("Initital Value:",x.size())
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        add_link_1 = x
        conved_add_link_1 = self.conv3x3_add_link_1(add_link_1)
#        print("add_link_1: ",add_link_1.size())
#        print("add_link_convolved",conved_add_link_1.size())
#        sys.exit()
        x = self.fire2(x)
        x = self.dropout_1(x)
        x = self.fire3(x)
        x = self.fire4(x)
        x = self.maxpool2(x)
        print("X output: ",x.size())
#        add_link_2 = x
#        conved_add_link_2 = self.conv3x3_add_link_2(add_link_2)
        x = torch.cat([x,conved_add_link_1],1)
        print("X concat output: ",x.size())
        sys.exit();
        x =  self.conv1x1_concat_1(x)
        x = self.fire5(x)
        x = self.fire6(x)
        x = self.fire7(x)
        x = self.dropout_2(x)
        x = self.fire8(x)
        x = self.maxpool3(x)
#        x = torch.cat([x,conved_add_link_2],1)
#        x = self.conv1x1_concat_2(x)
        x = self.fire9(x)
        x = self.conv2(x)
        x = self.avg_pool(x)
        x = self.softmax(x)
        return x

def fire_layer(inp, s, e):
    f = fire(inp, s, e)
    return f

def squeezenet(pretrained=False):
    net = SqueezeNet()
    # inp = Variable(torch.randn(64,3,32,32))
    # out = net.forward(inp)
    # print(out.size())
    return net

# if __name__ == '__main__':
#     squeezenet()
