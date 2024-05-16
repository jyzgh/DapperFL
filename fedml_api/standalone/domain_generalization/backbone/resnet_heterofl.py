import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
from math import ceil as up

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


'''
HeteroFL
ResNet for HeteroFL (BN is not tracked)
''' 

class ResNet_HeteroFL(nn.Module):  #  ResNethp, Width-varying ResNet
    def __init__(self, block, num_blocks, p_drop, num_classes=10, track=False):
        super(ResNet_HeteroFL, self).__init__()
        self.in_planes = up(64*p_drop)        
        # self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7,
        #                        stride=2, padding=3, bias=False)
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes, momentum=None, track_running_stats=track)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, up(64*p_drop), num_blocks[0], stride=1, track=track)
        self.layer2 = self._make_layer(block, up(128*p_drop), num_blocks[1], stride=2, track=track)
        self.layer3 = self._make_layer(block, up(256*p_drop), num_blocks[2], stride=2, track=track)
        self.layer4 = self._make_layer(block, up(512*p_drop), num_blocks[3], stride=2, track=track)
        self.fc = nn.Linear(up(512*block.expansion*p_drop), num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, track):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, track))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        logits = self.fc(out)
        probas = F.softmax(logits, dim=1)
        return logits, probas


class ResNet_c_HeteroFL(nn.Module): # ResNetchp, Width-varying ResNet designed for CIFAR-10, less parameters with deeper network
    def __init__(self, block, num_blocks, p_drop, num_classes=10, track=False):
        super(ResNet_c_HeteroFL, self).__init__()
        self.in_planes = up(16*p_drop)

        self.conv1 = nn.Conv2d(3, up(16*p_drop), kernel_size=3,stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes, momentum=None, track_running_stats=track)
        self.layer1 = self._make_layer(block, up(16*p_drop), num_blocks[0], stride=1, track=track)
        self.layer2 = self._make_layer(block, up(32*p_drop), num_blocks[1], stride=2, track=track)
        self.layer3 = self._make_layer(block,up(64*p_drop), num_blocks[2], stride=2, track=track)
        self.fc = nn.Linear(up(64*block.expansion*p_drop), num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, track):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, track))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        logits = self.fc(out)
        probas = F.softmax(logits, dim=1)
        return logits, probas
    