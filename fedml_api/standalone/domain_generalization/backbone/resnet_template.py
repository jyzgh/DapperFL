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


class ResNet(nn.Module):
    '''
    Vanilla ResNet, but different kernel with He's "deep residual networks for image recognition"
    '''
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        # self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7,
                            #    stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        # out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        logits = self.linear(out)
        probas = F.softmax(logits, dim=1)
        return logits, probas


class ResNet_c(nn.Module):
    '''
    Vanilla ResNet for CIFAR-10, less parameters with deeper network (narrow & deep)
    '''
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_c, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.fc = nn.Linear(64*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
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

    
##########################################################################################
        
    
class ResNet_WD(nn.Module): 
    '''
    Depth, width-varying ResNet w/ learnable step option
    '''
    def __init__(self, block, num_blocks, step_size_2d_list, p_drop, learnable_step=True, num_classes=10, width_per_group=64):
        super(ResNet_WD, self).__init__()
        # self.base_width = width_per_group
        self.in_planes = up(64*p_drop)
        # self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7,
        #                        stride=2, padding=3, bias=False)
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, up(64*p_drop), num_blocks[0], step_size_2d_list[0], learnable_step=learnable_step, stride=1, base_width=width_per_group)
        self.layer2 = self._make_layer(block, up(128*p_drop), num_blocks[1], step_size_2d_list[1], learnable_step=learnable_step, stride=2, base_width=width_per_group)
        self.layer3 = self._make_layer(block, up(256*p_drop), num_blocks[2], step_size_2d_list[2], learnable_step=learnable_step, stride=2, base_width=width_per_group)
        self.layer4 = self._make_layer(block, up(512*p_drop), num_blocks[3], step_size_2d_list[3], learnable_step=learnable_step, stride=2, base_width=width_per_group)

        # self.fc = nn.Linear(up(512*block.expansion*p_drop), num_classes)
        self.fc = nn.Linear(self.in_planes, num_classes)

    def _make_layer(self, block, planes, num_blocks, step_size_1d_list, learnable_step, stride, base_width):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        i = 0
        for stride in strides:
            layers.append(block(self.in_planes, planes, step_size_1d_list[i], learnable_step=learnable_step, stride=stride, base_width=base_width))
            self.in_planes = planes * block.expansion
            i += 1
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        # out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        logits = self.fc(out)
        probas = F.softmax(logits, dim=1)
        return logits, probas


class ResNet_c_WD(nn.Module):
    '''
    Depth, width-varying ResNet designed for CIFAR w/ learnable step option
    '''
    def __init__(self, block, num_blocks, step_size_2d_list, p_drop, learnable_step=True, num_classes=10):
        super(ResNet_c_WD, self).__init__()
        self.in_planes = up(16*p_drop)

        self.conv1 = nn.Conv2d(3, up(16*p_drop), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(up(16*p_drop))
        self.layer1 = self._make_layer(block, up(16*p_drop), num_blocks[0], step_size_2d_list[0], learnable_step=learnable_step, stride=1)
        self.layer2 = self._make_layer(block, up(32*p_drop), num_blocks[1], step_size_2d_list[1], learnable_step=learnable_step, stride=2)
        self.layer3 = self._make_layer(block, up(64*p_drop), num_blocks[2], step_size_2d_list[2], learnable_step=learnable_step, stride=2)
        self.fc = nn.Linear(up(64*block.expansion*p_drop), num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, step_size_1d_list, learnable_step, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        i = 0
        for stride in strides:
            layers.append(block(self.in_planes, planes, step_size_1d_list[i], learnable_step=learnable_step, stride=stride))
            self.in_planes = planes * block.expansion
            i += 1
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


class ResNet_WD224(nn.Module): 
    '''
    Depth, width-varying ResNet w/ learnable step option
    '''
    def __init__(self, block, num_blocks, step_size_2d_list, p_drop, learnable_step=True, num_classes=10, width_per_group=64):
        super(ResNet_WD224, self).__init__()
        # self.base_width = width_per_group
        self.in_planes = up(64*p_drop)
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7,
                               stride=2, padding=3, bias=False)
        # self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3,
        #                        stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, up(64*p_drop), num_blocks[0], step_size_2d_list[0], learnable_step=learnable_step, stride=1, base_width=width_per_group)
        self.layer2 = self._make_layer(block, up(128*p_drop), num_blocks[1], step_size_2d_list[1], learnable_step=learnable_step, stride=2, base_width=width_per_group)
        self.layer3 = self._make_layer(block, up(256*p_drop), num_blocks[2], step_size_2d_list[2], learnable_step=learnable_step, stride=2, base_width=width_per_group)
        self.layer4 = self._make_layer(block, up(512*p_drop), num_blocks[3], step_size_2d_list[3], learnable_step=learnable_step, stride=2, base_width=width_per_group)

        # self.fc = nn.Linear(up(512*block.expansion*p_drop), num_classes)
        self.fc = nn.Linear(self.in_planes, num_classes)

    def _make_layer(self, block, planes, num_blocks, step_size_1d_list, learnable_step, stride, base_width):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        i = 0
        for stride in strides:
            layers.append(block(self.in_planes, planes, step_size_1d_list[i], learnable_step=learnable_step, stride=stride, base_width=base_width))
            self.in_planes = planes * block.expansion
            i += 1
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