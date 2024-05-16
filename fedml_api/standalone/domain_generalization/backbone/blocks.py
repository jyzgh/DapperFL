from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
import torch


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

class BasicBlock(nn.Module):  # Vanilla
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option = 'B'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            if option == 'A':
                self.downsample = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes,
                            kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


'''
DepthFL
'''

class BasicBlockD(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, step_size, learnable_step=False, stride=1, base_width=64, option = 'B'):
        super(BasicBlockD, self).__init__()
        width = int(planes * (base_width / 64.0))
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.step_size = step_size
        # if step_size:
        #     self.step_size = nn.Parameter(torch.ones(1, requires_grad=True)*step_size)
        #     if not learnable_step:
        #         self.step_size.requires_grad = False
        # else:
        #     self.step_size = nn.Parameter(torch.ones(1, requires_grad=False)*step_size)
        #     self.step_size.requires_grad = False

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            if option == 'A':
                self.downsample = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes,
                            kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.step_size*self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out
    

class BottleneckD(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, step_size, learnable_step=True, stride=1, base_width=64):
        super(BottleneckD, self).__init__()
        width = int(planes * (base_width / 64.0))
        self.conv1 = nn.Conv2d(in_planes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.step_size = step_size
        # if step_size:
        #     self.step_size = nn.Parameter(torch.ones(1, requires_grad=True)*step_size)
        #     if not learnable_step:
        #         self.step_size.requires_grad = False
        # else:
        #     self.step_size = nn.Parameter(torch.ones(1, requires_grad=False)*step_size)
        #     self.step_size.requires_grad = False
                    
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.step_size*self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


'''
HeteroFL
'''

class BasicBlockH(nn.Module):  # Basic Block for HeteroFL (BN is not tracked)
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, track=False):
        super(BasicBlockH, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=None, track_running_stats=track)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=None, track_running_stats=track)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                        kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes, momentum=None, track_running_stats=track)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out    
    

class BottleneckH(nn.Module): # Bottleneck Block for HeteroFL (BN is not tracked)
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, base_width=64, track=False):
        super(BottleneckH, self).__init__()
        width = int(planes * (base_width / 64.0))
        self.conv1 = nn.Conv2d(in_planes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width, momentum=None, track_running_stats=track)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width, momentum=None, track_running_stats=track)
        self.conv3 = nn.Conv2d(width, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes, momentum=None, track_running_stats=track)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes, momentum=None, track_running_stats=track)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.step_size*self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out
    
'''
Learnable step
'''    
class BasicBlockM(nn.Module):  # NeFL - step_size learnable
    expansion = 1

    def __init__(self, in_planes, planes, step_size, learnable_step=True, stride=1, base_width=64, option = 'B'):
        super(BasicBlockM, self).__init__()
        width = int(planes * (base_width / 64.0))
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        if step_size:
            self.step_size = nn.Parameter(torch.ones(1, requires_grad=True)*step_size)
            if not learnable_step:
                self.step_size.requires_grad = False
        else:
            self.step_size = nn.Parameter(torch.ones(1, requires_grad=False)*step_size)
            self.step_size.requires_grad = False

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            if option == 'A':
                self.downsample = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes,
                            kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.step_size*self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out
    
        
class BottleneckM(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, step_size, learnable_step=True, stride=1, base_width=64):
        super(BottleneckM, self).__init__()
        width = int(planes * (base_width / 64.0))
        self.conv1 = nn.Conv2d(in_planes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        if step_size:
            self.step_size = nn.Parameter(torch.ones(1, requires_grad=True)*step_size)
            if not learnable_step:
                self.step_size.requires_grad = False
        else:
            self.step_size = nn.Parameter(torch.ones(1, requires_grad=False)*step_size)
            self.step_size.requires_grad = False
                    
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.step_size*self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out
    
