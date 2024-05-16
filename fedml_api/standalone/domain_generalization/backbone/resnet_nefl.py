from .blocks import *
from .resnet_template import *
from .resnet_heterofl import *


'''
Vanilla
'''
def resnet18(num_classes): # Vanilla ResNet
    """Constructs a ResNet-18 model."""
    model = ResNet(block=BasicBlock, 
                   num_blocks=[2, 2, 2, 2],
                   num_classes=num_classes,
                   )
    return model


def resnet56(num_classes):
    model = ResNet_c(block=BasicBlock,
                   num_blocks=[9, 9, 9],
                   num_classes=num_classes,
                   )
    return model


def resnet110(num_classes):
    model = ResNet_c(block=BasicBlock, 
                   num_blocks=[18, 18, 18],
                   num_classes=num_classes,
                   )
    return model


'''
NeFL-WD
'''
def resnet10wd(step_size_2d_list, p, learnable_step, num_classes): ## NeFL-WD (kernel 3)
    model = ResNet_WD(BasicBlockM,
                   [1, 1, 1, 1], step_size_2d_list,
                   p_drop = p, learnable_step=learnable_step, num_classes=num_classes
                   )
    return model


def resnet18wd(step_size_2d_list, p, learnable_step, num_classes): ## NeFL-WD (kernel 3)
    model = ResNet_WD(BasicBlockM,
                   [2, 2, 2, 2], step_size_2d_list,
                   p_drop = p, learnable_step=learnable_step, num_classes=num_classes
                   )
    return model


def resnet34wd(step_size_2d_list, p, learnable_step, num_classes): ## NeFL-WD (kernel 3)
    model = ResNet_WD(BasicBlockM,
                   [3, 4, 6, 3], step_size_2d_list,
                   p_drop = p, learnable_step=learnable_step, num_classes=num_classes
                   )
    return model


def resnet56wd(step_size_2d_list, p, learnable_step, num_classes): # NeFL-WD (kernel 3)
    model = ResNet_c_WD(BasicBlockM, 
                   [9, 9, 9], step_size_2d_list,
                   p_drop = p, learnable_step=learnable_step, num_classes=num_classes
                   )
    return model


def resnet110wd(step_size_2d_list, p, learnable_step, num_classes):
    model = ResNet_c_WD(BasicBlockM, 
                   [18, 18, 18], step_size_2d_list,
                   p_drop = p, learnable_step=learnable_step, num_classes=num_classes
                   )
    return model

def resnet101wd(step_size_2d_list, p, learnable_step, num_classes):
    model = ResNet_WD(BottleneckM,
                   [3, 4, 23, 3], step_size_2d_list,
                   p_drop = p, learnable_step=learnable_step, num_classes=num_classes
                   )
    return model


def resnet101_2wd(step_size_2d_list, p, learnable_step, num_classes):
    model = ResNet_WD224(BottleneckM,
                   [3, 4, 23, 3], step_size_2d_list,
                   p_drop = p, learnable_step=learnable_step, num_classes=num_classes, width_per_group=128
                   )
    return model


'''
DepthFL
'''
def resnet18DFs(step_size_2d_list, num_classes): # submodel
    model = ResNet_WD(BasicBlockD,
                   [2, 2, 2, 2], step_size_2d_list,
                   p_drop = 1, learnable_step=False, num_classes=num_classes
                   )
    return model


def resnet34DFs(step_size_2d_list, num_classes): # submodel
    model = ResNet_WD(BasicBlockD,
                   [3, 4, 6, 3], step_size_2d_list,
                   p_drop = 1, learnable_step=False, num_classes=num_classes
                   )
    return model


def resnet56DFs(step_size_2d_list, num_classes): # submodel
    model = ResNet_c_WD(BasicBlockD,
                   [9, 9, 9], step_size_2d_list,
                   p_drop = 1, learnable_step=False, num_classes=num_classes
                   )
    return model


def resnet110DFs(step_size_2d_list, num_classes): # submodel
    model = ResNet_c_WD(BasicBlockD,
                   [18, 18, 18], step_size_2d_list,
                   p_drop = 1, learnable_step=False, num_classes=num_classes
                   )
    return model


def resnet101_2DFs(step_size_2d_list, num_classes):
    model = ResNet_WD(BottleneckD,
                   [3, 4, 23, 3], step_size_2d_list,
                   p_drop = 1, learnable_step=False, num_classes=num_classes, width_per_group=128
                   )
    return model

'''
HeteroFL
'''
def resnet18_HeteroFL(num_classes, p): # resnet18hf
    model = ResNet_HeteroFL(block=BasicBlockH, 
                   num_blocks=[2, 2, 2, 2], p_drop = p,
                   num_classes=num_classes
                   )
    return model


def resnet34_HeteroFL(num_classes, p): # resnet34hf
    model = ResNet_HeteroFL(block=BasicBlockH,
                   num_blocks=[3, 4, 6, 3], p_drop = p,
                   num_classes=num_classes
                   )
    return model


def resnet56_HeteroFL(num_classes, p):
    model = ResNet_c_HeteroFL(block=BasicBlockH, 
                   num_blocks=[9, 9, 9], p_drop = p,
                   num_classes=num_classes
                   )
    return model


def resnet110_HeteroFL(num_classes, p):
    model = ResNet_c_HeteroFL(block=BasicBlockH, 
                   num_blocks=[18, 18, 18], p_drop = p,
                   num_classes=num_classes
                   )
    return model


def resnet101_2_HeteroFL(numclasses, p):
    model = ResNet_HeteroFL(block=BottleneckH,
                   num_blocks=[3, 4, 23, 3], p_drop = p,
                   num_classes=num_classes
                   )
    return model