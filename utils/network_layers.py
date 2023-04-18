import numpy as np
import torch.nn as nn
from LocallyConnected2d import LocallyConnected2d

def locallyconn2_layer(in_channels, out_channels, kernel_size, stride=(1, 1), activation=None, bias=False):
    output_size = (in_channels[0] * (kernel_size[0] - 1) // stride[0] + 1)            
    output_size = _pair(output_size)
    conv =  LocallyConnected2d(in_channels, out_channels, output_size, kernel_size, stride, bias)
    act = activationLayer(activation)
    return nn.Sequential(conv, act)

# NOTE: Dense layer in tensor flow is same as the linear layer in pytorch
# https://stackoverflow.com/questions/66626700/difference-between-tensorflows-tf-keras-layers-dense-and-pytorchs-torch-nn-lin
def dense_layer(units, activation=None, use_bias=False):
    linear = nn.Linear(units, use_bias=use_bias)
    act = activationLayer(activation)
    return nn.Sequential(linear, act)
                
# TODO: Should name and name2 be added and layer_i value be added
def conv2_layer(in_channels, out_channels, kernel_size, activation=None,
            strides=(1,1), dilation_rate=(1,1), data_format='channels_last',
            use_bias=False):
    conv = nn.Conv2d(in_channels, out_channels, kernel_size,
            strides=strides, dilation=dilation_rate, use_bias=use_bias)
    act = activationLayer(activation)
    # NOTE: This is same as the varianve_scaling_initializer in tensorflow
    # Ref: https://stats.stackexchange.com/questions/319323/whats-the-difference-between-variance-scaling-initializer-and-xavier-initialize
    nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
    return nn.Sequential(conv, act)

                
def activationLayer(activation):
    if isinstance(activation, str):
        if activation == 'relu':
            return nn.ReLU(),
        elif activation == 'tanh':
            return nn.Tanh(),
        else:
            assert False, 'Unknown activation function'   