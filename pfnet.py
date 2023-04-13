import torch.nn as nn
import numpy as np
from transformer.spatial_transformer import transformer
from utils.network_layers import conv2_layer


# Develops a RNN
class PFCell(nn.Module):
    def __init__():
        pass

    @staticmethod
    def map_features(local_maps):
        # This assert might have to be changed -- the values are out of a list?
        assert local_maps.shape[1:3] == 28, 28
        # TODO: Need to determine the number of input layers -- this is not given in flow
        # TODO: Must make sure the format of the output us (batchsize, height, width, channels)

        layer_i  =1
    
        # TODO: Can this be changed to sequential operation, How to add (local_maps to it)
        convs = [
            conv2_layer(
                24, (3, 3), activation=None, padding='same',
                use_bias=True, layer_i=layer_i),
            conv2_layer(
                16, (5, 5), activation=None, padding='same',
                use_bias=True, layer_i=layer_i),
            conv2_layer(
                8, (7, 7), activation=None, padding='same', 
                use_bias=True, layer_i=layer_i),
            conv2_layer(
                8, (7, 7), activation=None, padding='same',
                dilation_rate=(2, 2), use_bias=True, layer_i=layer_i),
            conv2_layer(
                8, (7, 7), activation=None, padding='same'
                dilation_rate=(3, 3), use_bias=True, layer_i=layer_i),
        ]


        
