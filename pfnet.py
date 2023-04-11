import torch.nn as nn
import numpy as np
from transformer.spatial_transformer import transformer

# Develops a RNN
class PFCell(nn.Module):
    def __init__():
        pass

    @staticmethod
    def map_features(local_maps):
        # This assert might have to be changed
        assert local_maps.shape[1:3] == 28, 28
        data_format - 'channels_last'

        
