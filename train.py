# This is the starting of the code for training the neural network

import os
import numpy as np
import torch
from datetime import datetime

import pfnet
from arguments import command_parser
from preprocess import get_dataflow

def run_training(params):
    # TODO: Is the following statement needed in pytorch implementation?
    # with tf.Graph().as_default():

    # TODO: Why is this random seed generation different from the numpy random seed?
    if params.seed is not None:
        torch.random_seed(params.seed)
    
    # training data and network
    train_data, num_train_data = get_dataflow(params.trainfiles, params, is_training=True)
    train_brain = pfnet.PFNet(inputs=train_data[1:], labels=train_data[0], params=params, is_training=True)
    
    



if __name__ == '__main__':
    params = command_parser()

    # fix numpy seed if it is needed
    if params.seed is not None and params.seed >= 0:
        np.random.seed(params.seed)
    
    # convert multi-input fields to numpy arrays
    params.transition_std = np.array(params.transition_std, np.float32)
    params.init_particles_std = np.array(params.init_particles_std, np.float32)

    # convert boolean fields
    if params.resample not in ['false', 'true']:
        print ("The value of resample must be either 'false' or 'true'")
        raise ValueError
    params.resample = (params.resample == 'true')

    # Make logpath to store the errors during the training  
    params.logpath = os.path.join(params.logpath, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.mkdir(params.logpath)


    run_training(params)