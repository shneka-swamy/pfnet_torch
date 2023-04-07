# Preprocesses the 2D input and the 3D input data ??
# TODO: The training file for the neural network to learn is not present 
# Must get this file downloaded later -- for now running the training with
# the test dataset


# NOTE: bptt is used (backpropagation through time) to train the neural network 
# This method is used mostly to train RNNs. 

import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import cv2

def decode_image( image, resize=None):
        img_str = cv2.imdecode(image, -1)
        if resize is not None:
            img_str = cv2.resize(img_str, resize)
        return img_str

def raw_images_to_array(images):
    image_list = []
    for image_str in images:
        image = decode_image(image_str, (56, 56))
        image = scale_observation(np.atleast_3d(image.astype(np.float32)))
        image_list.append(image)

    return np.stack(image_list, axis=0)

def scale_observation(x):
    if x.ndim == 2 or x.shape[2] == 1:  # depth
        return x * (2.0 / 100.0) - 1.0
    else:  # rgb
        return x * (2.0/255.0) - 1.0

class House3DTrajData():
    def __init__(self, files, params, init_particles_cov, seed=None):
        self.files = files
        self.mapmode = params.mapmode
        self.obsmode = params.obsmode
        self.trajlen = params.trajlen
        self.num_particles = params.num_particles
        self.init_particles_distr = params.init_particles_distr
        self.init_particles_cov = init_particles_cov
        self.seed = seed
        
    def process_wall_map(self, wallmap_feature):
        floormap = np.atleast_3d(decode_image(wallmap_feature))
        # transpose and invert
        floormap = 255 - np.transpose(floormap, axes=[1, 0, 2])
        return floormap

    def process_door_map(self, doormap_feature):
        return self.process_wall_map(doormap_feature)

    def process_roomtype_map(self, roomtypemap_feature):
        binary_map = np.fromstring(roomtypemap_feature, np.uint8)
        binary_map = cv2.imdecode(binary_map, 2) # Create a 16 bit image
        assert binary_map.dtype == np.uint16 and binary_map.ndim == 2

        # Binary encoding from bit 0 ... 9
        room_map = np.zeros((binary_map.shape[0], binary_map.shape[1], 9), dtype=np.unit8)
        for i in range(9):
            # TODO: What is the requirement of this statement ?
            room_map[:, :, i] = np.array((np.bitwise_and(binary_map, (1 << i)) > 0), dtype=np.uint8)
        room_map *= 255

        # transpose and invert
        room_map = np.transpose(room_map, axes=[1, 0, 2])
        return room_map

    def process_roomid_map(self, roomid_features):
        return np.atleast_3d(decode_image(roomid_features))

    # NOTE: This is a generator method that gives one training/test dataset at a time
    # TODO: Use the generator method to get only one data point at a time
    # NOTE: This will help to not load the entire dataset in an array
    def get_data(self):
        """
        The dataset that is maintained has the following dimension
        
        true states: (trajlen, 3)

        globalmap: (n, m, ch)

        initial particles: (num_particles, 3)

        observations: (trajlen, 56, 56, ch)
 
        odometries: (trajlen, 3) -- relative motion in the robot coordinate frame        
        """
        
        



def get_dataflow(files, params, is_training):
    batchsize = params.batchsize
    bptt_steps = params.bptt_steps
    mapmode = params.mapmode
    num_particles = params.num_particles
    obsmode = params.obsmode

    # NOTE: Initial covariance matrix is a multivariate gaussian distribution 
    # The center is perturbed by the initial particles std
    particle_std = params.init_particles_std.copy()
    # Converts from 'm' to pixel coordiantes
    particle_std[0] /= params.map_pixel_in_meters
    particle_std2 = np.square(particle_std)
    # Form the covariance matrix -- x, y, angle (Converting from std to covariance)
    init_particles_cov = np.diag(particle_std2[(0, 0, 1),])

     # TODO: This value is set 3 times -- is this required ??
        # Setting the value for seed
    if params.seed is not None and params.seed > 0:
        seed = params.seed
    else:
        if not is_training:
            seed = params.validseed
        else:
            seed = None

    # TODO: The df object is used to traverse through the dataset
    # TODO: The files here is the path to the .npy files
    df = House3DTrajData(files, params, init_particles_cov, seed)

    # TODO: Line number 447 - 466 in the method is not implemented

    obs_ch = {'rgb': 3, 'depth': 1, 'rgb_depth': 4}
    map_ch = {'wall': 1, 'wall_door': 2, 'wall_roomtype': 10, 'wall_door_roomtype': 11}
    types = [torch.float32, torch.float32, torch.float32, torch.float32, torch.float32, torch.bool]
    # TODO: What is the requirement of using the sizes variable
    sizes = [(batchsize, bptt_steps, 3), 
             (batchsize, None, None, map_ch[mapmode]),
             (batchsize, num_particles, 3),
             (batchsize, bptt_steps, 56, 56, obs_ch[obsmode]),
             (batchsize, bptt_steps, 3),
             (),]
    
    # TODO: Must implement nextdata
    nextdata = None

    return nextdata, num_particles

