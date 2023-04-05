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

# The following 3 functions are used to change the images to handle format

def decode_image(image, resize=None):
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

# The files for both the annotations and the images are direclty given
# TODO: The way files are read in this part might have to be changed
# TODO: The way that the image and the label are returned might have to be changed
class House3DTrajData(Dataset):
    def __init__(self, annotation_file, image_file, transform=None, target_transform=None):
        self.annotations = pd.read_csv(annotation_file)
        self.image_dir = pd.read_csv(image_file)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(self.annotations)
        return image, label

# NOTE: This class controls the flow of data -- the House3DTrajData class is used for that purpose
class House3DTrajProcess():
    def __init__(self, files, params, init_particles_cov):
        self.files = files
        self.mapmode = params.mapmode
        self.obsmode = params.obsmode
        self.trajlen = params.trajlen
        self.num_particles = params.num_particles
        self.init_particles_distr = params.init_particles_distr
        self.init_particles_cov = init_particles_cov

        # count the number of enteries in the dataset
        # TODO: Check this part after converting the dataset from tensorflow format to pytorch format    
        # TODO: Finish the visualize the dataset part before this 
        self.count = 0
        for file in self.files:
            self.count += House3DTrajData(file, params.mapdir).__len__()

    def size(self):
        return self.count

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
    df = House3DTrajProcess(files, params, init_particles_cov)

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

