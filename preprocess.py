# Preprocesses the 2D input and the 3D input data ??
# TODO: The training file for the neural network to learn is not present 
# Must get this file downloaded later -- for now running the training with
# the test dataset


# NOTE: bptt is used (backpropagation through time) to train the neural network 
# This method is used mostly to train RNNs. 

import numpy as np

# NOTE: This class controls the flow of data
class House3DTrajData():
    def __init__(self, files, params, init_particles_cov):
        self.files = files
        self.mapmode = params.mapmode
        self.obsmode = params.obsmode
        self.trajlen = params.trajlen
        self.num_particles = params.num_particles
        self.init_particles_distr = params.init_particles_distr
        self.init_particles_cov = init_particles_cov

        # count the number of enteries in the dataset
        # TODO: Work on this part after converting the dataset from tensorflow format to pytorch format    
        


def get_dataflow(files, params, is_training):
    batchsize = params.batchsize
    bptt_steps = params.bptt_steps

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



    # TODO: This section of the code needs to be continued 
    df = House3DTrajData(files, params, init_particles_cov)