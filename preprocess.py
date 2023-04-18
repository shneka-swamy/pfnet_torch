# Preprocesses the 2D input and the 3D input data ??
# TODO: The training file for the neural network to learn is not present 
# Must get this file downloaded later -- for now running the training with
# the test dataset


# NOTE: bptt is used (backpropagation through time) to train the neural network 
# This method is used mostly to train RNNs. 


from torch.utils.data import Dataset, DataLoader
from tfrecord.torch.dataset import TFRecordDataset
import numpy as np
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


def bounding_box(img):
    """
    Bounding box of non-zeros in an array (inclusive). Used with 2D maps
    :param img: numpy array
    :return: inclusive bounding box indices: top_row, bottom_row, leftmost_column, rightmost_column
    """
    # helper function to
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax
 
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

    @staticmethod
    def get_sample_seed(seed, data_i):
        """
        Defines a random seed for each datapoint in a deterministic manner.
        :param seed: int or None, defining a random seed
        :param data_i: int, the index of the current data point
        :return: None if seed is None, otherwise an int, a fixed function of both seed and data_i inputs.
        """
        return (None if (seed is None or seed == 0) else ((data_i + 1) * 113 + seed))

    # TODO: How are random particles generated and what is the use?
    @staticmethod
    def random_particles(state, distr, particles_cov, num_particles, roomidmap, seed=None):
        """
        Generate a random set of particles
        :param state: true state, numpy array of x,y,theta coordinates
        :param distr: string, type of distribution. Possible values: tracking / one-room.
        For 'tracking' the distribution is a Gaussian centered near the true state.
        For 'one-room' the distribution is uniform over states in the room defined by the true state.
        :param particles_cov: numpy array of shape (3,3), defines the covariance matrix if distr == 'tracking'
        :param num_particles: number of particles
        :param roomidmap: numpy array, map of room ids. Values define a unique room id for each pixel of the map.
        :param seed: int or None. If not None, the random seed will be fixed for generating the particle.
        The random state is restored to its original value.
        :return: numpy array of particles (num_particles, 3)
        """
        assert distr in ["tracking", "one-room"]  #TODO add support for two-room and all-room

        particles = np.zeros((num_particles, 3), np.float32)

        if distr == "tracking":
            # fix seed
            if seed is not None:
                random_state = np.random.get_state()
                np.random.seed(seed)

            # sample offset from the Gaussian
            center = np.random.multivariate_normal(mean=state, cov=particles_cov)

            # restore random seed
            if seed is not None:
                np.random.set_state(random_state)

            # sample particles from the Gaussian, centered around the offset
            particles = np.random.multivariate_normal(mean=center, cov=particles_cov, size=num_particles)

        elif distr == "one-room":
            # mask the room the initial state is in
            masked_map = (roomidmap == roomidmap[int(np.rint(state[0])), int(np.rint(state[1]))])

            # get bounding box for more efficient sampling
            rmin, rmax, cmin, cmax = bounding_box(masked_map)

            # rejection sampling inside bounding box
            sample_i = 0
            while sample_i < num_particles:
                particle = np.random.uniform(low=(rmin, cmin, 0.0), high=(rmax, cmax, 2.0*np.pi), size=(3, ),)
                # reject if mask is zero
                if not masked_map[int(np.rint(particle[0])), int(np.rint(particle[1]))]:
                    continue
                particles[sample_i] = particle
                sample_i += 1
        else:
            raise ValueError

        return particles


    # This generator function returns one data from the dataset at a time
    def get_data(self):
        """
        The dataset that is maintained has the following dimension
        
        true states: (trajlen, 3)

        globalmap: (n, m, ch)

        initial particles: (num_particles, 3)

        observations: (trajlen, 56, 56, ch)
 
        odometries: (trajlen, 3) -- relative motion in the robot coordinate frame        
        """

        tfrecord_path = self.files
        batch_size = 1
        index_path = None
        description = {'states': 'byte', 'map_roomid': 'byte', 'map_wall': 'byte', 'houseID': 'byte', 'roomID': 'byte', 'map_roomtype': 'byte', 
                    'depth': 'byte', 'rgb': 'byte', 'map_door': 'byte', 'odometry': 'byte'}
        dataset = TFRecordDataset(tfrecord_path, index_path, description)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        for features in loader:
            # process maps
            map_wall = self.process_wall_map(features['map_wall'].bytes_list.value[0])
            global_map_list = [map_wall]
            if 'door' in self.mapmode:
                map_door = self.process_door_map(features['map_door'].bytes_list.value[0])
                global_map_list.append(map_door)
            if 'roomtype' in self.mapmode:
                map_roomtype = self.process_roomtype_map(features['map_roomtype'].bytes_list.value[0])
                global_map_list.append(map_roomtype)
            if self.init_particles_distr == 'tracking':
                map_roomid = None
            else:
                map_roomid = self.process_roomid_map(features['map_roomid'].bytes_list.value[0])

            # input global map is a concatentation of semantic channels
            global_map = np.concatenate(global_map_list, axis=-1)

            # rescale to 0..2 range. this way zero padding will produce the equivalent of obstacles
            global_map = global_map.astype(np.float32) * (2.0 / 255.0)

            # process true states
            true_states = features['states'].bytes_list.value[0]
            true_states = np.frombuffer(true_states, np.float32).reshape((-1, 3))

            # trajectory may be longer than what we use for training
            data_trajlen = true_states.shape[0]
            assert data_trajlen >= self.trajlen
            true_states = true_states[:self.trajlen]

            # process odometry
            odometry = features['odometry'].bytes_list.value[0]
            odometry = np.frombuffer(odometry, np.float32).reshape((-1, 3))

            # process observations
            assert self.obsmode in ['rgb', 'depth', 'rgb-depth']  #TODO support for lidar
            if 'rgb' in self.obsmode:
                rgb = raw_images_to_array(list(features['rgb'].bytes_list.value)[:self.trajlen])
                observation = rgb
            if 'depth' in self.obsmode:
                depth = raw_images_to_array(list(features['depth'].bytes_list.value)[:self.trajlen])
                observation = depth
            if self.obsmode == 'rgb-depth':
                observation = np.concatenate((rgb, depth), axis=-1)

            # generate particle states
            init_particles = self.random_particles(true_states[0], self.init_particles_distr,
                                                    self.init_particles_cov, self.num_particles,
                                                    roomidmap=map_roomid,
                                                    seed=self.get_sample_seed(self.seed, data_i), )

            yield (true_states, global_map, init_particles, observation, odometry)


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
    # TODO: The files here is the path
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

