import torch.nn as nn
import torch
import numpy as np
from transformer.spatial_transformer import transformer
from utils.network_layers import conv2_layer, locallyconn2_layer, dense_layer
import torch.nn.functional as F

# TODO: Check if all the layer norms are set to ReLU activation
# Develops a RNN
class PFCell(torch.nn.RNNCellBase):

    def __init__(self, global_maps, params, batch_size, num_particles):
        super(PFCell, self).__init__()
        self.params = params
        self.batch_size = batch_size
        self.num_particles = num_particles
        self.global_maps = global_maps
        self.state_shape = (batch_size, num_particles, 3)
        self.weight_shape = (batch_size, num_particles, )

    def forward(self, inputs, states):
        """
        This function implements the __call__ function of the RNNCellBase class

        """
        
        pass 




    # TODO: The transition model might need to be changed -- the split function might not be correct
    # TODO: finish this part 
    def transition_model(self, paticle_states, odometry):
        """
        Implements stochastic transition model of the particle filter.
        In this function the particles are updated with the new odometry measurement.
        The odometry measurement, that is the motion in x, y and theta -- can be used to
        get the change in the pose
        This value is further used to update the pose        
        """

        translational_std = self.params.translational_std[0] / self.params.map_pixel_in_meters # in pixels
        rotational_std = self.params.translational_std[1] # in radians

        part_x, part_y, part_theta = torch.split(paticle_states, 1, dim=2)
        odom_x, odom_y, odom_theta = torch.split(odometry, 1, dim=2)

        noise_th = torch.random_normal(part_theta.shape(), mean=0.0, stddev=1.0)*rotational_std
        part_theta += noise_th

        cos_th = torch.cos(part_theta)
        sin_th = torch.sin(part_theta)
        delta_x = cos_th * odom_x - sin_th * odom_y
        delta_y = sin_th * odom_x + cos_th * odom_y
        delta_th = odom_theta

        delta_x += torch.random_normal(delta_x.shape, mean=0.0, stddev=1.0)
        delta_y += torch.random_normal(delta_y.shape, mean=0.0, stddev=1.0)

        return torch.stack([part_x+delta_x, part_y+delta_y, part_theta+delta_th])

    # TODO: Must understand this function
    @staticmethod
    def transform_maps(global_maps, particle_states, local_map_size):
        """
        Implements global to local map transformation.
        """
        batch_size, num_particles = particle_states.shape[0], particle_states.shape[1]
        total_samples = batch_size * num_particles
        flat_states = particle_states.reshape(total_samples, 3)

        # defining helper variables
        input_shape = global_maps.shape
        global_height = input_shape[1].to(torch.float32)
        global_width = input_shape[2].to(torch.float32)
        height_inverse = 1.0 / global_height
        width_inverse = 1.0 / global_width

        zero = torch.zeros(total_samples, dtype=torch.float32)
        one = torch.ones(total_samples, dtype=torch.float32)

        # Normalizing the orientation and precomput the sin and cos
        window_scalar = 8.0
        theta = -flat_states[:, 2] - 0.5 * np.pi
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        
        # Constructing the affine transformation matrix 
        # 1. Translate the globa map s.t. the center is at the particle state
        translation_x = (flat_states[:, 0] * width_inverse * 2.0) - 1.0
        translation_y = (flat_states[:, 1] * height_inverse * 2.0) - 1.0

        transm1 = torch.stack([one, zero, translation_x, zero, one, translation_y, zero, zero, one], dim=1)
        transm1 = transm1.reshape(total_samples, 3, 3)

        # 2. Rotate the ma s.t the orientation matches that of the particles
        rotm = torch.stack([cos_theta, sin_theta, zero, -sin_theta, cos_theta, zero, zero, zero, one], dim=1)
        rotm = rotm.reshape(total_samples, 3, 3)

        # 3. Scale down the map
        scale_x = torch.full(total_samples, window_scalar * local_map_size[1] * width_inverse, dtype=torch.float32)
        scale_y = torch.full(total_samples, window_scalar * local_map_size[0] * height_inverse, dtype=torch.float32)

        scalem = torch.stack([scale_x, zero, zero, zero, scale_y, zero, zero, zero, one], dim=1)
        scalem = scalem.reshape(total_samples, 3, 3)

        # 4. translate the locaal map s.t the particles defines the bottom midpoint
        translate_y2 = torch.full((total_samples,), -1.0, dtype=torch.float32)
    
        transm2 = torch.stack([one, zero, zero, zero, one, translate_y2, zero, zero, one], dim=1)
        transm2 = transm2.reshape(total_samples, 3, 3)

        # Chain the translation matrices into one: translate + rotate + scale + translate
        transform_m = torch.matmul (torch.matmul(scalem, torch.matmul(rotm, transm1)), transm2)

        # reshape the format expected by the spatial transform network
        # TODO: Check if this is correct
        transform_m = transform_m[:, :2].reshape(batch_size, num_particles, 6)

        # do the image transformation using the spatial transformer network
        # iterate over partilcles to avoid tiling large global maps
        output_list = []
        for i in range(num_particles):
            output_list.append(transformer(global_maps, transform_m[:, i], local_map_size))
        
        local_maps = torch.stack(output_list, dim=1)
        local_maps = local_maps.reshape(batch_size, num_particles, local_map_size[0], local_map_size[1], global_maps.shape[-1])
        return local_maps
    

    def observation_model(self, global_maps, particle_states, observation):
        local_maps = self.transform_maps(global_maps, particle_states, (28, 28))
    


    @staticmethod
    def map_features(local_maps):
        # This assert might have to be changed -- the values are out of a list?
        print("Shape of the local is: ", local_maps.shape)

        assert local_maps.shape[1:3] == [28, 28], "Error in pfnet:map_features -- paramer has the wrong dimension"
        # TODO: Need to determine the number of input layers -- this is not given in flow
        # TODO: Must make sure the format of the output us (batchsize, height, width, channels)

        layer_i  =1
        # Replacing tensorflow functionality of padding ='same'
        x = local_maps
        F.pad(x, (0, 0, 2, 1))

        # TODO: Assuming that the number of input channels is 2, this must be changed later
        # NOTE: These convolutional layers are on the local maps. These are not sequential operation
        convs = [
            conv2_layer(
                2, 24, (3, 3), activation=None,
                use_bias=True, layer_i=layer_i)(x),
            conv2_layer(
                24, 16, (5, 5), activation=None, padding='same',
                use_bias=True, layer_i=layer_i)(x),
            conv2_layer(
                16, 8, (7, 7), activation=None, padding='same', 
                use_bias=True, layer_i=layer_i)(x),
            conv2_layer(
                8, 8, (7, 7), activation=None, padding='same',
                dilation_rate=(2, 2), use_bias=True, layer_i=layer_i)(x),
            conv2_layer(
                8, 8, (7, 7), activation=None, padding='same',
                dilation_rate=(3, 3), use_bias=True, layer_i=layer_i)(x), 
        ]

        # NOTE: The output of all the convolutions are concatenated
        x = torch.cat([conv(x) for conv in convs], dim=1)
        layer_norm = nn.LayerNorm(x.shape[1:])
        x = layer_norm(x)
        assert x.get_shape().as_list()[1:4] == [28, 28, 384], "Error in pfnet:map_features, possible problem in layer_norm or concatenation"
        
        # Max pool operation
        F.pad(x, (0, 0, 2, 1))
        x = F.max_pool2d(x, kernel_size=(3, 3), stride=(2, 2))
        print("The shape of x after max pool is: ", x.shape)

        # TODO: The input to this layer might be wrong
        convs = [
            conv2_layer(384, 4, (3, 3), activation= None, use_bias=True),
            conv2_layer(4, 4, (3, 3), activation= None, use_bias=True),
        ]

        x = torch.cat([conv(x) for conv in convs], dim=1)
        layer_norm = nn.LayerNorm(x.shape[1:])

        assert x.get_shape().as_list()[1:4] == [14, 14, 16], "Error in pfnet:map_features, possible problem with the second convolutional layer"
    
        return x

    @staticmethod
    def observation_features(observation):
        x = observation
        F.pad(x, (0, 0, 2, 1))
        convs= [
            conv2_layer(3, 128, (3, 3), activation=None, use_bias=True)(x),
            conv2_layer(128, 128, (5, 5), activation=None, use_bias=True)(x),
            conv2_layer(128, 64, (5, 5), activation=None, use_bias=True, dilation_rate=(2, 2))(x),
            conv2_layer(64, 64, (5, 5), activation=None, use_bias=True, dilation_rate=(4, 4))(x),
            ]
        x = torch.cat([conv(x) for conv in convs], dim=1)
        F.pad(x, (0, 0, 2, 1))
        x =  F.max_pool2d(x, kernel_size=(3, 3), stride=(2, 2))
        layer_norm = nn.LayerNorm(x.shape[1:])
        x = layer_norm(x)

        assert x.shape[1:4] == [28, 28, 384], "Error in the observation layer, after the first convolution" 

        F.pad(x, (0, 0, 2, 1))
        x = conv2_layer(384, 16, (3,3), activation=None, use_bias=True)(x)
        x =  F.max_pool2d(x, kernel_size=(3, 3), stride=(2, 2))
        layer_norm = nn.LayerNorm(x.shape[1:])
        x = layer_norm(x)

        assert x.shape[1:4] == [14, 14, 16], "Error in the observation layer, after the second convolution"

        return x

    @staticmethod
    def joint_matrix_features(joint_matrix):
        assert joint_matrix.shape[1:4] == [14, 14, 24], "Error in the joint matrix features, the shape of the joint matrix is wrong"
        x = joint_matrix
        F.pad(x, ([[0, 0], [1, 1], [1, 1], [0, 0]]))

        convs = [
            locallyconn2_layer(24, 8, (3,3), activation=None, use_bias=True)(x),
            locallyconn2_layer(8, 8, (5,5), activation=None, use_bias=True)(x),
        ]
        x = torch.cat([conv(x) for conv in convs], dim=1)
        # Padding "Valid" is no padding in Tensor flow -- so no padding is done here
        x = F.max_pool2d(x, kernel_size=(3, 3), stride=(2, 2))
        assert x.shape[1:4] == [5, 5, 16], "Error in the joint matrix features, after the first convolution"
        return x
    
    @staticmethod
    def joint_vector_features(joint_vector):
        x = joint_vector
        x = dense_layer(1, activation=None, use_bias=True)(x)
        return x


# TODO: The following line must be changed based on the optimizer chosen
# TODO: The weight_decay is the L2 regularization parameter
#optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
        
