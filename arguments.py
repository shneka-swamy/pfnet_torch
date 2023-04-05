# Use command parser to get input from the user
# NOTE: This function is directly taken from the original code implemented in tensorflow
import argparse

def command_parser():
    parser = argparse.ArgumentParser(description="Particle Filter Network -- using Pytorch")

    # TODO: How to use .config file to train the model ?? 
    parser.add_argument('--config', required=True, help='Config file. use ./config/train.conf for training')

    # Get the training and the validation files in the .tfrecord format
    # nargs='*' means that the user can input multiple files
    parser.add_argument('--trainfiles', nargs='*', help='Data file(s) for training (tfrecord).')
    parser.add_argument('--testfiles', nargs='*', help='Data file(s) for validation or evaluation (tfrecord).')

    # input configuration
    parser.add_argument('--obsmode', type=str, default='rgb', 
                        help='Observation input type. Possible values: rgb / depth / rgb-depth / vrf.')
    parser.add_argument('--mapmode', type=str, default='wall', 
                        help='Map input type with different (semantic) channels. ' + 'Possible values: wall / wall-door / wall-roomtype / wall-door-roomtype')
    parser.add_argument('--map_pixel_in_meters', type=float, default=0.02, 
                        help='The width (and height) of a pixel of the map in meters. Defaults to 0.02 for House3D data.')

    # Initialize the particles distribution in how many rooms
    parser.add_argument('--init_particles_distr', type=str, default='tracking', 
                        help='Distribution of initial particles. Possible values: tracking / one-room / two-rooms / all-rooms')
    # tracking setting, 30cm, 30deg
    parser.add_argument('--init_particles_std', nargs='*', default=["0.3", "0.523599"], 
                        help='Standard deviations for generated initial particles. Only applies to the tracking setting.' + 
                        'Expects two float values: translation std (meters), rotation std (radians)')
    parser.add_argument('--trajlen', type=int, default=24, 
                        help='Length of the trajectory to be predicted.Assumes lower or equal to the trajectory length in the input data.')
    

    # PF-net configuration
    parser.add_argument('--num_particles', type=int, default=100,
                        help='Number of particles to use in the particle filter.') 
    parser.add_argument('--resample', type=str, default='false',
                        help='Whether to resample particles in the PFNet. Possible values: true / false')
    parser.add_argument('--alpha_resample_ratio', type=float, default=1.0,
          help='Trade-off parameter for soft-resampling in PF-net. Only effective if resample == true. '
               'Assumes values 0.0 < alpha <= 1.0. Alpha equal to 1.0 corresponds to hard-resampling.')
    parser.add_argument('--transition_std', nargs='*', default=["0.0", "0.0"],
                help='Standard deviations for transition model. Expects two float values: ' +
                     'translation std (meters), rotatation std (radians). Defaults to zeros.')

    # training configuration
    parser.add_argument('--batchsize', type=int, default=24, help='Minibatch size for training. Must be 1 for evaluation.')
    parser.add_argument('--bptt_steps', type=int, default=4,
          help='Number of backpropagation steps for training with backpropagation through time (BPTT). '
               'Assumed to be an integer divisor of the trajectory length (--trajlen).')
    parser.add_argument('--learningrate', type=float, default=0.0025, help='Initial learning rate for training.')
    parser.add_argument('--l2scale', type=float, default=4e-6, help='Scaling term for the L2 regularization loss.')
    parser.add_argument('--epochs', metavar='epochs', type=int, default=1, help='Number of epochs for training.')
    parser.add_argument('--decaystep', type=int, default=4, help='Decay the learning rate after every N epochs.')
    parser.add_argument('--decayrate', type=float, help='Rate of decaying the learning rate.')

    parser.add_argument('--load', type=str, default="", help='Load a previously trained model from a checkpoint file.')
    parser.add_argument('--logpath', type=str, default='',
          help='Specify path for logs. Makes a new directory under ./log/ if empty (default).')
    parser.add_argument('--seed', type=int, help='Fix the random seed of numpy and torch if set to larger than zero.')
    parser.add_argument('--validseed', type=int,
          help='Fix the random seed for validation if set to larger than zero. ' +
               'Useful to evaluate with a fixed set of initial particles, which reduces the validation error variance.')
    parser.add_argument('--gpu', type=int, default=0, help='Select a gpu on a multi-gpu machine. Defaults to zero.')

    return parser.parse_args()

