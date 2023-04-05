# This is the first part in which the code starts -- for the evaluation part
import numpy as np

import pfnet
from arguments import command_parser
from preprocess import get_dataflow

def run_evaluation(params):
    """Run evaluation with the parsed arguments"""

    # overwrite for evaluation
    params.batchsize = 1
    params.bptt_steps = params.trajlen
    



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

    run_evaluation(params)


