# This code is used to visualize the dataset
# In this code one data from a set of data is shown for visualization
# This code is independent of all the other code and is required for the training to happen

import torch
from tfrecord.torch.dataset import TFRecordDataset
import numpy as np
import matplotlib.pyplot as plt
import argparse
from preprocess import decode_image, raw_images_to_array

# Path to store the dataset in the pytorch readable format
def command_parser():
    parser = argparse.ArgumentParser(description='Visualize the dataset and convert it to pytorch readable format')
    parser.add_argument('--path_to_dataset', type=str, default='/home/pecs/DeepRob/test.tfrecords')
    parser.add_argument('--path_to_save', type=str, default='/home/pecs/DeepRob/valid.npy')
    parser.add_argument('--v', action='store_true', default=False, help='verbose')
    parser.add_argument('--vv', action='store_true', default=False, help='verbose verbose')
    return parser.parse_args()


def display_data(args):
    # NOTE: The batch_size value can be changed
    tfrecord_path = args.path_to_dataset
    batch_size = 1
    index_path = None
    description = {'states': 'byte', 'map_roomid': 'byte', 'map_wall': 'byte', 'houseID': 'byte', 'roomID': 'byte', 'map_roomtype': 'byte', 
                    'depth': 'byte', 'rgb': 'byte', 'map_door': 'byte', 'odometry': 'byte'}
    dataset = TFRecordDataset(tfrecord_path, index_path, description)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    data = next(iter(loader))
    
    map_wall = decode_image(data['map_wall'].numpy())
    map_door = decode_image(data['map_door'].numpy())
    
    # There are 8 possible room types
    map_roomtype = decode_image(data['map_roomtype'].numpy())
    map_roomid = decode_image(data['map_roomid'].numpy())

    # true states
    # (x, y, theta). x,y: pixel coordinates; theta: radians
    true_states = data['states'].numpy()
    true_states = np.frombuffer(true_states, np.float32).reshape((-1, 3))

    # odometry
    # each entry is true_states[i+1]-true_states[i].
    # last row is always [0,0,0]
    odometry = data['odometry'].numpy()
    odometry = np.frombuffer(odometry, np.float32).reshape((-1, 3))

    print("The first three true states are:", true_states[:3])
    print("The first three odometry values are:", odometry[:3])

    rgb = raw_images_to_array(data['rgb'].numpy())

    depth = raw_images_to_array(data['depth'].numpy())
    
    if args.vv:
        print("Size of the keys that are considered")
        print("Changing nothing to the data: ")
        print("Size states: ", true_states.shape)
        print("Size odometry", odometry.shape)
        print("After applying the decode_image function")
        print("Size map_wall", map_wall.shape)
        print("Size map_door", map_door.shape)
        print("Size map_roomtype", map_roomtype.shape)
        print("Size map_roomid", map_roomid.shape)
        print("After applying raw_images_to_array function")
        print("Size rgb", rgb.shape)
        print("Size depth", depth.shape)

    if args.v:
        print("Plot the map wall and the first observation")
        plt.figure()
        plt.imshow(rgb[0])
        plt.show()

        plt.figure()
        plt.imshow(map_door.transpose())
        plt.show()

def main():
    args = command_parser()
    display_data(args)

if __name__ == '__main__':
    main()
