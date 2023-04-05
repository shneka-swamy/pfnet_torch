# This code is used to visualize the dataset
# In this code this part is used to convert the .tfrecord file to a file
# that can be used by Torch. 

# NOTE: The dataset is stored in the same format as the reading result from the .tfrecord is
# Thus, the 'decode_image' and the 'raw_images_to_array' functions must be used.

import torch
from tfrecord.torch.dataset import TFRecordDataset
import numpy as np
import matplotlib.pyplot as plt
import argparse
from preprocess import raw_images_to_array, decode_image

# Path to store the dataset in the pytorch readable format
def command_parser():
    parser = argparse.ArgumentParser(description='Visualize the dataset and convert it to pytorch readable format')
    parser.add_argument('--path_to_dataset', type=str, default='/home/pecs/DeepRob/test.tfrecords')
    parser.add_argument('--path_to_save', type=str, default='/home/pecs/DeepRob/data.npy')
    parser.add_argument('--v', action='store_true', default=False, help='check if the dataset is stored in .npy file properly')

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

    print("Plot the map wall and the first observation")
    plt.figure()
    plt.imshow(rgb[0])
    plt.show()

    plt.figure()
    plt.imshow(map_door.transpose())
    plt.show()

    # Storing data in the .npy format
    np.save(args.path_to_save, data)

# Check if the stored data can be loaded properly
def check_data(args):
    data = np.load(args.path_to_save, allow_pickle=True)
    print("The loaded data is:")
    # NOTE : The way to access the data in the .npy file
    print(data.item()['states'])

def main():
    args = command_parser()
    display_data(args)
    if args.v:
        check_data(args)

if __name__ == '__main__':
    main()
