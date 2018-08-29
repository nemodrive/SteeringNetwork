import os
import cv2
import sys
import h5py
import imageio
import numpy as np
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser


def generate_data(dir, args):
    video_file = os.path.join(args.src_dir, dir, dir + '.mov')
    data_file = os.path.join(args.src_dir, dir, dir + '_sampled.csv')

    # Get number of frames to be extracted
    df = pd.read_csv(data_file)
    new_nframes = len(df['steer'])

    # Get video data
    vid = imageio.get_reader(video_file, 'ffmpeg')
    metadata = vid.get_meta_data()
    nframes = metadata['nframes']
    height, width = metadata['source_size']

    # Resized width and height
    w = int(width / args.resize_factor)
    h = int(height / args.resize_factor)

    dataset_shape = (new_nframes, h, w, 3)
    extracted_frames = np.zeros(dataset_shape, dtype=np.int32)
    frame_indices = list(map(lambda x: x[0], np.array_split(range(nframes), new_nframes)))

    # Extract video frames
    curr_index = 0
    count = 0
    for data in vid.iter_data():
        # Stop if we have extracted all the frames that we need
        if curr_index == new_nframes:
            break
        if count == frame_indices[curr_index]:
            # Resize image before saving it
            extracted_frames[curr_index, :, :, :] = cv2.resize(data, (w, h))
            curr_index += 1
        count += 1

    # Group video data from csv by frame
    targets_data = []
    for i in range(len(df['steer'])):
        single_target = []
        single_target += [df['acceleration_x'][i], df['acceleration_y'][i], df['acceleration_z'][i]]
        single_target += [df['gps_lat'][i], df['gps_long'][i]]
        single_target += [df['gyro_x'][i], df['gyro_y'][i], df['gyro_z'][i]]
        single_target += [df['speed_x'][i], df['speed_y'][i]]
        single_target.append(df['steer'][i] / args.max_steer)
        single_target.append(df['turn'][i])
        targets_data.append(single_target)

    # Create h5 file
    out_file = h5py.File(os.path.join(args.dst_dir, dir + '.h5'), 'w')
    rgb = out_file.create_dataset('rgb', dataset_shape, data=extracted_frames, dtype='i1')
    targets = out_file.create_dataset('targets', (new_nframes, 12), data=targets_data, dtype='f')


if __name__ == '__main__':
    arg_parser = ArgumentParser()

    arg_parser.add_argument(
        '--src_dir',
        type=str,
        default='data_and_video',
        help='directory where the data and videos are stored')
    arg_parser.add_argument(
        '--dst_dir',
        type=str,
        default='bddv_dataset',
        help='directory where the dataset will be stored')
    arg_parser.add_argument(
        '--max_steer',
        type=float,
        default=35,
        help='the maximum value of the steering angle')
    arg_parser.add_argument(
        '--resize_factor',
        type=float,
        default=4,
        help='by how many times to reduce the image width and height')

    args = arg_parser.parse_args()

    if not os.path.exists(args.src_dir):
        print('Inexistent source directory')
        exit(1)
    if not os.path.exists(args.dst_dir):
        os.makedirs(args.dst_dir)

    # Find the directories containing data and videos
    dirs = os.listdir(args.src_dir)

    for i in tqdm(range(len(dirs)), file=sys.stdout):
        generate_data(dirs[i], args)
