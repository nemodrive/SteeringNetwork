import os
import random
from shutil import copy
from argparse import ArgumentParser

TRAIN = 1
VALID = 1 - TRAIN


def split_data(args):
    info_files = os.listdir(args.info_dir)
    train_len = int(TRAIN * len(info_files))
    valid_len = int(VALID * len(info_files))

    video_dict = {}

    with open(args.video_index, 'r') as f:
        data = f.readlines()

    for line in data:
        name = line.split('/')[-1].split('.')[0] + '.csv'
        #print(name)
        video_dict[name] = line[:-1]

    for dir_name in ['train', 'test', 'validation']:
        for subdirs in ['info', 'data']:
            try:
                os.makedirs(os.path.join(args.destination, dir_name, subdirs))
            except FileExistsError:
                # directory already exists
                pass

    train_inds = sorted(random.sample(range(len(info_files)), train_len))
    for ind in reversed(train_inds):
        vid_name = info_files[ind].split('.')[0] + ".mov"
        print(vid_name)
        info_file = os.path.join(args.info_dir, info_files[ind])
        vid_file = video_dict[info_files[ind]]

        copy(info_file, os.path.join(args.destination, 'train/info/', info_files[ind]))
        copy(vid_file, os.path.join(args.destination, 'train/data/', vid_name))

        #del info_files[ind]

    valid_inds = sorted(random.sample(range(len(info_files)), valid_len))
    for ind in reversed(valid_inds):
        vid_name = info_files[ind].split('.')[0] + ".mov"
        info_file = os.path.join(args.info_dir, info_files[ind])
        vid_file = video_dict[info_files[ind]]

        copy(info_file, os.path.join(args.destination, 'validation/info/', info_files[ind]))
        copy(vid_file, os.path.join(args.destination, 'validation/data/', vid_name))
        #del info_files[ind]
    '''
    for file in info_files:
        vid_name = file.split('.')[0] + ".mov"
        info_file = os.path.join(args.info_dir, file)
        vid_file = video_dict[info_files[ind]]

        copy(info_file, os.path.join(args.destination, 'test_demo/info/', info_files[ind]))
        copy(vid_file, os.path.join(args.destination, 'test_demo/data/', vid_name))
    '''
if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--info_dir',
        type=str,
        default='/home/alexm/Desktop/SteeringNetwork/data_loading/bdd_dataset_helper/video_data/original',
        help='path to the directory containing info about the videos')
    arg_parser.add_argument('--video_index',
        type=str,
        default='/home/alexm/Desktop/hal_data/samples-1k/good_videos',
        help='path to the directory containing the videos')
    arg_parser.add_argument('--destination',
        type=str,
        default='/home/nemodrive3/workspace/andreim/upb_data/dataset_demo',
        help='path to the destination directory')

    args = arg_parser.parse_args()
    split_data(args)
