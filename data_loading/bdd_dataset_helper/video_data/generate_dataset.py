import os
import random
from shutil import copy
from argparse import ArgumentParser

TRAIN = 0.7
VALID = 0.3


def split_data(args):
    info_files = os.listdir(args.info_dir)
    train_len = int(TRAIN * len(info_files))
    valid_len = int(VALID * len(info_files))

    train_inds = sorted(random.sample(range(len(info_files)), train_len))
    for ind in reversed(train_inds):
        vid_name = info_files[ind].split('.')[0] + ".mov"
        info_file = os.path.join(args.info_dir, info_files[ind])
        vid_file = os.path.join(args.video_dir, vid_name)

        copy(info_file, 'train/info.csv')
        copy(vid_file, 'train/data.mov')

        del info_files[ind]

    valid_inds = sorted(random.sample(range(len(info_files)), valid_len))
    for ind in reversed(valid_inds):
        vid_name = info_files[ind].split('.')[0] + ".mov"
        info_file = os.path.join(args.info_dir, info_files[ind])
        vid_file = os.path.join(args.video_dir, vid_name)

        copy(info_file, 'validation/info.csv')
        copy(vid_file, 'validation/data.mov')

        del info_files[ind]

    for file in info_files:
        vid_name = file.split('.')[0] + ".mov"
        info_file = os.path.join(args.info_dir, file)
        vid_file = os.path.join(args.video_dir, vid_name)

        copy(info_file, 'test/info.csv')
        copy(vid_file, 'test/data.mov')


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--info_dir',
        type=str,
        default='/home/alexm/Desktop/SteeringNetwork/data_loading/bdd_dataset_helper/video_data/original',
        help='path to the directory containing info about the videos')
    arg_parser.add_argument('--video_dir',
        type=str,
        default='/home/alexm/Desktop/hal_data/samples-1k/good_videos',
        help='path to the directory containing the videos')

    args = arg_parser.parse_args()
    split_data(args)
