import os
import random
from shutil import copy
from argparse import ArgumentParser

TRAIN = 0.8
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
        print(name)
        video_dict[name] = line[:-1]

    for dir_name in ['train', 'test', 'validation']:
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
            os.mkdir(dir_name + '/data')
            os.mkdir(dir_name + '/info')
        else:
            if not os.path.exists(dir_name + '/data'):
                os.mkdir(dir_name + '/data')
            if not os.path.exists(dir_name + '/info'):
                os.mkdir(dir_name + '/info')

    train_inds = sorted(random.sample(range(len(info_files)), train_len))
    for ind in reversed(train_inds):
        vid_name = info_files[ind].split('.')[0] + ".mov"
        info_file = os.path.join(args.info_dir, info_files[ind])
        vid_file = video_dict[info_files[ind]]

        copy(info_file, 'train/info/' + info_files[ind])
        copy(vid_file, 'train/data/' + vid_name)

        #del info_files[ind]

    valid_inds = sorted(random.sample(range(len(info_files)), valid_len))
    for ind in reversed(valid_inds):
        vid_name = info_files[ind].split('.')[0] + ".mov"
        info_file = os.path.join(args.info_dir, info_files[ind])
        vid_file = video_dict[info_files[ind]]

        copy(info_file, 'validation/info/' + info_files[ind])
        copy(vid_file, 'validation/data/' + vid_name)
        #del info_files[ind]

    for file in info_files:
        vid_name = file.split('.')[0] + ".mov"
        info_file = os.path.join(args.info_dir, file)
        vid_file = video_dict[info_files[ind]]

        copy(info_file, 'test/info/' + info_files[ind])
        copy(vid_file, 'test/data/' + vid_name)

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

    args = arg_parser.parse_args()
    split_data(args)
