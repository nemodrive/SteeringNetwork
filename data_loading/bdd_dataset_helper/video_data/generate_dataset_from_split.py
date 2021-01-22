import os
import random
from shutil import copy
from argparse import ArgumentParser

TRAIN = 0.8
VALID = 1. - TRAIN

train_split_file = ''
test_split_file = ''


def split_data(args):
    info_files = os.listdir(args.info_dir)

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

    print(video_dict)

    with open(args.train_split, 'r') as f:
        train_names = f.readlines()

    # trim the newline from the end
    train_names = [x[:-1] for x in train_names]

    train_len = int(TRAIN * len(train_names))
    valid_len = int(VALID * len(train_names))

    train_inds = sorted(random.sample(range(len(train_names)), train_len))
    for ind in reversed(train_inds):
        vid_name = train_names[ind] + ".mov"
        print(vid_name)
        info_file = os.path.join(args.info_dir, train_names[ind])
        vid_file = video_dict[train_names[ind] + '.csv']

        copy(info_file, os.path.join(args.destination, 'train/info/', info_files[ind]))
        copy(vid_file, os.path.join(args.destination, 'train/data/', vid_name))

        #del info_files[ind]

    valid_inds = list(set(range(len(train_names))).difference(set(train_inds)))
    for ind in reversed(valid_inds):
        vid_name = train_names[ind] + ".mov"
        info_file = os.path.join(args.info_dir, info_files[ind])
        vid_file = video_dict[train_names[ind] + '.csv']

        copy(info_file, os.path.join(args.destination, 'validation/info/', info_files[ind]))
        copy(vid_file, os.path.join(args.destination, 'validation/data/', vid_name))
        #del info_files[ind]

    with open(args.test_split, 'r') as f:
        test_names = f.readlines()

    # trim the newline from the end
    test_names = [x[:-1] for x in test_names]

    for file in test_names:
        vid_name = file.split + ".mov"
        info_file = os.path.join(args.info_dir, file)
        vid_file = video_dict[file + '.csv']

        copy(info_file, os.path.join(args.destination, 'test/info/', info_files[ind]))
        copy(vid_file, os.path.join(args.destination, 'test/data/', vid_name))

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
    arg_parser.add_argument('--train_split',
                            type=str,
                            default='/home/andrei/storage/nemodrive/upb_data/dataset',
                            help='path to the train split file')
    arg_parser.add_argument('--test_split',
                            type=str,
                            default='/home/andrei/storage/nemodrive/upb_data/dataset',
                            help='path to the test split file')
    arg_parser.add_argument('--destination',
        type=str,
        default='/home/andrei/storage/nemodrive/upb_data/dataset',
        help='path to the destination directory')

    args = arg_parser.parse_args()
    split_data(args)
