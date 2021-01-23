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


    with open(args.train_split, 'r') as f:
        train_names = f.readlines()

    # trim the newline from the end
    train_names = [x.replace('\n', '') for x in train_names]

    train_len = int(TRAIN * len(train_names))
    valid_len = int(VALID * len(train_names))

    train_inds = sorted(random.sample(range(len(train_names)), train_len))
    for ind in reversed(train_inds):
        vid_name = train_names[ind] + ".mov"
        print(vid_name)
        info_file = os.path.join(args.info_dir, train_names[ind] + '-0.csv')
        vid_file = video_dict[train_names[ind] + '-0.csv']

        copy(info_file, os.path.join(args.destination, 'train/info/', train_names[ind] + '-0.csv'))
        copy(vid_file, os.path.join(args.destination, 'train/data/', vid_name))

        #del info_files[ind]

    valid_inds = list(set(range(len(train_names))).difference(set(train_inds)))
    for ind in reversed(valid_inds):
        vid_name = train_names[ind] + ".mov"
        info_file = os.path.join(args.info_dir, train_names[ind] + '-0.csv')
        vid_file = video_dict[train_names[ind] + '-0.csv']

        copy(info_file, os.path.join(args.destination, 'validation/info/', train_names[ind] + '-0.csv'))
        copy(vid_file, os.path.join(args.destination, 'validation/data/', vid_name))
        #del info_files[ind]

    with open(args.test_split, 'r') as f:
        test_names = f.readlines()

    # trim the newline from the end
    test_names = [x.replace('\n', '') for x in test_names]

    for file in test_names:
        vid_name = file + ".mov"
        info_file = os.path.join(args.info_dir, file + '-0.csv')
        vid_file = video_dict[file + '-0.csv']

        copy(info_file, os.path.join(args.destination, 'test/info/', file + '-0.csv'))
        copy(vid_file, os.path.join(args.destination, 'test/data/', vid_name))

    return train_inds, valid_inds


def check_split(args, train_inds, valid_inds):
    video_dict = {}

    with open(args.video_index, 'r') as f:
        data = f.readlines()

    for line in data:
        name = line.split('/')[-1].split('.')[0] + '.csv'
        #print(name)
        video_dict[name] = line[:-1]

    with open(args.train_split, 'r') as f:
        train_names = f.readlines()

    # trim the newline from the end
    train_names = [x.replace('\n', '') for x in train_names]

    with open(args.test_split, 'r') as f:
        test_names = f.readlines()

    # trim the newline from the end
    test_names = [x.replace('\n', '') for x in test_names]

    train_names_orig = train_names
    train_names = [train_names_orig[ind] for ind in train_inds]

    valid_names = [train_names_orig[ind] for ind in valid_inds]

    train_names = set(train_names)
    valid_names = set(valid_names)
    test_names = set(test_names)

    train_data_dest = os.path.join(args.destination, 'train/data/')
    train_info_dest = os.path.join(args.destination, 'train/info/')
    valid_data_dest = os.path.join(args.destination, 'validation/data/')
    valid_info_dest = os.path.join(args.destination, 'validation/info/')
    test_data_dest = os.path.join(args.destination, 'test/data/')
    test_info_dest = os.path.join(args.destination, 'test/info/')

    train_data_names = set([x.split('.')[0] for x in os.listdir(train_data_dest)])
    train_info_names = set([x.split('.')[0][:-2] for x in os.listdir(train_info_dest)])
    valid_data_names = set([x.split('.')[0] for x in os.listdir(valid_data_dest)])
    valid_info_names = set([x.split('.')[0][:-2] for x in os.listdir(valid_info_dest)])
    test_data_names = set([x.split('.')[0] for x in os.listdir(test_data_dest)])
    test_info_names = set([x.split('.')[0][:-2] for x in os.listdir(test_info_dest)])


    # check 1: file names match in both original location and final dataset location
    assert train_names == train_data_names, "Train data names error"
    assert train_names == train_info_names, "Train info names error"
    assert valid_names == valid_data_names, "Valid data names error"
    assert valid_names == valid_info_names, "Valid info names error"
    assert test_names == test_data_names, "Test data names error"
    assert test_names == test_info_names, "Test info names error"

    # check if file contents are the same
    for split in ['train', 'validation', 'test']:
        for vid_name in os.listdir(os.path.join(args.destination, split, 'data')):
            file_name = vid_name.split('.')[0]
            vid_path = os.path.join(args.destination, split, 'data', vid_name)
            info_path = vid_path.replace('/data/', '/info/').replace('.mov', '-0.csv')
            orig_info_file = os.path.join(args.info_dir, file_name + '-0.csv')
            orig_vid_file = video_dict[file_name + '-0.csv']
            # print(vid_path, info_path, orig_vid_file, orig_info_file)
            stream = os.popen('diff {} {}'.format(vid_path, orig_vid_file))
            output = stream.read()
            assert output == '', 'Difference between {} and {}'.format(vid_path, orig_vid_file)
            stream = os.popen('diff {} {}'.format(info_path, orig_info_file))
            output = stream.read()
            assert output == '', 'Difference between {} and {}'.format(info_path, orig_info_file)

    print('Everything should be fine')

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
    train_inds, valid_inds = split_data(args)
    check_split(args, train_inds, valid_inds)
