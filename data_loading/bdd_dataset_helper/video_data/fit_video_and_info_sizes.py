import os
import pandas as pd
from argparse import ArgumentParser


def adjust_sizes(args):
    """
    Make the info files and the folders containing frames have the same number
    of entries
    """
    for file in os.listdir(args.video_dir):
        info_file = os.path.join(args.info_dir, file + '.csv')
        file = os.path.join(args.video_dir, file)
        info = pd.read_csv(info_file).iloc[:, 1:]
        nframes = len(os.listdir(file))

        if nframes < len(info):
            info = info.iloc[:nframes]
            info.to_csv(info_file)
        elif len(info) < nframes:
            for i in range(len(info), nframes):
                frame = os.path.join(file, '{}.jpg'.format(i))
                os.remove(frame)


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--video_dir',
        type=str,
        default='train/img_data',
        help='path to directory containing folders with video frames')
    arg_parser.add_argument('--info_dir',
        type=str,
        default='train/img_info',
        help='path to the directory containing info files')

    args = arg_parser.parse_args()

    adjust_sizes(args)
