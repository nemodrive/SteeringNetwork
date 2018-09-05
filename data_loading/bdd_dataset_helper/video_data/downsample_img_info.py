import os
import pandas as pd
import pickle as pkl
from argparse import ArgumentParser


def downsample(info_name, sample_rate, info_file, out_dir):
    info = pd.read_csv(info_file).iloc[:, 1:]
    columns = info.columns

    downsampled_data = []
    for i in range(len(info)):
        if i % sample_rate == 0:
            downsampled_data.append(list(info.iloc[i]))

    df = pd.DataFrame(downsampled_data, columns=columns)
    df.to_csv(os.path.join(out_dir, info_name))


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--info_dir',
        type=str,
        default='train/info',
        help='path to directory containing csv info files')
    arg_parser.add_argument('--sample_rate',
        type=int,
        default=3,
        help='select one in sample_rate frames')
    arg_parser.add_argument('--out_dir',
        type=str,
        default='train/img_info',
        help='path to the directory where to stored downsampled info')

    args = arg_parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    for file in os.listdir(args.info_dir):
        info_name = file
        file = os.path.join(args.info_dir, file)

        downsample(info_name, args.sample_rate, file, args.out_dir)
