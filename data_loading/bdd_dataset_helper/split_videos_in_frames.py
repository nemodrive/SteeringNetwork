import os
import cv2
import h5py
import subprocess
import pandas as pd
from argparse import ArgumentParser


def generate_frames(file, info, output_dir, sample_rate):
    # Get video frame rate
    cmd = ['ffmpeg', '-i', file]
    ffmpeg = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    _, err = ffmpeg.communicate()
    fps = round(float(str(err).split('fps')[0].split()[-1]))

    if fps == 60:
        sample_rate *= 2

    vid = cv2.VideoCapture(file)
    ret, img = vid.read()

    # Get frames and data
    targets_data = []
    frames = []

    cnt = 0
    frame_name = 0
    while ret:
        if cnt % sample_rate == 0:
            frames.append(img)
            frame_name += 1
            # Get info
            targets_data.append(list(info.iloc[cnt]))
        ret, img =  vid.read()
        cnt += 1

    # Create h5 file
    filename = file.split('/')[-1].split('.')[0]
    out_file = h5py.File(os.path.join(output_dir, filename + '.h5'), 'w')
    img_data_shape = (len(frames), 180, 320, 3)
    target_data_shape = (len(frames), 15)
    rgb = out_file.create_dataset('rgb', img_data_shape, data=frames, dtype='i1')
    targets = out_file.create_dataset('targets', target_data_shape, data=targets_data, dtype='f')


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--video_dir',
        type=str,
        default='video_data/train/data',
        help='path to directory containing the video files')
    arg_parser.add_argument('--info_dir',
        type=str,
        default='video_data/train/info',
        help='path to directory containing the info csv files')
    arg_parser.add_argument('--output_dir',
        type=str,
        default='video_data/train/img_data',
        help='path to directory where to store the frames')
    arg_parser.add_argument('--sample_rate',
        type=int,
        default=3,
        help='select one in sample_rate frames')

    args = arg_parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for file in os.listdir(args.video_dir):
        vid_name = file.split('.')[0]
        info_file = os.path.join(args.info_dir, vid_name + '.csv')
        info = pd.read_csv(info_file).iloc[:, 1:]
        file = os.path.join(args.video_dir, file)
        generate_frames(file, info, args.output_dir, args.sample_rate)
