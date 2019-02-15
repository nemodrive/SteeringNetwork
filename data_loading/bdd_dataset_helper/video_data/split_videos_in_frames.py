import os
import cv2
import subprocess
import pandas as pd
from argparse import ArgumentParser


def generate_frames(file, output_dir, sample_rate):
    if '.mov' in file:
        # Get video frame rate
        cmd = ['ffmpeg', '-i', file]
        ffmpeg = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        _, err = ffmpeg.communicate()
        print(err)
        fps = round(float(str(err).split('fps')[0].split()[-1]))

        if fps == 60:
            sample_rate *= 2

        vid = cv2.VideoCapture(file)
        ret, img = vid.read()

        # Get frames
        frames = []
        cnt = 0
        while ret:
            if cnt % sample_rate == 0:
                frames.append(img)
            ret, img =  vid.read()
            cnt += 1

        filename = file.split('/')[-1].split('.')[0]
        out_dir = os.path.join(output_dir, filename)

        # Create output directory directory
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # Store the frames
        for i in range(len(frames)):
            frame_file = os.path.join(out_dir, 'frame{}.jpg'.format(i))
            cv2.imwrite(frame_file, frames[i])


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--video_dir',
        type=str,
        default='train/data',
        help='path to directory containing the video files')
    arg_parser.add_argument('--output_dir',
        type=str,
        default='train/img_data',
        help='path to directory where to store the frames')
    arg_parser.add_argument('--sample_rate',
        type=int,
        default=1,
        help='select one in sample_rate frames')

    args = arg_parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for file in os.listdir(args.video_dir):
        vid_name = file.split('.')[0]
        file = os.path.join(args.video_dir, file)
        generate_frames(file, args.output_dir, args.sample_rate)
