import cv2
import os
import json
import numpy as np
import math
import argparse
import subprocess
import time
from process_steer import *
from frames_sensors import get_interpolated_sensors

# might need to switch to this get_interpolated_speed when replaying GPS
#from MKZ.nodes.json_to_speed import get_interpolated_speed

# constant for the low res resolution
pixelh = 216
pixelw = 384
# constant for the high resolution
HEIGHT = 720
WIDTH = 1280


def probe_file(filename):
    cmnd = ['ffprobe', '-show_format', '-show_streams', '-pretty', filename]
    p = subprocess.Popen(cmnd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #print filename
    out, err = p.communicate()
    content = out.split(b'\n')
    whole_time = 0
    rotate = None
    horizontal = True
    for item in content:
        name = item.split(b'=')[0]
        tag = name.split(b':')
        if tag[0] == 'TAG':
            tag = tag[1]
        if name == b'duration':
            time = item.split(b':')
            hour = time[-3].split(b'=')[1]
            minute = time[-2]
            second = time[-1]
        if name == b'width':
            im_w = int(item.split(b'=')[1])
        if name == b'height':
            im_h = int(item.split(b'=')[1])
        if tag == b'rotate':
            rotate = int(item.split(b'=')[1])
    if im_w <= im_h:
        if rotate is None or rotate == 180 or rotate == -180:
            horizontal = False
    else:
        if rotate == 90 or rotate == -90 or rotate == 270 or rotate == -270:
            horizontal = False
        #print hour, minute, second
    whole_time = float(hour) * 3600 + float(minute) * 60 + float(second)

    return whole_time, horizontal


def full_im(pixel, all_num):
    # whether this frame is full image or not
    num_l, num_r, num_l_u, num_r_u = all_num

    # num is average pixel intensity over the rest areas
    num = 1.0 * (np.sum(pixel) - num_l - num_r - num_l_u - num_r_u) / \
            (pixel.shape[0] * pixel.shape[1] * 3 - 4 * pixelh * pixelw * 3)
    #print(num)
    # return (is a full image)
    return num >= 1


def get_nr_frames(video_path):

    cmd = [
        'ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
        '-show_entries', 'stream=nb_frames', '-of',
        'default=nokey=1:noprint_wrappers=1', video_path
    ]

    pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**7)
    pout, perr = pipe.communicate()

    return int(pout)


def get_steer_values(speed_values, time_stamps):

    speed = speed_values[:args.truncate_frames, :]
    speed = speed[0::args.temporal_downsample_factor, :]

    time_stamps = time_stamps[:args.truncate_frames]
    time_stamps = time_stamps[0::args.temporal_downsample_factor]

    # from speed to stop labels
    #stop_label = speed_to_future_has_stop(speed, args.stop_future_frames,
    #                                      args.speed_limit_as_stop)

    # Note that the turning heuristic is tuned for 3Hz video and urban area
    # Note also that stop_future_frames is reused for the turn
    turn, steer_value = turn_future_smooth(speed, args.stop_future_frames,
                              args.speed_limit_as_stop, args)

    #locs = relative_future_location(
    #    speed, args.stop_future_frames,
    #    args.frame_rate / args.temporal_downsample_factor)

    return turn, steer_value


def process_video_info(video_path, args):
    fd, fprefix, out_name = parse_path(video_path, args)

    nr_frames = get_nr_frames(video_path)
    duration, _ = probe_file(video_path)
    # save the speed field
    json_path = os.path.join(os.path.dirname(fd), "info", fprefix + ".json")
    t_stamp, speed_f_steer, speed, gps, acc, gyro, err = get_interpolated_sensors(
        json_path, fprefix + ".mov", nr_frames)
    if err:
        # if speed is none, the error message is printed in other functions
        return 0, False

    turn, steer_value = get_steer_values(speed_f_steer, t_stamp)
    import pdb; pdb.set_trace()

    return True


def parse_path(video_path, args):
    fd, fname = os.path.split(video_path)
    fprefix = fname.split(".")[0]
    out_name = os.path.join(args.output_directory, fprefix + ".csv")

    # return all sorts of info:
    # video_base_path, video_name_wo_prefix, cache_path, out_tfrecord_path
    return (fd, fprefix, out_name)


def process_videos(args):
    with open(args.video_index) as f:
        content = f.readlines()
    content = [x.strip() for x in content]

    for video in content:
        process_video_info(video, args)

    print('Finished processing all files')



if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        '--video_index',
        type=str,
        default='/data/nx-bdd-20160929/video_filtered_index_38_60_sec.txt',
        help='filtered video indexing')

    arg_parser.add_argument(
        '--output_directory',
        type=str,
        default='/data/nx-bdd-20160929/tfrecord_fix_speed/',
        help='Training data directory')

    arg_parser.add_argument(
        '--num_threads',
        type=int,
        default=1,
        help='Number of threads to preporcess the images')
    arg_parser.add_argument(
        '--truncate_frames',
        type=int,
        default=36 * 15,
        help='Number of frames to leave in the video')
    arg_parser.add_argument(
        '--temp_dir_root',
        type=str,
        default='/tmp/',
        help='the temp dir to hold ffmpeg outputs')
    arg_parser.add_argument(
        '--low_res',
        type=bool,
        default=False,
        help='the data we want to use is low res')

    arg_parser.add_argument(
        '--decode_downsample_factor',
        type=int,
        default=1,
        help='The original high res video is 640*360. This param downsample the image during jpeg decode process')
    
    '''The original video is in 15 FPS, this flag optionally downsample the video temporally
       All other frame related operations are carried out after temporal downsampling'''
    arg_parser.add_argument(
        '--temporal_downsample_factor',
        type=int,
        default=5,
        help='The original video is in 15 FPS, this flag optionally downsample the video temporally')

    arg_parser.add_argument(
        '--speed_limit_as_stop',
        type=float,
        default=0.3,
        help='if speed is less than this, then it is considered to be stopping'
    )
    arg_parser.add_argument(
        '--stop_future_frames',
        type=int,
        default=2,
        help='Shift the stop labels * frames forward, to predict the future')

    arg_parser.add_argument(
        '--balance_drop_prob',
        type=float,
        default=-1.0,
        help='drop no stop seq with specified probability')

    arg_parser.add_argument(
        '--acceleration_thres',
        type=float,
        default=-1.0,
        help='acceleration threshold, minus value for not using it')

    arg_parser.add_argument(
        '--deceleration_thres',
        type=float,
        default=1.0,
        help='deceleration threshold, minus value for not using it')

    arg_parser.add_argument(
        '--non_random_temporal_downsample',
        type=bool,
        default=False,
        help='''if true, use fixed downsample method''')

    arg_parser.add_argument(
        '--frame_rate',
        type=float,
        default=15.0,
        help='the frame_rate we have for the videos')

    args = arg_parser.parse_args()

    if args.low_res:
        print("Warning: using low res specific settings")
    if not os.path.exists(args.output_directory):
        os.mkdir(args.output_directory)


    process_videos(args)