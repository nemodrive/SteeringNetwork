import cv2
import os
import json
import math
import argparse
import subprocess
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
from copy import deepcopy
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


def get_steer_values(original_data, fixed_data):
    speed = fixed_data['speed']
    time_stamps = fixed_data['timestamp']
    gps = fixed_data['gps']
    acc = fixed_data['accelerometer']
    gyro = fixed_data['gyroscope']
    # Trucate frames in case of a positive argument value
    if args.truncate_frames > 0:
        speed = fixed_data['speed'][:args.truncate_frames, :]
        time_stamps = fixed_data['timestamp'][:args.truncate_frames]
        gps = fixed_data['gps'][:args.truncate_frames, :]
        acc = fixed_data['accelerometer'][:args.truncate_frames, :]
        gyro = fixed_data['gyroscope'][:args.truncate_frames, :]

    # Compute turn and steering on original video frames
    turn, steer_value = turn_future_smooth(original_data['speed'],
        args.stop_future_frames, args.speed_limit_as_stop, args)
    original_data['turn'] = turn
    original_data['steer'] = steer_value

    # Compute turn and steering on 15FPS video frames
    turn, steer_value = turn_future_smooth(speed, args.stop_future_frames,
        args.speed_limit_as_stop, args)
    fixed_data['turn'] = turn
    fixed_data['steer'] = steer_value
    fixed_data = deepcopy(fixed_data)

    # Downsample data
    speed = speed[0::args.temporal_downsample_factor, :]
    time_stamps = time_stamps[0::args.temporal_downsample_factor]
    gps = gps[0::args.temporal_downsample_factor, :]
    acc = acc[0::args.temporal_downsample_factor, :]
    gyro = gyro[0::args.temporal_downsample_factor, :]

    # from speed to stop labels
    #stop_label = speed_to_future_has_stop(speed, args.stop_future_frames,
    #                                      args.speed_limit_as_stop)

    # Note that the turning heuristic is tuned for 3Hz video and urban area
    # Note also that stop_future_frames is reused for the turn
    turn, steer_value = turn_future_smooth(speed, args.stop_future_frames,
        args.speed_limit_as_stop, args)
    sampled_data = {
        'timestamp': time_stamps,
        'speed': speed,
        'gps': gps,
        'accelerometer': acc,
        'gyroscope': gyro,
        'turn': turn,
        'steer': steer_value
    }

    #locs = relative_future_location(
    #    speed, args.stop_future_frames,
    #    args.frame_rate / args.temporal_downsample_factor)

    return original_data, fixed_data, sampled_data


def process_video_info(video_path, args):
    fd, fprefix, fixed_out_name, original_out_name, sampled_out_name = \
        parse_path(video_path, args)

    nr_frames = get_nr_frames(video_path)
    duration, _ = probe_file(video_path)
    # save the speed field
    json_path = os.path.join(os.path.dirname(fd), "info", fprefix + ".json")
    fix_data, orig_data, err = get_interpolated_sensors(
        json_path, fprefix + ".mov", nr_frames)
    if err:
        # if speed is none, the error message is printed in other functions
        return False

    # orig_data, fix_data, samp_data = get_steer_values(orig_data, fix_data)
    if args.debug:
        import pdb; pdb.set_trace()

    # Prepare data to be transformed into csv
    full_data = {
        # 'fixed_data': fix_data,
        'original_data': orig_data,
        # 'sampled_data': samp_data
    }
    data_to_save = {}
    for data_type in full_data:
        curr_data = full_data[data_type]
        data_to_save[data_type] = {
            'timestamp': curr_data['timestamp'],
            'speed_x': curr_data['speed'][:, 0],
            'speed_y': curr_data['speed'][:, 1],
            'linear_speed': curr_data['linear_speed'],
            'course': curr_data['course'],
            # 'gps_lat': curr_data['gps'][:, 0],
            # 'gps_long': curr_data['gps'][:, 1],
            # 'acceleration_x': curr_data['accelerometer'][:, 0],
            # 'acceleration_y': curr_data['accelerometer'][:, 1],
            # 'acceleration_z': curr_data['accelerometer'][:, 2],
            # 'gyro_x': curr_data['gyroscope'][:, 0],
            # 'gyro_y': curr_data['gyroscope'][:, 1],
            # 'gyro_z': curr_data['gyroscope'][:, 2],
            # 'course': curr_data['course']
            # 'turn': curr_data['turn'],
            # 'steer': curr_data['steer']
        }

    # Create pandas dataframes with all the data
    # df = pd.DataFrame(data=data_to_save['fixed_data'])
    # df.to_csv(fixed_out_name)

    df = pd.DataFrame(data=data_to_save['original_data'])
    df.to_csv(original_out_name)

    # df = pd.DataFrame(data=data_to_save['sampled_data'])
    # df.to_csv(sampled_out_name)

    return True


def parse_path(video_path, args):
    '''
    Extract video directory path, video name and names of the data output
    files
    '''
    fd, fname = os.path.split(video_path)
    fprefix = fname.split(".")[0]
    original_out_name = os.path.join(args.output_directory, fprefix + ".csv")
    fixed_out_name = os.path.join(args.output_directory, 'fixed', fprefix + "_fixed.csv")
    sampled_out_name = os.path.join(args.output_directory, 'sampled', fprefix + "_sampled.csv")

    # return all sorts of info:
    # video_base_path, video_name_wo_prefix, cache_path, out_tfrecord_path
    return fd, fprefix, fixed_out_name, original_out_name, sampled_out_name


def process_videos(args):
    with open(args.video_index) as f:
        content = f.readlines()
    content = [x.strip() for x in content]

    for i in tqdm(range(len(content))):
        ok = process_video_info(content[i], args)
        if not ok:
            print("Video {} produced an error.".format(content[i]))
            continue

    print('Finished processing all files')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        '--video_index',
        type=str,
        default='/home/tempuser/workspace/alexm/samples-1k/filtered',
        help='filtered video indexing')

    arg_parser.add_argument(
        '--output_directory',
        type=str,
        default='/home/tempuser/workspace/alexm/SteeringNetwork/data_loading/bdd_dataset_helper/video_data/all_info',
        help='Training data directory')
    arg_parser.add_argument(
        '--debug',
        type=bool,
        default=False)

    arg_parser.add_argument(
        '--num_threads',
        type=int,
        default=1,
        help='Number of threads to preporcess the images')
    arg_parser.add_argument(
        '--truncate_frames',
        type=int,
        default=36 * 15,
        help='Number of frames to leave in the video. Negative value corresponds to no trucation')
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
