import os
import math
import numpy as np
import pandas as pd
from argparse import ArgumentParser


enum = {
        'straight': 0,
        'slow_or_stop': 1,
        'turn_left': 2,
        'turn_right': 3,
        'turn_left_slight': 4,
        'turn_right_slight': 5,
        }


def get_angle(sx, sy):
    '''Get the angle corresponding to a velocity vector'''
    pi = math.pi
    if sy == 0:
        if sx > 0:
            course = pi / 2
        elif sx == 0:
            course = None
        else:
            course = 3 * pi / 2
        return course

    course = math.atan(sx / sy)
    if sx >= 0 and sy < 0:
        # Second quadrant
        course = pi + course
    elif sx < 0 and sy < 0:
        # Third quadrant
        course = pi + course
    elif sx < 0 and sy > 0:
        # Fourth quadrant
        course = 2 * pi + course

    # Keep angles within [-pi, pi]
    if course > pi:
        course = 2 * pi - course

    assert not math.isnan(course)
    return course


def add_steering(file, tts, deceleration_thresh):
    df = pd.read_csv(file)
    sx = df['speed_x']
    sy = df['speed_y']
    s = df['linear_speed']
    tstamps = df['timestamp']

    # Angle thresholds
    thresh_low = (2 * math.pi / 360) * 2
    thresh_high = (2 * math.pi / 360) * 180
    thresh_slight_low = (2 * math.pi / 360) * 5

    next_ind = 0
    steer = np.zeros(len(s))
    for i in range(len(s)):
        while next_ind < len(s) and tstamps[next_ind] - tstamps[i] < tts:
            next_ind += 1
        if next_ind >= len(s):
            break
        curr_angle = get_angle(sx[i], sy[i])
        next_angle = get_angle(sx[next_ind], sy[next_ind])

        # The car is not moving or will not be moving
        if curr_angle is None or next_angle is None:
            steer[i] = enum['slow_or_stop']
            continue

        course_diff = next_angle - curr_angle
        if s[next_ind] - s[i] < deceleration_thresh:
            steer[i] = enum['slow_or_stop']
        if thresh_low < course_diff < thresh_high:
            if thresh_slight_low < course_diff:
                steer[i] = enum['turn_right']
            else:
                steer[i] = enum['turn_right_slight']
        elif -thresh_high < course_diff < -thresh_low:
            if course_diff < -thresh_slight_low:
                steer[i] = enum['turn_left']
            else:
                steer[i] = enum['turn_left_slight']
        elif course_diff < -thresh_high or thresh_high < course_diff:
            print("WTF", (next_angle, sx[next_ind], sy[next_ind]), (curr_angle, sx[i], sy[i]))
            steer[i] = enum['slow_or_stop']
        else:
            steer[i] = enum['straight']

    # Replicate the last steering action over the last frames
    while i < len(s):
        steer[i] = steer[i - 1]
        i += 1

    df_steer = pd.DataFrame({'steer': steer})
    df = pd.concat([df, df_steer], axis=1)
    df = df.iloc[:, 1:]

    df.to_csv(file)


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--data_dir',
        type=str,
        default='original',
        help='path to directory containing the csv data files')
    arg_parser.add_argument('--time_to_steer',
        type=int,
        default=1000,
        help='time in ms between the pair of frames used for making the steering prediction')
    arg_parser.add_argument('--decel_thresh',
        type=float,
        default=-3.0,
        help='deceleration below which it is considered that we brake')

    args = arg_parser.parse_args()

    for file in os.listdir(args.data_dir):
        file = os.path.join(args.data_dir, file)
        add_steering(file, args.time_to_steer, args.decel_thresh)
