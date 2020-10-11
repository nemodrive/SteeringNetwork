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

    if 'steer' in df and 'course' in df and 'steer_angle' in df:
        return

    # Compute course
    # course = np.zeros(len(s))
    # for i in range(len(s)):
    #     course[i] = get_angle(sx[i], sy[i])
    course = df['course'] * math.pi / 180

    # Store steering angles
    steer_angle = np.zeros(len(s))

    # Angle thresholds
    thresh_low = (math.pi / 180) * 2
    thresh_high = (math.pi / 180) * 180
    thresh_slight_low = (math.pi / 180) * 5

    next_ind = 0
    steer = np.zeros(len(s))
    for i in range(len(s)):
        while next_ind < len(s) and tstamps[next_ind] - tstamps[i] <= tts:
            next_ind += 1
        if next_ind >= len(s):
            break
        curr_angle = course[i]
        next_angle = course[next_ind]

        # The car is not moving or will not be moving
        if curr_angle is None or next_angle is None:
            steer[i] = enum['slow_or_stop']
            continue

        course_diff = next_angle - curr_angle
        # Fix angle in case of steering angles around 360 degrees
        if course_diff < -math.pi:
            course_diff += 2 * math.pi
        elif math.pi < course_diff:
            course_diff -= 2 * math.pi

        steer_angle[i] = course_diff
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
            print("WTF", file, next_angle * 180 / math.pi, curr_angle * 180 / math.pi, course_diff * 180 / math.pi, "{} / {}".format(i, len(s)))
            steer[i] = enum['slow_or_stop']
        else:
            steer[i] = enum['straight']

    # Replicate the last steering action over the last frames
    if i < len(s):
        angle_unit = steer_angle[i - 1] / (len(s) - i)
    while i < len(s):
        steer[i] = steer[i - 1]
        steer_angle[i] = steer_angle[i - 1] - angle_unit
        i += 1

    if 'steer' not in df:
        df_steer = pd.DataFrame({'steer': steer})
        df = pd.concat([df, df_steer], axis=1)

    # if 'course' not in df:
    #     df_course = pd.DataFrame({'course': course})
    #     df = pd.concat([df, df_course], axis=1)

    if 'steer_angle' not in df:
        df_steer_angle = pd.DataFrame({'steer_angle': steer_angle})
        df = pd.concat([df, df_steer_angle], axis=1)

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
