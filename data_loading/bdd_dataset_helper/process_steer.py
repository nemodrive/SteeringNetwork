import tensorflow as tf
import numpy as np
import os, random
import scipy.misc as misc
import glob
import cv2
import ctypes
import math
from scipy import interpolate


turn_str2int = {
    'not_sure': -1,
    'straight': 0,
    'slow_or_stop': 1,
    'turn_left': 2,
    'turn_right': 3,
    'turn_left_slight': 4,
    'turn_right_slight': 5,
}
turn_int2str = {y: x for x, y in turn_str2int.items()}
naction = np.sum(np.less_equal(0, np.array(list(turn_str2int.values()))))


def speed_to_course(speed):
    pi = math.pi
    if speed[1] == 0:
        if speed[0] > 0:
            course = pi / 2
        elif speed[0] == 0:
            course = None
        elif speed[0] < 0:
            course = 3 * pi / 2
        return course

    course = math.atan(speed[0] / speed[1])
    # if course < 0:
    #     course = course + 2 * pi

    # if speed[1] < 0:
    #     course = pi + course
    #     if course > 2 * pi:
    #         course = course - 2 * pi

    if speed[0] >= 0 and speed[1] < 0:
        # Second quadrant
        course = pi + course
    elif speed[0] < 0 and speed[1] < 0:
        # Third quadrant
        course = pi + course
    elif speed[0] < 0 and speed[1] > 0:
        # Fourth quadrant
        course = 2 * pi + course

    assert not math.isnan(course)
    return course


def to_course_list(speed_list):
    l = speed_list.shape[0]
    course_list = []
    for i in range(l):
        speed = speed_list[i, :]
        course_list.append(speed_to_course(speed))
    return course_list


def diff(a, b):
    # return a-b \in -pi to pi
    d = a - b
    if d > math.pi:
        d -= math.pi * 2
    if d < -math.pi:
        d += math.pi * 2
    return d


def turning_heuristics(speed_list, args, speed_limit_as_stop=0):
    course_list = to_course_list(speed_list)
    speed_v = np.linalg.norm(speed_list, axis=1)
    l = len(course_list)
    action = np.zeros(l).astype(np.int32)
    course_diff = np.zeros(l).astype(np.float32)

    enum = turn_str2int

    thresh_low = (2 * math.pi / 360) * 1
    thresh_high = (2 * math.pi / 360) * 35
    thresh_slight_low = (2 * math.pi / 360) * 3

    for i in range(l):
        if i == 0:
            # Uncertainty to be solved at the end
            continue

        # the speed_limit_as_stop should be small,
        # this detect strict real stop
        if speed_v[i] < speed_limit_as_stop + 1e-3:
            # take the smaller speed as stop
            action[i] = enum['slow_or_stop']
            continue

        course = course_list[i]
        prev = course_list[i - 1]

        if course is None or prev is None:
            action[i] = enum['slow_or_stop']
            course_diff[i] = 9999
            continue

        curr_course_diff = diff(course, prev)

        course_diff[i] = curr_course_diff * 360 / (2 * math.pi)
        # Decide the type of steering action
        if thresh_high > curr_course_diff > thresh_low:
            if curr_course_diff > thresh_slight_low:
                action[i] = enum['turn_right']
            else:
                action[i] = enum['turn_right_slight']
        elif -thresh_high < curr_course_diff < -thresh_low:
            if curr_course_diff < -thresh_slight_low:
                action[i] = enum['turn_left']
            else:
                action[i] = enum['turn_left_slight']
        elif curr_course_diff >= thresh_high or curr_course_diff <= -thresh_high:
            action[i] = enum['not_sure']
        else:
            action[i] = enum['straight']

        # this detect significant slow down that is not due to going to turn
        if args.deceleration_thres < 0 and action[i] == enum['straight']:
            hz = args.frame_rate / args.temporal_downsample_factor
            acc_now = (speed_v[i] - speed_v[i - 1]) * hz
            if acc_now < args.deceleration_thres:
                action[i] = enum['slow_or_stop']
                continue


    # avoid the initial uncertainty
    action[0] = action[1]
    return action, course_diff


def turn_future_smooth(speed, nfuture, speed_limit_as_stop, args):
    # this function takes in the speed and output a smooth future action map
    turn, steer_values = turning_heuristics(speed, args, speed_limit_as_stop)
    return turn, steer_values


def future_smooth(actions, naction, nfuture):
    # TODO: could add weighting differently between near future and far future
    # given a list of actions, for each time step, return the distribution of future actions
    l = len(
        actions
    )  # action is a list of integers, from 0 to naction-1, negative values are ignored
    out = np.zeros((l, naction), dtype=np.float32)
    for i in range(l):
        # for each output position
        total = 0
        for j in range(min(nfuture, l - i)):
            # for each future position
            # current deal with i+j action
            acti = i + j
            if actions[acti] >= 0:
                out[i, actions[acti]] += 1
                total += 1
        if total == 0:
            out[i, turn_str2int['straight']] = 1.0
        else:
            out[i, :] = out[i, :] / total
    return out


def speed_to_future_has_stop(speed, nfuture, speed_limit_as_stop):
    # expect the stop_label to be 1 dimensional, representing the stop labels along time
    # nfuture is how many future stop labels to consider
    speed = np.linalg.norm(speed, axis=1)
    stop_label = np.less(speed, speed_limit_as_stop)
    stop_label = stop_label.astype(np.int32)

    # the naction=2 means: the number of valid actions are 2
    smoothed = future_smooth(stop_label, 2, nfuture)
    out = np.less(0, smoothed[:, 1]).astype(np.int32)
    return out


def no_stop_dropout_valid(stop_label, drop_prob):
    nbatch = stop_label.shape[0]
    ntime = stop_label.shape[1]
    out = np.ones(nbatch, dtype=np.bool)
    for i in range(nbatch):
        # determine whether this seq has stop
        has_stop = False
        for j in range(ntime):
            if stop_label[i, j]:
                has_stop = True
                break
        if not has_stop:
            if np.random.rand() < drop_prob:
                out[i] = False

    return out


def fix_none_in_course(course_list):
    l = len(course_list)

    # fix the initial None values
    not_none_value = 0
    for i in range(l):
        if course_list[i] is not None:
            not_none_value = course_list[i]
            break
    for i in range(l):
        if course_list[i] is None:
            course_list[i] = not_none_value
        else:
            break

    # a course could be None, use the previous course in that case
    for i in range(1, l):
        if course_list[i] is None:
            course_list[i] = course_list[i - 1]
    return course_list


def integral(speed, time0):
    out = np.zeros_like(speed)
    l = speed.shape[0]
    for i in range(l):
        s = speed[i, :]
        if i > 0:
            out[i, :] = out[i - 1, :] + s * time0
    return out


def relative_future_course_speed(speed, nfuture, sample_rate):
    def norm_course_diff(course):
        if course > math.pi:
            course = course - 2 * math.pi
        if course < -math.pi:
            course = course + 2 * math.pi
        return course

    # given the speed vectors, calculate the future location relative to
    # the current location, with facing considered
    course_list = to_course_list(speed)
    course_list = fix_none_in_course(course_list)

    # integrate the speed to get the location
    loc = integral(speed, 1.0 / sample_rate)

    out = np.zeros_like(loc)
    l = out.shape[0]
    for i in range(l):
        if i + nfuture < l:
            fi = min(i + nfuture, l - 1)
            # first is course diff
            out[i, 0] = norm_course_diff(course_list[fi] - course_list[i])
            # second is the distance
            out[i, 1] = np.linalg.norm(loc[fi, :] - loc[i, :])
        else:
            # at the end of the video, just use what has before
            out[i, :] = out[i - 1, :]

    # normalize the speed to be per second
    timediff = 1.0 * nfuture / sample_rate
    out = out / timediff

    return out


def relative_future_location(speed, nfuture, sample_rate):
    # given the speed vectors, calculate the future location relative to
    # the current location, with facing considered
    course_list = to_course_list(speed)
    course_list = fix_none_in_course(course_list)

    # integrate the speed to get the location
    loc = integral(speed, 1.0 / sample_rate)

    # project future motion on to the current facing direction
    # this is counter clock wise
    def rotate(vec, theta):
        c = math.cos(theta)
        s = math.sin(theta)
        xp = c * vec[0] - s * vec[1]
        yp = s * vec[0] + c * vec[1]
        return np.array([xp, yp])

    out = np.zeros_like(loc)
    l = out.shape[0]
    for i in range(l):
        future = loc[min(i + nfuture, l - 1), :]
        delta = future - loc[i, :]
        out[i, :] = rotate(delta, course_list[i])

    return out
