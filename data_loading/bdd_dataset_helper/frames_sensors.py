import json
import numpy as np
import math
from process_steer import *


def to_array(val, key):
    return np.array([x[key] for x in val]).ravel()


def check_allignment(res, video_filename):
    if res['timestamp'][0] - res['startTime'] > 2000:
        print('This is bad video because starting time too far ahead',
              video_filename)
        return False

    if res['endTime'] - res['timestamp'][-1] > 2000:
        print('This is bad video because ending time too far ahead',
              video_filename)
        return False

    return True


def read_gyro_json(json_file, video_filename):

    gyro = json_file['gyro']
    res = {}
    bad_video_t = 0
    bad_video_same = 0
    for ifile, f in enumerate(gyro):
        if ifile != 0:
            if int(f['timestamp']) - prev_t > 30:
                bad_video_t = 1
                break
            if abs(int(f['timestamp']) - int(prev_t)) < 1:
                bad_video_same = 1
                break
        prev_t = f['timestamp']

    if bad_video_t:
        print('This is a bad video because time sample for gyro not uniform',
              video_filename)
        return None
    if len(gyro) == 0:
        print('This is a bad video because no gyro data available',
              video_filename)
        return None
    if bad_video_same:
        print('This is a bad video because same timestamps for gyro',
              video_filename)
        return None

    for key in gyro[0].keys():
        res[key] = to_array(gyro, key)

    # add the starting time point and ending time point as well
    res['startTime'] = json_file['startTime']
    res['endTime'] = json_file['endTime']

    if check_allignment(res, video_filename) is False:
        return None

    return res


def read_acc_json(json_file, video_filename):

    acc = json_file['accelerometer']
    res = {}
    bad_video_t = 0
    bad_video_same = 0
    for ifile, f in enumerate(acc):
        if ifile != 0:
            if int(f['timestamp']) - prev_t > 30:
                bad_video_t = 1
                break
            if abs(int(f['timestamp']) - int(prev_t)) < 1:
                bad_video_same = 1
                break
        prev_t = f['timestamp']

    if bad_video_t:
        print(
            'This is a bad video because time sample for accelerometer not uniform',
            video_filename)
        return None
    if len(acc) == 0:
        print('This is a bad video because no accelerometer data available',
              video_filename)
        return None
    if bad_video_same:
        print('This is a bad video because same timestamps for accelerometer',
              video_filename)
        return None

    for key in acc[0].keys():
        res[key] = to_array(acc, key)

    # add the starting time point and ending time point as well
    res['startTime'] = json_file['startTime']
    res['endTime'] = json_file['endTime']

    if check_allignment(res, video_filename) is False:
        return None

    return res


def read_loc_json(json_file, video_filename):

    locs = json_file['locations']
    res = {}
    bad_video_c = 0
    bad_video_t = 0
    bad_video_same = 0
    for ifile, f in enumerate(locs):
        if int(f['course']) == -1 or int(f['speed']) == -1:
            bad_video_c += 1  # Changed for interpolation
            if bad_video_c >= 3:
                break
        if ifile != 0:
            if int(f['timestamp']) - prev_t > 1100:
                bad_video_t = 1
                break
            if abs(int(f['timestamp']) - int(prev_t)) < 1:
                bad_video_same = 1
                break
        prev_t = f['timestamp']
    if bad_video_c >= 3:
        print('This is a bad video because course or speed is -1', json_path,
              video_filename)
        return None
    if bad_video_t:
        print('This is a bad video because time sample not uniform', json_path,
              video_filename)
        return None
    if len(locs) == 0:
        print('This is a bad video because no location data available',
              json_path, video_filename)
        return None
    if bad_video_same:
        print('This is a bad video because same timestamps', json_path,
              video_filename)
        return None

    for key in locs[0].keys():
        res[key] = to_array(locs, key)

    # add the starting time point and ending time point as well
    res['startTime'] = json_file['startTime']
    res['endTime'] = json_file['endTime']

    if check_allignment(res, video_filename) is False:
        return None

    return res


def fill_missing_speeds_and_courses(values, show_warning):
    l = len(values)
    for i in range(l):
        if values[i] == -1:
            if show_warning:
                print(
                    "Warning: course==-1 appears, previous computation might not be reliable"
                )
            if i == (l - 1):
                values[i] = values[i - 1]
            else:
                if values[i + 1] == -1:
                    return None
                values[i] = values[i + 1]
    return values


def fill_missing_gps(values, max_value):
    l = len(values)
    for i in range(l):
        if values[i] < -max_value or values[i] > max_value:
            if i == (l - 1):
                values[i] = values[i - 1]
            else:
                if values[i + 1] == -1:
                    return None
                values[i] = values[i + 1]
    return values


def get_interpolated_acc(res, nr_frames):

    acc_x = res['x']
    acc_y = res['y']
    acc_z = res['z']

    acc = np.zeros((nr_frames, 3), dtype=np.float32)

    tot_ms = res['endTime'] - res['startTime']
    # total number of output

    # if time is t second, there should be t+1 points
    last_start = 0
    ts = res['timestamp']
    for i in range(nr_frames):
        # convert to ms timestamp
        timenow = i * tot_ms // nr_frames + res['startTime']

        while (last_start + 1 < len(ts)) and (ts[last_start + 1] < timenow):
            last_start += 1

        if last_start + 1 == len(ts):
            acc[i, :] = acc_x[last_start], acc_y[last_start], acc_z[last_start]
        elif timenow <= ts[0]:
            acc[i, :] = acc_x[0], acc_y[0], acc_z[0]

        else:
            time1 = timenow - ts[last_start]
            time2 = ts[last_start + 1] - timenow
            r1 = time2 / (time1 + time2)
            r2 = time1 / (time1 + time2)

            inter_x = r1 * acc_x[last_start] + r2 * acc_x[last_start + 1]
            inter_y = r1 * acc_y[last_start] + r2 * acc_y[last_start + 1]
            inter_z = r1 * acc_z[last_start] + r2 * acc_z[last_start + 1]
            acc[i, :] = inter_x, inter_y, inter_z

    return acc


def get_interpolated_gyro(res, nr_frames):

    gyro_x = res['x']
    gyro_y = res['y']
    gyro_z = res['z']

    gyro = np.zeros((nr_frames, 3), dtype=np.float32)

    tot_ms = res['endTime'] - res['startTime']
    # total number of output

    # if time is t second, there should be t+1 points
    last_start = 0
    ts = res['timestamp']
    for i in range(nr_frames):
        # convert to ms timestamp
        timenow = i * tot_ms // nr_frames + res['startTime']

        while (last_start + 1 < len(ts)) and (ts[last_start + 1] < timenow):
            last_start += 1

        if last_start + 1 == len(ts):
            gyro[i, :] = gyro_x[last_start], gyro_y[last_start], gyro_z[
                last_start]
        elif timenow <= ts[0]:
            gyro[i, :] = gyro_x[0], gyro_y[0], gyro_z[0]

        else:
            time1 = timenow - ts[last_start]
            time2 = ts[last_start + 1] - timenow
            r1 = time2 / (time1 + time2)
            r2 = time1 / (time1 + time2)

            inter_x = r1 * gyro_x[last_start] + r2 * gyro_x[last_start + 1]
            inter_y = r1 * gyro_y[last_start] + r2 * gyro_y[last_start + 1]
            inter_z = r1 * gyro_z[last_start] + r2 * gyro_z[last_start + 1]
            gyro[i, :] = inter_x, inter_y, inter_z

    return gyro


def get_interpolated_loc(res, nr_frames):

    # get the speed on OX and OY, because
    # course is the angle of the speed
    def vec(speed, course):
        t = math.radians(course)
        return np.array([math.sin(t) * speed, math.cos(t) * speed])

    hz = 15
    course = res['course']
    speed0 = res['speed']
    latitude = res['latitude']
    longitude = res['longitude']
    # first convert to speed vecs
    l = len(course)

    speed = np.zeros((l, 2), dtype=np.float32)
    for i in range(l):
        # interpolate when the number of missing speed is small
        speed0 = fill_missing_speeds_and_courses(speed0, False)
        course = fill_missing_speeds_and_courses(course, True)
        latitude = fill_missing_gps(latitude, 90.0)
        longitude = fill_missing_gps(longitude, 180.0)
        if (speed0 is None) or (course is None) or (latitude is None) or (
                longitude is None):
            return None

        speed[i, :] = vec(speed0[i], course[i])

    tot_ms = res['endTime'] - res['startTime']
    # total number of output

    nout = tot_ms * hz // 1000

    out = np.zeros((nout, 2), dtype=np.float32)
    tstamp = np.zeros((nout, 1), dtype=np.int64)
    # if time is t second, there should be t+1 points
    last_start = 0
    ts = res['timestamp']
    for i in range(nout):
        # convert to ms timestamp
        timenow = i * 1000.0 / hz + res['startTime']
        tstamp[i] = timenow
        while (last_start + 1 < len(ts)) and (ts[last_start + 1] < timenow):
            last_start += 1

        if last_start + 1 == len(ts):
            out[i, :] = speed[last_start, 0], speed[last_start, 1]
        elif timenow <= ts[0]:
            out[i, :] = speed[0, 0], speed[0, 1]
        else:
            time1 = timenow - ts[last_start]
            time2 = ts[last_start + 1] - timenow
            r1 = time2 / (time1 + time2)
            r2 = time1 / (time1 + time2)
            inter = r1 * speed[last_start, :] + r2 * speed[last_start + 1, :]
            out[i, :] = inter[0], inter[1]


    out_speed = np.zeros((nr_frames, 2), dtype=np.float32)
    out_gps = np.zeros((nr_frames, 2), dtype=np.float32)
    # if time is t second, there should be t+1 points
    last_start = 0
    ts = res['timestamp']
    for i in range(nr_frames):
        # convert to ms timestamp
        timenow = i * tot_ms // nr_frames + res['startTime']

        while (last_start + 1 < len(ts)) and (ts[last_start + 1] < timenow):
            last_start += 1

        if last_start + 1 == len(ts):
            out_speed[i, :] = speed0[last_start], course[last_start]
            out_gps[i, :] = latitude[last_start], longitude[last_start]
        elif timenow <= ts[0]:
            out_speed[i, :] = speed0[0], course[0]
            out_gps[i, :] = latitude[0], longitude[0]

        else:
            time1 = timenow - ts[last_start]
            time2 = ts[last_start + 1] - timenow
            r1 = time2 / (time1 + time2)
            r2 = time1 / (time1 + time2)
            inter_s = r1 * speed0[last_start] + r2 * speed0[last_start + 1]
            inter_r = r1 * course[last_start] + r2 * course[last_start + 1]
            out_speed[i, :] = inter_s, inter_r

            inter_lat = r1 * latitude[last_start] + r2 * latitude[last_start
                                                                  + 1]
            inter_lon = r1 * longitude[last_start] + r2 * longitude[last_start
                                                                    + 1]
            out_gps[i, :] = inter_lat, inter_lon

    return tstamp, out, out_speed, out_gps


def get_interpolated_sensors(json_path, video_filename, nr_frames):

    with open(json_path) as data_file:
        seg = json.load(data_file)

    #get gps and speed values for each frame
    res = read_loc_json(seg, video_filename)
    if res is None:
        return (None, None, None, None, -1)
    tstamp, speed_steer, out_speed, out_gps = get_interpolated_loc(res, nr_frames)
    if out_speed is None:
        out_speed = np.array([])
    if out_gps is None:
        out_gps = np.array([])
    if speed_steer is None:
        speed_steer = np.array([])

    #get accelerometer valus for each frame
    res = read_acc_json(seg, video_filename)
    if res is None:
        return (None, None, None, None, -1)
    out_acc = get_interpolated_acc(res, nr_frames)
    if out_acc is None:
        out_acc = np.array([])

    #get gyroscope values for each frame
    res = read_gyro_json(seg, video_filename)
    if res is None:
        return (None, None, None, None, -1)
    out_gyro = get_interpolated_gyro(res, nr_frames)
    if out_gyro is None:
        out_gyro = np.array([])

    return tstamp, speed_steer, out_speed, out_gps, out_acc, out_gyro, 0
