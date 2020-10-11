import json
import numpy as np
import math
from process_steer import *


# Fixed frame rate at which to interpolate data
HZ = 15


def to_array(val, key):
    return np.array([x[key] for x in val]).ravel()


def check_allignment(res, video_filename):
    if res['timestamp'][0] - res['startTime'] > 2000:
        print('{} is bad video because starting time too far ahead'.format(video_filename))
        return False

    if res['endTime'] - res['timestamp'][-1] > 2000:
        print('{} is bad video because ending time too far ahead'.format(video_filename))
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
        print('{} is a bad video because time sample for gyro not uniform'.format(video_filename))
        return None
    if len(gyro) == 0:
        print('{} is a bad video because no gyro data available'.format(video_filename))
        return None
    if bad_video_same:
        print('{} is a bad video because same timestamps for gyro'.format(video_filename))
        return None

    for key in gyro[0].keys():
        res[key] = to_array(gyro, key)

    # add the starting time point and ending time point as well
    res['startTime'] = json_file['startTime']
    res['endTime'] = json_file['endTime']

    if check_allignment(res, video_filename) is False:
        return None

    return res


def read_steer_json(json_file, video_filename):
    steer = json_file['steer_data']['steer']
    timestamps = json_file['steer_data']['tp']
    steers = []
    for (s, t) in zip(steer, timestamps):
        steers.append({'steer': s, 'timestamp': t})
    res = {}
    bad_video_t = 0
    bad_video_same = 0
    for ifile, f in enumerate(steers):
        if ifile != 0:
            if int(f['timestamp']) - prev_t > 30:
                bad_video_t = 1
                break
            if abs(int(f['timestamp']) - int(prev_t)) < 0:
                bad_video_same = 1
                break
        prev_t = f['timestamp']

    if bad_video_t:
        print(
            '{} is a bad video because time sample for steer not uniform'.format(video_filename))
        return None
    if len(steer) == 0:
        print('{} is a bad video because no steer data available'.format(video_filename))
        return None
    if bad_video_same:
        print('{} is a bad video because same timestamps for steer'.format(video_filename))
        return None

    for key in steers[0].keys():
        res[key] = to_array(steers, key)

    # add the starting time point and ending time point as well
    res['startTime'] = json_file['startTime'] / 1000
    res['endTime'] = json_file['endTime'] / 1000

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
            '{} is a bad video because time sample for accelerometer not uniform'.format(video_filename))
        return None
    if len(acc) == 0:
        print('{} is a bad video because no accelerometer data available'.format(video_filename))
        return None
    if bad_video_same:
        print('{} is a bad video because same timestamps for accelerometer'.format(video_filename))
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
            if int(f['timestamp']) - prev_t > 1500:
                bad_video_t = 1
                break
            # modified to < 0 because the timestamps were floats not ints
            if abs(int(f['timestamp']) - int(prev_t)) < 0:
                bad_video_same = 1
                break
        prev_t = f['timestamp']
    if bad_video_c >= 3:
        print('{} is a bad video because course or speed is -1'.format(video_filename))
        return None
    if bad_video_t:
        print('{} is a bad video because time sample not uniform'.format(video_filename))
        return None
    if len(locs) == 0:
        print('{} is a bad video because no location data available'.format(video_filename))
        return None
    if bad_video_same:
        print('{} is a bad video because same timestamps'.format(video_filename))
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
                if values[i + 1] < -max_value or values[i + 1] > max_value:
                    return None
                values[i] = values[i + 1]
    return values


def get_interpolated_acc(res, nr_frames):
    tot_ms = res['endTime'] - res['startTime']

    def interpolate(res, nout, time_unit):
        acc_x = res['x']
        acc_y = res['y']
        acc_z = res['z']
        acc = np.zeros((nout, 3), dtype=np.float32)

        # if time is t second, there should be t+1 points
        last_start = 0
        ts = res['timestamp']
        for i in range(nout):
            # convert to ms timestamp
            timenow = i * time_unit + res['startTime']

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

    fixed = None
    # fixed = interpolate(res, tot_ms * HZ // 1000, 1000.0 / HZ)
    original = interpolate(res, nr_frames, tot_ms / nr_frames)

    return fixed, original


def get_interpolated_gyro(res, nr_frames):
    tot_ms = res['endTime'] - res['startTime']

    def interpolate(res, nout, time_unit):
        gyro_x = res['x']
        gyro_y = res['y']
        gyro_z = res['z']

        gyro = np.zeros((nout, 3), dtype=np.float32)

        # if time is t second, there should be t+1 points
        last_start = 0
        ts = res['timestamp']
        for i in range(nout):
            # convert to ms timestamp
            timenow = i * time_unit + res['startTime']

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

    fixed = None
    # fixed = interpolate(res, tot_ms * HZ // 1000, 1000.0 / HZ)
    original = interpolate(res, nr_frames, tot_ms / nr_frames)

    return fixed, original


def get_interpolated_loc(res, nr_frames):
    # Get the speed on OX and OY, because course is the angle of the speed
    def vec(speed, course):
        t = math.radians(course)
        return np.array([math.sin(t) * speed, math.cos(t) * speed])

    # Interpolate timestamp, speed and gps
    def interpolate(speed, latitude, longitude, northing, easting, course, res, nout, time_unit):
        tstamp_out = np.zeros((nout,), dtype=np.int64)
        speed_out = np.zeros((nout, 2), dtype=np.float32)
        gps_out = np.zeros((nout, 2), dtype=np.float32)
        course_out = np.zeros((nout,), dtype=np.float32)
        pos_out = np.zeros((nout, 2), dtype=np.float32)
        # If time is t second, there should be t+1 points
        last_start = 0
        ts = res['timestamp']
        for i in range(nout):
            # Convert to ms timestamp
            timenow = i * time_unit + res['startTime']
            tstamp_out[i] = timenow
            while (last_start + 1 < len(ts)) and (ts[last_start + 1] < timenow):
                last_start += 1

            if last_start + 1 == len(ts):
                speed_out[i, :] = speed[last_start]
                gps_out[i, :] = latitude[last_start], longitude[last_start]
                course_out[i] = course[last_start]
                pos_out[i] = northing[last_start], easting[last_start]
            elif timenow <= ts[0]:
                speed_out[i, :] = speed[0]
                gps_out[i, :] = latitude[0], longitude[0]
                course_out[i] = course[0]
                pos_out[i] = northing[0], easting[0]
            else:
                # Compute the speed components
                time1 = timenow - ts[last_start]
                time2 = ts[last_start + 1] - timenow
                r1 = time2 / (time1 + time2)
                r2 = time1 / (time1 + time2)
                inter = r1 * speed[last_start, :] + r2 * speed[last_start + 1, :]
                speed_out[i, :] = inter

                # Compute the course
                curr_course = course[last_start]
                next_course = course[last_start + 1]
                if next_course - curr_course < -180:
                    next_course += 360
                elif next_course - curr_course > 180:
                    curr_course += 360
                course_out[i] = r1 * curr_course + r2 * next_course
                if course_out[i] > 360:
                    course_out[i] -= 360

                # Compute latitude and longitude
                lat_avg_speed = (speed[last_start + 1, 1] + speed[last_start, 1]) / 2
                lng_avg_speed = (speed[last_start + 1, 0] + speed[last_start, 0]) / 2
                lat_diff_speed = speed[last_start + 1, 1] - speed[last_start, 1]
                lng_diff_speed = speed[last_start + 1, 0] - speed[last_start, 0]
                total_time = ts[last_start + 1] - ts[last_start]
                lat_diff = latitude[last_start + 1] - latitude[last_start]
                lng_diff = longitude[last_start + 1] - longitude[last_start]

                lat_radius = 0
                lng_radius = 0
                if lat_diff > 0:
                    lat_radius = lat_avg_speed * total_time / lat_diff
                if lng_diff > 0:
                    lng_radius = lng_avg_speed * total_time / lng_diff

                delta_lat = 0
                delta_lng = 0
                if lat_radius > 0:
                    delta_lat = (speed[last_start, 1] * time1 + lat_diff_speed * time1 ** 2 / (2 * total_time)) / lat_radius
                if lng_radius > 0:
                    delta_lng = (speed[last_start, 0] * time1 + lng_diff_speed * time1 ** 2 / (2 * total_time)) / lng_radius

                new_lat = latitude[last_start] + delta_lat
                new_lng = longitude[last_start] + delta_lng
                gps_out[i, :] = new_lat, new_lng

                # Compute northing and easting
                northing_avg_speed = (speed[last_start + 1, 1] + speed[last_start, 1]) / 2
                easting_avg_speed = (speed[last_start + 1, 0] + speed[last_start, 0]) / 2
                northing_diff_speed = speed[last_start + 1, 1] - speed[last_start, 1]
                easting_diff_speed = speed[last_start + 1, 0] - speed[last_start, 0]
                total_time = ts[last_start + 1] - ts[last_start]
                northing_diff = northing[last_start + 1] - northing[last_start]
                easting_diff = easting[last_start + 1] - easting[last_start]

                northing_radius = 0
                easting_radius = 0
                if northing_diff > 0:
                    northing_radius = northing_avg_speed * total_time / northing_diff
                if easting_diff > 0:
                    easting_radius = easting_avg_speed * total_time / easting_diff

                delta_northing = 0
                delta_easting = 0
                if northing_radius > 0:
                    delta_northing = (speed[last_start, 1] * time1 + northing_diff_speed * time1 ** 2 / (
                            2 * total_time)) / northing_radius
                if easting_radius > 0:
                    delta_easting = (speed[last_start, 0] * time1 + easting_diff_speed * time1 ** 2 / (
                            2 * total_time)) / easting_radius

                new_northing = northing[last_start] + delta_northing
                new_easting = easting[last_start] + delta_easting
                pos_out[i, :] = new_northing, new_easting

        return tstamp_out, speed_out, gps_out, course_out, pos_out

    course = res['course']
    speed0 = res['speed']
    latitude = res['latitude']
    longitude = res['longitude']
    northing = res['northing']
    easting = res['easting']
    # First convert to speed vecs
    l = len(course)

    # Interpolate when the number of missing speed is small
    speed0 = fill_missing_speeds_and_courses(speed0, False)
    course = fill_missing_speeds_and_courses(course, True)
    latitude = fill_missing_gps(latitude, 90.0)
    longitude = fill_missing_gps(longitude, 180.0)
    if (speed0 is None) or (course is None) or (latitude is None) or (longitude is None):
        return None, None

    speed = np.zeros((l, 2), dtype=np.float32)
    for i in range(l):
        speed[i, :] = vec(speed0[i], course[i])

    tot_ms = res['endTime'] - res['startTime']
    # Get fixed frame rate stats
    # nout = tot_ms * HZ // 1000
    # tstamp_hz, speed_hz, gps_hz, course_hz = \
    #     interpolate(speed, latitude, longitude, course, res, nout, 1000.0 / HZ)

    # Get stats for each frame
    tstamp_orig, speed_orig, gps_orig, course_orig, pos_orig = \
        interpolate(speed, latitude, longitude, northing, easting, course, res, nr_frames, tot_ms / nr_frames)

    fixed = {
        # 'timestamp': tstamp_hz,
        # 'speed': speed_hz,
        # 'linear_speed': np.linalg.norm(speed_hz, axis=1),
        # 'course': course_hz,
        # 'gps': gps_hz
    }
    original = {
        'timestamp': tstamp_orig,
        'speed': speed_orig,
        'linear_speed': np.linalg.norm(speed_orig, axis=1),
        'course': course_orig,
        'gps': gps_orig,
        'pos': pos_orig
    }

    return fixed, original


def get_interpolated_sensors(json_path, video_filename, nr_frames):
    with open(json_path) as data_file:
        seg = json.load(data_file)

    # Generate data for a fixed frame rate and for every frame in the video
    fixed_data = {}
    original_data = {}

    # Get gps and speed values for each frame
    res = read_loc_json(seg, video_filename)
    if res is None:
        return None, None, -1
    fixed_loc, original_loc = get_interpolated_loc(res, nr_frames)
    if fixed_loc is None or original_loc is None:
        print("{} has bad localization".format(video_filename))
        return None, None, -1
    for key in original_loc:
        # fixed_data[key] = fixed_loc[key]
        original_data[key] = original_loc[key]

    # Get steer values for each frame
    res = read_steer_json(seg, video_filename)
    if res is None:
        return None, None, -1
    fixed_acc, original_acc = get_interpolated_steer(res, nr_frames)
    # fixed_data['accelerometer'] = fixed_acc
    original_data['steer'] = original_acc

    # Get accelerometer valus for each frame
    # res = read_acc_json(seg, video_filename)
    # if res is None:
    #     return None, None, -1
    # fixed_acc, original_acc = get_interpolated_acc(res, nr_frames)
    # # fixed_data['accelerometer'] = fixed_acc
    # original_data['accelerometer'] = original_acc

    # Get gyroscope values for each frame
    # res = read_gyro_json(seg, video_filename)
    # if res is None:
    #     return None, None, -1
    # fixed_gyro, original_gyro = get_interpolated_gyro(res, nr_frames)
    # # fixed_data['gyroscope'] = fixed_gyro
    # original_data['gyroscope'] = original_gyro

    return fixed_data, original_data, 0
