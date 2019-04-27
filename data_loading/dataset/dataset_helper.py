import math
import random
import subprocess
import numpy as np
import pandas as pd


class DatasetHelper(object):
    def __init__(self, video_data, start_buckets, video_metadata, cfg):
        self.video_data = video_data
        self.start_buckets = start_buckets
        self.video_metadata = video_metadata
        self.cfg = cfg
        self.turn_str2int = {
            'straight': 0,
            'slow_or_stop': 1,
            'turn_left': 2,
            'turn_right': 3,
            'turn_left_slight': 4,
            'turn_right_slight': 5,
        }

    def get_video_index(self, bucket_index):
        '''Do a binary search over the start bucket indices to find which video
        the bucket_index belongs to'''
        l, r = 0, len(self.start_buckets) - 1

        while l + 1 < r:
            m = (l + r) // 2
            if bucket_index >= self.start_buckets[m]:
                l = m
            else:
                r = m

        return l

    def get_data(self, vid_index, bucket_index, nr_frames, frame_size):
        video_items = list(self.video_data.items())
        vid_name = video_items[vid_index][0]
        meta = self.video_metadata[vid_name]
        nr_bucks = self.start_buckets[vid_index + 1] - self.start_buckets[vid_index]
        img_per_buck = meta['nframes'] // nr_bucks

        # Fix bucket start position to be able to extract nr_frames
        if bucket_index + nr_frames > self.start_buckets[vid_index + 1]:
            bucket_index = self.start_buckets[vid_index + 1] - nr_frames

        # Choose images at random from buckets
        img_indices = []
        for i in range(nr_frames):
            img_index = (bucket_index + i) * img_per_buck + random.randint(0, img_per_buck - 1)
            img_indices.append(img_index)

        # Open video and iterate to the desired frames
        images = []
        fps = meta['nframes'] / meta['duration']
        for ind in img_indices:
            cmd = ['ffmpeg',
                   '-loglevel', 'fatal',
                   '-ss', str(ind / fps),
                   '-i', video_items[vid_index][1][0],
                   '-vframes', '1',
                   '-f', 'image2pipe',
                   '-pix_fmt', 'bgr24',
                   '-vcodec', 'rawvideo', '-']

            ffmpeg = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            out, err = ffmpeg.communicate()

            if err or len(out) == 0:
                print('Bad frame request')
                return None, None

            img = np.fromstring(out, dtype='uint8').reshape(frame_size)
            images.append(img)

        # Get the steering data using the csv info file
        steer_dist = self.cfg.steer_dist
        info_file = video_items[vid_index][1][1]
        df = pd.read_csv(info_file)
        s = df['linear_speed']
        course = df['course']

        # Determine the indices of the frames we use for computing the steering
        dist = np.zeros(len(s))
        time_unit = self.video_metadata[vid_name]['duration'] / self.video_metadata[vid_name]['nframes']
        curr_ind = img_indices[0] + 1
        while curr_ind < len(s):
            if curr_ind > img_indices[-1] and dist[curr_ind] - dist[img_indices[-1]] > steer_dist:
                break
            dist[curr_ind] = dist[curr_ind - 1] + s[curr_ind - 1] * time_unit
            curr_ind += 1

        steer_indices = np.zeros((len(img_indices), 2))
        steer_indices[:, 1] = img_indices
        moving_ind = img_indices[0] + 1
        steer_ind = 0
        while steer_ind < len(img_indices) and moving_ind < len(s):
            if dist[moving_ind] - dist[img_indices[steer_ind]] >= steer_dist:
                steer_indices[steer_ind, 0] = moving_ind
                steer_ind += 1
            moving_ind += 1

        # Determine the steering for each video frame selected
        steer_angles, steer_cmds = self.get_steer(steer_indices, course, s)

        # Construct the target vectors
        target_vectors = []
        for i in range(len(images)):
            target_vector = list(df.iloc[img_indices[i]][1:])
            target_vector.append(steer_angles[i])
            target_vector.append(steer_cmds[i])
            target_vectors.append(target_vector)

        # Convert outputs to numpy arrays
        images = np.array(images, dtype=np.float)
        target_vectors = np.array(target_vectors)

        return images, target_vectors

    def get_steer(self, steer_indices, course, s):
        '''Determine the steer angle and the steer command'''
        angles = np.zeros(len(steer_indices))
        cmds = np.zeros(len(steer_indices))
        enum = self.turn_str2int

        curr_course = 0
        next_course = 0
        for i in range(len(steer_indices)):
            next_frame, curr_frame = steer_indices[i]
            if next_frame == 0:
                if i == 0:
                    break
                else:
                    steer_indices[i, 0] = steer_indices[i - 1, 0]

            # Check if the current state is stop
            if s[curr_frame] < self.cfg.speed_limit_as_stop:
                angles[i] = 0
                cmds[i] = enum['slow_or_stop']
                continue

            # Check if the next state is stop
            if s[next_frame] < self.cfg.speed_limit_as_stop:
                angles[i] = 0
                cmds[i] = enum['slow_or_stop']
                continue

            # Angle thresholds
            thresh_low = (2 * math.pi / 360) * 2
            thresh_high = (2 * math.pi / 360) * 180
            thresh_slight_low = (2 * math.pi / 360) * 5

            # Compute the angle between the velocity vector of the current
            # frame and the next frame
            curr_course = course[curr_frame]
            next_course = course[next_frame]
            angles[i] = next_course - curr_course

            # Decide on the type of action
            if s[next_frame] - s[curr_frame] < self.cfg.deceleration_thresh:
                cmds[i] = enum['slow_or_stop']
            elif thresh_low < angles[i] < thresh_high:
                if thresh_slight_low < angles[i]:
                    cmds[i] = enum['turn_right']
                else:
                    cmds[i] = enum['turn_right_slight']
            elif -thresh_high < angles[i] < -thresh_low:
                if angles[i] < -thresh_slight_low:
                    cmds[i] = enum['turn_left']
                else:
                    cmds[i] = enum['turn_left_slight']
            elif angles[i] < -thresh_high or thresh_high < angles[i]:
                cmds[i] = enum['slow_or_stop']
            else:
                cmds[i] = enum['straight']

        i += 1
        # Fill the remaining commands with the same as the last one
        if i < len(steer_indices):
            angle_unit = (next_course - curr_course) / (len(steer_indices) - i)
            for j in range(i, len(steer_indices)):
                angles[j] = angles[j - 1] - angle_unit
                cmds[j] = cmds[j - 1]

        return angles, cmds
