import cv2
import math
import numpy as np
import pandas as pd
import pickle as pkl
from collections import OrderedDict
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

from ..image_tools import *
import random

NR_IMAGES_PER_FILE = 200


def gaussian_distribution(x, mean, div):
    div **= 2
    ct_term = 1. / np.sqrt(2 * np.pi * div)
    return ct_term * np.exp(-np.power(x - mean, 2) / (2 * div))


class BDDVDataset(Dataset):
    def __init__(self,
                 cfg,
                 video_data,
                 tag_names,
                 image_width,
                 image_height,
                 train=True,
                 nr_bins=1,
                 transform=None,
                 augmentation=None):

        self.cfg = cfg
        self.video_data = video_data
        self.tag_names = tag_names
        self.train = train
        self.image_width = image_width
        self.image_height = image_height
        self.transform = transform
        self.augmentation = augmentation
        self._batch = self._regression_batch

        if nr_bins > 1:
            self._bins = np.arange(-1.0, 1.0, 2.0 / nr_bins)
            self._batch = self._clasification_batch

        # Read video metadata
        with open(cfg.video_metadata_path, 'rb') as f:
            self.video_metadata = pkl.load(f)

        # Get dataset length in terms of video buckets and start buckets for each video
        self.start_buckets = []
        self.len = 0
        for key in self.video_data:
            self.start_buckets.append(self.len)
            vid_duration_ms = self.video_metadata[key]['duration'] * 1000
            interval = cfg.video_bucket_ms
            n_bucks = int(vid_duration_ms / interval)
            self.len += n_bucks
        self.start_buckets.append(self.len)

        self.helper = DatasetHelper(self.video_data, self.start_buckets, self.video_metadata, cfg)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        # Determine which video to extract the sequence of frames from
        video_file_index = self.helper.get_video_index(index)
        bucket_index = index - self.start_buckets[video_file_index]
        images, target_vectors = self.helper.get_data(video_file_index,
            bucket_index, self.cfg.frame_seq_len)

        if self.train:
            for i in range(len(images)):
                images[i] = self.augmentation(images[i])

        if self.transform is not None:
            for i in range(len(images)):
                images[i] = self.transform(images[i])

        cmd_signals = [t[self.tag_names.index('Control Signal')] for t in target_vectors]
        cmd_signals = [x if x > 2 else 2 for x in cmd_signals]

        return self._batch(images, target_vectors, cmd_signals)

    def _clasification_batch(self, images, target_vectors, cmd_signals):
        # control_cmds = []

        # control_cmds.append(self.tag_names.index('Gas'))
        # control_cmds.append(self.tag_names.index('Brake'))
        # target_vect = target_vector[control_cmds]

        speed = np.array([t[self.tag_names.index('Linear Speed')] for t in target_vectors])

        # Compute steer distribution as a gaussian
        steer = np.array([t[self.tag_names.index('Steer')] for t in target_vectors])
        steer_bin_no = np.digitize(steer, self._bins)
        steer_mean_bin = self._bins[steer_bin_no - 1] + 1.0 / len(self._bins)

        steer_distribution = gaussian_distribution(
            np.copy(self._bins) + 1.0 / len(self._bins), steer_mean_bin,
            self.cfg.dispersion)

        # imgs = [np.transpose(img, (2, 0, 1)) for img in images]
        return np.transpose(images, (0, 3, 1, 2)), speed / 90.0, \
                steer_distribution, cmd_signals

    def _regression_batch(self, images, target_vectors, cmd_signals):
        control_cmds = []

        control_cmds.append(self.tag_names.index('Steer'))
        # control_cmds.append(self.tag_names.index('Gas'))
        # control_cmds.append(self.tag_names.index('Brake'))
        target_vect = np.array([t[control_cmds] for t in target_vectors])

        speed = np.array([t[self.tag_names.index('Speed')] for t in target_vectors])

        return np.transpose(image, (0, 3, 1, 2)), speed / 90.0, target_vect, \
                cmd_signals


class BDDVSampler(Sampler):
    def __init__(self, cfg, video_data, tag_names, seed, prob_weights):
        random.seed(seed)
        self.video_data = video_data
        self.tag_names = tag_names
        self.cfg = cfg

        # Read video metadata
        with open(cfg.video_metadata_path, 'rb') as f:
            self.video_metadata = pkl.load(f)

        # Get dataset length in terms of video buckets and start buckets for each video
        self.start_buckets = []
        self.len = 0
        for key in self.video_data:
            self.start_buckets.append(self.len)
            vid_duration_ms = self.video_metadata[key]['duration'] * 1000
            interval = cfg.video_bucket_ms
            n_bucks = int(vid_duration_ms / interval)
            self.len += n_bucks
        self.start_buckets.append(self.len)

        self.helper = DatasetHelper(self.video_data, self.start_buckets, self.video_metadata, None)

        self.seen = 0
        self.samples_cmd = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
        self.samples_idx = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        self._population = [0, 1, 2, 3, 4, 5]
        self._weights = prob_weights
        self._split_samples()

        for key in self.samples_cmd.keys():
            random.shuffle(self.samples_cmd[key])

    def __len__(self):
        return self.len

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        sample_type = random.choices(self._population, self._weights)[0]
        idx = self.samples_cmd[sample_type][self.samples_idx[sample_type]]

        self.samples_idx[sample_type] += 1
        if self.samples_idx[sample_type] >= len(self.samples_cmd[sample_type]):
            self.samples_idx[sample_type] = 0
            random.shuffle(self.samples_cmd[sample_type])

        self.seen += 1
        if self.seen >= self.len:
            for key in self.samples_cmd.keys():
                random.shuffle(self.samples_cmd[key])
                self.samples_idx[key] = 0
            self.seen = 0
            raise StopIteration

        return idx

    def _split_samples(self):
        index = 0
        for i in range(len(self.video_data)):
            # Get the info related to the video
            info_file = list(self.video_data.items())[i][1][1]
            info = pd.read_csv(info_file)
            # Determine the number of frames in a bucket
            key = info_file.split('.')[0].split('/')[-1]
            vid_duration_ms = self.video_metadata[key]['duration'] * 1000
            interval = self.cfg.video_bucket_ms
            n_bucks = int(vid_duration_ms / interval)
            frames_per_buck = len(info) // n_bucks
            # For each bucket, set the steering command for the bucket as the
            # most common command for the frames in the bucket
            for j in range(0, n_bucks * frames_per_buck, frames_per_buck):
                cmds = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0}
                for _ in range(j, j + frames_per_buck):
                    cmds[info['steer'][j]] += 1
                max_cmd = 0
                for k in range(1, 6):
                    if cmds[max_cmd] < cmds[k]:
                        max_cmd = k
                self.samples_cmd[max_cmd].append(index)
                index += 1


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

    def get_data(self, vid_index, bucket_index, nr_frames):
        video_items = list(self.video_data.items())
        vid_name = video_items[vid_index][0]
        nr_bucks = self.start_buckets[vid_index + 1] - self.start_buckets[vid_index]
        img_per_buck = self.video_metadata[vid_name]['nframes'] // nr_bucks

        # Fix bucket start position to be able to extract nr_frames
        if bucket_index + nr_frames > self.start_buckets[vid_index + 1]:
            bucket_index = self.start_buckets[vid_index + 1] - nr_frames

        # Choose images at random from buckets
        img_indices = []
        for i in range(nr_frames):
            img_index = (bucket_index + i) * img_per_buck + random.randint(0, img_per_buck)
            img_indices.append(img_index)

        # Open video and iterate to the desired frames
        images = []
        vid = cv2.VideoCapture(video_items[vid_index][1][0])
        cnt = 0
        for ind in img_indices:
            while cnt <= ind:
                cnt += 1
                ret, img = vid.read()
                if not ret:
                    print("Request for bad frame")
                    return None, None
            images.append(img)

        # Get the steering data using the csv info file
        steer_dist = self.cfg.steer_dist
        info_file = video_items[vid_index][1][1]
        df = pd.read_csv(info_file)
        s = df['linear_speed']
        sx = df['speed_x']
        sy = df['speed_y']

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
        steer_angles, steer_cmds = self._get_steer(steer_indices, sx, sy, s, self.cfg)

        # Construct the target vectors
        target_vectors = []
        for i in range(len(images)):
            target_vector = list(df.iloc[img_indices[i]])
            target_vector.append(steer_angles[i])
            target_vector.append(steer_cmds[i])
            target_vectors.append(target_vector)

        # Convert outputs to numpy arrays
        images = np.array(images, dtype=np.float)
        target_vectors = np.array(target_vectors)

        return images, target_vectors

    def _get_angle(self, sx, sy):
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

        assert not math.isnan(course)
        return course

    def _get_steer(self, steer_indices, sx, sy, s, cfg):
        '''Determine the steer angle and the steer command'''
        angles = np.zeros(len(steer_indices))
        cmds = np.zeros(len(steer_indices))
        enum = self.turn_str2int

        for i in range(len(steer_indices)):
            next_frame, curr_frame = steer_indices[i]
            if next_frame == 0:
                if i == 0:
                    break
                else:
                    steer_indices[i, 0] = steer_indices[i - 1, 0]

            # Check if the current state is stop
            if s[curr_frame] < cfg.speed_limit_as_stop:
                angles[i] = 0
                cmds[i] = enum['slow_or_stop']
                continue

            # Check if the next state is stop
            if s[next_frame] < cfg.speed_limit_as_stop:
                angles[i] = 0
                cmds[i] = enum['slow_or_stop']
                continue

            # Angle thresholds
            thresh_low = (2 * math.pi / 360) * 2
            thresh_high = (2 * math.pi / 360) * 180
            thresh_slight_low = (2 * math.pi / 360) * 5

            # Compute the angle between the velocity vector of the current
            # frame and the next frame
            curr_sx = sx[curr_frame]
            curr_sy = sy[curr_frame]
            next_sx = sx[next_frame]
            next_sy = sy[next_frame]

            curr_course = self._get_angle(curr_sx, curr_sy)
            next_course = self._get_angle(next_sx, next_sy)
            angles[i] = next_course - curr_course

            # Decide on the type of action
            if s[next_frame] - s[curr_frame] < cfg.deceleration_thresh:
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

        return angles, cmds
