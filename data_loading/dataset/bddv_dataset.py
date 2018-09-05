import numpy as np
import pandas as pd
import pickle as pkl
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

from dataset_helper import DatasetHelper
import random


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
        vidsize = self.cfg.data_info.image_shape
        vidsize = (vidsize[1], vidsize[2], vidsize[0])
        images, target_vectors = None, None
        no_fails = -1
        while images is None or target_vectors is None:
            no_fails += 1
            images, target_vectors = self.helper.get_data(video_file_index,
                bucket_index, self.cfg.data_info.frame_seq_len, vidsize)
            if no_fails == 5:
                print("Failed too many times to retrieve a frame")
                return None

        if self.train:
            for i in range(len(images)):
                images[i] = self.augmentation(images[i])

        if self.transform is not None:
            for i in range(len(images)):
                images[i] = self.transform(images[i])

        # Concatenate channels from several images
        images = images.reshape(images.shape[1], images.shape[2],
                images.shape[3] * self.cfg.data_info.frame_seq_len)

        cmd_signals = [t[self.tag_names.index('Control Signal')] for t in target_vectors]
        cmd_signals = np.array(cmd_signals, dtype=np.int)

        return self._batch(images, target_vectors, cmd_signals)

    def _clasification_batch(self, images, target_vectors, cmd_signals):
        # control_cmds = []

        # control_cmds.append(self.tag_names.index('Gas'))
        # control_cmds.append(self.tag_names.index('Brake'))
        # target_vect = target_vector[control_cmds]

        speed = np.array([t[self.tag_names.index('Linear Speed')] for t in target_vectors])

        # Compute steer distribution as a gaussian
        steer = np.array([t[self.tag_names.index('Control Signal')] for t in target_vectors])
        steer_bin_no = np.digitize(steer, self._bins)
        steer_mean_bin = self._bins[steer_bin_no - 1] + 1.0 / len(self._bins)

        steer_distribution = gaussian_distribution(
            np.copy(self._bins) + 1.0 / len(self._bins), steer_mean_bin,
            self.cfg.dispersion)

        return np.transpose(images, (2, 0, 1)), speed / 80.0, \
                steer_distribution, cmd_signals

    def _regression_batch(self, images, target_vectors, cmd_signals):
        control_cmds = []

        control_cmds.append(self.tag_names.index('Control Signal'))
        # control_cmds.append(self.tag_names.index('Gas'))
        # control_cmds.append(self.tag_names.index('Brake'))
        target_vect = np.array([t[control_cmds] for t in target_vectors])

        speed = np.array([t[self.tag_names.index('Linear Speed')] for t in target_vectors])

        return np.transpose(image, (2, 0, 1)), speed / 90.0, target_vect, \
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
