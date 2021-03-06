import os
import cv2
import math
import random
import numpy as np
import pandas as pd
import pickle as pkl
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

from utils import transformation

import matplotlib.pyplot as plt


def gaussian_distribution(x, mean, div):
    div **= 2
    ct_term = 1. / np.sqrt(2 * np.pi * div)
    distribution = ct_term * np.exp(-np.power(x - mean, 2) / (2 * div))
    #distribution /= distribution.sum()
    return distribution


class BDDVImageDataset(Dataset):
    def __init__(self,
                 cfg,
                 image_data,
                 tag_names,
                 image_width,
                 image_height,
                 train=True,
                 nr_bins=1,
                 transform=None,
                 augmentation=None):

        self.cfg = cfg
        self.image_data = image_data
        self.tag_names = tag_names
        self.train = train
        self.image_width = image_width
        self.image_height = image_height
        self.transform = transform
        self.augmentation = augmentation

        if nr_bins > 1:
            self._bins = np.arange(-1.0, 1.0, 2.0 / nr_bins)
            self._batch = self._clasification_batch

        # Get dataset length in terms of video frames and start frame for each video
        self.start_frames = []
        self.len = 0
        for key in self.image_data:
            self.start_frames.append(self.len)
            self.len += len(os.listdir(self.image_data[key][0]))
        self.start_frames.append(self.len)

#        for key in self.image_data:
#            nframes = len(os.listdir(self.image_data[key][0]))
#            info = pd.read_csv(self.image_data[key][1])
#            print(self.image_data[key][0].split('/')[-1])
#            for i in range(nframes):
#                frame = cv2.imread(os.path.join(self.image_data[key][0], '{}.jpg'.format(i)))
#                print(info['course'][i])
#                plt.imshow(frame)
#                plt.show()

    def __len__(self):
        return self.len

    def _normalize(self, img):
        return (img - 128.0) / 128.0

    def __getitem__(self, index):
        # Determine which video to extract the sequence of frames from
        video_file_index = self._get_video_index(index)
        frame_index = index - self.start_frames[video_file_index]

        # Fix frame number to prevent overflow
        if index + self.cfg.data_info.frame_seq_len > self.start_frames[video_file_index + 1]:
            frame_index -= index + self.cfg.data_info.frame_seq_len - self.start_frames[video_file_index + 1]

        # Get frames and frame info
        viditems = list(self.image_data.items())[video_file_index]
        video_dir = viditems[1][0]

        images = []
        target_vectors = []
        for i in range(0, self.cfg.data_info.frame_seq_len):
            # read data
            frame = cv2.imread(os.path.join(video_dir, 'frame' + str(frame_index + i) + '.jpg'))
            info = list(pd.read_csv(viditems[1][1]).iloc[frame_index, 1:])

            # add frame
            images.append(frame)
            target_vectors.append(info)

        # Augment data
        if self.train:
            for i in range(len(images)):
                course = np.rad2deg(target_vectors[i][self.tag_names.index("Steer Angle")])
                speed = target_vectors[i][self.tag_names.index('Linear Speed')]

                # augment image
                images[i], course = self.augmentation((images[i], course, speed))
                target_vectors[i][-1] = np.deg2rad(course)

                images[i] = self._normalize(images[i])
                images[i] = transformation.Crop.crop_center(images[i], down=0.4, up=0.1)
        else:
            for i in range(len(images)):
                images[i] = self._normalize(images[i])
                images[i] = transformation.Crop.crop_center(images[i], down=0.4, up=0.1)
        images = np.array(images)

        # Concatenate channels from several images
        images = images.reshape(self.image_height, self.image_width, 3 * self.cfg.data_info.frame_seq_len)
        return self._batch(images, target_vectors)

    def _get_video_index(self, bucket_index):
        '''Do a binary search over the start bucket indices to find which video
        the bucket_index belongs to'''
        l, r = 0, len(self.start_frames) - 1

        while l + 1 < r:
            m = (l + r) // 2
            if bucket_index >= self.start_frames[m]:
                l = m
            else:
                r = m

        return l

    def _clasification_batch(self, images, target_vectors):
        speed = np.array([t[self.tag_names.index('Linear Speed')] for t in target_vectors])

        def normalize_angle(x):
            x /= math.pi / 2
            x = min(1, max(-1, x))
            return x

        # Compute steer distribution as a gaussian
        # steer_cmd = np.array([t[self.tag_names.index('Steer')] for t in target_vectors])
        steer = np.array([normalize_angle(t[self.tag_names.index('Steer Angle')]) for t in target_vectors])
        steer_bin_no = np.digitize(steer, self._bins)
        steer_mean_bin = self._bins[steer_bin_no - 1] + 1.0 / len(self._bins)
        steer_distribution = gaussian_distribution(
            np.copy(self._bins) + 1.0 / len(self._bins), steer_mean_bin,
            self.cfg.dispersion)

        return np.transpose(images, (2, 0, 1)), speed, steer_distribution


class BDDVImageSampler(Sampler):
    def __init__(self, cfg, image_data, tag_names, seed, prob_weights):
        random.seed(seed)
        self.image_data = image_data
        self.tag_names = tag_names
        self.cfg = cfg

        # Get dataset length in terms of video frames and start frame for each video
        self.start_frames = []
        self.len = 0
        for key in self.image_data:
            self.start_frames.append(self.len)
            self.len += len(os.listdir(self.image_data[key][0]))
        self.start_frames.append(self.len)

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
        # added this while because samples_cmd[sample_type] could be empty
        while True:
            sample_type = random.choices(self._population, self._weights)[0]
            if self.samples_cmd[sample_type]:
                break
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
        for i in range(len(self.image_data)):
            # Get the info related to the video
            info_file = list(self.image_data.items())[i][1][1]
            info = pd.read_csv(info_file)
            # Distribute frames in buckets corresponding to commands
            for j in range(len(info)):
                cmd = info['steer'][j]
                self.samples_cmd[cmd].append(index)
                index += 1