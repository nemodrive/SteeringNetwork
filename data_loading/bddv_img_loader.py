from __future__ import print_function
import os
import glob
import time
import imgaug as ig
from imgaug import augmenters as iga

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from collections import OrderedDict

from .data_loader import DataLoaderBase
from .dataset import get_sampler


class BDDVImageAugmentation(object):
    def __init__(self, seed):
        st = lambda aug: iga.Sometimes(0.4, aug)
        oc = lambda aug: iga.Sometimes(0.3, aug)
        rl = lambda aug: iga.Sometimes(0.09, aug)
        self.seq = iga.Sequential(
            [
                rl(iga.GaussianBlur(
                    (0, 1.5))),  # blur images with a sigma between 0 and 1.5
                rl(
                    iga.AdditiveGaussianNoise(
                        loc=0, scale=(0.0, 0.05),
                        per_channel=0.5)),  # add gaussian noise to images
                oc(iga.Dropout((0.0, 0.10), per_channel=0.5)
                   ),  # randomly remove up to X% of the pixels
                oc(
                    iga.CoarseDropout(
                        (0.0, 0.10), size_percent=(0.08, 0.2), per_channel=0.5)
                ),  # randomly remove up to X% of the pixels
                oc(
                    iga.Add((-40, 40), per_channel=0.5)
                ),  # change brightness of images (by -X to Y of original value)
                st(iga.Multiply((0.10, 2.5), per_channel=0.2)
                   ),  # change brightness of images (X-Y% of original value)
                rl(iga.ContrastNormalization(
                    (0.5, 1.5),
                    per_channel=0.5)),  # improve or worsen the contrast
            ],
            random_order=True)

        ig.seed(seed)

    def __call__(self, image):
        return self.seq.augment_image(image)


class BDDVImageLoader(DataLoaderBase):
    def __init__(self, cfg):
        super(BDDVImageLoader, self).__init__(cfg)
        self.channel, self.image_height, self.image_width = cfg.data_info.image_shape
        self.transformations = self.get_transformations()
        self.augmentation = self.get_augmentation()
        self._info_path = cfg.dataset.info_path
        self._info_path_test = cfg.dataset.info_eval_path
        self._sampler = get_sampler(self._dataset_cfg.sampler)
        self._nr_bins = cfg.model.nr_bins
        self.load_data()

    @staticmethod
    def get_transformations():
        return transforms.Compose(
            [transforms.Lambda(lambda x: x / 127.5 - 1.0)])

    def get_augmentation(self):
        return transforms.Compose(
            [BDDVImageAugmentation(self._seed)])

    def load_data(self):
        data_train = OrderedDict()
        video_train = os.listdir(self._dataset_path)
        for vid in video_train:
            vidname = vid
            vid = os.path.join(self._dataset_path, vid)
            info = os.path.join(self._info_path, vidname + '.csv')
            data_train[vidname] = (vid, info)
        data_eval = OrderedDict()
        video_eval = os.listdir(self._dataset_path_test)
        for vid in video_eval:
            vidname = vid
            vid = os.path.join(self._dataset_path_test, vid)
            info = os.path.join(self._info_path_test, vidname + '.csv')
            data_eval[vidname] = (vid, info)
        tag_names = [
            'Acceleration x', 'Acceleration y', 'Acceleration z', 'Course', 'Gps Lat',
            'Gps Long', 'Gyro x', 'Gyro y', 'Gyro z', 'Linear Speed', 'Speed x',
            'Speed y', 'Timestamp', 'Steer', 'Steer Angle'
        ]

        self.data = {
            'train': {
                'data': data_train,
                'tags': tag_names
            },
            'test': {
                'data': data_eval,
                'tags': tag_names
            }
        }

    def get_train_loader(self):
        train_data = self.data['train']
        train_dataset = self._dataset(
            self._dataset_cfg,
            train_data['data'],
            train_data['tags'],
            self.image_width,
            self.image_height,
            nr_bins=self._nr_bins,
            train=True,
            transform=self.transformations,
            augmentation=self.augmentation)

        sampler = self._sampler(
            self._dataset_cfg,
            train_data['data'],
            train_data['tags'],
            self._seed,
            self._dataset_cfg.sampler.weights)

        return DataLoader(
            train_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=self._no_workers)

    def get_test_loader(self):
        test_data = self.data['test']
        test_dataset = self._dataset(
            self._dataset_cfg,
            test_data['data'],
            test_data['tags'],
            self.image_width,
            self.image_height,
            nr_bins=self._nr_bins,
            train=False,
            transform=self.transformations)

        return DataLoader(
            test_dataset,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            num_workers=self._no_workers)
