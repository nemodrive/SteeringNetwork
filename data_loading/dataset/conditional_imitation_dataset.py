import h5py
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

from ..image_tools import *
import random

NR_IMAGES_PER_FILE = 200


def gaussian_distribution(x, mean, div):
    div **= 2
    ct_term = 1. / np.sqrt(2 * np.pi * div)
    return ct_term * np.exp(-np.power(x - mean, 2) / (2 * div))


class ConditionalImitationLearningDataset(Dataset):
    def __init__(self,
                 cfg,
                 image_files,
                 tag_names,
                 image_width,
                 image_height,
                 train=True,
                 nr_bins=1,
                 transform=None,
                 augmentation=None):

        self.cfg = cfg
        self.image_files = image_files
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

    def __len__(self):
        return NR_IMAGES_PER_FILE * len(self.image_files)

    def __getitem__(self, index):

        image_file_index = index // 200
        image_index = index % 200
        try:
            data = h5py.File(self.image_files[image_file_index], 'r')
            image = data['rgb'][image_index]
            target_vector = data['targets'][image_index]
        except:
            print(image_file_index, self.image_files[image_file_index])

        if self.train:
            image = self.augmentation(image)

        if self.transform is not None:
            image = self.transform(image)

        cmd_signal = target_vector[self.tag_names.index("Control signal")]
        if int(cmd_signal) < 2:
            cmd_signal = 2.00

        return self._batch(image, target_vector, cmd_signal)

    '''
    def _clasification_batch(self, image, target_vector, cmd_signal):

        control_cmds = []

        control_cmds.append(self.tag_names.index('Steer'))
        control_cmds.append(self.tag_names.index('Gas'))
        control_cmds.append(self.tag_names.index('Brake'))

        speed = target_vector[self.tag_names.index("Speed")]

        target_vect = target_vector[control_cmds]
        target_vect[0] = np.digitize(target_vect[0], self._bins) - 1
        if target_vect[0] < 0:
            target_vect[0] = 0
        if target_vect[0] > 179:
            target_vect[0] = 179

        return np.transpose(
            image, (2, 0, 1)), speed / 90.0, target_vect, cmd_signal
    '''

    def _clasification_batch(self, image, target_vector, cmd_signal):

        control_cmds = []

        control_cmds.append(self.tag_names.index('Gas'))
        control_cmds.append(self.tag_names.index('Brake'))
        target_vect = target_vector[control_cmds]

        speed = target_vector[self.tag_names.index("Speed")]

        #compute steer distribution as a gaussian
        steer = target_vector[self.tag_names.index('Steer')]
        steer_bin_no = np.digitize(steer, self._bins)
        steer_mean_bin = self._bins[steer_bin_no - 1] + 1.0 / len(self._bins)

        steer_distribution = gaussian_distribution(
            np.copy(self._bins) + 1.0 / len(self._bins), steer_mean_bin,
            self.cfg.dispersion)

        return np.transpose(
            image,
            (2, 0,
             1)), speed / 90.0, steer_distribution, target_vect, cmd_signal

    def _regression_batch(self, image, target_vector, cmd_signal):

        control_cmds = []

        control_cmds.append(self.tag_names.index('Steer'))
        control_cmds.append(self.tag_names.index('Gas'))
        control_cmds.append(self.tag_names.index('Brake'))

        speed = target_vector[self.tag_names.index("Speed")]

        return np.transpose(
            image,
            (2, 0, 1)), speed / 90.0, target_vector[control_cmds], cmd_signal


class ConditionalImitationLearningSampler(Sampler):
    def __init__(self, image_files, tag_names, seed, prob_weights):

        random.seed(seed)

        self.image_files = image_files
        self.tag_names = tag_names
        self.length = NR_IMAGES_PER_FILE * len(self.image_files)
        self.seen = 0
        self.samples_cmd = {2: [], 3: [], 4: [], 5: []}
        self.samples_idx = {2: 0, 3: 0, 4: 0, 5: 0}
        self._population = [2, 3, 4, 5]
        self._weights = prob_weights
        #self._weights = [0.23, 0.298, 0.242, 0.23]
        #self._weights = [0.283, 0.234, 0.241, 0.242]
        self._split_samples()
        for key in self.samples_cmd.keys():
            random.shuffle(self.samples_cmd[key])

    def __len__(self):
        return self.length

    def __iter__(self):
        return self

    def next(sel):
        return self.__next__()

    def __next__(self):

        sample_type = random.choices(self._population, self._weights)[0]
        idx = self.samples_cmd[sample_type][self.samples_idx[sample_type]]

        self.samples_idx[sample_type] += 1
        if self.samples_idx[sample_type] >= len(self.samples_cmd[sample_type]):
            self.samples_idx[sample_type] = 0
            random.shuffle(self.samples_cmd[sample_type])

        self.seen += 1
        if self.seen >= self.length:
            for key in self.samples_cmd.keys():
                random.shuffle(self.samples_cmd[key])
                self.samples_idx[key] = 0
            self.seen = 0
            raise StopIteration

        return idx

    def _split_samples(self):

        for idx_file in range(len(self.image_files)):
            try:
                data = h5py.File(self.image_files[idx_file], 'r')
                target_vector = data['targets']
            except:
                print(image_file_index, self.image_files[image_file_index])
                raise NameError

            for img_nr in range(NR_IMAGES_PER_FILE):

                cmd = target_vector[img_nr][self.tag_names.index(
                    "Control signal")]
                if int(cmd) < 2:
                    cmd = 2.0
                self.samples_cmd[int(cmd)].append(idx_file * 200 + img_nr)
