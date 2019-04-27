import pandas as pd
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from .image_tools import *
from .data_loader import DataLoaderBase


class UdacityCloningDataLoader(DataLoaderBase):
    def __init__(self, cfg):
        super(UdacityCloningDataLoader, self).__init__(cfg)

        self.channel, self.image_width, self.image_height = cfg.data_info.image_shape
        self.transformations = self.get_transformations()

        self.load_data()

    @staticmethod
    def get_transformations():
        return transforms.Compose([transforms.Lambda(lambda x: x / 127.5 - 1)])

    def load_data(self):
        data_df = pd.read_csv(
            os.path.join(self._dataset_path, 'interpolated.csv'))
        data_df = data_df[['timestamp', 'frame_id', 'filename', 'angle']]

        data_np = data_df.values
        data_np_processed = np.empty(
            (int(data_np.shape[0] / 3), 5), dtype=object)

        for i in range(int(data_np.shape[0] / 3)):
            timestamp = data_np[3 * i, 0]
            camera_center, camera_left, camera_right = data_np[3 * i:3 * i + 3,
                                                               2]
            steering = np.mean(data_np[3 * i:3 * i + 3, 3])
            data_np_processed[i] = np.asarray(
                [
                    timestamp, camera_left, camera_right, camera_center,
                    steering
                ],
                dtype=object)

        x = data_np_processed[:, :-1]
        y = data_np_processed[:, -1]

        train_data, test_data = self.split_data(x, y)
        self.data = (train_data, test_data)

    def get_train_loader(self):
        train_dataset = self._dataset(
            self._dataset_cfg,
            self.data[0][0],
            self.data[0][1],
            self._dataset_path,
            self.image_width,
            self.image_height,
            train=True,
            transform=self.transformations)

        return DataLoader(
            train_dataset,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            num_workers=self._no_workers)

    def get_test_loader(self):
        test_dataset = self._dataset(
            self._dataset_cfg,
            self.data[1][0],
            self.data[1][1],
            self._dataset_path,
            self.image_width,
            self.image_height,
            train=False,
            transform=self.transformations)

        return DataLoader(
            test_dataset,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            num_workers=self._no_workers)
