from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from .data_loader import DataLoaderBase
from .image_tools import *
import json


class LargeScaleE2ELoader(DataLoaderBase):
    def __init__(self, cfg):
        super(LargeScaleE2ELoader, self).__init__(cfg)

        self.videos_subpath = self._dataset_cfg.dataset_video_subpath
        self.channel, self.image_width, self.image_height = cfg.data_info.image_shape
        self.transformations = self.get_transformations()
        self.load_data()

    @staticmethod
    def get_transformations():
        return transforms.Compose([transforms.Resize((640, 360))])

    def get_all_files_from(self, videos_path):
        from os import walk
        video_filenames = []
        for (dirpath, dirnames, filename) in walk(videos_path):
            video_filenames.extend(filename)
        video_filenames = [x.split(".")[0] for x in video_filenames]
        return video_filenames


    def load_data_for_steering2(self):
        cfg = self._dataset_cfg

        if cfg.use_preprocessed:
            interpolated_speeds = json.load(open(self._dataset_path + cfg.dataset_preprocessed + cfg.dataset_preprocessed_speeds, "r"))
            videos_info = json.load(open(self._dataset_path + cfg.dataset_preprocessed + cfg.dataset_preprocessed_videoinfo, "r"))
        else:
            from data_loading.bdd_dataset_helper.store_dataset_info import get_preprocessed_info
            from data_loading.bdd_dataset_helper.store_dataset_info import get_interpolated_info
            from data_loading.bdd_dataset_helper.store_dataset_info import save_preprocessed_info2
            from data_loading.bdd_dataset_helper.store_dataset_info import save_interpolated_info2

            interpolated_speeds = get_interpolated_info(self._dataset_path)
            save_path = self._dataset_path + cfg.dataset_preprocessed + cfg.dataset_preprocessed_speeds
            save_interpolated_info2(interpolated_speeds, save_path)
            videos_info = get_preprocessed_info(self._dataset_path)
            save_path = self._dataset_path + cfg.dataset_preprocessed + cfg.dataset_preprocessed_videoinfo
            save_preprocessed_info2(videos_info, save_path)

        x = list(videos_info.keys())
        y = [interpolated_speeds[xx] for xx in x]

        train_data, test_data = self.split_data(x, y)

        train_videos_info = {}
        train_videos_speeds = {}
        for key in train_data[0]:
            train_videos_info[key] = videos_info[key]
            train_videos_speeds[key] = interpolated_speeds[key]

        test_videos_info = {}
        test_videos_speeds = {}
        for key in test_data[0]:
            test_videos_info[key] = videos_info[key]
            test_videos_speeds[key] = interpolated_speeds[key]

        self.data = ((train_videos_info, train_videos_speeds), (test_videos_info, test_videos_speeds))


    def load_data_for_steering(self):
        self.preprocessed = os.path.exists(self._dataset_path + "/intermediary/interpolated.json")

        interpolated_speeds = {}
        if self.preprocessed:
            interpolated_speeds = json.load(open(self._dataset_path + "/intermediary/interpolated.json", "r"))

        videos_info = json.load(open(self._dataset_path + "/intermediary/video_infos.json", "r"))

        x = list(videos_info.keys())
        y = [interpolated_speeds[xx] for xx in x]

        #x, y, videos_info, interpolated_speeds =


        train_data, test_data = self.split_data(x, y)

        train_videos_info = {}
        train_videos_speeds = {}
        for key in train_data[0]:
            train_videos_info[key] = videos_info[key]
            train_videos_speeds[key] = interpolated_speeds[key]

        test_videos_info = {}
        test_videos_speeds = {}
        for key in test_data[0]:
            test_videos_info[key] = videos_info[key]
            test_videos_speeds[key] = interpolated_speeds[key]

        self.data = ((train_videos_info, train_videos_speeds), (test_videos_info, test_videos_speeds))


    def load_data_for_object_detection(self):
        train_images_path = "/images/100k/train/"
        train_labels_path = "/labels/100k/train/"

        train_images_paths = []
        train_labels_paths = []

        for image_name in self.get_all_files_from(self._dataset_path + train_images_path):
            train_images_paths.append(train_images_path + image_name + ".jpg")
            train_labels_paths.append(train_labels_path + image_name + ".json")

        valid_images_path = "/images/100k/val/"
        valid_labels_path = "/labels/100k/val/"

        valid_images_paths = []
        valid_labels_paths = []

        for image_name in self.get_all_files_from(valid_images_path):
            valid_images_paths.append(valid_images_path + image_name + ".jpg")
            valid_labels_paths.append(valid_labels_path + image_name + ".json")

        self.data = ((train_images_paths, train_labels_paths), (valid_images_paths, valid_labels_paths))

    def load_data(self):
        self.load_data_for_steering2()
        #self.load_data_for_steering()

    def get_train_loader(self):
        train_dataset = self._dataset(
            self._dataset_cfg,
            self.data[0][0],
            self.data[0][1],
            self._dataset_path + self.videos_subpath,
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
            self._dataset_path + self.videos_subpath,
            self.image_width,
            self.image_height,
            train=False,
            transform=self.transformations)

        return DataLoader(
            test_dataset,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            num_workers=self._no_workers)