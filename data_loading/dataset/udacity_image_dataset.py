from torch.utils.data import Dataset

from ..image_tools import *


class UdacityImageSteerDataset(Dataset):
    def __init__(self, cfg, x, y, data_dir, image_width, image_height, train=True, transform=None):
        self.cfg = cfg
        self.X = x
        self.y = y
        self.train = train
        self.image_width = image_width
        self.image_height = image_height
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        center, left, right = self.X[index, 1:4]
        steering_angle = self.y[index]

        # if testing, only use the center image and its steering angle
        if self.train:
            # TODO 0.1 very questionable
            image_left, steering_angle_left = augment_img(load_image(self.data_dir, left),
                                                          steering_angle + 0.1,
                                                          self.image_width, self.image_height)

            image_center, steering_angle_center = augment_img(load_image(self.data_dir, center),
                                                              steering_angle,
                                                              self.image_width, self.image_height)

            image_right, steering_angle_right = augment_img(load_image(self.data_dir, right),
                                                            steering_angle - 0.1,
                                                            self.image_width, self.image_height)

            if self.transform is not None:
                image_left = self.transform(image_left)
                image_center = self.transform(image_center)
                image_right = self.transform(image_right)

            return (image_center, steering_angle_center),\
                   (image_left, steering_angle_left),\
                   (image_right, steering_angle_right)

        image = load_image(self.data_dir, center)
        image = preprocess(image, self.image_width, self.image_height)

        if self.transform is not None:
            image = self.transform(image)

        return image, steering_angle
