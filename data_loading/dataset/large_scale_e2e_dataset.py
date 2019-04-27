from torch.utils.data import Dataset
from ..image_tools import *
import json
import os
import cv2
import PIL


class LargeScaleE2EDataset(Dataset):
    def __init__(self,
                 cfg,
                 x,
                 y,
                 data_dir,
                 image_width,
                 image_height,
                 train=True,
                 transform=None):
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
        image = self.X[index]
        label = self.y[index]

        if self.train:
            return image, label

        return image, label


class LargeScaleE2EDatasetSteering(Dataset):
    def __init__(self,
                 cfg,
                 x,
                 y,
                 data_dir,
                 image_width,
                 image_height,
                 train=True,
                 transform=None):
        self.cfg = cfg
        self.X = x
        self.y = y
        self.train = train
        self.image_width = image_width
        self.image_height = image_height
        self.data_dir = data_dir
        self.transform = transform

        self.chunks = self.make_chunks(x, y)

    def make_chunks(self, videos_info, interpolated_speeds):
        chunks = []
        for video_name in videos_info.keys():
            if interpolated_speeds[video_name] is None:
                pass
            else:
                end_offset = (self.cfg.temporal_stride * self.cfg.
                              sequence_length / self.cfg.playback_fps) + 1
                video_length = int(
                    len(interpolated_speeds[video_name]) /
                    self.cfg.playback_fps) - int(end_offset)
                chunks.extend([(video_name, second)
                               for second in range(0, video_length)])

        return chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, index):
        chunk = self.chunks[index]

        video_speeds = self.y[chunk[
            0]]  # get speeds associated with this video name

        # load frames with asociated speeds
        video_name = chunk[0]
        time_point = chunk[1]

        video_full_path = self.data_dir + "/" + video_name + ".mov"
        capture = cv2.VideoCapture(video_full_path)

        images = []
        speeds = []

        time_stride = self.cfg.temporal_stride
        seq_length = self.cfg.sequence_length

        frame_stride = self.X[video_name]["fps"] * time_stride
        frame_no = time_point * self.X[video_name]["fps"] / self.X[video_name][
            "nframes"]

        for seq_idx in range(seq_length):
            capture.set(1, frame_no)
            ret, img = capture.read()
            img = PIL.Image.fromarray(img)
            img = self.transform(img)
            speed = video_speeds[int(frame_no * len(video_speeds))]

            images.append(np.array(img))
            speeds.append(np.array(speed))

            frame_no += frame_stride / self.X[video_name]["nframes"]

        return np.array(images), np.array(speeds)


class LargeScaleE2EDatasetObjDetection(Dataset):
    def __init__(self,
                 cfg,
                 x,
                 y,
                 data_dir,
                 image_width,
                 image_height,
                 train=True,
                 transform=None):
        self.cfg = cfg
        self.X = x
        self.y = y
        self.train = train
        self.image_width = image_width
        self.image_height = image_height
        self.data_dir = data_dir
        self.transform = transform

    @staticmethod
    def json_to_nparray(json_data):
        def hot_encoding(classint, classcnt):
            arr = [.0 for _ in range(classcnt)]
            arr[classint] = 1.0
            return arr

        name_to_int = {
            "traffic sign": 0,
            "car": 1,
            "track": 2,
            "pedestrian": 3
        }
        classes = len(name_to_int.keys())

        objects = json_data["objects"]

        objlist = []
        for object in objects:
            x1 = object["box2d"]["x1"]
            y1 = object["box2d"]["y1"]
            x2 = object["box2d"]["x2"]
            y2 = object["box2d"]["y2"]
            clazz = object["category"]

            height, width = abs(x1 - x2), abs(y1 - y2)

            objlist.append(
                [x1, y1, width, height] + hot_encoding(name_to_int[clazz]))

        return np.array(objlist)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        image = self.X[index]
        label = self.y[index]

        image = load_image(self.data_dir, image)
        label = json.load(os.path.join(self.data_dir, label))
        label = self.json_to_nparray(label)

        return image, label
