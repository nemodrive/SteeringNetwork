from tqdm import tqdm
from .json_to_speed import get_interpolated_speed
import imageio
import json

# TODO: remove hardcodes

def get_all_files_from(videos_path):
    from os import walk
    video_filenames = []
    for (dirpath, dirnames, filename) in walk(videos_path):
        video_filenames.extend(filename)
    video_filenames = [x.split(".")[0] for x in video_filenames]
    return video_filenames

def get_preprocessed_info(dataset_path):
    all_video_paths = get_all_files_from(dataset_path)

    all_video_info = {}

    for video_path in tqdm(all_video_paths):
        video_info = imageio.get_reader(dataset_path + video_path + ".mov", 'ffmpeg').get_meta_data()
        all_video_info[ video_path] = video_info

    return all_video_info

def save_preprocessed_info(dataset_path, all_info_path):
    all_video_info = get_preprocessed_info(dataset_path)

    with open(all_info_path, 'w') as outfile:
        json.dump(all_video_info, outfile)

def save_preprocessed_info2(all_video_info, all_info_path):
    with open(all_info_path, 'w') as outfile:
        json.dump(all_video_info, outfile)


def get_interpolated_info(dataset_path):

    all_video_paths = get_all_files_from(dataset_path + "/videos")
    all_frames_data = {}
    for video_path in tqdm(all_video_paths):
        interpolated = get_interpolated_speed(dataset_path + "/info/" + video_path + ".json",
                                              dataset_path + "/videos/" + video_path + ".mov", 15)
        all_frames_data[video_path] = interpolated

    return all_frames_data

def save_interpolated_info(dataset_path, destination):
    all_frames_data = get_interpolated_info(dataset_path)
    with open(destination, 'w') as outfile:
        json.dump(all_frames_data, outfile)

def save_interpolated_info2(all_frames_data, destination):
    with open(destination, 'w') as outfile:
        json.dump(all_frames_data, outfile)



if __name__ == "__main__":
    import sys
    print(sys.argv)
    save_preprocessed_info(sys.argv[1], sys.argv[2])
    #save_interpolated_info(sys.argv[1], sys.argv[2])