import os
import subprocess
import pickle as pkl
from argparse import ArgumentParser


def get_video_metadata(args):
    """
    Extract and store video metadata useful for training from video
    """
    video_metadata = {}
    # Get the metadata for each video file
    for video in os.listdir(args.video_dir):
        if '.mov' in video:
            vidname = os.path.join(args.video_dir, video)

            # Get the number of frames in the video
            nframes_cmd = ['ffprobe', '-v', 'error', '-count_frames',
                '-select_streams', 'v:0', '-show_entries', 'stream=nb_frames',
                '-of', 'default=nokey=1:noprint_wrappers=1', vidname]
            p = subprocess.Popen(nframes_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = p.communicate()
            nframes = int(out)

            # Get the duration of each video
            duration_cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream=duration', '-of',
                'default=noprint_wrappers=1:nokey=1', vidname]
            p = subprocess.Popen(duration_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = p.communicate()
            duration = float(out)

            # Save the metadata for the current video
            video_metadata[video] = {
                'nframes': nframes,
                'duration': duration
            }

    # Store the metadata for all videos in a pickle file
    pkl.dump(video_metadata, open(args.data_filename, 'wb'))


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--video_dir',
        type=str,
        default='/home/alexm/Desktop/hal_data/samples-1k/good_videos',
        help='path to directory containing videos')
    arg_parser.add_argument('--data_filename',
        type=str,
        default='video_metadata.pkl',
        help='name of the file that will be generated')

    args = arg_parser.parse_args()

    get_video_metadata(args)
