import os
import subprocess
import pickle as pkl
from argparse import ArgumentParser


arg_parser = ArgumentParser()
arg_parser.add_argument('--video_data_dir',
    type=str,
    default='original',
    help='path to directory containing files with data about videos')
arg_parser.add_argument('--video_dir',
    type=str,
    default='/home/alexm/Desktop/hal_data/samples-1k/videos',
    help='path to directory containing videos')
arg_parser.add_argument('--data_filename',
    type=str,
    default='video_metadata.pkl',
    help='name of the file that will be generated')

args = arg_parser.parse_args()

video_metadata = {}
videos = [x.split('_')[0] for x in os.listdir(args.video_data_dir)]
for video in videos:
    vidname = os.path.join(args.video_dir, video + '.mov')

    nframes_cmd = ['ffprobe', '-v', 'error', '-count_frames', '-select_streams',
        'v:0', '-show_entries', 'stream=nb_frames', '-of',
        'default=nokey=1:noprint_wrappers=1', vidname]
    p = subprocess.Popen(nframes_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    nframes = int(out)

    duration_cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=duration', '-of',
        'default=noprint_wrappers=1:nokey=1', vidname]
    p = subprocess.Popen(duration_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    duration = float(out)

    video_metadata[video] = {
        'nframes': nframes,
        'duration': duration
    }

pkl.dump(video_metadata, open(args.data_filename, 'wb'))
