import cv2
import os
import shutil
import argparse
from subprocess import call
import subprocess
import time
from json_to_speed import get_interpolated_speed
import numpy as np
from scipy.misc import imresize
from process_steer import *


# might need to switch to this get_interpolated_speed when replaying GPS
#from MKZ.nodes.json_to_speed import get_interpolated_speed

# constant for the low res resolution
pixelh = 216
pixelw = 384
# constant for the high resolution
HEIGHT = 720
WIDTH = 1280


def probe_file(filename):
    cmnd = ['ffprobe', '-show_format', '-show_streams', '-pretty', filename]
    p = subprocess.Popen(cmnd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #print filename
    out, err = p.communicate()
    content = out.split(b'\n')
    whole_time = 0
    rotate = None
    horizontal = True
    for item in content:
        name = item.split(b'=')[0]
        tag = name.split(b':')
        if tag[0] == 'TAG':
            tag = tag[1]
        if name == b'duration':
            time = item.split(b':')
            hour = time[-3].split(b'=')[1]
            minute = time[-2]
            second = time[-1]
        if name == b'width':
            im_w = int(item.split(b'=')[1])
        if name == b'height':
            im_h = int(item.split(b'=')[1])
        if tag == b'rotate':
            rotate = int(item.split(b'=')[1])
    if im_w <= im_h:
        if rotate is None or rotate == 180 or rotate == -180:
            horizontal = False
    else:
        if rotate == 90 or rotate == -90 or rotate == 270 or rotate == -270:
            horizontal = False
        #print hour, minute, second
    whole_time = float(hour) * 3600 + float(minute) * 60 + float(second)

    return whole_time, horizontal


def full_im(pixel, all_num):
    # whether this frame is full image or not
    num_l, num_r, num_l_u, num_r_u = all_num

    # num is average pixel intensity over the rest areas
    num = 1.0 * (np.sum(pixel) - num_l - num_r - num_l_u - num_r_u) / \
            (pixel.shape[0] * pixel.shape[1] * 3 - 4 * pixelh * pixelw * 3)
    #print(num)
    # return (is a full image)
    return num >= 1


def read_one_video(video_path, jobid, args):
    fd, fprefix, cache_images, out_name = parse_path(video_path, jobid, args)
    
    FNULL = open(os.devnull, 'w')
    hz_res = 1 if args.low_res else 15
    ratio = False

    # save the speed field
    json_path = os.path.join(os.path.dirname(fd), "info", fprefix + ".json")
    speeds = get_interpolated_speed(json_path, fprefix + ".mov", hz_res)
    if speeds is None:
        # if speed is none, the error message is printed in other functions
        return 0, False

    if speeds.shape[0] < args.truncate_frames:
        print("skipping since speeds are too short!")
        return 0, False

    # filter the too short videos
    duration, ratio = probe_file(video_path)
    if duration < (args.truncate_frames + 1) * 1.0 / hz_res:
        print('the video duration is too short')
        return 0, False

    if abs(speeds.shape[0] - duration * hz_res) > 2 * hz_res:
        # allow one second of displacement
        print("skipping since unequal speed length and image_list length")
        return 0, False

    speeds = speeds[:args.truncate_frames, :]

    image_list = []
    if args.low_res:
        cmnd = [
            'ffmpeg', '-i', video_path, '-f', 'image2pipe', '-loglevel',
            'panic', '-pix_fmt', 'rgb24', '-r', '1', '-vcodec', 'rawvideo', '-'
        ]
        pipe = subprocess.Popen(cmnd, stdout=subprocess.PIPE, bufsize=10**7)
        pout, perr = pipe.communicate()
        image_buff = np.fromstring(pout, dtype='uint8')
        if image_buff.size < args.truncate_frames * HEIGHT * WIDTH * 3:
            print(jobid, video_path, image_buff.size,
                  'Insufficient video size.')
            return 0, False
        image_buff = image_buff[0:args.truncate_frames * HEIGHT * WIDTH * 3]
        image_buff = image_buff.reshape(args.truncate_frames, HEIGHT, WIDTH, 3)

        for i in range(args.truncate_frames):
            image = image_buff[i, :, :, :]
            image_left = image[HEIGHT - pixelh:HEIGHT, 0:pixelw]
            image_right = image[HEIGHT - pixelh:HEIGHT, WIDTH - pixelw:WIDTH]
            image_left_up = image[0:pixelh, 0:pixelw]
            image_right_up = image[0:pixelh, WIDTH - pixelw:WIDTH]

            all_im = [image_left, image_right, image_left_up, image_right_up]
            all_num = [
                np.sum(image_left),
                np.sum(image_right),
                np.sum(image_left_up),
                np.sum(image_right_up)
            ]
            rank = np.argsort(all_num)
            if i % 5 != 0:
                img = all_im[rank[-1]]
            else:
                full = full_im(image, all_num)

                if full:
                    img = imresize(
                        image, (pixelh, pixelw, 3), interp='nearest')
                else:
                    print(video_path,
                          'It is not full screen at frames % 5 == 0')
                    return 0, False

            # RGB to BGR
            img = img[:, :, [2, 1, 0]]
            st = cv2.imencode(".JPEG", img, [cv2.IMWRITE_JPEG_QUALITY, 50])
            # convert the tuple to the second arg, which is the image content
            contents = st[1].tostring("C")

            image_list.append(contents)
    else:
        # generate the video to images to this dir
        if os.path.exists(cache_images):
            shutil.rmtree(cache_images)
        os.mkdir(cache_images)

        call(
            [
                'ffmpeg', '-i', video_path, '-r', '15', '-qscale:v', '10',
                '-s', '640*360', '-threads', '4', cache_images + '/%04d.jpg'
            ],
            stdout=FNULL,
            stderr=FNULL)

        for subdir, dirs, files in os.walk(cache_images):
            for f in sorted(files):
                image_data = cv2.imread(os.path.join(subdir, f))
                image_list.append(image_data)

        if len(image_list) < args.truncate_frames:
            print('Insufficient video size.')
            return 0, False
        image_list = image_list[0:args.truncate_frames]

    if args.low_res:
        example = {
            'height': pixelh,
            'width': pixelw,
            'channel': 3,
            'full_height': HEIGHT,
            'full_width': WIDTH,
            'video_name': video_path,
            'format': 'JPEG',
            'encoded': image_list,
            'speeds': speeds.ravel().tolist()
        }
    else:
        example = {
            'height': 360,
            'width': 640,
            'channel': 3,
            'video_name': video_path,
            'format': 'JPEG',
            'encoded': image_list,
            'speeds': speeds.ravel().tolist()
        }

    print(video_path)
    return example, True


def parse_path(video_path, jobid, args):
    fd, fname = os.path.split(video_path)
    fprefix = fname.split(".")[0]
    cache_images = os.path.join(args.temp_dir_root,
                                "prepare_records_image_temp_" + str(jobid))
    out_name = os.path.join(args.output_directory, fprefix + ".records")

    # return all sorts of info:
    # video_base_path, video_name_wo_prefix, cache_path, out_tfrecord_path
    return (fd, fprefix, cache_images, out_name)


'''
def convert_one(video_path, jobid):
    fd, fprefix, cache_images, out_name = parse_path(video_path, jobid)
    if not os.path.exists(out_name):
        example, state = read_one_video(video_path, jobid)
        if state:
            writer = tf.python_io.TFRecordWriter(out_name)
            writer.write(example.SerializeToString())
            writer.close()


def p_convert(video_path_list, jobid):
    #start = time.time()
    for video_path in video_path_list:
        fd, fprefix, cache_images, out_name = parse_path(video_path, jobid, args)
        mod = int(fprefix[0:3], 16) % args.num_threads
        if mod == jobid:
            convert_one(video_path, jobid, args)


def parallel_run(args):
    with open(args.video_index) as f:
        content = f.readlines()
    content = [x.strip() for x in content]

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    threads = []
    for i in range(args.num_threads):
        # arguments are (the range to process, train phase or test phase)
        args = (content, i, args)
        t = multiprocessing.Process(target=p_convert, args=args)
        #t = threading.Thread(target=p_convert, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print('Finished processing all files')
    sys.stdout.flush()
'''


def convert_one(video_path, jobid, args):
    fd, fprefix, cache_images, out_name = parse_path(video_path, jobid, args)
    example, state = read_one_video(video_path, jobid, args)
    return example


def p_convert(video_path, jobid, args):

    fd, fprefix, cache_images, out_name = parse_path(video_path, jobid, args)
    example = convert_one(video_path, jobid, args)

    if args.non_random_temporal_downsample:
        tstart = 0
    else:
        '''
        tstart = tf.random_uniform(
            [],
            minval=0,
            maxval=args.temporal_downsample_factor,
            dtype=tf.int32) '''
        tstart = 0
    len_downsampled = args.truncate_frames // args.temporal_downsample_factor

    encoded = example['encoded'][:args.truncate_frames]
    encoded_sub = encoded[tstart::args.temporal_downsample_factor]

    speed = example['speeds']
    speed = np.reshape(speed, [-1, 2])
    speed = speed[:args.truncate_frames, :]
    speed = speed[tstart::args.temporal_downsample_factor, :]

    # from speed to stop labels
    stop_label = speed_to_future_has_stop(speed, args.stop_future_frames,
                                          args.speed_limit_as_stop)

    # Note that the turning heuristic is tuned for 3Hz video and urban area
    # Note also that stop_future_frames is reused for the turn
    turn, steer_value = turn_future_smooth(speed, args.stop_future_frames,
                              args.speed_limit_as_stop, args, 1, 1)
    import pdb; pdb.set_trace()
    preview_data(encoded_sub, turn, steer_value)
    locs = relative_future_location(
        speed, args.stop_future_frames,
        args.frame_rate / args.temporal_downsample_factor)


def preview_data(images, turn, steer_value):

    print(turn_int2str)
    for nr_image in range(len(images)):

        img = cv2.resize(images[nr_image], (720, 460), cv2.INTER_AREA)
        cv2.imshow("Bla", img)
        print(turn[nr_image], steer_value[nr_image])
        time.sleep(0.2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()



def process_videos(args):
    with open(args.video_index) as f:
        content = f.readlines()
    content = [x.strip() for x in content]

    for video in content:
        p_convert(video, 1, args)

    print('Finished processing all files')


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        '--video_index',
        type=str,
        default='/data/nx-bdd-20160929/video_filtered_index_38_60_sec.txt',
        help='filtered video indexing')

    arg_parser.add_argument(
        '--output_directory',
        type=str,
        default='/data/nx-bdd-20160929/tfrecord_fix_speed/',
        help='Training data directory')

    arg_parser.add_argument(
        '--num_threads',
        type=int,
        default=1,
        help='Number of threads to preporcess the images')
    arg_parser.add_argument(
        '--truncate_frames',
        type=int,
        default=36 * 15,
        help='Number of frames to leave in the video')
    arg_parser.add_argument(
        '--temp_dir_root',
        type=str,
        default='/tmp/',
        help='the temp dir to hold ffmpeg outputs')
    arg_parser.add_argument(
        '--low_res',
        type=bool,
        default=False,
        help='the data we want to use is low res')

    arg_parser.add_argument(
        '--decode_downsample_factor',
        type=int,
        default=1,
        help='The original high res video is 640*360. This param downsample the image during jpeg decode process')
    
    '''The original video is in 15 FPS, this flag optionally downsample the video temporally
       All other frame related operations are carried out after temporal downsampling'''
    arg_parser.add_argument(
        '--temporal_downsample_factor',
        type=int,
        default=5,
        help='The original video is in 15 FPS, this flag optionally downsample the video temporally')

    arg_parser.add_argument(
        '--speed_limit_as_stop',
        type=float,
        default=0.3,
        help='if speed is less than this, then it is considered to be stopping'
    )
    arg_parser.add_argument(
        '--stop_future_frames',
        type=int,
        default=2,
        help='Shift the stop labels * frames forward, to predict the future')

    arg_parser.add_argument(
        '--balance_drop_prob',
        type=float,
        default=-1.0,
        help='drop no stop seq with specified probability')

    arg_parser.add_argument(
        '--acceleration_thres',
        type=float,
        default=-1.0,
        help='acceleration threshold, minus value for not using it')

    arg_parser.add_argument(
        '--deceleration_thres',
        type=float,
        default=1.0,
        help='deceleration threshold, minus value for not using it')

    arg_parser.add_argument(
        '--non_random_temporal_downsample',
        type=bool,
        default=False,
        help='''if true, use fixed downsample method''')

    arg_parser.add_argument(
        '--frame_rate',
        type=float,
        default=15.0,
        help='the frame_rate we have for the videos')

    args = arg_parser.parse_args()

    if args.low_res:
        print("Warning: using low res specific settings")
    if not os.path.exists(args.output_directory):
        os.mkdir(args.output_directory)


    process_videos(args)