# BDD steering information extraction

 Steering data consists of the following information:

 - acceleration
 - course (angle given by the magnetometer)
 - gps
 - gyro
 - linear speed
 - speed, as a 2D vector
 - timestamp

 In order to generate steering data from videos, run the *extract_frame_info.py* script. The script takes the following arguments:

 - video_index - the path to a file containing the paths of the video files that we want to process.
 - output_directory - the path to a directory where we want to dump all the steering data. If the directory does not exist, it will be created, given that the path to where we want to create it is valid.
 - debug - flag used to start debugging mode after processing the steering information for a video
 - truncate_frames - the number of frames at which to truncate all the videos. If a negative value is passed, the videos will not be truncated.
 - temporal_downsample_factor - the rate at which we will sample data from the video (actually from a set of data collected at 15 FPS from the video).
 - speed_limit_as_stop - a threshold under which we consider that the car has stopped.

#### Note: I have commented the part where I was generating video data at a fixed 15 fps rate and the one for generating data at a sampled rate, as we were ultimately interested only in the original data.

We now head to the *video_data* folder.
After running *extract_frame_info.py* we obtain a list of csv files in the *output_directory* which contain the above described data. Apart from this, we also want additional information regarding the steering, namely:
- steer:
     - 0: straight
     - 1: slow_or_stop
     - 2: turn_left
     - 3: turn_right
     - 4: turn_left_slight
     - 5: turn_right_slight
- steer_angle - positive angle for right steer and negative angle for left steer.

In order to add this information, we run the *add_steering_prediction.py* script, which takes the following arguments:
- data_dir - path to the directory containing the csv files that we want to add the steering data to.
- time_to_steer - time in milliseconds between the current frame and the next frame used for making the steering prediction (for example, if we set 1000, we want to know the difference in the direction of the car after a second and the current direction of the car).
- decel_thresh - deceleration in m/s below which it is considered that we brake (we compare tbe speed difference between the frame after time_to_steer milliseconds and the current frame with the deceleration threshold).

Further, we want to downsample the information, such that it matches the video frames that we extracted. To do this, we run the *downsample_img_info.py* script, which takes the following arguments:
- data_dir - path to the directory containing the csv files we want to downsample.
- output_dir - path to the directory where we want to store the downsampled info files.
- sample_rate - the rate at which we want to downsample (select one in sample_rate frames). Should be set to the same downsampling rate like the video.


# BDD video processing

We first generate the dataset for training from videos, after which we can generate the dataset for training from images. Training from images is recommended from a training time point of view, but it implies storing a significantly larger amount of data.
At this step it is supposed that we have the videos csv info files containing all the information needed. We refer to the folder containing the info about all frames as *info* and to the one containing downsampled info as *img_info*.

### Generate dataset for video training

We again work in the *video_data* folder.
To generate our train, test and validation datasets, we run the *generate_dataset.py* script, which takes the following arguments:
- info_dir - path to the directory containing the info files. We use *info*.
- video_dir - path to the the directory containing the video files

After running the script folders, *train*, *test* and *validation* will result, each containing an *info* and a *data* folder for the csv and video files respectively.

For training from video, we will also need a file containing metadata about the videos, namely the duration and the number of frames. For generating this file, we run the *group_video_metadata.py* script, taking the arguments:
- video_dir - path to the directory containing the videos
- data_filename - name of the file containing the metadata


### Generate dataset for image training

Having generated the dataset for video training, we are now ready to generate the image dataset as well.
To get the video frames, we run the *split_videos_in_frames.py* script with the following arguments:
- video_dir - path to the directory containing the videos to split into frames
- output_dir - path to the directory where to put the frames
- sample_rate - downsampling rate of the video

The frames for each video will be stored in a directory with the video's name.

In the end, run the *fit_video_and_info_sizes.py* script, to make the number of entries in the info files and the number of frames stored for each video respectively equal. The script takes the following arguments:
- video_dir - path to the directory containing the folders with video frames
- info_dir - path to the directory containing the csv info files
