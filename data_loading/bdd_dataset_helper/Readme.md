# BDD steering information extractor

Steering data consists of the following information:

- acceleration
- gps
- gyro
- speed, as a 2D vector
- steering angle (negative for left steering, positive for right steering)
- timestamp
- turn: 
    - -1: not sure
    - 0: straight
    - 1: slow_or_stop
    - 2: turn_left
    - 3: turn_right
    - 4: turn_left_slight
    - 5: turn_right_slight

In order to generate steering data from videos, run the extract_frame_info.py script. The script takes the following arguments:

- video_index - the path to a file containing the paths of the video files that we want to process
- output_directory - the path to a directory where we want to dump all the steering data. If the directory does not exist, it will be created, given that the path to where we want to create it is valid.
- debug - flag used to start debugging mode after processing the steering information for a video
- truncate_frames - the number of frames at which to truncate all the videos. If a negative value is passed, the videos will not be truncated
- temporal_downsample_factor - the rate at which we will sample data from the video (actually from a set of data collected at 15 FPS from the video)
- speed_limit_as_stop - a threshold under which we consider that the car has stopped
