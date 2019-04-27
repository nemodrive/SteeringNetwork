import cv2
import json
import math
import numpy as np
import simulator
import NemoDriveSimulator.steering as steering
import NemoDriveSimulator.transformation as transformation


class AugmentationEvaluator:
    def __init__(self, json, translation_threshold=1.5, rotation_threshold=math.pi / 4., time_penalty=6):
        """
        :param json: path to json file
        :param translation_threshold: translation threshold on OX axis
        :param rotation_threshold: rotation threshold relative to OY axis
        :param time_penalty: time penalty for human intervention
        """
        self.json = json
        self.translation_threshold = translation_threshold
        self.rotation_threshold = rotation_threshold
        self.time_penalty = time_penalty

        self._read_json()
        self._reset()

        # initialize simulator
        self.simulator = simulator.Simulator(
            self.K,
            self.M,
            time_penalty=self.time_penalty,
            distance_limit=self.translation_threshold,
            angle_limit=self.rotation_threshold
        )

    def _read_json(self):
        # get data from json
        with open(self.json) as f:
            self.data = json.load(f)

        # get cameras
        self.center_camera = self.data['cameras'][0]
        # self.left_camera = self.data['cameras'][1]
        # self.right_camera = self.data['cameras'][2]

        self.Res = np.array([
            [360. / 1080., 0., 0.],
            [0., 640. / 1920., 0.],
            [0., 0., 1.]
        ])

        # intrinsic camera parameters
        self.K = np.array(
            self.center_camera['cfg_extra']['camera_matrix']
        )

        self.K = np.matmul(self.Res, self.K)

        # extrinsic camera parameters
        camera_position = self.center_camera['cfg_extra']['camera_position']
        self.M = np.array([
            [1.0, 0.0, 0.0, camera_position[0]],
            [0.0, 1.0, 0.0, camera_position[1]],
            [0.0, 0.0, 1.0, camera_position[2]]
        ])

        # read locations list
        self.locations = self.data['locations']

    def _reset(self):
        self.center_capture = cv2.VideoCapture(self.center_camera['video_path'])
        # self.left_capture = cv2.VideoCapture(self.left_camera['video_path'])
        # self.right_camera = cv2.VideoCapture(self.right_camera['video_path'])
        self.frame_index = 0
        self.locations_index = 0

    @staticmethod
    def get_steer(course, speed, dt, eps=1e-12):
        sgn = np.sign(course)
        dist = speed * dt
        R = dist / (np.deg2rad(abs(course)) + eps)
        delta, _, _ = steering.get_delta_from_radius(R)
        steer = sgn * steering.get_steer_from_delta(delta)
        return steer

    @staticmethod
    def get_course(steer, speed, dt):
        dist = speed * dt
        delta = steering.get_delta_from_steer(steer)
        R = steering.get_radius_from_delta(delta)
        rad_course = dist / R
        course = np.rad2deg(rad_course)
        return course

    @staticmethod
    def get_relative_course(prev_course, crt_course):
        a = crt_course - prev_course
        a = (a + 180) % 360 - 180
        return a

    def _get_closest_location(self, tp):
        return min(self.locations, key=lambda x: abs(x['timestamp'] - tp))

    def get_next_image(self, predicted_course=0.):
        """
        :param predicted_course: predicted course by nn in degrees
        :return: augmented image corresponding to predicted course or empty np.array in case the video ended
        """
        ret, frame = self.center_capture.read()
        dt = 1. / self.center_capture.get(cv2.CAP_PROP_FPS)

        # check if the video ended
        if not ret:
            self._reset()
            return np.array([])

        # for the first frame return
        if self.frame_index == 0:
            self.prev_course = self.locations[self.frame_index]['course']
            self.frame_index += 1
            return frame

        # read course and speed for previous frame
        location = self._get_closest_location(1000 * dt * (self.frame_index - 1) + self.locations[0]['timestamp'])
        course = location['course']
        speed = location['speed']

        # compute relative course and save current course
        rel_course = AugmentationEvaluator.get_relative_course(self.prev_course, course)
        self.prev_course = course

        # compute steering from course, speed, dt
        steer = AugmentationEvaluator.get_steer(rel_course, speed, dt)
        predicted_steer = AugmentationEvaluator.get_steer(predicted_course, speed, dt)
        # print("STEER", steer)

        # run augmentator
        args = [frame, steer, speed, dt, predicted_steer]
        sim_frame = self.simulator.run(
            args,
        )

        # increase the frame index
        self.frame_index += 1
        return sim_frame

    def get_autonomy(self):
        total_time = (self.data['endTime'] - self.data['startTime']) / 1000
        return self.simulator.get_autonomy(
            total_time=total_time
        )

    def get_number_interventions(self):
        return self.simulator.get_number_interventions()


if __name__ == "__main__":
    # initialize evaluator
    # check multiple parameters like time_penalty, distance threshold and angle threshold
    # in the original paper time_penalty was 6s
    augm = AugmentationEvaluator("./test_data/1c820d64b4af4c85.json", time_penalty=6)
    predicted_course = 0.0

    # get first frame of the video
    frame = augm.get_next_image()

    while True:
        # make prediction based on frame
        # predicted_course = 0.01 * np.random.randn(1)[0]
        predicted_course = -0.1 * np.random.rand(1)[0]

        # get next frame corresponding to current prediction
        frame = augm.get_next_image(predicted_course)
        if frame.size == 0:
            break

        # eliminate car
        frame = transformation.Crop.crop_center(frame, down=0.4, up=0.05)

        # show augmented frmae
        cv2.imshow("Augmented frame", frame)
        cv2.waitKey(33)

    # print autonomy and number of interventions
    print("Autonomy:", augm.get_autonomy())
    print("#Interventions:", augm.get_number_interventions())
