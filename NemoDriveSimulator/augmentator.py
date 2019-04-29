import NemoDriveSimulator.transformation as transformation
import NemoDriveSimulator.steering as steering
import numpy as np
import math
import json

class Augmentator:
    def __init__(self, json):
        """
        Constructor
        :param K: camera intrinsic matrix
        :param M: camera extrinsic matrix
        """
        self.json = json
        self._read_json()

        # object for image transformation (translation & rotation)
        # K, M are constructed in read_json
        self.transform = transformation.Transformation(self.K, self.M)

    def _read_json(self):
        # get data from json
        with open(self.json) as f:
            self.data = json.load(f)

        # get cameras
        self.center_camera = self.data['cameras'][0]
        # self.left_camera = self.data['cameras'][1]
        # self.right_camera = self.data['cameras'][2]

        self.Res = np.array([
            [180. / 1080., 0., 0.],
            [0., 320. / 1920., 0.],
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

    def augment(self, data, translation, rotation, intersection_distance=7.5):
        """
        Augment a frame
        Warning!!! this augmentation may work only for turns less than 180 degrees. For bigger turns, although it
        reaches the same point, it may not follow the real car's trajectory

        :param data: [frame, steer, velocity, delta_time]
        :param translation: ox translation, be aware that positive values mean right translation
        :param rotation: rotation angle, be aware that positive valuea mean right rotation
        :param intersection_distance: distance where the simualted car and real car will intersect
        :return: the augmented frame, steer for augmented frame
        """
        assert abs(rotation) < math.pi / 2, "The angle in absolute value must be less than Pi/2"

        frame, steer, _ = data
        eps = 1e-12

        # compute wheel angle and radius of the real car
        steer = eps if abs(steer) < eps else steer
        wheel_angle = steering.get_delta_from_steer(steer + eps)
        R = steering.get_radius_from_delta(wheel_angle)

        # estimate the future position of the real car
        alpha = intersection_distance / R  # or may try velocity * delta_time / R
        P1 = np.array([R * (1 - np.cos(alpha)), R * np.sin(alpha)])

        # determine the point where the simulated car is
        P2 = np.array([translation, 0.0])

        # compute the line parameters that passes through simulated point and is
        # perpendicular to it's orientation
        d = np.zeros((3,))
        rotation = eps if abs(rotation) < eps else rotation
        d[0] = np.sin(rotation)
        d[1] = np.cos(rotation)
        d[2] = -d[0] * translation

        # we need to find the circle center (Cx, Cy) for the simulated car
        # we have the equations
        # (P11 - Cx)**2 + (P12 - Cy)**2 = (P21 - Cx)**2 + (P22 - Cy)**2
        # d0 * Cx + d1 * Cy + d2 = 0
        # to solve this, we substitute Cy with -d0/d1 * Cx - d2/d1
        a = P1[0]**2 + (P1[1] + d[2]/d[1])**2 - P2[0]**2 - (P2[1] + d[2]/d[1])**2
        b = -2 * P2[0] + 2 * d[0]/d[1] * (P2[1] + d[2]/d[1]) + 2 * P1[0] - 2 * d[0]/d[1] * (P1[1] + d[2]/d[1])
        Cx = a / b
        Cy = -d[0]/d[1] * Cx - d[2]/d[1]
        C = np.array([Cx, Cy])

        # determine the radius
        sim_R = np.linalg.norm(C - P2)
        assert np.isclose(sim_R, np.linalg.norm(C - P1)), "The points P1 and P2 are not on the same circle"

        # determine the "sign" of the radius
        # sgn = 1 if np.cross(w2, w1) >= 0 else -1
        w1 = np.array([np.sin(rotation), np.cos(rotation)])
        w2 = P1 - P2
        sgn = 1 if np.cross(w2, w1) >= 0 else -1
        sim_R = sgn * sim_R

        # determine wheel angle
        sim_delta, _, _ = steering.get_delta_from_radius(sim_R)
        sim_steer = steering.get_steer_from_delta(sim_delta)

        # simulate transformation
        sim_frame = self.transform.rotate_image(frame, rotation)
        sim_frame = self.transform.translate_image(sim_frame, translation)

        return sim_frame, sim_steer, sim_delta, sim_R, C
