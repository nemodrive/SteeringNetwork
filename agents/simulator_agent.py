from carla.agent.agent import Agent
from carla.client import VehicleControl
from logbook import Logger
import cv2

NAMESPACE = 'simulator_agent'
log = Logger(NAMESPACE)


class SimulatorAgent(Agent):
    def __init__(self):

        super(SimulatorAgent, self).__init__()
        self._image_cut = None

    def set_simulator(self, cfg):
        self._image_cut = cfg.simulator.image_cut
        self.set_eval_mode()

    def _run_step(self, measurements, sensor_data, directions, target):
        control = self._compute_action(
            sensor_data['CameraRGB'].data,
            measurements.player_measurements.forward_speed, directions)

        return control

    def _compute_action(self, rgb_image, speed, directions=None):

        #shit for demo
        img = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(self.img_dir + "/image_" + str(self.nr_img) + ".jpg", img)
        #shit for demo

        rgb_image = rgb_image[self._image_cut[0]:self._image_cut[1], :]

        steer, acc, brake = self._control_function(rgb_image, speed, directions)

        # This a bit biased, but is to avoid fake breaking
        if brake < 0.1:
            brake = 0.0

        if acc > brake:
            brake = 0.0

        # We limit speed to 35 km/h to avoid
        if speed > 10.0 and brake == 0:
            acc = 0.0

        control = VehicleControl()
        control.steer = steer
        control.throttle = acc
        control.brake = brake
        control.hand_brake = 0
        control.reverse = 0

        return control

    def _control_function(self, image_input_raw, speed, control_input):
        #return predicted_steers, predicted_acc, predicted_brake
        raise NotImplemented
