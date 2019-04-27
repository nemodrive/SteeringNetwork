import subprocess
from threading import Thread
from logbook import Logger, FileHandler
import carla.driving_benchmark.experiment_suites as bms

NAMESPACE = 'run_simulator'
log = Logger(NAMESPACE)

SIMULATOR_EXE = "./CarlaUE4.sh"
SIMULATOR_SERVER = "-carla-server"
SIMULATOR_WINDOWED = "-windowed"
RESX = "-ResX"
RESY = "-ResY"
BENCHMARK = "-benchmark"
SIMULATOR_SETTINGS = "-carla-settings"
SIMULATOR_LOG_FILE = "simulator_log_file"
FPS = "-fps"

BENCHMARKS = {
    "CoRL2017": bms.CoRL2017,
    "BasicExperimentSuite": bms.BasicExperimentSuite
}


def get_benchmark(cfg):
    assert hasattr(cfg, "benchmark") and cfg.benchmark in BENCHMARKS, "Please provide a valid benchmark name."
    return BENCHMARKS[cfg.benchmark](cfg.town)


class Simulator(Thread):
    def __init__(self, cfg, path, local_log):

        super(Simulator, self).__init__()

        self._path_to_experiments = path
        self._simulator_path = cfg.carla_settings.exe_path
        self._town_name = cfg.carla_settings.town
        self._resX = cfg.carla_settings.resX
        self._resY = cfg.carla_settings.resY
        self._fps = cfg.carla_settings.fps
        self._carla_settings_file = cfg.carla_settings.settings_file

        # also log everything in a file
        file_handler = FileHandler(
            path + "/" + cfg.simulator.log_name, level='INFO', bubble=True)
        file_handler.push_application()
        local_log.handlers.append(file_handler)

        self._copy_carla_settings()
        self._process = None

    def _copy_carla_settings(self):

        path_to_file = self._simulator_path + "/" + self._carla_settings_file
        path_to_copy_file = self._path_to_experiments + "/" + self._carla_settings_file

        with open(path_to_copy_file, "wt") as cf:
            with open(path_to_file, "rt") as f:
                cf.write(f.read())

    def run(self):

        simulator_log_file = self._path_to_experiments + "/" + SIMULATOR_LOG_FILE
        with open(simulator_log_file, "wt") as log_file:

            carla_exe = SIMULATOR_EXE
            carla_map = self._town_name
            carla_server = SIMULATOR_SERVER
            carla_windowed = SIMULATOR_WINDOWED
            carla_res_x = RESX + "=" + self._resX
            carla_res_y = RESY + "=" + self._resY
            carla_settings = SIMULATOR_SETTINGS + "=" + self._carla_settings_file
            carla_benchmark = BENCHMARK
            carla_fps = FPS + "=" + self._fps
            carla_process = [
                carla_exe, carla_map, carla_server, carla_windowed, carla_res_x,
                carla_res_y, carla_benchmark, carla_fps, carla_settings
            ]
            print(carla_process)
            self._process = subprocess.run(
                carla_process,
                stdout=log_file,
                stderr=log_file,
                cwd=self._simulator_path)

    def kill_process(self):
        if self._process is not None:
            self._process.kil()
