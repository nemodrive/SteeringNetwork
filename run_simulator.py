import os
import sys
import logging
from logbook import Logger, StreamHandler
from logbook.compat import redirect_logging
import torch.multiprocessing as mp
from carla.driving_benchmark import run_driving_benchmark

try:
    from carla import carla_server_pb2 as carla_protocol
except ImportError:
    raise RuntimeError(
        'cannot import "carla_server_pb2.py", run the protobuf compiler to generate this file'
    )

from utils.config import generate_configs, save_config
from utils import utils
from agents import get_agent
from simulator.simulator import Simulator, get_benchmark
from carla_benchmark import CarlaBenchmark
from demo_benchmark import DemoBenchmark


NAMESPACE = 'run_simulator'
log = Logger(NAMESPACE)


def run_once(args):

    cfg, run_id, path = args

    sim_path = path + "/" + cfg.simulator.save_folder
    if not os.path.exists(sim_path):
        os.makedirs(sim_path)

    simulator = Simulator(cfg,sim_path, log)
    simulator.start()

    # -- Set seed
    cfg.general.seed = utils.set_seed(cfg.general.seed)

    # -- Load simulator
    # TODO 2 start server with config
    # TODO 2 Save simulator config in path ( see line 41 with save_config(

    # -- Resume agent and metrics if checkpoints are available
    resume_path = path + "/" + cfg.checkpoint
    if resume_path:
        log.info("Resuming training ...")
        cfg.agent.resume = resume_path
    logging.info('listening to server %s:%s', cfg.simulator.host,
                 cfg.simulator.port)

    # -- Get agent
    agent = get_agent(cfg.agent)
    agent.set_simulator(cfg)

    os.chdir(sim_path)

    benchmark_agent = DemoBenchmark(cfg.simulator.town)

    # -- Init finished
    #save_config(os.path.join(cfg.general.common.save_path, "ran_cfg"), cfg)

    # Now actually run the driving_benchmark
    #import pdb; pdb.set_trace()
    run_driving_benchmark(agent, benchmark_agent, cfg.simulator.town,
                          cfg.simulator.carla_log_name,
                          cfg.simulator.continue_experiment,
                          cfg.simulator.host, cfg.simulator.port)

    simulator.kill_process()


if __name__ == '__main__':

    # Initialize logger properties
    StreamHandler(sys.stdout).push_application()
    log.info("[MODE] Train")
    log.warn('Logbook is too awesome for most applications')

    # -- Parse config file & generate
    procs_no, arg_list = generate_configs()

    log.info("Starting simulation...")
    redirect_logging()
    if len(arg_list) > 1:
        # Run batch of experiments:
        pool = mp.Pool(procs_no)
        pool.map(run_once, arg_list)
    else:
        run_once(arg_list[0])
