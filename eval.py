import os
from logbook import Logger, StreamHandler
import sys

from utils.config import generate_configs, save_config, resume
from utils import utils
from agents import get_agent
from data_loading import get_data_loader
import torch.multiprocessing as mp

# TODO must modify other stuff just to report evaluation

NAMESPACE = 'eval'
log = Logger(NAMESPACE)


def run_once(args):
    cfg, run_id, path = args

    # -- Set seed
    cfg.general.seed = utils.set_seed(cfg.general.seed)

    # -- Resume agent and metrics if checkpoints are available
    # TODO Resume
    resume_path = path + "/" + cfg.checkpoint
    if resume_path:
        log.info("Resuming training ...")
        cfg.agent.resume = resume_path

    # -- Get agent
    agent = get_agent(cfg.agent)

    # -- Should have some kind of reporting agent
    # TODO Implement reporting agent

    # -- Init finished
    save_config(os.path.join(cfg.general.common.save_path, "ran_cfg"), cfg)

    agent.eval_agent()


if __name__ == '__main__':
    # Initialize logger properties
    StreamHandler(sys.stdout).push_application()
    log.info("[MODE] Eval agent only")
    log.warn('Logbook is too awesome for most applications')

    # -- Parse config file & generate
    procs_no, arg_list = generate_configs()
    log.info("Starting...")

    if len(arg_list) > 1:
        # Run batch of experiments:
        pool = mp.Pool(procs_no)
        pool.map(run_once, arg_list)
    else:
        run_once(arg_list[0])
