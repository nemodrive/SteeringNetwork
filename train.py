from logbook import Logger, StreamHandler
import sys
import os
import torch.multiprocessing as mp
import pickle

from utils.config import generate_configs, save_config, resume
from utils import utils
from agents import get_agent
from data_loading import get_data_loader


NAMESPACE = 'train'
log = Logger(NAMESPACE)


def run_once(args):
    cfg, run_id, path = args

    # -- Set seed
    cfg.general.seed = utils.set_seed(cfg.general.seed)

    # -- Get data loaders
    data_loader = get_data_loader(cfg.data_loader)

    train_data = data_loader.get_train_loader()
    test_data = data_loader.get_test_loader()

    # -- Resume agent and metrics if checkpoints are available
    # TODO Resume
    if cfg.checkpoint != "":
        resume_path = path + "/" + cfg.checkpoint
        log.info("Resuming training ...")
        cfg.agent.resume = resume_path

    # -- Get agent
    agent = get_agent(cfg.agent)

    # -- Should have some kind of reporting agent
    # TODO Implement reporting agent

    # -- Init finished
    save_config(os.path.join(cfg.general.common.save_path, "ran_cfg"), cfg)

    eval_freq = cfg.train.eval_freq
    no_epochs = cfg.train.no_epochs - agent.get_train_epoch()

    for epoch in range(no_epochs):
        log.info("Train epoch: {}".format(epoch))
        agent.train(train_data)
        if epoch % eval_freq == 0:
            agent.test(test_data)
        print("Finished an epoch :D")

    with open(path + "/loss_values_train", "wb") as f:
        pickle.dump(agent.loss_values_train, f)

    with open(path + "/loss_values_test", "wb") as f:
        pickle.dump(agent.loss_values_test, f)

    agent.eval_agent()


if __name__ == '__main__':
    # Initialize logger properties
    StreamHandler(sys.stdout).push_application()
    log.info("[MODE] Train")
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
