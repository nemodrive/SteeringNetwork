import torch
from logbook import Logger

from .learning_agent import LearningAgent
from models import get_models

NAMESPACE = '<name_of_class>'  # ++ Identifier name for logging
log = Logger(NAMESPACE)


class DemoAgentWiki(LearningAgent):  # ++ Extend Learning agent
    def __init__(self, cfg):
        super(DemoAgentWiki, self).__init__(cfg)

        use_cuda = self._use_cuda   # ++ Parent class already saves some configuration variables
        # ++ All parent variables should start with _.

        # -- Get necessary variables from cfg
        self.train_cfg = cfg.train

        # -- Initialize model
        model_class = get_models(cfg.model)
        self.model = model_class[0](cfg.model,
                                    torch.zeros(3, 224, 224), torch.zeros(10))
        # ++ All models receive as parameters (configuration namespace, input data size,
        # ++ output data size)

        self._models.append(
            self.model
        )  # -- Add models & optimizers to base for saving

        # ++ After adding model you can set the agent to cuda mode
        # ++ Parent class already makes some adjustments. E.g. turns model to cuda mode
        if use_cuda:
            self.cuda()

        # -- Initialize optimizers
        self.optimizer = self.get_optim(cfg.train.algorithm,
                                        cfg.train.algorithm_args, self.model)
        self._optimizers.append(
            self.optimizer)  # -- Add models & optimizers to base for saving

        # -- Initialize criterion
        self.criterion = getattr(torch.nn, cfg.train.criterion)()

        # -- Change settings from parent class
        # ++ Parent class automatically initializes 4 metrics: loss/acc for train/test
        # ++ E.g switch metric slope
        self.set_eval_metric_comparison(True)

        # ++ E.g. to add variable name to be saved at checkpoints
        self._save_data.append("train_cfg")

        super(DemoAgentWiki, self).__end_init__()

    def _session_init(self):
        """
        Called each train/test stage
        """

    def _train(self, dataloader):
        """
        Considering a dataloader (loaded from config.)
        Implement the training loop.
            
        E.g.: 
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # optimize
           
        :return training loss metric & other information 
        """

        train_loss = None

        return train_loss, {}

    def _test(self, dataloader):
        """
        Considering a dataloader (loaded from config.)
        Implement the testing loop.
        """
        test_loss = 0

        return test_loss, None, {}

    def _control_function(self, image_input_raw, speed, control_input):
        """
        Implement for carla simulator run.
        :return: steer, acc, brake
        """
        #  outputs = self.model(input)
        steer, acc, brake = None, None, None
        return steer, acc, brake

    def _set_eval_mode(self):
        """
        Custom configuration when changing to evaluation mode
        """
        pass

    def _set_train_mode(self):
        """
        Custom configuration when changing to train mode
        """
        pass

    def _save(self, save_data, path):
        """
        Called when saving agent state. Agent already saves variables defined in the list
        self._save_data and other default options.
        :param save_data: Pre-loaded dictionary with saved data. Append here other data
        :param path: Path to folder where other custom data can be saved
        :return: should return default save_data dictionary to be saved
        """
        return save_data

    def _resume(self, agent_check_point_path, saved_data):
        """
        Custom resume scripts should pe implemented here
        :param agent_check_point_path: Path of the checkpoint resumed
        :param saved_data: loaded checkpoint data (dictionary of variables)
        """
        pass
