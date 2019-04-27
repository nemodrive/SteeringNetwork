from argparse import Namespace
from termcolor import colored as clr
import os
import yaml
from logbook import Logger
from time import time

NAMESPACE = 'config.py'
log = Logger(NAMESPACE)

def namespace_to_dct(namespace):
    dct = {}
    for key, value in namespace.__dict__.items():
        if isinstance(value, Namespace):
            dct[key] = namespace_to_dct(value)
        else:
            dct[key] = value
    return dct


def value_of(cfg, name, default=None):
    return getattr(cfg, name) if hasattr(cfg, name) else default


def _update_config(default_cfg, diff_cfg):
    """Updates @default_cfg with values from @diff_cfg"""

    for key, value in diff_cfg.__dict__.items():
        if isinstance(value, Namespace):
            if hasattr(default_cfg, key):
                _update_config(getattr(default_cfg, key), value)
            else:
                setattr(default_cfg, key, value)
        else:
            setattr(default_cfg, key, value)


def config_to_string(cfg, indent=0):
    """Creates a multi-line string with the contents of @cfg."""

    text = ""
    for key, value in cfg.__dict__.items():
        text += " " * indent + clr(key, "yellow") + ": "
        if isinstance(value, Namespace):
            text += "\n" + config_to_string(value, indent + 2)
        else:
            text += clr(str(value), "red") + "\n"
    return text


def dict_to_namespace(dct):
    namespace = Namespace()
    for key, value in dct.items():
        name = key.rstrip("_")
        if isinstance(value, dict) and not key.endswith("_"):
            setattr(namespace, name, dict_to_namespace(value))
        else:
            setattr(namespace, name, value)
    return namespace


def parse_args():
    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        '-dcf' '--default_config_fle',
        default='default',
        dest='default_config_file',
        help='Default configuration file'
    )
    arg_parser.add_argument(
        '-cf' '--config_file',
        default=['default'],
        nargs="+",
        dest='config_files',
        help='Configuration file.'
    )
    arg_parser.add_argument(
        '-id' '--id',
        default=0, type=int,
        dest='run_id',
        help='Id for current run.'
    )
    arg_parser.add_argument(
        '-e' '--experiment',
        type=str,
        default="",
        dest="experiment",
        help="Experiment. Overrides cf and dcf"
    )
    arg_parser.add_argument(
        '--resume',
        default=None,
        dest="resume"
    )
    arg_parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Name of checkpoint for resumed experiment"
    )
    arg_parser.add_argument(
        "-va", "--visualise_activations",
        default=False,
        action="store_true",
        dest="activations",
        help='show network activations for seen images'
    )
    arg_parser.add_argument(
        "--no_cuda",
        default=False,
        action="store_true",
        dest="no_cuda",
        help="overwrite use_cuda from config.yaml")

    arg_parser.add_argument(
        "--eval",
        default=False,
        action="store_true",
        dest="eval_model",
        help="check if the agent is in eval mode or train mode")
    arg_parser.add_argument(
        "--path_to_image",
        type=str,
        default="",
        help="path to a specific image you want to evaluate the agent on")
    arg_parser.add_argument(
        "--image_number",
        type=int,
        default=-1,
        help="If image is for h5 file, specify the image number")

    arg_parser.add_argument(
        "--save_image",
        type=str,
        default="NetworkActivation",
        help="Save the image as the name given")

    return arg_parser.parse_args()


def read_config():
    """Reads an YAML config file and transforms it to a Namespace

    It reads two config files and combines their info into a Namespace. One is
    supposed to be  the default configuration with the common settings for most
    experiments, while the other specifies what changes. The YAML structure is
    transformed into a Namespace excepting keys ending with '_'. The reason for
    this behaviour is the following: sometimes you want to overwrite a full
    dictionary, not just specific values (e.g. args for optimizer).

    """
    import yaml
    import os.path

    args = parse_args()

    if args.resume:
        if os.path.isdir(args.resume):
            root_path = args.resume
            assert os.path.isdir(root_path)
            algos = os.listdir(root_path)

            # Only use works if there's only one algorithm in the experiment folder
            # TODO could use --experiment to select particular experiment
            assert len(algos) == 1
            path = os.path.join(root_path, algos[0])
            config_files = ["cfg.yaml"]
            default_config_file = 'cfg.yaml'
            # Resume experiment from this folder
        else:
            # Check for last experiment with name...
            # , weird stuff here
            from os import listdir
            name = args.resume
            all_exps = os.listdir("./experiments/")
            runs = [f for f in all_exps if f.endswith(name)]
            assert len(runs) > 0
            last_time = str(max([int(f.split("_")[0]) for f in runs]))
            exp_folder = [f for f in runs if f == (last_time + "_" + name)]
            assert len(exp_folder) == 1
            root_path = os.path.join("experiments", exp_folder[0])
            assert os.path.isdir(root_path)
            algos = os.listdir(root_path)

            # Only use works if there's only one algorithm in the experiment folder
            # TODO could use --experiment to select particular experiment
            assert len(algos) == 1
            path = os.path.join(root_path, algos[0])
            config_files = ["cfg.yaml"]
            default_config_file = 'cfg.yaml'
    elif args.experiment:
        from os import listdir
        path = f"./configs/{args.experiment:s}/"
        config_files = [f for f in listdir(path)
                        if f.startswith(args.experiment + "_")
                        and f.endswith(".yaml")]
        default_config_file = "default.yaml"
        print(f"Found {len(config_files):d} experiments!")
    else:
        path = f"./configs/"
        config_files = [f + ".yaml" for f in args.config_files]
        default_config_file = args.default_config_file + ".yaml"

    cfgs = []
    for config_file in config_files:
        with open(os.path.join(path, config_file)) as handler:
            config_data = yaml.load(handler, Loader=yaml.SafeLoader)
        cfg = dict_to_namespace(config_data)

        if default_config_file != config_file:
            with open(os.path.join(path, default_config_file)) as handler:
                default_cfg_data = yaml.load(handler, Loader=yaml.SafeLoader)
            default_cfg = dict_to_namespace(default_cfg_data)
            _update_config(default_cfg, cfg)
            cfg = default_cfg

        cfg.resume = args.resume
        cfg.run_id = args.run_id
        cfg.agent.activations = args.activations
        cfg.checkpoint = args.checkpoint
        cfg.eval_model = args.eval_model
        cfg.path_to_image = args.path_to_image
        cfg.image_number = args.image_number
        cfg.save_image = args.save_image
        if args.no_cuda:
            cfg.agent.use_cuda = False
        cfgs.append(cfg)

        if cfg.verbose > 0:
            import sys
            sys.stdout.write(f"{clr('[Config]', 'red'):s} ")
            if default_config_file != config_file:
                print(f"Read {config_file:s} over {default_config_file:s}.")
            else:
                print(f"Read {config_file:s}.")

    return cfgs[0] if len(cfgs) == 1 else cfgs


def add_common(name_space, common_namespace):
    for k in name_space.__dict__.keys():
        if isinstance(getattr(name_space, k), Namespace):
            getattr(name_space, k).common = common_namespace


def save_config(path, cfg, override=False):
    cfg_file = path + ".yaml"

    if not os.path.isfile(cfg_file) or not override:
        with open(cfg_file, "w") as yaml_file:
            yaml.safe_dump(namespace_to_dct(cfg), yaml_file,
                           default_flow_style=False)

def resume(path):
    resume_checkpoint = None
    resume_prefix = None
    checkpoints = list(map(lambda x: int(x.split('_')[2]), os.listdir(path)))

    if len(checkpoints) > 0:
        resume_checkpoint = max(checkpoints)
        resume_prefix = os.path.join(path, f"step_{resume_checkpoint:d}__")

        files_for_checkpoint = map(lambda x: f"step_{resume_checkpoint:d}__{x:s}", ['metrics.pkl', 'results.pkl'])
        for f in files_for_checkpoint:
            assert f in os.listdir(path), "The checkpoint should have all the necessary files"

        log.info(f"Resuming agent from checkpoint {resume_checkpoint:d}")

    return resume_prefix


def generate_configs():
    """@Tudor: Generate multiple configs with results folders """

    cfgs = read_config()
    cfg0 = cfgs if type(cfgs) != list else cfgs[0]
    cfgs = [cfgs] if type(cfgs) != list else cfgs

    # Common namespace added to all config first order children
    common_namespace = Namespace()

    procs_no = cfg0.procs_no
    results_folder = cfg0.save_folder + "/"
    root_path = None

    if cfg0.resume:
        if os.path.isdir(cfg0.resume):
            root_path = cfg0.resume
            log.info("Resuming", root_path, "!")
        else:
            name = cfg0.name
            all_exps = os.listdir(results_folder)
            runs = [f for f in all_exps if f.endswith(name)]
            if len(runs) > 0:
                last_time = str(max([int(f.split("_")[0]) for f in runs]))
                log.info("Resuming", last_time, "!")
                exp_folder = [f for f in all_exps if f == (last_time + "_" + name)]
                assert len(exp_folder) == 1
                root_path = os.path.join(results_folder, exp_folder[0])
                assert os.path.isdir(root_path)

    if root_path is None:
        root_path = results_folder + f"{int(time()):d}_{cfg0.name:s}/"
        assert not os.path.exists(root_path)
        os.makedirs(root_path)
    args = []
    for j, cfg in enumerate(cfgs):
        title = cfg.title
        for c in " -.,=:;/()":
            title = title.replace(c, "_")
        alg_path = os.path.join(root_path, f"{j:d}_{title:s}")
        if not os.path.isdir(alg_path):
            os.makedirs(alg_path, exist_ok=True)
        cfg_file = os.path.join(alg_path, "cfg")
        save_config(cfg_file, cfg)

        for run_id in range(cfg.runs):
            exp_path = os.path.join(alg_path, f"{run_id:d}")
            if not os.path.isdir(exp_path):
                os.makedirs(exp_path)
            common_namespace.save_path = exp_path

            add_common(cfg, common_namespace)

            args.append((cfg, run_id, exp_path))

    return procs_no, args


