from .udacity_cloning_loader import UdacityCloningDataLoader
from .conditional_imitation_loader import ConditionalImitationLoader
DATA_LOADERS = {
    "UdacityCloningDataLoader": UdacityCloningDataLoader,
    "ConditionalImitationDataLoader": ConditionalImitationLoader
}


def get_data_loader(cfg):
    assert hasattr(cfg, "name") and cfg.name in DATA_LOADERS,\
        "Please provide a valid agent name."
    return DATA_LOADERS[cfg.name](cfg)
