from .udacity_cloning_loader import UdacityCloningDataLoader
from .conditional_imitation_loader import ConditionalImitationLoader
from .bddv_loader import BDDVLoader

DATA_LOADERS = {
    "UdacityCloningDataLoader": UdacityCloningDataLoader,
    "ConditionalImitationDataLoader": ConditionalImitationLoader,
    "BDDVLoader": BDDVLoader
}


def get_data_loader(cfg):
    assert hasattr(cfg, "name") and cfg.name in DATA_LOADERS,\
        "Please provide a valid loader name."
    return DATA_LOADERS[cfg.name](cfg)
