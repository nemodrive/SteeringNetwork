from .udacity_image_dataset import UdacityImageSteerDataset
from .conditional_imitation_dataset import ConditionalImitationLearningDataset
from .conditional_imitation_dataset import ConditionalImitationLearningSampler
from .bddv_dataset import BDDVDataset, BDDVSampler

DATA_SET = {
    "UdacityImageSteerDataset": UdacityImageSteerDataset,
    "ConditionalImitationLearningDataset": ConditionalImitationLearningDataset,
    "BDDVDataset": BDDVDataset
}

SAMPLER = {
    "ConditionalImitationLearningSampler": ConditionalImitationLearningSampler,
    "BDDVSampler": BDDVSampler
}


def get_dataset(cfg):
    assert hasattr(
        cfg, "name"
    ) and cfg.name in DATA_SET, "Please provide a valid dataset name."
    return DATA_SET[cfg.name]


def get_sampler(cfg):
    assert hasattr(
        cfg, "sampler_name"
    ) and cfg.sampler_name in SAMPLER, "Please provide a valid sampler name"
    return SAMPLER[cfg.sampler_name]
