# Andrei@Ian2018
from . import cloning_nvidia_model
from . import cloning_simple_model
from . import conditional_imitation_model
from . import classification_conditional_imitation_model
""" Each model script should have the method get_models() which returns a list of models """

ALL_MODELS = {
    "CloningSimpleModel":
    cloning_simple_model,
    "CloningNVIDIAModel":
    cloning_nvidia_model,
    "ConditionalImitationModel":
    conditional_imitation_model,
    "ClassificationConditionalImitationModel":
    classification_conditional_imitation_model,
}


def get_models(cfg):
    # @name         : name of the model
    assert hasattr(
        cfg, "name"
    ) and cfg.name in ALL_MODELS, "Please provide a valid model name."
    return ALL_MODELS[cfg.name].get_models()
