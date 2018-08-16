from . import demo_agent
from . import conditional_imitation_agent
from . import classification_conditional_imitation_agent

AGENTS = {
    "DemoAgent":
    demo_agent.DemoAgent,
    "ConditionalImitationAgent":
    conditional_imitation_agent.ConditionalImitationAgent,
    "ClassificationConditionalImitationAgent":
    classification_conditional_imitation_agent.
    ClassificationConditionalImitationAgent,
}


def get_agent(cfg):
    assert hasattr(
        cfg,
        "name") and cfg.name in AGENTS, "Please provide a valid agent name."
    return AGENTS[cfg.name](cfg)
