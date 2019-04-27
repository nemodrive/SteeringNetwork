import torch


def set_seed(seed, use_cuda=True):
    import random
    import numpy as np

    if seed == 0:
        seed = random.randint(1, 9999999)

    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    return seed


def report_net_param(model):
    param_avg, grad_avg = .0, .0
    param_max, grad_max = None, None
    param_groups_count = .0
    for p in model.parameters():
        param_avg += p.data.abs().mean()
        grad_avg += p.grad.data.abs().mean()
        p_max = p.data.abs().max()
        g_max = p.grad.data.abs().max()
        param_max = max(p_max, param_max) if param_max else p_max
        grad_max = max(g_max, grad_max) if grad_max else g_max
        param_groups_count += 1

    param_avg = param_avg / param_groups_count
    grad_avg = grad_avg / param_groups_count
    return param_max, param_avg, grad_max, grad_avg
