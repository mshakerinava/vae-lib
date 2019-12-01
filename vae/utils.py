import torch


def reparameterize_normal(mean, log_var, eps):
    std = torch.exp(log_var / 2)
    return mean + eps * std


def kl_normal(mean_1, log_var_1, mean_2, log_var_2):
    return (torch.exp(log_var_1) + (mean_1 - mean_2) ** 2) / (2 * torch.exp(log_var_2)) \
           + 0.5 * (log_var_2 - log_var_1) - 0.5


def check_and_flatten(x, y):
    assert x.shape == y.shape
    B = x.shape[0]
    x = x.view(B, -1)
    y = y.view(B, -1)
    return x, y


def remove_prefix(kwargs, prefix):
    assert len(prefix) > 0
    ret = {}
    for key, val in kwargs.items():
        if key.startswith(prefix):
            ret[key[len(prefix):]] = val
    return ret
