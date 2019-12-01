import torch.nn.functional as F
from vae import utils


def bernoulli_nll(y_logits, y_target):
    y_logits, y_target = utils.check_and_flatten(y_logits, y_target)
    # return data to range [0, 1]
    y_target = y_target + 0.5
    return F.binary_cross_entropy_with_logits(input=y_logits, target=y_target, reduction='none').sum(dim=1)


def squared_error(y, y_target):
    y, y_target = utils.check_and_flatten(y, y_target)
    return ((y - y_target) ** 2).sum(dim=1)
