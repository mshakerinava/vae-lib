import sys
import json
import inspect
import argparse
import torch
import torch.nn as nn
import numpy as np
import vae
from vae import loss
from vae import utils


def add_main_arguments(arg_parser, prefix=''):
    arg_parser.add_argument('--%senc' % prefix, type=str, default='MLP', help='VAE encoder network')
    arg_parser.add_argument('--%sdec' % prefix, type=str, default='MLP', help='VAE decoder network')
    arg_parser.add_argument('--%sloss-fn' % prefix, type=str, default='squared_error', help='VAE reconstruction loss function')
    arg_parser.add_argument('--%scode-size' % prefix, type=int, default=2, help='VAE code size (size of z)')
    arg_parser.add_argument('--%skl-tolerance' % prefix, type=float, default=0, help='Only optimize KL divergence if its mean is above this threshold')


def parse_sub_arguments(module_name, argv, prefix=''):
    arg_parser = argparse.ArgumentParser(allow_abbrev=False)
    module = eval('vae.' + module_name)
    module.add_arguments(arg_parser, prefix)
    args, _ = arg_parser.parse_known_args(argv)
    print('parsed sub-arguments = %s' % json.dumps(vars(args), sort_keys=True, indent=4), file=sys.stderr)
    args = vars(args)
    if prefix != '':
        prefix = prefix.replace('-', '_')
        args = utils.remove_prefix(args, prefix)
    return args


def load(file):
    d = torch.load(file)
    vae_module = VAE(**d['init_kwargs'])
    vae_module.load_state_dict(d['net_state_dict'])
    return vae_module


class VAE(nn.Module):
    def __init__(self, enc, enc_kwargs, dec, dec_kwargs, data_shape, code_size, loss_fn, kl_tolerance):
        super().__init__()
        self.init_kwargs = locals().copy()
        self.init_kwargs.pop('__class__')
        self.init_kwargs.pop('self')
        self.enc = eval('vae.enc.%s' % enc)(data_shape, code_size, **enc_kwargs)
        self.dec = eval('vae.dec.%s' % dec)(data_shape, code_size, **dec_kwargs)
        self.data_shape = data_shape
        self.code_size = code_size
        self.loss_fn = eval('loss.%s' % loss_fn)
        self.kl_tolerance = kl_tolerance
        self.opt = None

    def save(self, file):
        torch.save({
            'init_kwargs': self.init_kwargs,
            'net_state_dict': self.state_dict(),
        }, file)

    def set_optimizer(self, opt):
        self.opt = opt

    def get_device(self):
        # NOTE: This method only makes sense when all module parameters reside on the **same** device.
        return list(self.parameters())[0].device

    def forward(self, x, eps):
        mean, log_var = self.enc(x)
        z = utils.reparameterize_normal(mean, log_var, eps)
        y = self.dec(z)
        return mean, log_var, z, y

    def loss(self, x, mean, log_var, y):
        B = x.shape[0]
        assert x[0].shape == self.data_shape
        assert mean.shape == (B, self.code_size)
        assert log_var.shape == (B, self.code_size)
        coding_loss = utils.kl_normal(mean, log_var, torch.zeros_like(mean), torch.zeros_like(log_var)).sum(dim=1)
        thresh = torch.tensor(self.kl_tolerance * self.code_size, dtype=torch.float32, device=self.get_device())
        coding_loss_thresh = torch.max(coding_loss, thresh).mean()
        coding_loss = coding_loss.mean()
        reconstruction_loss = self.loss_fn(y, x).mean()
        total_loss = coding_loss_thresh + reconstruction_loss
        return total_loss, coding_loss, reconstruction_loss

    def train_dataset(self, dataset, num_passes, batch_size, opt=None):
        self.train()
        m = dataset.shape[0]
        assert batch_size <= m
        opt = opt or self.opt
        assert opt
        total_loss_list = []
        coding_loss_list = []
        reconstruction_loss_list = []
        while num_passes > 0:
            np.random.shuffle(dataset)
            cur_passes = min(num_passes, m // batch_size)
            num_passes -= cur_passes
            for i in range(cur_passes):
                x = torch.tensor(dataset[i * batch_size: (i + 1) * batch_size], device=self.get_device())
                eps = torch.randn([batch_size, self.code_size], device=self.get_device())
                mean, log_var, z, y = self.forward(x, eps)
                total_loss, coding_loss, reconstruction_loss = self.loss(x, mean, log_var, y)
                opt.zero_grad()
                total_loss.backward()
                opt.step()
                total_loss_list.append(total_loss.item())
                coding_loss_list.append(coding_loss.item())
                reconstruction_loss_list.append(reconstruction_loss.item())
        return total_loss_list, coding_loss_list, reconstruction_loss_list

    def eval_dataset(self, dataset, batch_size):
        self.eval()
        m = dataset.shape[0]
        total_loss_sum = 0
        coding_loss_sum = 0
        reconstruction_loss_sum = 0
        with torch.no_grad():
            for i in range(m // batch_size + 1 if m % batch_size != 0 else 0):
                x = torch.tensor(dataset[i * batch_size: min(m, (i + 1) * batch_size)], device=self.get_device())
                cur_batch_size = x.shape[0]
                eps = torch.zeros([cur_batch_size, self.code_size], device=self.get_device())
                mean, log_var, z, y = self.forward(x, eps)
                total_loss, coding_loss, reconstruction_loss = self.loss(x, mean, log_var, y)
                total_loss_sum += total_loss.item() * cur_batch_size
                coding_loss_sum += coding_loss.item() * cur_batch_size
                reconstruction_loss_sum += reconstruction_loss.item() * cur_batch_size
        total_loss = total_loss_sum / m
        coding_loss = coding_loss_sum / m
        reconstruction_loss = reconstruction_loss_sum / m
        return total_loss, coding_loss, reconstruction_loss

    def sample(self, n=1):
        z = torch.randn((n, self.code_size), device=self.get_device())
        y = self.dec(z)
        return y
