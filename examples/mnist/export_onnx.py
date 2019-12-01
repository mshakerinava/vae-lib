import sys
import torch
import torch.nn as nn
import numpy as np

sys.path.append('../..')
import vae


class Net(nn.Module):
    def __init__(self, dec):
        super().__init__()
        self.dec = dec

    def forward(self, x):
        y_logits = self.dec(x)
        y = torch.sigmoid(y_logits) * 255.0
        return y


# load VAE
vae_module = vae.load('vae_best.tar')
net = Net(dec=vae_module.dec)

# export to ONNX
dummy_x = torch.randn(1, 2)
torch.onnx.export(net, dummy_x, "vae-dec.onnx", verbose=True)
