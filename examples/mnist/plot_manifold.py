import sys
import argparse
import torch
import numpy as np
from scipy.special import erfinv
from PIL import Image

sys.path.append('../..')
import vae

parser = argparse.ArgumentParser()
parser.add_argument('--num-rows', default=10, type=int)
parser.add_argument('--num-cols', default=10, type=int)
parser.add_argument('--margin', default=0.5, type=float)
parser.add_argument('--scale', default=4, type=float)
parser.add_argument('--cpu', action='store_false', dest='gpu')
args = parser.parse_args()

# create VAE
DEVICE = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
print('DEVICE = %s' % str(DEVICE))
vae_module = vae.load('vae_best.tar')
vae_module.to(DEVICE)


def inv_cdf(x):
    return np.sqrt(2) * erfinv(2 * x - 1)


z = np.zeros((args.num_rows, args.num_cols, vae_module.code_size))
for i in range(args.num_rows):
    for j in range(args.num_cols):
        z[i, j] = np.array([
            inv_cdf((i + args.margin) / (args.num_rows + 2 * args.margin)),
            inv_cdf((j + args.margin) / (args.num_cols + 2 * args.margin))
        ])

z = np.reshape(z, (args.num_rows * args.num_cols, vae_module.code_size))
z = torch.tensor(z, dtype=torch.float32, device=DEVICE)
with torch.no_grad():
    y_logits = vae_module.dec(z)
y = torch.sigmoid(y_logits)
y = np.uint8(y.cpu().numpy() * 255)
height, width = vae_module.data_shape[1:]
y = np.reshape(y, (args.num_rows, args.num_cols, height, width))
y = np.transpose(y, (1, 2, 0, 3))
y = np.reshape(y, (args.num_rows * height, args.num_cols * width))
img_array = y

img = Image.fromarray(img_array, mode='L')
img = img.resize((img_array.shape[1] * args.scale, img_array.shape[0] * args.scale), resample=Image.BILINEAR)
img.save('manifold.png')
