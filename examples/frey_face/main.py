import sys
import json
import random
import argparse
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from scipy.io import loadmat
from log_utils import log, log_tabular, clear_logs

sys.path.append('../..')
import vae

# parse main arguments
argv = sys.argv[1:]
parser = argparse.ArgumentParser()
parser.add_argument('--num-epochs', type=int, default=1000, help='Number of epochs to train')
parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training and evaluation')
parser.add_argument('--seed', type=int, default=1, help='Manual PRNG seed for reproducibility')
parser.add_argument('--cpu', action='store_false', dest='gpu', help='Only use CPUs')
vae.add_main_arguments(arg_parser=parser)
# can potentially add arguments for more VAEs with different prefixes here...
args, _ = parser.parse_known_args(argv)
log('parsed main-arguments = %s' % json.dumps(vars(args), sort_keys=True, indent=4))
# prefixes can be removed from `args` with `vae.utils.remove_prefix` here, if necessary.

# set RNG seeds
random.seed(args.seed, version=2)
np.random.seed(random.randint(0, 2**32 - 1))
torch.manual_seed(random.randint(0, 2**32 - 1))

# ensure reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# parse sub-arguments
enc_kwargs = vae.parse_sub_arguments(module_name='enc.' + args.enc, prefix='enc-', argv=argv)
dec_kwargs = vae.parse_sub_arguments(module_name='dec.' + args.dec, prefix='dec-', argv=argv)

# load dataset
HEIGHT, WIDTH = 28, 20
data = loadmat('../../data/frey_rawface.mat', squeeze_me=True, struct_as_record=False)['ff']
data = data.T.reshape(-1, 1, HEIGHT, WIDTH)
data = data.astype(np.float32) / 255.0 - 0.5
np.random.shuffle(data)

VAL_SIZE = data.shape[0] // 20
TRAIN_SIZE = data.shape[0] - VAL_SIZE
data_val = data[:VAL_SIZE]
data_train = data[VAL_SIZE:]

# create VAE
DEVICE = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
print('DEVICE = %s' % str(DEVICE))

vae_module = vae.VAE(enc=args.enc, enc_kwargs=enc_kwargs, dec=args.dec, dec_kwargs=dec_kwargs,
    data_shape=(1, HEIGHT, WIDTH), code_size=args.code_size, loss_fn=args.loss_fn, kl_tolerance=args.kl_tolerance)
vae_module.to(DEVICE)
opt = optim.Adam(vae_module.parameters(), lr=1e-3)
vae_module.set_optimizer(opt)
log(vae_module)

# train VAE
keys = ['Epoch', 'Coding Loss', 'Reconstruction Loss', 'Total Loss']
formats = ['%04d', '%9.4f', '%9.4f', '%9.4f']
clear_logs()
log_tabular(vals=keys)
total_loss, coding_loss, reconstruction_loss = vae_module.eval_dataset(dataset=data_val, batch_size=args.batch_size)
log_tabular(vals=[0, coding_loss, reconstruction_loss, coding_loss + reconstruction_loss], keys=keys, formats=formats)

best_loss = np.inf
total_loss_list = [total_loss]
coding_loss_list = [coding_loss]
reconstruction_loss_list = [reconstruction_loss]
for epoch in range(args.num_epochs):
    num_passes = TRAIN_SIZE // args.batch_size
    vae_module.train_dataset(dataset=data_train, num_passes=num_passes, batch_size=args.batch_size)
    total_loss, coding_loss, reconstruction_loss = vae_module.eval_dataset(dataset=data_val, batch_size=args.batch_size)
    total_loss_list.append(total_loss)
    coding_loss_list.append(coding_loss)
    reconstruction_loss_list.append(reconstruction_loss)
    log_tabular(vals=[epoch + 1, coding_loss, reconstruction_loss, total_loss], keys=keys, formats=formats)
    if total_loss < best_loss:
        best_loss = total_loss
        vae_module.save('vae_best.tar')

vae_module.save('vae_latest.tar')
total_loss_avg = np.mean(total_loss_list[-10:])
coding_loss_avg = np.mean(coding_loss_list[-10:])
reconstruction_loss_avg = np.mean(reconstruction_loss_list[-10:])
log('\nLast 10 Epochs | Coding Loss: %f | Reconstruction Loss: %f | Total Loss: %f' % (
    coding_loss_avg, reconstruction_loss_avg, total_loss_avg))
