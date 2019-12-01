import os
import glob
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--log-file', default='log.csv', type=str)
parser.add_argument('--offset', default=5, type=int)
args = parser.parse_args()


def parse_csv(file):
    ret = {}
    with open(file, mode='r') as f:
        lines = [x.strip() for x in f.readlines()]
        keys = [x.strip().lower() for x in lines[0].split(',')]
        for key in keys:
            ret[key] = []
        for line in lines[1:]:
            vals = [x.strip() for x in line.split(',')]
            assert len(vals) == len(keys)
            for i in range(len(vals)):
                ret[keys[i]].append(vals[i])
    return ret


d = parse_csv(args.log_file)
x = np.array([int(x) for x in d['epoch']])

with plt.style.context('seaborn'):
    plt.rcParams.update({'font.size': 22})
    fig = plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    y = np.array([float(a) for a in d['total loss']])
    plt.plot(x[args.offset:], y[args.offset:])
    plt.title('Total Loss')
    plt.xlabel('Epochs')
    plt.ylabel('NLL + KL')

    plt.subplot(1, 3, 2)
    y = np.array([float(a) for a in d['reconstruction loss']])
    plt.plot(x[args.offset:], y[args.offset:])
    plt.title('Reconstruction Loss')
    plt.xlabel('Epochs')
    plt.ylabel('NLL')
    
    plt.subplot(1, 3, 3)
    y = np.array([float(a) for a in d['coding loss']])
    plt.plot(x[args.offset:], y[args.offset:])
    plt.title('Coding Loss')
    plt.xlabel('Epochs')
    plt.ylabel('KL')

    plt.tight_layout()
    fig.savefig('plot_loss.png', bbox_inches='tight', dpi=600)
    fig.savefig('plot_loss.svg', bbox_inches='tight')
    plt.close()
