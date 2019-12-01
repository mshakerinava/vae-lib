#!/bin/bash

# MNIST (~11 MB)
mkdir -p mnist
cd mnist
wget --continue http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget --continue http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget --continue http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget --continue http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
cd ..
