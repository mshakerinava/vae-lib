#!/bin/bash

# MNIST (~11 MB)
mkdir -p mnist
cd mnist
wget --continue http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget --continue http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget --continue http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget --continue http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
cd ..

# Fashion MNIST (~30 MB)
mkdir -p fashion_mnist
cd fashion_mnist
wget --continue http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
wget --continue http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
wget --continue http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
wget --continue http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
cd ..
