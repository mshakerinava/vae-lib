#!/bin/bash

python3 main.py --num-epochs 1000 --batch-size 128 --enc-widths 500 --dec-widths 500 --kl-tolerance 0.5 --loss-fn bernoulli_nll --seed 123
