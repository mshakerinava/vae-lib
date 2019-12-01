#!/bin/bash

python3 main.py --num-epochs 10000 --batch-size 128 --enc-widths 200 --dec-widths 200 --kl-tolerance 0.5 --loss-fn bernoulli_nll --seed 123
