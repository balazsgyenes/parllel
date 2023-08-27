#!/bin/bash

python benchmarks/sb3/train_sac_cheetah_parllel.py -m +iteration="range(5)"
python benchmarks/sb3/train_sac_hopper_parllel.py -m +iteration="range(5)"
python benchmarks/sb3/train_sac_cheetah_sb3.py -m +iteration="range(5)"
python benchmarks/sb3/train_sac_hopper_sb3.py -m +iteration="range(5)"
