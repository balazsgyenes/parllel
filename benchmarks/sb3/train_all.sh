#!/bin/bash

# get path to this script
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

python $SCRIPT_DIR/train_sac_cheetah_parllel.py -m +iteration="range(5)"
python $SCRIPT_DIR/train_sac_hopper_parllel.py -m +iteration="range(5)"
python $SCRIPT_DIR/train_sac_cheetah_sb3.py -m +iteration="range(5)"
python $SCRIPT_DIR/train_sac_hopper_sb3.py -m +iteration="range(5)"
