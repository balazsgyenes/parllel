#!/bin/bash

# get path to this script
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

python $SCRIPT_DIR/train_sac_parllel.py --config-name sac_cheetah_parllel -m +iteration="range(5)"
python $SCRIPT_DIR/train_sac_parllel.py --config-name sac_hopper_parllel -m +iteration="range(5)"
python $SCRIPT_DIR/train_sac_parllel.py --config-name sac_walker_parllel -m +iteration="range(5)"

python $SCRIPT_DIR/train_sac_sb3.py --config-name sac_cheetah_sb3 -m +iteration="range(5)"
python $SCRIPT_DIR/train_sac_sb3.py --config-name sac_hopper_sb3 -m +iteration="range(5)"
python $SCRIPT_DIR/train_sac_sb3.py --config-name sac_walker_sb3 -m +iteration="range(5)"

python $SCRIPT_DIR/train_ppo_parllel.py --config-name ppo_cheetah_parllel -m +iteration="range(5)"

python $SCRIPT_DIR/train_ppo_sb3.py --config-name ppo_cheetah_sb3 -m +iteration="range(5)"
