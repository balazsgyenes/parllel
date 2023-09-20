#!/bin/bash

# get path to this script
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

python $SCRIPT_DIR/train_sac_parllel.py --config-name sac_parllel_cheetah -m +iteration="range(5)"
python $SCRIPT_DIR/train_sac_parllel.py --config-name sac_parllel_hopper -m +iteration="range(5)"
python $SCRIPT_DIR/train_sac_parllel.py --config-name sac_parllel_walker -m +iteration="range(5)"

python $SCRIPT_DIR/train_ppo_parllel.py --config-name ppo_parllel_cheetah -m +iteration="range(5)"
python $SCRIPT_DIR/train_ppo_parllel.py --config-name ppo_parllel_hopper -m +iteration="range(5)"
python $SCRIPT_DIR/train_ppo_parllel.py --config-name ppo_parllel_walker -m +iteration="range(5)"

python $SCRIPT_DIR/train_sac_sb3.py --config-name sac_sb3_cheetah -m +iteration="range(5)"
python $SCRIPT_DIR/train_sac_sb3.py --config-name sac_sb3_hopper -m +iteration="range(5)"
python $SCRIPT_DIR/train_sac_sb3.py --config-name sac_sb3_walker -m +iteration="range(5)"

python $SCRIPT_DIR/train_ppo_sb3.py --config-name ppo_sb3_cheetah -m +iteration="range(5)"
python $SCRIPT_DIR/train_ppo_sb3.py --config-name ppo_sb3_hopper -m +iteration="range(5)"
python $SCRIPT_DIR/train_ppo_sb3.py --config-name ppo_sb3_walker -m +iteration="range(5)"
