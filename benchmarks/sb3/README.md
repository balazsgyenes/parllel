## Benchmarking against SB3

[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) is a very popular RL library with broad acceptance across the community.

To benchmark against SB3, we train policies on [`Hopper-v4`](https://gymnasium.farama.org/environments/mujoco/hopper/) and [`HalfCheetah-v4`](https://gymnasium.farama.org/environments/mujoco/half_cheetah/) using both SB3 and `parllel` and compare the results. Hyperparameters are taken from [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo) and modified slightly, such that the configurations are as similar as possible for both frameworks. Here, the focus is not to train with optimal parameters, but to minimize configuration differences between SB3 and `parllel`.

We cannot directly compare against previously published performance metrics for SB3 (e.g. [here](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html#results), with full training curves [here](https://github.com/DLR-RM/stable-baselines3/issues/48)), because these values depend heavily on hyperparameters (e.g. gSDE) and the exact environment version (`HalfCheetahBulletEnv-v0` vs. `HalfCheetah-v4`). Instead, we train policies from scratch.

## Running the Benchmarks

Run a benchmarking script once using: `python train_[algo]_[framework].py --config-name [algo]_[framework]_[env_name]`

For example, train SAC on HalfCheetah using parllel using: `python train_sac_parllel.py --config-name sac_parllel_cheetah`

Run a benchmarking script with several (e.g. 5) random seeds using: `python train_[algo]_[framework].py --config-name [algo]_[framework]_[env_name] -m +iteration="range(5)"`

Run all benchmarking scripts with 5 random seeds each using: `bash train_all.sh`

# Benchmarking Results

The training results using 5 random seeds for each framework can be accessed at this [public WandB report](https://api.wandb.ai/links/gyenes/96m63ytt).
