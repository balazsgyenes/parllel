import multiprocessing as mp

import torch
import wandb

from parllel.configuration import add_default_config_fields
import parllel.logger as logger
from parllel.logger import Verbosity
from parllel.torch.algos.ppo import add_default_ppo_config

from train_ppo import build


if __name__ == "__main__":
    mp.set_start_method("fork")

    config = dict(
        parallel = True,
        batch_T = 128,
        batch_B = 16,
        discount = 0.99,
        learning_rate = 0.001,
        gae_lambda = 0.95,
        reward_clip_min = -5,
        reward_clip_max = 5,
        obs_norm_initial_count = 10000,
        normalize_advantage = True,
        max_steps_decorrelate = 50,
        env = dict(
            max_episode_steps = 1000,
        ),
        device = "cuda:0" if torch.cuda.is_available() else "cpu",
        model = dict(
            hidden_sizes = [64, 64],
            hidden_nonlinearity=torch.nn.Tanh,
        ),
        runner = dict(
            n_steps = 50 * 16 * 128,
            log_interval_steps = 5 * 16 * 128,
        ),
    )

    config = add_default_ppo_config(config)
    config = add_default_config_fields(config)

    run = wandb.init(
        anonymous="allow", # for this example, wandb should not be mandatory
        project="CartPole",
        group="PPO",
        tags=["discrete", "state-based", "ppo"],
        config=config,
        sync_tensorboard=True,  # auto-upload any values logged to tensorboard
        monitor_gym=True,  # auto-upload any videos recorded by gym's VideoRecorder
        save_code=True,  # save script used to start training, git commit, and patch
    )

    logger.init(
        tensorboard=True,
        wandb=run,
        output_files={
            "txt": "log.txt",
            # "csv": "progress.csv",
        },
        config=config,
        model_save_path="model.pt",
        # verbosity=Verbosity.DEBUG,
    )

    # # record videos of the policy/environment during training
    # config["video_recorder"] = dict(
    #     video_folder=f"videos/{run.id}",
    #     step_trigger=lambda x: x % 2000 == 0, # record every 2000 steps/env
    #     video_length=200,
    # )
    
    # TODO: suppress warning about overwriting folder
    # TODO: do not record decorrelation steps
    # TODO: log videos without calling `log`, which causes tensorboard to go out of sync

    with build(config) as runner:
        runner.run()

    run.finish()

    # TODO: minimal
    # add support for recording videos of rollouts
    # update other example scripts

    # TODO: future
    # detect if wandb was initialized after parllel logging was initialized
    # add separate wandb writer, making tensorboard optional
    # add additional serializers for config files
        # investigate using pickle to serialize classes (but needs to be detectable on read)
        # ensure no collisions with wandb's config.yaml file

    # TODO: test
    # custom fields in traj_info for logging
    # do not init logging
    # init wandb second
    # verify that writing to wandb folder is equivalent to wandb.save(policy="live")
    # issues with pickling lambda for record trigger? (try also in spawn mode)
    # no log_dir
    # no model_save_path
    # no config
    # wandb disabled
    # explicit config_save_path
    # Why does sb3 use cloud pickle?
