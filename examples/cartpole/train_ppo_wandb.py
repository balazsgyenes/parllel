from datetime import datetime
import multiprocessing as mp
from pathlib import Path

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
        anonymous="must", # for this example, send to wandb dummy account
        project="CartPole",
        group="PPO",
        tags=["discrete", "state-based", "ppo"],
        config=config,
        sync_tensorboard=True,  # auto-upload any values logged to tensorboard
        monitor_gym=True,  # auto-upload any videos recorded by gym's VideoRecorder
        save_code=True,  # save script used to start training, git commit, and patch
    )

    logger.init(
        # log_dir=Path(f"log_data/cartpole-ppo/{datetime.now().strftime('%Y-%m-%d_%H-%M')}"),
        tensorboard=True,
        wandb_run=run,
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
    # TODO: test pickling lambda for record trigger (try also in spawn mode)

    with build(config) as runner:
        runner.run()

    run.finish()
