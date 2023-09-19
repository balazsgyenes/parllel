from contextlib import contextmanager
from functools import partial
from typing import Iterator

# isort: off
import gymnasium as gym
import torch
import wandb
from gymnasium import spaces
from omegaconf import DictConfig

# isort: on
import parllel.logger as logger
from parllel.callbacks import RecordingSchedule
from parllel.patterns import (
    add_advantage_estimation,
    add_agent_info,
    add_bootstrap_value,
    add_obs_normalization,
    add_reward_clipping,
    add_reward_normalization,
    build_cages,
    build_sample_tree,
)
from parllel.replays import BatchedDataLoader
from parllel.runners import RLRunner
from parllel.samplers import BasicSampler
from parllel.samplers.eval import EvalSampler
from parllel.torch.agents.gaussian import GaussianPgAgent
from parllel.torch.algos.ppo import PPO, build_loss_sample_tree
from parllel.torch.distributions import Gaussian
from parllel.transforms import RecordVectorizedVideo
from parllel.types import BatchSpec

# isort: split
from common.traj_infos import SB3EvalTrajInfo, SB3TrajInfo
from models.gaussian_pg_mlp import GaussianPgMlpModel


@contextmanager
def build(config: DictConfig) -> Iterator[RLRunner]:
    build_env = partial(gym.make, config["env_name"])
    parallel = config["parallel"]
    batch_spec = BatchSpec(
        config["batch_T"],
        config["batch_B"],
    )

    # create all environments before initializing pytorch models
    cages, metadata = build_cages(
        EnvClass=build_env,
        n_envs=batch_spec.B,
        TrajInfoClass=SB3TrajInfo,
        parallel=parallel,
    )
    video_config = config.get("video")
    eval_cages, eval_metadata = build_cages(
        EnvClass=build_env,
        n_envs=config["eval"]["n_eval_envs"],
        TrajInfoClass=SB3EvalTrajInfo,
        parallel=parallel,
        render_mode="rgb_array" if video_config is not None else None,
    )

    sample_tree, metadata = build_sample_tree(
        env_metadata=metadata,
        batch_spec=batch_spec,
        parallel=parallel,
    )
    obs_space, action_space = metadata.obs_space, metadata.action_space
    assert isinstance(obs_space, spaces.Box)
    assert isinstance(action_space, spaces.Box)

    # instantiate models
    model = GaussianPgMlpModel(
        obs_space=obs_space,
        action_space=action_space,
        **config["model"],
    )
    distribution = Gaussian(dim=action_space.shape[0], **config["distribution"])
    device = config["device"] or ("cuda:0" if torch.cuda.is_available() else "cpu")
    wandb.config.update({"device": device}, allow_val_change=True)
    device = torch.device(device)

    # instantiate agent
    agent = GaussianPgAgent(
        model=model,
        distribution=distribution,
        example_obs=metadata.example_obs_batch,
        device=device,
    )

    # add agent info, which stores value predictions
    sample_tree = add_agent_info(sample_tree, agent, metadata.example_obs_batch)

    # for advantage estimation, we need to estimate the value of the last
    # state in the batch
    sample_tree = add_bootstrap_value(sample_tree)

    # add several helpful transforms
    batch_transforms, step_transforms = [], []

    sample_tree, step_transforms = add_obs_normalization(
        sample_tree,
        step_transforms,
        initial_count=config["obs_norm_initial_count"],
    )

    sample_tree, batch_transforms = add_reward_normalization(
        sample_tree,
        batch_transforms,
        discount=config["algo"]["discount"],
    )

    sample_tree, batch_transforms = add_reward_clipping(
        sample_tree,
        batch_transforms,
        reward_clip_min=config["reward_clip_min"],
        reward_clip_max=config["reward_clip_max"],
    )

    # add advantage normalization, required for PPO
    sample_tree, batch_transforms = add_advantage_estimation(
        sample_tree,
        batch_transforms,
        discount=config["algo"]["discount"],
        gae_lambda=config["algo"]["gae_lambda"],
        normalize=config["algo"]["normalize_advantage"],
    )

    sampler = BasicSampler(
        batch_spec=batch_spec,
        envs=cages,
        agent=agent,
        sample_tree=sample_tree,
        max_steps_decorrelate=config["max_steps_decorrelate"],
        get_bootstrap_value=True,
        step_transforms=step_transforms,
        batch_transforms=batch_transforms,
    )

    loss_sample_tree = build_loss_sample_tree(sample_tree)
    loss_sample_tree = loss_sample_tree.to_ndarray()
    loss_sample_tree = loss_sample_tree.apply(torch.from_numpy)

    dataloader = BatchedDataLoader(
        tree=loss_sample_tree,
        sampler_batch_spec=batch_spec,
        batch_size=config["algo"]["batch_size"],
        pre_batches_transform=lambda tree: tree.to(device=device),
    )

    optimizer = torch.optim.Adam(
        agent.model.parameters(),
        lr=config["algo"]["learning_rate"],
        **config.get("optimizer", {}),
    )

    if config["algo"]["learning_rate_type"] == "linear":
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=max(1, int(config["runner"]["n_steps"] // batch_spec.size)),
        )
    else:
        lr_scheduler = None

    # create algorithm
    algorithm = PPO(
        agent=agent,
        dataloader=dataloader,
        optimizer=optimizer,
        learning_rate_scheduler=lr_scheduler,
        **config["algo"],
    )

    # env_info is different between sampling envs and eval envs, so build eval sample tree from
    # scratch
    eval_sample_tree, eval_metadata = build_sample_tree(
        env_metadata=eval_metadata,
        batch_spec=BatchSpec(1, config["eval"]["n_eval_envs"]),
        parallel=parallel,
    )

    if video_config is not None:
        output_dir = (
            log_dir / "videos" if (log_dir := logger.log_dir) is not None else "videos"
        )
        video_recorder = RecordVectorizedVideo(
            output_dir=output_dir,
            sample_tree=eval_sample_tree,
            buffer_key_to_record="env_info.rendering",
            env_fps=eval_metadata.gym_metadata["render_fps"],
            use_wandb=True,
            **video_config,
        )
        eval_transforms = [video_recorder]
    else:
        video_recorder = None
        eval_transforms = None

    eval_sampler = EvalSampler(
        max_traj_length=config["eval"]["max_traj_length"],
        max_trajectories=config["eval"]["max_trajectories"],
        envs=eval_cages,
        agent=agent,
        sample_tree=eval_sample_tree,
        step_transforms=eval_transforms,
    )

    if video_recorder is not None:
        callbacks = [
            RecordingSchedule(
                video_recorder_transform=video_recorder,
                cages=eval_cages,
                trigger="on_eval",
            )
        ]
    else:
        callbacks = None

    # create runner
    runner = RLRunner(
        sampler=sampler,
        agent=agent,
        algorithm=algorithm,
        batch_spec=batch_spec,
        eval_sampler=eval_sampler,
        callbacks=callbacks,
        logger_algo_prefix="train",
        **config["runner"],
    )

    try:
        yield runner

    finally:
        eval_sampler.close()
        eval_sample_tree.close()
        sampler.close()
        sample_tree.close()
        agent.close()
        for cage in eval_cages:
            cage.close()
        for cage in cages:
            cage.close()
