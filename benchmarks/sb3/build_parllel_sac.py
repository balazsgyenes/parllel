import itertools
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
from parllel.patterns import build_cages, build_sample_tree
from parllel.replays.replay import ReplayBuffer
from parllel.runners import RLRunner
from parllel.samplers import BasicSampler
from parllel.samplers.eval import EvalSampler
from parllel.torch.agents.sac_agent import SacAgent
from parllel.torch.algos.sac import SAC, build_replay_buffer_tree
from parllel.torch.distributions.squashed_gaussian import SquashedGaussian
from parllel.transforms import RecordVectorizedVideo
from parllel.types import BatchSpec

# isort: split
from common.traj_infos import SB3EvalTrajInfo, SB3TrajInfo
from models.sac_q_and_pi import PiMlpModel, QMlpModel


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

    replay_length = int(config["algo"]["replay_size"]) // batch_spec.B
    replay_length = (replay_length // batch_spec.T) * batch_spec.T
    sample_tree, metadata = build_sample_tree(
        env_metadata=metadata,
        batch_spec=batch_spec,
        parallel=parallel,
        full_size=replay_length,
    )
    obs_space, action_space = metadata.obs_space, metadata.action_space
    assert isinstance(obs_space, spaces.Box)
    assert isinstance(action_space, spaces.Box)

    # instantiate models
    pi_model = PiMlpModel(
        obs_space=obs_space,
        action_space=action_space,
        **config["pi_model"],
    )
    q1_model = QMlpModel(
        obs_space=obs_space,
        action_space=action_space,
        **config["q_model"],
    )
    q2_model = QMlpModel(
        obs_space=obs_space,
        action_space=action_space,
        **config["q_model"],
    )
    model = torch.nn.ModuleDict(
        {
            "pi": pi_model,
            "q1": q1_model,
            "q2": q2_model,
        }
    )
    distribution = SquashedGaussian(
        dim=action_space.shape[0],
        scale=action_space.high[0],
        **config["distribution"],
    )
    device = config["device"] or ("cuda:0" if torch.cuda.is_available() else "cpu")
    wandb.config.update({"device": device}, allow_val_change=True)
    device = torch.device(device)

    # instantiate agent
    agent = SacAgent(
        model=model,
        distribution=distribution,
        device=device,
        learning_starts=config["algo"]["random_explore_steps"],
    )

    # SAC requires no agent_info, so no need to add to sample_tree

    sampler = BasicSampler(
        batch_spec=batch_spec,
        envs=cages,
        agent=agent,
        sample_tree=sample_tree,
    )

    replay_buffer_tree = build_replay_buffer_tree(sample_tree)
    # because we are only using standard Array types which behave the same as
    # torch Tensors, we can torchify the entire replay buffer here instead of
    # doing it for each batch individually
    replay_buffer_tree = replay_buffer_tree.to_ndarray()
    replay_buffer_tree = replay_buffer_tree.apply(torch.from_numpy)

    replay_buffer = ReplayBuffer(
        tree=replay_buffer_tree,
        sampler_batch_spec=batch_spec,
        size_T=replay_length,
        replay_batch_size=config["algo"]["batch_size"],
        newest_n_samples_invalid=0,
        oldest_n_samples_invalid=1,
        batch_transform=lambda tree: tree.to(device=device),
    )

    q_optimizer = torch.optim.Adam(
        itertools.chain(
            agent.model["q1"].parameters(),
            agent.model["q2"].parameters(),
        ),
        lr=config["algo"]["learning_rate"],
        **config.get("optimizer", {}),
    )
    pi_optimizer = torch.optim.Adam(
        agent.model["pi"].parameters(),
        lr=config["algo"]["learning_rate"],
        **config.get("optimizer", {}),
    )

    if config["algo"]["learning_rate_type"] == "linear":
        lr_schedulers = [
            torch.optim.lr_scheduler.LinearLR(
                pi_optimizer,
                start_factor=1.0,
                end_factor=0.0,
                # TODO: adjust total iters for delayed learning start
                total_iters=max(1, int(config["runner"]["n_steps"] // batch_spec.size)),
            ),
            torch.optim.lr_scheduler.LinearLR(
                q_optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=max(1, int(config["runner"]["n_steps"] // batch_spec.size)),
            ),
        ]
    else:
        lr_schedulers = None

    # create algorithm
    algorithm = SAC(
        batch_spec=batch_spec,
        agent=agent,
        replay_buffer=replay_buffer,
        q_optimizer=q_optimizer,
        pi_optimizer=pi_optimizer,
        learning_rate_schedulers=lr_schedulers,
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
