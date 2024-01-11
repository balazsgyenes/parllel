import itertools
import multiprocessing as mp
from contextlib import contextmanager
from typing import Iterator

# isort: off
import hydra
import torch
import wandb
from gymnasium import spaces
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

# isort: on
import parllel.logger as logger
from parllel.cages import TrajInfo
from parllel.logger import Verbosity
from parllel.patterns import build_cages, build_eval_sample_tree, build_sample_tree
from parllel.replays.replay import ReplayBuffer
from parllel.runners import RLRunner
from parllel.samplers import BasicSampler, EvalSampler
from parllel.torch.agents.sac_agent import SacAgent
from parllel.torch.algos.sac import SAC, build_replay_buffer_tree
from parllel.torch.distributions.squashed_gaussian import SquashedGaussian
from parllel.types import BatchSpec

# isort: split
from envs.continuous_cartpole import build_cartpole
from models.sac_q_and_pi import PiMlpModel, QMlpModel


@contextmanager
def build(config: DictConfig) -> Iterator[RLRunner]:
    parallel = config["parallel"]
    batch_spec = BatchSpec(
        config["batch_T"],
        config["batch_B"],
    )
    TrajInfo.set_discount(config["algo"]["discount"])

    # create all environments before initializing pytorch models
    cages, metadata = build_cages(
        EnvClass=build_cartpole,
        n_envs=batch_spec.B,
        env_kwargs=OmegaConf.to_container(config["env"], throw_on_missing=True),
        TrajInfoClass=TrajInfo,
        parallel=parallel,
    )
    eval_cages, _ = build_cages(
        EnvClass=build_cartpole,
        n_envs=config["eval"]["n_eval_envs"],
        env_kwargs=OmegaConf.to_container(config["env"], throw_on_missing=True),
        TrajInfoClass=TrajInfo,
        parallel=parallel,
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
        learning_starts=config["algo"]["learning_starts"],
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

    # create algorithm
    algorithm = SAC(
        batch_spec=batch_spec,
        agent=agent,
        replay_buffer=replay_buffer,
        q_optimizer=q_optimizer,
        pi_optimizer=pi_optimizer,
        action_space=action_space,
        **config["algo"],
    )

    eval_sample_tree = build_eval_sample_tree(
        sample_tree=sample_tree,
        n_eval_envs=config["eval"]["n_eval_envs"],
    )

    eval_sampler = EvalSampler(
        max_traj_length=config["eval"]["max_traj_length"],
        max_trajectories=config["eval"]["max_trajectories"],
        envs=eval_cages,
        agent=agent,
        sample_tree=eval_sample_tree,
    )

    # create runner
    runner = RLRunner(
        sampler=sampler,
        agent=agent,
        algorithm=algorithm,
        batch_spec=batch_spec,
        eval_sampler=eval_sampler,
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


@hydra.main(version_base=None, config_path="conf", config_name="train_sac")
def main(config: DictConfig) -> None:
    run = wandb.init(
        anonymous="must",  # for this example, send to wandb dummy account
        project="parllel examples",
        tags=["continuous cartpole", "sac"],
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
        sync_tensorboard=True,  # auto-upload any values logged to tensorboard
        save_code=True,  # save script used to start training, git commit, and patch
    )

    logger.init(
        wandb_run=run,
        # this log_dir is used if wandb is disabled (using `wandb disabled`)
        log_dir=HydraConfig.get().runtime.output_dir,
        tensorboard=True,
        output_files={
            "txt": "log.txt",
            # "csv": "progress.csv",
        },
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
        model_save_path="model.pt",
        # verbosity=Verbosity.DEBUG,
    )

    with build(config) as runner:
        runner.run()

    logger.close()
    run.finish()


if __name__ == "__main__":
    mp.set_start_method("forkserver")
    main()
