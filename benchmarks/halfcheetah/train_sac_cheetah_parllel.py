# fmt: off
import itertools
import multiprocessing as mp
from contextlib import contextmanager
from functools import partial
from typing import Iterator

# isort: off
import gymnasium as gym
import hydra
import torch
import wandb
from gymnasium import spaces
from omegaconf import DictConfig, OmegaConf

# isort: on
import parllel.logger as logger
from parllel.cages import TrajInfo
from parllel.logger import Verbosity
from parllel.patterns import build_cages_and_sample_tree, build_eval_sampler
from parllel.replays.replay import ReplayBuffer
from parllel.runners import RLRunner
from parllel.samplers import BasicSampler
from parllel.torch.agents.sac_agent import SacAgent
from parllel.torch.algos.sac import SAC, build_replay_buffer_tree
from parllel.torch.distributions.squashed_gaussian import SquashedGaussian
from parllel.types import BatchSpec

# isort: split
from models.sac_q_and_pi import PiMlpModel, QMlpModel


# fmt: on
@contextmanager
def build(config: DictConfig) -> Iterator[RLRunner]:
    build_env = partial(gym.make, "HalfCheetah-v4")

    parallel = config["parallel"]
    batch_spec = BatchSpec(
        config["batch_T"],
        config["batch_B"],
    )
    TrajInfo.set_discount(config["algo"]["discount"])

    replay_length = int(config["algo"]["replay_size"]) // batch_spec.B
    replay_length = (replay_length // batch_spec.T) * batch_spec.T
    cages, sample_tree, metadata = build_cages_and_sample_tree(
        EnvClass=build_env,
        env_kwargs={},
        TrajInfoClass=TrajInfo,
        reset_automatically=True,
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

    optimizers = {
        "pi": torch.optim.Adam(
            agent.model["pi"].parameters(),
            lr=config["algo"]["learning_rate"],
            **config.get("optimizer", {}),
        ),
        "q": torch.optim.Adam(
            itertools.chain(
                agent.model["q1"].parameters(),
                agent.model["q2"].parameters(),
            ),
            lr=config["algo"]["learning_rate"],
            **config.get("optimizer", {}),
        ),
    }

    # create algorithm
    algorithm = SAC(
        batch_spec=batch_spec,
        agent=agent,
        replay_buffer=replay_buffer,
        optimizers=optimizers,
        **config["algo"],
    )

    eval_sampler, eval_sample_tree = build_eval_sampler(
        sample_tree=sample_tree,
        agent=agent,
        CageCls=type(cages[0]),
        EnvClass=build_env,
        env_kwargs={},
        TrajInfoClass=TrajInfo,
        **config["eval_sampler"],
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
        eval_cages = eval_sampler.envs
        eval_sampler.close()
        for cage in eval_cages:
            cage.close()
        eval_sample_tree.close()

        sampler.close()
        agent.close()
        for cage in cages:
            cage.close()
        sample_tree.close()


@hydra.main(version_base=None, config_path="conf", config_name="sac_cheetah_parllel")
def main(config: DictConfig) -> None:
    run = wandb.init(
        project="parllel",
        tags=["continuous", "state-based", "sac", "feedforward", "cheetah"],
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
        sync_tensorboard=True,  # auto-upload any values logged to tensorboard
        save_code=True,  # save script used to start training, git commit, and patch
    )

    logger.init(
        wandb_run=run,
        tensorboard=True,
        output_files={
            "txt": "log.txt",
        },
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
        model_save_path="model.pt",
    )

    with build(config) as runner:
        runner.run()

    run.finish()


if __name__ == "__main__":
    mp.set_start_method("forkserver")
    main()
