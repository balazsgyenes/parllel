import multiprocessing as mp
from contextlib import contextmanager
from typing import Iterator

import gymnasium as gym
import hydra
from omegaconf import DictConfig, OmegaConf

from parllel.cages import TrajInfo
from parllel.cages.tests.dummy import DummyEnv
from parllel.patterns import build_cages_and_sample_tree
from parllel.samplers.profiling import ProfilingSampler
from parllel.types import BatchSpec


def make_cartpole() -> gym.Env:
    return gym.make("CartPole-v1")


@contextmanager
def build(config: DictConfig) -> Iterator[ProfilingSampler]:
    parallel = config["parallel"]
    batch_spec = BatchSpec(
        config["batch_T"],
        config["batch_B"],
    )
    TrajInfo.set_discount(0.99)

    env_config = OmegaConf.to_container(config["env"], throw_on_missing=True)
    EnvClass = env_config.pop("EnvClass")
    EnvClass = globals()[EnvClass]

    cages, sample_tree, metadata = build_cages_and_sample_tree(
        EnvClass=EnvClass,
        env_kwargs=env_config,
        TrajInfoClass=TrajInfo,
        reset_automatically=True,
        batch_spec=batch_spec,
        parallel=parallel,
    )

    sampler = ProfilingSampler(
        batch_spec=batch_spec,
        envs=cages,
        sample_tree=sample_tree,
        n_iterations=config["n_iterations"],
        # profile_path=config["profile_path"],
    )

    try:
        yield sampler

    finally:
        for cage in cages:
            cage.close()
        sample_tree.close()


@hydra.main(version_base=None, config_path="conf", config_name="profile_sampling")
def main(config: DictConfig) -> None:
    with build(config) as sampler:
        sampler.collect_batch()


if __name__ == "__main__":
    import platform

    if platform.system() == "Darwin":
        mp.set_start_method("spawn")

    main()
