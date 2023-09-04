import multiprocessing as mp
from contextlib import contextmanager
from typing import Iterator

# isort: off
import hydra
import torch
import wandb
from gymnasium.spaces import Discrete
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

# isort: on
import parllel.logger as logger
from parllel import Array, ArrayDict, dict_map
from parllel.cages import TrajInfo
from parllel.logger import Verbosity
from parllel.patterns import (
    add_advantage_estimation,
    add_agent_info,
    add_bootstrap_value,
    add_reward_clipping,
    add_reward_normalization,
    build_cages,
    build_sample_tree,
)
from parllel.replays import BatchedDataLoader
from parllel.runners import RLRunner
from parllel.samplers import BasicSampler
from parllel.torch.agents.categorical import CategoricalPgAgent
from parllel.torch.algos.ppo import PPO, build_loss_sample_tree
from parllel.torch.distributions import Categorical
from parllel.types import BatchSpec

# isort: split
from envs.dummy import DummyEnv
from models.pointnet_pg_model import PointNetPgModel
from pointcloud import PointCloudSpace


@contextmanager
def build(config: DictConfig) -> Iterator[RLRunner]:
    parallel = config["parallel"]
    batch_spec = BatchSpec(
        config["batch_T"],
        config["batch_B"],
    )
    TrajInfo.set_discount(config["algo"]["discount"])

    env_kwargs = OmegaConf.to_container(config["env"], throw_on_missing=True)
    env_kwargs["actions"] = "discrete"
    cages, metadata = build_cages(
        EnvClass=DummyEnv,
        n_envs=batch_spec.B,
        env_kwargs=env_kwargs,
        TrajInfoClass=TrajInfo,
        parallel=parallel,
    )

    sample_tree, metadata = build_sample_tree(
        env_metadata=metadata,
        batch_spec=batch_spec,
        parallel=parallel,
        keys_to_skip="observation",  # we will allocate this ourselves
    )
    obs_space, action_space = metadata.obs_space, metadata.action_space
    assert isinstance(obs_space, PointCloudSpace)
    assert isinstance(action_space, Discrete)

    sample_tree["observation"] = dict_map(
        Array.from_numpy,
        metadata.example_obs,
        batch_shape=tuple(batch_spec),
        max_mean_num_elem=obs_space.max_num_points,
        kind="jagged",
        storage="shared" if parallel else "local",
        padding=1,
    )
    sample_tree["observation"][0] = obs_space.sample()
    metadata.example_obs_batch = sample_tree["observation"][0]

    # instantiate model
    model = PointNetPgModel(
        obs_space=obs_space,
        action_space=action_space,
        **config["model"],
    )
    distribution = Categorical(dim=action_space.n)
    device = config["device"] or ("cuda:0" if torch.cuda.is_available() else "cpu")
    wandb.config.update({"device": device}, allow_val_change=True)
    device = torch.device(device)

    # instantiate agent
    agent = CategoricalPgAgent(
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

    def batch_transform(tree: ArrayDict[Array]) -> ArrayDict[torch.Tensor]:
        tree = tree.to_ndarray()
        tree = tree.apply(torch.from_numpy)
        return tree.to(device=device)

    dataloader = BatchedDataLoader(
        tree=loss_sample_tree,
        sampler_batch_spec=batch_spec,
        n_batches=config["algo"]["minibatches"],
        batch_transform=batch_transform,
    )

    optimizer = torch.optim.Adam(
        agent.model.parameters(),
        lr=config["algo"]["learning_rate"],
        **config.get("optimizer", {}),
    )

    # create algorithm
    algorithm = PPO(
        agent=agent,
        dataloader=dataloader,
        optimizer=optimizer,
        **config["algo"],
    )

    # create runner
    runner = RLRunner(
        sampler=sampler,
        agent=agent,
        algorithm=algorithm,
        batch_spec=batch_spec,
        **config["runner"],
    )

    try:
        yield runner

    finally:
        sampler.close()
        agent.close()
        for cage in cages:
            cage.close()
        sample_tree.close()


@hydra.main(version_base=None, config_path="conf", config_name="train_ppo")
def main(config: DictConfig) -> None:
    run = wandb.init(
        anonymous="must",  # for this example, send to wandb dummy account
        project="parllel examples",
        tags=["pointcloud dummy env", "ppo"],
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
    mp.set_start_method("fork")
    main()
