import multiprocessing as mp
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

import parllel.logger as logger
from parllel.arrays import Array, buffer_from_dict_example, buffer_from_example
from parllel.buffers import (AgentSamples, Buffer, EnvSamples, Samples,
                             buffer_asarray, buffer_method)
from parllel.cages import ProcessCage, SerialCage, TrajInfo
from parllel.logger import Verbosity
from parllel.patterns import (add_advantage_estimation, add_bootstrap_value,
                              add_reward_clipping, add_reward_normalization)
from parllel.replays import BatchedDataLoader
from parllel.runners import OnPolicyRunner
from parllel.samplers import BasicSampler
from parllel.torch.agents.categorical import CategoricalPgAgent
from parllel.torch.algos.ppo import PPO, build_dataloader_buffer
from parllel.torch.distributions import Categorical
from parllel.torch.handler import TorchHandler
from parllel.torch.utils import buffer_to_device, torchify_buffer
from parllel.transforms import Compose
from parllel.types import BatchSpec

from envs.dummy import build_dummy
from models.pointnet_pg_model import PointNetPgModel


@contextmanager
def build(config: Dict) -> OnPolicyRunner:
    parallel = config["parallel"]
    batch_spec = BatchSpec(
        config["batch_T"],
        config["batch_B"],
    )
    TrajInfo.set_discount(config["algo"]["discount"])

    ## copied from build_cages_and_env_buffers

    EnvClass = build_dummy
    env_kwargs = config["env"]
    TrajInfoClass = TrajInfo
    reset_automatically = True
    batch_spec = batch_spec
    parallel = parallel
    full_size = None

    if parallel:
        CageCls = ProcessCage
        storage = "shared"
    else:
        CageCls = SerialCage
        storage = "local"

    cage_kwargs = dict(
        EnvClass=EnvClass,
        env_kwargs=env_kwargs,
        TrajInfoClass=TrajInfoClass,
        reset_automatically=reset_automatically,
    )

    # create example env
    example_cage = CageCls(**cage_kwargs)

    # get example output from env
    example_cage.random_step_async()
    action, obs, reward, terminated, truncated, info = example_cage.await_step()

    spaces = example_cage.spaces
    obs_space, action_space = spaces.observation, spaces.action

    example_cage.close()

    if full_size is not None:
        logger.debug(f"Allocating replay buffer of size {batch_spec.B * full_size}")
    else:
        logger.debug("Allocating batch buffer.")

    # allocate batch buffer based on examples
    np_obs = np.asanyarray(obs)
    if (dtype := np_obs.dtype) == np.float64:
        dtype = np.float32
    elif dtype == np.int64:
        dtype = np.int32
    batch_observation = Array(
        shape=(150,) + obs_space.shape,
        dtype=dtype,
        batch_shape=tuple(batch_spec),
        kind="jagged",
        storage=storage,
        padding=1,
        full_size=full_size,
    )

    # in case environment creates rewards of shape (1,) or of integer type,
    # force to be correct shape and type
    batch_reward = buffer_from_dict_example(
        reward,
        tuple(batch_spec),
        name="reward",
        shape=(),
        dtype=np.float32,
        storage=storage,
        full_size=full_size,
    )
    batch_terminated = buffer_from_example(
        terminated,
        tuple(batch_spec),
        shape=(),
        dtype=bool,
        storage=storage,
    )
    batch_truncated = buffer_from_example(
        truncated,
        tuple(batch_spec),
        shape=(),
        dtype=bool,
        storage=storage,
    )
    batch_done = buffer_from_example(
        truncated,
        tuple(batch_spec),
        shape=(),
        dtype=bool,
        storage=storage,
        padding=1,
        full_size=full_size,
    )
    batch_info = buffer_from_dict_example(
        info, tuple(batch_spec), name="envinfo", storage=storage
    )
    batch_env = EnvSamples(
        batch_observation,
        batch_reward,
        batch_done,
        batch_terminated,
        batch_truncated,
        batch_info,
    )

    # in discrete problems, integer actions are used as array indices during
    # optimization. Pytorch requires indices to be 64-bit integers, so we
    # force actions to be 32 bits only if they are floats
    batch_action = buffer_from_dict_example(
        action,
        tuple(batch_spec),
        name="action",
        force_32bit="float",
        storage=storage,
        full_size=full_size,
    )

    # pass batch buffers to Cage on creation
    if CageCls is ProcessCage:
        cage_kwargs["buffers"] = (
            batch_action,
            batch_observation,
            batch_reward,
            batch_done,
            batch_terminated,
            batch_truncated,
            batch_info,
        )

    logger.debug(f"Instantiating {batch_spec.B} environments...")

    # create cages to manage environments
    cages = [CageCls(**cage_kwargs) for _ in range(batch_spec.B)]

    logger.debug("Environments instantiated.")

    ## end copied from build_cages_and_env_buffers

    spaces = cages[0].spaces
    obs_space, action_space = spaces.observation, spaces.action

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

    # write dict into namedarraytuple and read it back out. this ensures the
    # example is in a standard format (i.e. namedarraytuple).
    batch_env.observation[0] = obs_space.sample()
    example_obs = batch_env.observation[0]

    # instantiate agent
    agent = CategoricalPgAgent(
        model=model,
        distribution=distribution,
        example_obs=example_obs,
        device=device,
    )
    agent = TorchHandler(agent=agent)

    # get example output from agent
    _, agent_info = agent.step(example_obs)

    # allocate batch buffer based on examples
    batch_agent_info = buffer_from_example(agent_info, (batch_spec.T,))
    batch_agent = AgentSamples(batch_action, batch_agent_info)
    batch_buffer = Samples(batch_agent, batch_env)

    # for advantage estimation, we need to estimate the value of the last
    # state in the batch
    batch_buffer = add_bootstrap_value(batch_buffer)

    # add several helpful transforms
    batch_transforms, step_transforms = [], []

    batch_buffer, batch_transforms = add_reward_normalization(
        batch_buffer,
        batch_transforms,
        discount=config["algo"]["discount"],
    )

    batch_buffer, batch_transforms = add_reward_clipping(
        batch_buffer,
        batch_transforms,
        reward_clip_min=config["reward_clip_min"],
        reward_clip_max=config["reward_clip_max"],
    )

    # add advantage normalization, required for PPO
    batch_buffer, batch_transforms = add_advantage_estimation(
        batch_buffer,
        batch_transforms,
        discount=config["algo"]["discount"],
        gae_lambda=config["algo"]["gae_lambda"],
        normalize=config["algo"]["normalize_advantage"],
    )

    sampler = BasicSampler(
        batch_spec=batch_spec,
        envs=cages,
        agent=agent,
        sample_buffer=batch_buffer,
        max_steps_decorrelate=config["max_steps_decorrelate"],
        get_bootstrap_value=True,
        obs_transform=Compose(step_transforms),
        batch_transform=Compose(batch_transforms),
    )

    dataloader_buffer = build_dataloader_buffer(batch_buffer)

    def batch_transform(x: Buffer):
        x = buffer_asarray(x)
        x = torchify_buffer(x)
        return buffer_to_device(x, device=device)

    dataloader = BatchedDataLoader(
        buffer=dataloader_buffer,
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
    runner = OnPolicyRunner(
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
        buffer_method(batch_buffer, "close")
        buffer_method(batch_buffer, "destroy")


@hydra.main(version_base=None, config_path="conf", config_name="train_ppo")
def main(config: DictConfig) -> None:
    mp.set_start_method("fork")

    run = wandb.init(
        anonymous="must",  # for this example, send to wandb dummy account
        project="PointCloudRL",
        tags=["discrete", "state-based", "ppo", "feedforward"],
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
        sync_tensorboard=True,  # auto-upload any values logged to tensorboard
        save_code=True,  # save script used to start training, git commit, and patch
    )

    logger.init(
        wandb_run=run,
        # this log_dir is used if wandb is disabled (using `wandb disabled`)
        log_dir=Path(f"log_data/pointcloud-ppo/{datetime.now().strftime('%Y-%m-%d_%H-%M')}"),
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

    run.finish()


if __name__ == "__main__":
    main()
