from contextlib import contextmanager
from datetime import datetime
import multiprocessing as mp
from pathlib import Path
from typing import Dict

import torch

from parllel.arrays import (Array, RotatingArray, SharedMemoryArray, 
    RotatingSharedMemoryArray, buffer_from_example)
from parllel.buffers import AgentSamples, buffer_method, Samples
from parllel.cages import TrajInfo
from parllel.configuration import add_default_config_fields, add_metadata
from parllel.logging import init_log_folder, log_config
from parllel.patterns import (add_advantage_estimation, add_bootstrap_value,
    add_obs_normalization, add_reward_clipping, add_reward_normalization,
    build_cages_and_env_buffers)
from parllel.runners import OnPolicyRunner
from parllel.samplers import BasicSampler
from parllel.torch.agents.categorical import CategoricalPgAgent
from parllel.torch.algos.ppo import PPO, add_default_ppo_config
from parllel.torch.distributions import Categorical
from parllel.torch.handler import TorchHandler
from parllel.transforms import Compose
from parllel.types import BatchSpec

from envs.cartpole import build_cartpole
from models.model import CartPoleFfPgModel


@contextmanager
def build(config: Dict) -> OnPolicyRunner:

    init_log_folder(config["log_dir"])
    log_config(config, config["log_dir"])

    parallel = config["parallel"]
    batch_spec = BatchSpec(
        config["batch_T"],
        config["batch_B"],
    )
    traj_info_kwargs = {
        "discount": config["discount"],
    }

    if parallel:
        ArrayCls = SharedMemoryArray
        RotatingArrayCls = RotatingSharedMemoryArray
    else:
        ArrayCls = Array
        RotatingArrayCls = RotatingArray

    with build_cages_and_env_buffers(
            EnvClass=build_cartpole,
            env_kwargs=config["env"],
            TrajInfoClass=TrajInfo,
            traj_info_kwargs=traj_info_kwargs,
            wait_before_reset=False,
            batch_spec=batch_spec,
            parallel=parallel,
        ) as (cages, batch_action, batch_env):

        obs_space, action_space = cages[0].spaces

        # instantiate model and agent
        model = CartPoleFfPgModel(
            obs_space=obs_space,
            action_space=action_space,
            **config["model"],
        )
        distribution = Categorical(dim=action_space.n)
        device = torch.device("cuda", index=0) if torch.cuda.is_available() else torch.device("cpu")

        # instantiate model and agent
        agent = CategoricalPgAgent(
            model=model,
            distribution=distribution,
            observation_space=obs_space,
            action_space=action_space,
            n_states=batch_spec.B,
            device=device,
        )
        agent = TorchHandler(agent=agent)

        # write dict into namedarraytuple and read it back out. this ensures the
        # example is in a standard format (i.e. namedarraytuple).
        batch_env.observation[0] = obs_space.sample()
        example_obs = batch_env.observation[0]

        # get example output from agent
        _, agent_info = agent.step(example_obs)

        # allocate batch buffer based on examples
        batch_agent_info = buffer_from_example(agent_info, (batch_spec.T,), ArrayCls)
        batch_agent = AgentSamples(batch_action, batch_agent_info)
        batch_buffer = Samples(batch_agent, batch_env)

        # for advantage estimation, we need to estimate the value of the last
        # state in the batch
        batch_buffer = add_bootstrap_value(batch_buffer)

        # add several helpful transforms
        batch_transforms, step_transforms = [], []

        batch_buffer, step_transforms = add_obs_normalization(
            batch_buffer,
            step_transforms,
            initial_count=config["obs_norm_initial_count"],
        )

        batch_buffer, batch_transforms = add_reward_normalization(
            batch_buffer,
            batch_transforms,
            discount=config["discount"],
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
            discount=config["discount"],
            gae_lambda=config["gae_lambda"],
            normalize=config["normalize_advantage"],
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

        optimizer = torch.optim.Adam(
            agent.model.parameters(),
            lr=config["learning_rate"],
            **config["optimizer"],
        )
        
        # create algorithm
        algorithm = PPO(
            batch_spec=batch_spec,
            agent=agent,
            optimizer=optimizer,
            **config["algo"],
        )

        # create runner
        runner = OnPolicyRunner(
            sampler=sampler,
            agent=agent,
            algorithm=algorithm,
            batch_spec=batch_spec,
            log_dir=config["log_dir"],
            **config["runner"],
        )

        try:
            yield runner
        
        finally:
            sampler.close()
            agent.close()
            buffer_method(batch_buffer, "close")
            buffer_method(batch_buffer, "destroy")
    

if __name__ == "__main__":
    mp.set_start_method("fork")

    config = dict(
        # log_dir = Path(f"log_data/cartpole-ppo/{datetime.now().strftime('%Y-%m-%d_%H-%M')}"),
        log_dir = None,
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
        model = dict(
            hidden_sizes = [64, 64],
            hidden_nonlinearity=torch.nn.Tanh,
        ),
        runner = dict(
            n_steps = 50 * 16 * 128,
            log_interval_steps = 5 * 16 * 128,
        ),
        meta = dict(
            name = "[Example 1] CartPole with PPO",
        ),
    )

    config = add_default_ppo_config(config)
    config = add_metadata(config, build)
    config = add_default_config_fields(config)

    with build(config) as runner:
        runner.run()
