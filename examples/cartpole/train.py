from contextlib import contextmanager
import multiprocessing as mp

import numpy as np
import torch

from parllel.arrays import (Array, RotatingArray, SharedMemoryArray, 
    RotatingSharedMemoryArray, buffer_from_example)
from parllel.buffers import buffer_map, buffer_method
from parllel.patterns import add_bootstrap_value, build_cages_and_env_buffers
from parllel.runners.onpolicy import OnPolicyRunner
from parllel.samplers.basic import BasicSampler
from parllel.buffers import Samples, AgentSamples
from parllel.cages import TrajInfo
from parllel.torch.agents.categorical import CategoricalPgAgent
from parllel.torch.algos.ppo import PPO
from parllel.torch.distributions.categorical import Categorical
from parllel.torch.handler import TorchHandler
from parllel.torch.utils import numpify_buffer, torchify_buffer
from parllel.transforms import Compose
from parllel.transforms.advantage import EstimateAdvantage
from parllel.transforms.clip_rewards import ClipRewards
from parllel.transforms.norm_obs import NormalizeObservations
from parllel.transforms.norm_rewards import NormalizeRewards
from parllel.types import BatchSpec

from build.make_env import make_env
from build.model import CartPoleFfCategoricalPgModel


@contextmanager
def build():

    batch_B = 16
    batch_T = 128
    batch_spec = BatchSpec(batch_T, batch_B)
    parallel = True
    EnvClass=make_env
    env_kwargs={
        "max_episode_steps": 1000,
    }
    TrajInfoClass = TrajInfo
    traj_info_kwargs = {}
    discount = 0.99
    gae_lambda = 0.95
    reward_min = -5.
    reward_max = 5.
    learning_rate = 0.001
    n_steps = 200 * batch_spec.size


    if parallel:
        ArrayCls = SharedMemoryArray
        RotatingArrayCls = RotatingSharedMemoryArray
    else:
        ArrayCls = Array
        RotatingArrayCls = RotatingArray

    with build_cages_and_env_buffers(
            EnvClass=EnvClass,
            env_kwargs=env_kwargs,
            TrajInfoClass=TrajInfoClass,
            traj_info_kwargs=traj_info_kwargs,
            wait_before_reset=False,
            batch_spec=batch_spec,
            parallel=parallel,
        ) as (cages, batch_action, batch_env):

        obs_space, action_space = cages[0].spaces

        # instantiate model and agent
        model = CartPoleFfCategoricalPgModel(
            obs_space=obs_space,
            action_space=action_space,
            hidden_sizes=[64, 64],
            hidden_nonlinearity=torch.nn.Tanh,
            )
        distribution = Categorical(dim=action_space.n)
        device = torch.device("cpu")

        # instantiate model and agent
        agent = CategoricalPgAgent(model=model, distribution=distribution, device=device)
        handler = TorchHandler(agent=agent)

        # write dict into namedarraytuple and read it back out. this ensures the
        # example is in a standard format (i.e. namedarraytuple).
        batch_env.observation[0, 0] = obs_space.sample()
        example_obs = batch_env.observation[0, 0]

        # get example output from agent
        example_obs = torchify_buffer(example_obs)
        example_agent_step = agent.dry_run(n_states=batch_spec.B,
            observation=example_obs)
        agent_info, rnn_state = numpify_buffer(example_agent_step)

        # allocate batch buffer based on examples
        batch_agent_info = buffer_from_example(agent_info, tuple(batch_spec), ArrayCls)
        batch_agent = AgentSamples(batch_action, batch_agent_info)
        batch_buffer = Samples(batch_agent, batch_env)

        batch_buffer = add_bootstrap_value(batch_buffer)

        obs_transform = NormalizeObservations(initial_count=10000)
        batch_buffer = obs_transform.dry_run(batch_buffer)

        reward_norm_transform = NormalizeRewards(discount=discount)
        batch_buffer = reward_norm_transform.dry_run(batch_buffer, RotatingArrayCls)

        reward_clip_transform = ClipRewards(reward_min=reward_min,
            reward_max=reward_max)
        batch_buffer = reward_clip_transform.dry_run(batch_buffer)

        advantage_transform = EstimateAdvantage(discount=discount,
            gae_lambda=gae_lambda)
        batch_buffer = advantage_transform.dry_run(batch_buffer, ArrayCls)

        batch_transform = Compose([
            reward_norm_transform,
            reward_clip_transform,
            advantage_transform,
        ])

        sampler = BasicSampler(
            batch_spec=batch_spec,
            envs=cages,
            agent=handler,
            batch_buffer=batch_buffer,
            max_steps_decorrelate=50,
            get_bootstrap_value=True,
            obs_transform=obs_transform,
            batch_transform=batch_transform,
        )

        optimizer = torch.optim.Adam(
            agent.parameters(),
            lr=learning_rate,
        )
        
        # create algorithm
        algorithm = PPO(
            batch_spec=batch_spec,
            agent=handler,
            optimizer=optimizer,
        )

        # create runner
        runner = OnPolicyRunner(sampler=sampler, agent=handler, algorithm=algorithm,
                                n_steps = n_steps, batch_spec=batch_spec)

        try:
            yield runner
        
        finally:
            sampler.close()
            handler.close()
            buffer_method(batch_buffer, "close")
            buffer_method(batch_buffer, "destroy")
    

if __name__ == "__main__":
    mp.set_start_method("fork")
    with build() as runner:
        runner.run()
