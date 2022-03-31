from contextlib import contextmanager
import multiprocessing as mp

import numpy as np
import torch

from parllel.buffers import buffer_map, buffer_method
from parllel.arrays import (Array, RotatingArray, ManagedMemoryArray,
    RotatingManagedMemoryArray, buffer_from_example, buffer_from_dict_example)
from parllel.cages import Cage, ProcessCage
from parllel.runners.onpolicy import OnPolicyRunner
from parllel.samplers.basic import BasicSampler
from parllel.samplers.collections import Samples, AgentSamplesWBootstrap, EnvSamples
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
from parllel.types import BatchSpec, TrajInfo

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
    TrajInfoClass=TrajInfo
    traj_info_kwargs={}
    wait_before_reset=False
    discount = 0.99
    gae_lambda = 0.95
    reward_min = -5.
    reward_max = 5.
    learning_rate = 0.001
    n_steps = 200 * batch_spec.size


    if parallel:
        CageCls = ProcessCage
        ArrayCls = ManagedMemoryArray
        RotatingArrayCls = RotatingManagedMemoryArray
    else:
        CageCls = Cage
        ArrayCls = Array
        RotatingArrayCls = RotatingArray

    # create cages to manage environments
    cages = [
        CageCls(
            EnvClass = EnvClass,
            env_kwargs = env_kwargs,
            TrajInfoClass = TrajInfoClass,
            traj_info_kwargs = traj_info_kwargs,
            wait_before_reset = wait_before_reset,
        )
        for _ in range(batch_spec.B)
    ]

    # get example output from env
    example_env_output = cages[0].get_example_output()
    obs, reward, done, info = example_env_output

    # allocate batch buffer based on examples
    batch_observation = buffer_from_dict_example(obs, tuple(batch_spec), RotatingArrayCls, name="obs", padding=1)
    batch_reward = buffer_from_dict_example(reward, tuple(batch_spec), ArrayCls, name="reward", force_float32=True)
    batch_done = buffer_from_dict_example(done, tuple(batch_spec), RotatingArrayCls, name="done", padding=1)
    batch_info = buffer_from_dict_example(info, tuple(batch_spec), ArrayCls, name="envinfo")
    batch_env_samples = EnvSamples(batch_observation, batch_reward, batch_done, batch_info)

    # write dict into namedarraytuple and read it back out. this ensures the
    # example is in a standard format (i.e. namedarraytuple).
    batch_env_samples[0, 0] = example_env_output
    example_env_output = batch_env_samples[0, 0]

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

    # get example output from agent
    example_obs, _, _, _ = example_env_output
    example_obs = torchify_buffer(buffer_map(np.asarray, example_obs))
    example_agent_step = agent.dry_run(n_states=batch_spec.B,
        observation=example_obs)
    action, agent_info = numpify_buffer(example_agent_step)

    # allocate batch buffer based on examples
    batch_action = buffer_from_example(action, tuple(batch_spec), ArrayCls)
    batch_agent_info = buffer_from_example(agent_info, tuple(batch_spec), ArrayCls)
    batch_bootstrap_value = buffer_from_example(agent_info.value, (batch_spec.B,), ArrayCls)
    batch_agent_samples = AgentSamplesWBootstrap(batch_action, batch_agent_info, batch_bootstrap_value)

    batch_samples = Samples(batch_agent_samples, batch_env_samples)

    for cage in cages:
        cage.set_samples_buffer(batch_action, *batch_env_samples)

    obs_transform = NormalizeObservations(initial_count=10000)
    batch_samples = obs_transform.dry_run(batch_samples)

    reward_norm_transform = NormalizeRewards(discount=discount)
    batch_samples = reward_norm_transform.dry_run(batch_samples, RotatingArrayCls)

    reward_clip_transform = ClipRewards(reward_min=reward_min,
        reward_max=reward_max)
    batch_samples = reward_clip_transform.dry_run(batch_samples)

    advantage_transform = EstimateAdvantage(discount=discount,
        gae_lambda=gae_lambda)
    batch_samples = advantage_transform.dry_run(batch_samples, ArrayCls)

    batch_transform = Compose([
        reward_norm_transform,
        reward_clip_transform,
        advantage_transform,
    ])

    sampler = BasicSampler(batch_spec=batch_spec,
                          envs=cages,
                          agent=handler,
                          batch_buffer=batch_samples,
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
        agent.close()
        for cage in cages:
            cage.close()
        buffer_method(batch_samples, "close")
        buffer_method(batch_samples, "destroy")
    

if __name__ == "__main__":
    mp.set_start_method("spawn")
    with build() as runner:
        runner.run()
