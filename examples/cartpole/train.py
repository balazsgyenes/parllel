from contextlib import contextmanager
import multiprocessing as mp

import torch

from parllel.arrays import (Array, RotatingArray, SharedMemoryArray, 
    RotatingSharedMemoryArray, buffer_from_example)
from parllel.buffers import AgentSamples, buffer_method, Samples
from parllel.cages import TrajInfo
from parllel.patterns import (add_advantage_estimation, add_bootstrap_value,
    add_obs_normalization, add_reward_clipping, add_reward_normalization,
    build_cages_and_env_buffers)
from parllel.runners.onpolicy import OnPolicyRunner
from parllel.samplers.basic import BasicSampler
from parllel.torch.agents.categorical import CategoricalPgAgent
from parllel.torch.algos.ppo import PPO
from parllel.torch.distributions import Categorical
from parllel.torch.handler import TorchHandler
from parllel.transforms import Compose
from parllel.types import BatchSpec

from envs.cartpole import build_cartpole
from models.model import CartPoleFfPgModel


@contextmanager
def build():

    batch_B = 16
    batch_T = 128
    batch_spec = BatchSpec(batch_T, batch_B)
    parallel = True
    EnvClass = build_cartpole
    env_kwargs = {
        "max_episode_steps": 1000,
    }
    discount = 0.99
    TrajInfoClass = TrajInfo
    traj_info_kwargs = {
        "discount": discount,
    }
    gae_lambda = 0.95
    reward_min = -5.
    reward_max = 5.
    learning_rate = 0.001
    n_steps = 50 * batch_spec.size
    log_interval_steps = 5 * batch_spec.size


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
        model = CartPoleFfPgModel(
            obs_space=obs_space,
            action_space=action_space,
            hidden_sizes=[64, 64],
            hidden_nonlinearity=torch.nn.Tanh,
            )
        distribution = Categorical(dim=action_space.n)
        device = torch.device("cuda", index=0) if torch.cuda.is_available() else torch.device("cpu")

        # instantiate model and agent
        agent = CategoricalPgAgent(
            model=model, distribution=distribution, observation_space=obs_space,
            action_space=action_space, n_states=batch_spec.B, device=device)
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
            initial_count=10000,
        )

        batch_buffer, batch_transforms = add_reward_normalization(
            batch_buffer,
            batch_transforms,
            discount=discount,
        )

        batch_buffer, batch_transforms = add_reward_clipping(
            batch_buffer,
            batch_transforms,
            reward_min=reward_min,
            reward_max=reward_max,
        )

        # add advantage normalization, required for PPO
        batch_buffer, batch_transforms = add_advantage_estimation(
            batch_buffer,
            batch_transforms,
            discount=discount,
            gae_lambda=gae_lambda,
            normalize=True,
        )

        sampler = BasicSampler(
            batch_spec=batch_spec,
            envs=cages,
            agent=agent,
            batch_buffer=batch_buffer,
            max_steps_decorrelate=50,
            get_bootstrap_value=True,
            obs_transform=Compose(step_transforms),
            batch_transform=Compose(batch_transforms),
        )

        optimizer = torch.optim.Adam(
            agent.parameters(),
            lr=learning_rate,
        )
        
        # create algorithm
        algorithm = PPO(
            batch_spec=batch_spec,
            agent=agent,
            optimizer=optimizer,
        )

        # create runner
        runner = OnPolicyRunner(
            sampler=sampler,
            agent=agent,
            algorithm=algorithm,
            n_steps=n_steps,
            batch_spec=batch_spec,
            log_interval_steps=log_interval_steps,
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
    with build() as runner:
        runner.run()
