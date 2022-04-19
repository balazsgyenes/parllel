from contextlib import contextmanager
import multiprocessing as mp

import torch

from parllel.arrays import (Array, RotatingArray, SharedMemoryArray, 
    RotatingSharedMemoryArray, buffer_from_example)
from parllel.buffers import AgentSamples, buffer_method, Samples
from parllel.cages import TrajInfo
from parllel.patterns import (add_advantage_estimation, add_bootstrap_value,
    add_reward_clipping, add_reward_normalization, add_valid,
    build_cages_and_env_buffers, add_initial_rnn_state)
from parllel.runners.onpolicy import OnPolicyRunner
from parllel.samplers.recurrent import RecurrentSampler
from parllel.torch.agents.categorical import CategoricalPgAgent
from parllel.torch.agents.ensemble import AgentProfile
from parllel.torch.agents.independent import IndependentPgAgents
from parllel.torch.algos.ppo import PPO
from parllel.torch.distributions import Categorical
from parllel.torch.handler import TorchHandler
from parllel.transforms import Compose
from parllel.types import BatchSpec

from hera_gym.builds.multi_agent_cartpole import build_multi_agent_cartpole
from models.atari_lstm_model import AtariLstmPgModel


@contextmanager
def build():

    batch_B = 16
    batch_T = 128
    batch_spec = BatchSpec(batch_T, batch_B)
    parallel = False
    EnvClass = build_multi_agent_cartpole
    env_kwargs = {
        "max_episode_steps": 1000,
        "headless": True,
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
            wait_before_reset=True,
            batch_spec=batch_spec,
            parallel=parallel,
        ) as (cages, batch_action, batch_env):

        obs_space, action_space = cages[0].spaces

        # instantiate model and agent
        device = torch.device("cuda", index=0) if torch.cuda.is_available() else torch.device("cpu")
        ## cart
        cart_model = AtariLstmPgModel(
            obs_space=obs_space,
            action_space=action_space["cart"],
            pre_lstm_hidden_sizes=32,
            lstm_size=16,
            post_lstm_hidden_sizes=32,
            hidden_nonlinearity=torch.nn.Tanh,
            )
        cart_distribution = Categorical(dim=action_space["cart"].n)
        cart_agent = CategoricalPgAgent(
            model=cart_model, distribution=cart_distribution, observation_space=obs_space,
            action_space=action_space["cart"], n_states=batch_spec.B, device=device,
            recurrent=True)
        cart_profile = AgentProfile(instance=cart_agent, action_key="cart")

        ## camera
        camera_model = AtariLstmPgModel(
            obs_space=obs_space,
            action_space=action_space["camera"],
            pre_lstm_hidden_sizes=32,
            lstm_size=16,
            post_lstm_hidden_sizes=32,
            hidden_nonlinearity=torch.nn.Tanh,
            )
        camera_distribution = Categorical(dim=action_space["camera"].n)
        camera_agent = CategoricalPgAgent(
            model=camera_model, distribution=camera_distribution, observation_space=obs_space,
            action_space=action_space["camera"], n_states=batch_spec.B, device=device,
            recurrent=True)
        camera_profile = AgentProfile(instance=camera_agent, action_key="camera")

        agent = IndependentPgAgents([cart_profile, camera_profile])
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

        # for recurrent problems, we need to save the initial state at the 
        # beginning of the batch
        batch_buffer = add_initial_rnn_state(batch_buffer, agent)

        # for advantage estimation, we need to estimate the value of the last
        # state in the batch
        batch_buffer = add_bootstrap_value(batch_buffer)
        
        # for recurrent problems, compute mask that zeroes out samples after
        # environments are done before they can be reset
        batch_buffer = add_valid(batch_buffer)

        # add several helpful transforms
        batch_transforms, step_transforms = [], []

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

        sampler = RecurrentSampler(
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
        runner = OnPolicyRunner(sampler=sampler, agent=agent, algorithm=algorithm,
                                n_steps = n_steps, batch_spec=batch_spec)

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
