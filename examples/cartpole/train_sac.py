from contextlib import contextmanager
import multiprocessing as mp

import torch

from parllel.arrays import (Array, RotatingArray, SharedMemoryArray, 
    RotatingSharedMemoryArray, buffer_from_example)
from parllel.buffers import AgentSamples, buffer_method, Samples
from parllel.cages import TrajInfo
from parllel.patterns import add_bootstrap_value, build_cages_and_env_buffers
from parllel.runners import OffPolicyRunner
from parllel.samplers.basic import BasicSampler
from parllel.torch.agents.sac_agent import SacAgent
from parllel.torch.algos.sac import SAC
from parllel.torch.distributions.squashed_gaussian import SquashedGaussian
from parllel.torch.handler import TorchHandler
from parllel.torch.utils import numpify_buffer, torchify_buffer
from parllel.transforms import (ClipRewards, Compose,
    NormalizeObservations, NormalizeRewards)
from parllel.types import BatchSpec

from envs.continuous_cartpole import build_cartpole
from models.sac_q_and_pi import QMlpModel, PiMlpModel


@contextmanager
def build():

    batch_B = 16
    batch_T = 128
    batch_spec = BatchSpec(batch_T, batch_B)
    parallel = True
    EnvClass=build_cartpole
    env_kwargs={
        "max_episode_steps": 1000,
    }
    TrajInfoClass = TrajInfo
    traj_info_kwargs = {}
    discount = 0.99
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
        pi_model = PiMlpModel(
            obs_space=obs_space,
            action_space=action_space,
            hidden_sizes=[64, 64],
            hidden_nonlinearity=torch.nn.Tanh,
        )
        q1_model = QMlpModel(
            obs_space=obs_space,
            action_space=action_space,
            hidden_sizes=[64, 64],
            hidden_nonlinearity=torch.nn.Tanh,
        )
        q2_model = QMlpModel(
            obs_space=obs_space,
            action_space=action_space,
            hidden_sizes=[64, 64],
            hidden_nonlinearity=torch.nn.Tanh,
        )
        model = torch.ModuleDict{
            "pi": pi_model,
            "q1": q1_model,
            "q2": q2_model,
        }
        distribution = SquashedGaussian(dim=action_space.shape[0], scale=action_space.high[0])
        device = torch.device("cpu")

        # instantiate model and agent
        agent = SacAgent(model=model, distribution=distribution, device=device,
            obs_space=obs_space, action_space=action_space)
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

        batch_transform = Compose([
            reward_norm_transform,
            reward_clip_transform,
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
            agent.model.parameters(),
            lr=learning_rate,
        )
        
        # create algorithm
        algorithm = SAC(
            batch_spec=batch_spec,
            agent=handler,
            optimizer=optimizer,
        )

        # create runner
        runner = OffPolicyRunner(sampler=sampler, agent=handler, algorithm=algorithm,
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
