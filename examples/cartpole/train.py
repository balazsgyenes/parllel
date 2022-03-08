from contextlib import contextmanager

import torch

from parllel.buffers import buffer_from_example, buffer_from_dict_example, buffer_method
from parllel.arrays import Array, RotatingArray, ManagedMemoryArray, RotatingManagedMemoryArray
from parllel.cages import Cage, ParallelProcessCage
# from parllel.runners.onpolicy_runner import OnPolicyRunner
from parllel.samplers import MiniSampler
from parllel.samplers.collections import Samples, AgentSamples, EnvSamples
from parllel.torch.agents.pg.categorical import CategoricalPgAgent
# from parllel.torch.algos.pg.ppo import PPO
from parllel.torch.distributions.categorical import Categorical
from parllel.torch.handler import TorchHandler
from parllel.types.traj_info import TrajInfo

from build.make_env import make_env
from build.model import CartPoleFfCategoricalPgModel


@contextmanager
def build():

    batch_B = 8
    batch_T = 64
    parallel = True
    EnvClass=make_env
    env_kwargs={}
    TrajInfoClass=TrajInfo
    traj_info_kwargs={}
    wait_before_reset=False

    if parallel:
        CageCls = ParallelProcessCage
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
        for _ in range(batch_B)
    ]

    # get example output from env
    example_env_output = cages[0].get_example_output()
    obs, reward, done, info = example_env_output

    # allocate batch buffer based on examples
    batch_observation = buffer_from_dict_example(obs, (batch_T, batch_B), RotatingArrayCls, name="obs", padding=1)
    batch_reward = buffer_from_dict_example(reward, (batch_T, batch_B), ArrayCls, name="reward", force_float32=True)
    batch_done = buffer_from_dict_example(done, (batch_T, batch_B), ArrayCls, name="done")
    batch_info = buffer_from_dict_example(info, (batch_T, batch_B), ArrayCls, name="envinfo")
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
    action, agent_info = handler.dry_run(n_states=batch_B, observation=example_obs)

    # allocate batch buffer based on examples
    batch_action = buffer_from_example(action, (batch_T, batch_B), ArrayCls)
    batch_agent_info = buffer_from_example(agent_info, (batch_T, batch_B), ArrayCls)
    batch_agent_samples = AgentSamples(batch_action, batch_agent_info)

    batch_samples = Samples(batch_agent_samples, batch_env_samples)

    # TODO: move into sampler init
    if parallel:
        for cage in cages:
            cage.set_samples_buffer(batch_samples)

    sampler = MiniSampler(batch_T=batch_T, batch_B=batch_B, envs=cages, agent=handler,
                          batch_buffer=batch_samples, get_bootstrap_value=False)

    try:
        yield sampler

        # create algorithm
        # algorithm = PPO()

        # create runner
        # runner = OnPolicyRunner(sampler=sampler, agent=handler, algorithm=algorithm,
        #                         n_steps = 5e6)
    finally:
        cages, handler, batch_samples = sampler.close()
        agent.close()
        for cage in cages:
            cage.close()
        buffer_method(batch_samples, "close")
        buffer_method(batch_samples, "destroy")
    

if __name__ == "__main__":
    with build() as sampler:
        samples, completed_trajectories = sampler.collect_batch(elapsed_steps=0)

        print(samples)
        print(completed_trajectories)
