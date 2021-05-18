import torch

from parllel.cages.cage import Cage
from parllel.runners.onpolicy_runner import OnPolicyRunner
from parllel.samplers.serial import Sampler
from parllel.torch.agents.pg.categorical import CategoricalPgAgent
from parllel.torch.algos.pg.ppo import PPO
from parllel.torch.distributions.categorical import Categorical
from parllel.torch.handlers.handler import Handler
from parllel.torch.models.classic import ClassicControlFfModel
from parllel.types.traj_info import TrajInfo

from build.make_env import make_env


def train():

    # TODO: seeding

    batch_B = 8
    batch_T = 64

    # create runner
    runner = OnPolicyRunner(n_steps = 5e6)
    
    # create dependencies of runner: sampler, agent, and algorithm
    sampler = Sampler(batch_T=batch_T, batch_B=batch_B, get_bootstrap_value=True)
    agent = CategoricalPgAgent()
    algorithm = PPO()

    # create cages to manage environments
    envs = [Cage(
            EnvClass=make_env,
            env_kwargs={},
            TrajInfoClass=TrajInfo,
            traj_info_kwargs = {},
            wait_before_reset=False)
        for _ in range(batch_B)]

    # create a cage without pre-allocated buffer for example generation
    example_env = Cage(EnvClass=make_env, env_kwargs={}, TrajInfoClass=None, traj_info_kwargs={})
    example_env.initialize()

    # TODO: make obs and action spaces available via the cage
    obs_space, action_space = example_env.spaces

    # get example output from both env and agent
    prev_action = action_space.sample()
    example_env.step_async(prev_action)
    example_env_output = example_env.await_step()
    obs, reward, done, info = example_env_output

    # instantiate model and agent
    model = ClassicControlFfModel(
        observation_shape=obs_space.shape[0],
        output_size=action_space.n,
        hidden_sizes=[64, 64],
        hidden_nonlinearity=torch.nn.Tanh,
        )
    distribution = Categorical(dim=action_space.n)
    device = torch.device("cuda")
    action, agent_info = agent.initialize(model=model, device=device, distribution=distribution, n_states=batch_B)

    # allocate batch buffer based on examples

    sampler.initialize(agent=agent, envs=envs, batch_buffer=batch_buffer)

    algorithm.initialize()

    runner.initialize(sampler, agent, algorithm)

    runner.run()