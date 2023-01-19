from contextlib import contextmanager
from datetime import datetime
import itertools
import multiprocessing as mp
from pathlib import Path
from typing import Dict

import torch

from parllel.arrays import buffer_from_example
from parllel.arrays import buffer_from_dict_example
from parllel.arrays.large import LargeArray
from parllel.arrays.sharedmemory import LargeSharedMemoryArray
from parllel.buffers import AgentSamples, buffer_method, Samples
from parllel.buffers import EnvSamples
from parllel.cages import TrajInfo
from parllel.cages import SerialCage, ProcessCage
from parllel.configuration import add_default_config_fields
import parllel.logger as logger
from parllel.logger import Verbosity
from parllel.patterns import (build_cages_and_env_buffers, build_eval_sampler)
from parllel.replays.replay import ReplayBuffer
from parllel.runners import OffPolicyRunner
from parllel.samplers import BasicSampler
from parllel.torch.agents.sac_agent import SacAgent
from parllel.torch.algos.sac import SAC, add_default_sac_config, SamplesForLoss
from parllel.torch.distributions.squashed_gaussian import SquashedGaussian
from parllel.torch.handler import TorchHandler
from parllel.torch.utils import torchify_buffer
from parllel.types import BatchSpec

from envs.continuous_cartpole import build_cartpole
from models.sac_q_and_pi import QMlpModel, PiMlpModel


@contextmanager
def build(config: Dict) -> OffPolicyRunner:

    parallel = config["parallel"]
    batch_spec = BatchSpec(
        config["batch_T"],
        config["batch_B"],
    )
    TrajInfo.set_discount(config["discount"])

    replay_buffer_dims = (config["replay_length"], batch_spec.B)

    EnvClass = build_cartpole
    env_kwargs = config["env"]
    TrajInfoClass = TrajInfo
    reset_automatically = True
    batch_spec = batch_spec
    parallel = parallel

    if parallel:
        CageCls = ProcessCage
        LargeArrayCls = LargeSharedMemoryArray
    else:
        CageCls = SerialCage
        LargeArrayCls = LargeArray

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
    action, obs, reward, done, info = example_cage.await_step()

    example_cage.close()

    # allocate batch buffer based on examples
    batch_observation = buffer_from_dict_example(obs, replay_buffer_dims, LargeArrayCls, name="obs", padding=1, apparent_size=batch_spec.T)
    batch_reward = buffer_from_dict_example(reward, replay_buffer_dims, LargeArrayCls, name="reward", apparent_size=batch_spec.T)
    batch_done = buffer_from_dict_example(done, replay_buffer_dims, LargeArrayCls, name="done", padding=1, apparent_size=batch_spec.T)
    batch_info = buffer_from_dict_example(info, tuple(batch_spec), LargeArrayCls, name="envinfo")
    batch_env = EnvSamples(batch_observation, batch_reward, batch_done, batch_info)

    """In discrete problems, integer actions are used as array indices during
    optimization. Pytorch requires indices to be 64-bit integers, so we do not
    convert here.
    """
    batch_action = buffer_from_dict_example(action, replay_buffer_dims, LargeArrayCls, name="action", force_32bit=False, apparent_size=batch_spec.T)

    # pass batch buffers to Cage on creation
    if CageCls is ProcessCage:
        cage_kwargs["buffers"] = (batch_action, batch_observation, batch_reward, batch_done, batch_info)
    
    # create cages to manage environments
    cages = [CageCls(**cage_kwargs) for _ in range(batch_spec.B)]

    spaces = cages[0].spaces
    obs_space, action_space = spaces.observation, spaces.action

    # instantiate model and agent
    pi_model = PiMlpModel(
        obs_space=obs_space,
        action_space=action_space,
        **config["pi_model"],
    )
    q1_model = QMlpModel(
        obs_space=obs_space,
        action_space=action_space,
        **config["q_model"],
    )
    q2_model = QMlpModel(
        obs_space=obs_space,
        action_space=action_space,
        **config["q_model"],
    )
    model = torch.nn.ModuleDict({
        "pi": pi_model,
        "q1": q1_model,
        "q2": q2_model,
    })
    distribution = SquashedGaussian(
        dim=action_space.shape[0],
        scale=action_space.high[0],
    )
    device = torch.device(config["device"])

    # instantiate model and agent
    agent = SacAgent(
        model=model,
        distribution=distribution,
        observation_space=obs_space,
        action_space=action_space,
        device=device,
        **config["agent"],
    )
    agent = TorchHandler(agent=agent)

    # write dict into namedarraytuple and read it back out. this ensures the
    # example is in a standard format (i.e. namedarraytuple).
    batch_env.observation[0] = obs_space.sample()
    example_obs = batch_env.observation[0]

    # get example output from agent
    _, agent_info = agent.step(example_obs)

    # allocate batch buffer based on examples
    batch_agent_info = buffer_from_example(agent_info, (batch_spec.T,), LargeArrayCls)
    batch_agent = AgentSamples(batch_action, batch_agent_info)
    batch_buffer = Samples(batch_agent, batch_env)

    sampler = BasicSampler(
        batch_spec=batch_spec,
        envs=cages,
        agent=agent,
        sample_buffer=batch_buffer,
        max_steps_decorrelate=config["max_steps_decorrelate"],
        get_bootstrap_value=False,
    )

    batch_obs = batch_buffer.env.observation
    replay_buffer = SamplesForLoss(
        observation=batch_obs.full,
        action=batch_buffer.agent.action.full,
        reward=batch_buffer.env.reward.full,
        done=batch_buffer.env.done.full,
        next_observation=batch_obs.full.next,
    )
    # because we are not using frame stacks, we can optionally convert the
    # entire replay buffer to torch Tensors here
    # replay_buffer = torchify_buffer(replay_buffer)

    replay_buffer = ReplayBuffer(
        buffer=replay_buffer,
        sampler_batch_spec=batch_spec,
        leading_dim=config["replay_length"],
        n_samples=config["batch_size"],
        newest_n_samples_invalid=0,
        oldest_n_samples_invalid=1,
    )

    optimizers = {
        "pi": torch.optim.Adam(
            agent.model["pi"].parameters(),
            lr=config["learning_rate"],
            **config["optimizer"],
        ),
        "q": torch.optim.Adam(
            itertools.chain(
                agent.model["q1"].parameters(),
                agent.model["q2"].parameters(),
            ),
            lr=config["learning_rate"],
            **config["optimizer"],
        ),
    }
    
    # create algorithm
    algorithm = SAC(
        batch_spec=batch_spec,
        agent=agent,
        replay_buffer=replay_buffer,
        optimizers=optimizers,
        discount=config["discount"],
        **config["algo"],
    )

    eval_sampler, step_buffer = build_eval_sampler(
        samples_buffer=batch_buffer,
        agent=agent,
        CageCls=type(cages[0]),
        EnvClass=build_cartpole,
        env_kwargs=config["env"],
        TrajInfoClass=TrajInfo,
        **config["eval_sampler"],
    )

    # create runner
    runner = OffPolicyRunner(
        sampler=sampler,
        agent=agent,
        algorithm=algorithm,
        batch_spec=batch_spec,
        eval_sampler=eval_sampler,
        **config["runner"],
    )

    try:
        yield runner
    
    finally:
        eval_cages = eval_sampler.envs
        eval_sampler.close()
        for cage in eval_cages:
            cage.close()
        buffer_method(step_buffer, "close")
        buffer_method(step_buffer, "destroy")
    
        sampler.close()
        agent.close()
        for cage in cages:
            cage.close()
        buffer_method(batch_buffer, "close")
        buffer_method(batch_buffer, "destroy")
    

if __name__ == "__main__":
    mp.set_start_method("fork")

    config = dict(
        parallel=False,
        batch_T=128,
        batch_B=16,
        discount=0.99,
        learning_rate=0.001,
        max_steps_decorrelate=50,
        env=dict(
            max_episode_steps=1000,
            reward_type="sparse",
        ),
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        pi_model=dict(
            hidden_sizes=[64, 64],
            hidden_nonlinearity=torch.nn.Tanh,
        ),
        q_model=dict(
            hidden_sizes=[64, 64],
            hidden_nonlinearity=torch.nn.Tanh,
        ),
        agent=dict(
            learning_starts=1e4,
        ),
        replay_length=20 * 128,
        algo=dict(
            learning_starts=int(1e4),
            replay_ratio=64,
            target_update_tau=0.01
        ),
        eval_sampler=dict(
            max_traj_length=2000,
            min_trajectories=20,
            n_eval_envs=16,
        ),
        runner=dict(
            n_steps=100 * 16 * 128,
            log_interval_steps=5 * 16 * 128,
        ),
    )

    config = add_default_sac_config(config)
    config = add_default_config_fields(config)

    logger.init(
        log_dir=Path(f"log_data/cartpole-sac/{datetime.now().strftime('%Y-%m-%d_%H-%M')}"),
        tensorboard=True,
        output_files={
            "txt": "log.txt",
            # "csv": "progress.csv",
        },
        config=config,
        model_save_path="model.pt",
        # verbosity=Verbosity.DEBUG,
    )

    with build(config) as runner:
        runner.run()
