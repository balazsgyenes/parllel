from contextlib import contextmanager
from datetime import datetime
import itertools
import multiprocessing as mp
from pathlib import Path
from typing import Dict

import torch

from parllel.arrays import (Array, RotatingArray, SharedMemoryArray, 
    RotatingSharedMemoryArray, buffer_from_example)
from parllel.buffers import AgentSamples, buffer_method, Samples, NamedArrayTupleClass
from parllel.cages import TrajInfo
from parllel.configuration import add_default_config_fields
import parllel.logger as logger
from parllel.logger import Verbosity
from parllel.patterns import (add_reward_clipping, add_reward_normalization,
    build_cages_and_env_buffers, build_eval_sampler)
from parllel.replays.replay import ReplayBuffer
from parllel.runners import OffPolicyRunner
from parllel.samplers import BasicSampler
from parllel.torch.agents.sac_agent import SacAgent
from parllel.torch.algos.sac import SAC, add_default_sac_config
from parllel.torch.distributions.squashed_gaussian import SquashedGaussian
from parllel.torch.handler import TorchHandler
from parllel.transforms import Compose
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

    if parallel:
        ArrayCls = SharedMemoryArray
        RotatingArrayCls = RotatingSharedMemoryArray
    else:
        ArrayCls = Array
        RotatingArrayCls = RotatingArray

    cages, batch_action, batch_env = build_cages_and_env_buffers(
        EnvClass=build_cartpole,
        env_kwargs=config["env"],
        TrajInfoClass=TrajInfo,
        wait_before_reset=False,
        batch_spec=batch_spec,
        parallel=parallel,
    )

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
    batch_agent_info = buffer_from_example(agent_info, (batch_spec.T,), ArrayCls)
    batch_agent = AgentSamples(batch_action, batch_agent_info)
    batch_buffer = Samples(batch_agent, batch_env)

    # add several helpful transforms
    batch_transforms = []

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

    sampler = BasicSampler(
        batch_spec=batch_spec,
        envs=cages,
        agent=agent,
        sample_buffer=batch_buffer,
        max_steps_decorrelate=config["max_steps_decorrelate"],
        get_bootstrap_value=False,
        batch_transform=Compose(batch_transforms),
    )

    # create the replay buffer as a longer version of the batch buffer
    replay_length = config["replay_length"]
    replay_buffer = buffer_from_example(batch_buffer[0], (replay_length,))
    ReplayBufferSamples = NamedArrayTupleClass("ReplayBufferSamples",
        ["observation", "action", "reward", "done", "next_observation"])
    replay_obs = replay_buffer.env.observation
    replay_buffer_samples = ReplayBufferSamples(
        observation=replay_obs,
        action=replay_buffer.agent.action,
        reward=replay_buffer.env.reward,
        done=replay_buffer.env.done,
        # TODO: replace with replay_obs.next
        next_observation=replay_obs[1 : replay_obs.last + 2],
    )

    replay_buffer = ReplayBuffer(
        buffer_to_append=replay_buffer,
        buffer_to_sample=replay_buffer_samples,
        batch_spec=batch_spec,
        length_T=replay_length,
        newest_n_samples_invalid=1, # next_observation not set yet
        # actually, all samples have a next_observation already, but it is not
        # copied into the replay buffer because of conversion to ndarray
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
        parallel=True,
        batch_T=128,
        batch_B=16,
        discount=0.99,
        learning_rate=0.001,
        reward_clip_min=-5,
        reward_clip_max=5,
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
            learning_starts=1e4,
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
