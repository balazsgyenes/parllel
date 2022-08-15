from contextlib import contextmanager
from os import PathLike
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from parllel.arrays import Array, RotatingArray, buffer_from_dict_example
from parllel.buffers import AgentSamples, EnvSamples, buffer_method, Samples
from parllel.buffers.void import VoidBuffer
from parllel.cages import Cage, TrajInfo
from parllel.logging import MODEL_FILENAME, load_config
from parllel.runners import ShowPolicy
from parllel.samplers import EvalSampler
from parllel.torch.agents.categorical import CategoricalPgAgent
from parllel.torch.agents.ensemble import AgentProfile
from parllel.torch.agents.independent import IndependentPgAgents
from parllel.torch.distributions import Categorical
from parllel.torch.handler import TorchHandler
from parllel.types import BatchSpec

from hera_gym.builds.multi_agent_cartpole import build_multi_agent_cartpole
from models.atari_lstm_model import AtariLstmPgModel


@contextmanager
def build(config: Dict, model_checkpoint_path: PathLike) -> ShowPolicy:

    if config["max_steps"] is None:
        config["max_steps"] = np.iinfo(int).max

    traj_info_kwargs = {
        "discount": config["discount"],
    }

    batch_spec = BatchSpec(
        T=config["max_steps"],
        B=1,
    )

    cage_kwargs = dict(
        EnvClass = build_multi_agent_cartpole,
        env_kwargs = config["env"],
        TrajInfoClass = TrajInfo,
        traj_info_kwargs = traj_info_kwargs,
        wait_before_reset = False,
    )

    cage = Cage(**cage_kwargs)

    # get example output from env
    cage.random_step_async()
    action, obs, reward, done, info = cage.await_step()

    obs_space, action_space = cage.spaces

    # allocate batch buffer based on examples
    step_observation = buffer_from_dict_example(obs, (1, 1), RotatingArray, name="obs", padding=1)
    step_reward = buffer_from_dict_example(reward, (1, 1), Array, name="reward")
    step_done = buffer_from_dict_example(done, (1, 1), RotatingArray, name="done", padding=1)
    step_info = buffer_from_dict_example(info, (1, 1), Array, name="envinfo")
    step_env = EnvSamples(step_observation, step_reward, step_done, step_info)

    step_action = buffer_from_dict_example(action, (1, 1), Array, name="action")
    step_agent = AgentSamples(step_action, VoidBuffer())

    step_buffer = Samples(step_agent, step_env)

    # instantiate model and agent
    device = torch.device("cuda", index=0) if torch.cuda.is_available() else torch.device("cpu")
    ## cart
    cart_model = AtariLstmPgModel(
        obs_space=obs_space,
        action_space=action_space["cart"],
        **config["cart_model"],
    )
    cart_distribution = Categorical(dim=action_space["cart"].n)
    cart_agent = CategoricalPgAgent(
        model=cart_model,
        distribution=cart_distribution,
        observation_space=obs_space,
        action_space=action_space["cart"],
        n_states=batch_spec.B,
        device=device,
        recurrent=True,
    )
    cart_profile = AgentProfile(instance=cart_agent, action_key="cart")

    ## camera
    camera_model = AtariLstmPgModel(
        obs_space=obs_space,
        action_space=action_space["camera"],
        **config["camera_model"],
    )
    camera_distribution = Categorical(dim=action_space["camera"].n)
    camera_agent = CategoricalPgAgent(
        model=camera_model,
        distribution=camera_distribution,
        observation_space=obs_space,
        action_space=action_space["camera"],
        n_states=batch_spec.B,
        device=device,
        recurrent=True,
    )
    camera_profile = AgentProfile(instance=camera_agent, action_key="camera")

    agent = IndependentPgAgents(
        agent_profiles=[cart_profile, camera_profile],
        observation_space=obs_space,
        action_space=action_space,
    )
    agent = TorchHandler(agent=agent)

    agent.load_model(model_checkpoint_path)

    sampler = EvalSampler(
        max_traj_length=config["max_steps"],
        min_trajectories=config["min_trajectories"],
        envs=[cage],
        agent=agent,
        step_buffer=step_buffer,
        # add obs_transforms loaded from snapshot
    )

    # create runner
    runner = ShowPolicy(
        sampler=sampler,
        agent=agent,
    )

    try:
        yield runner
    
    finally:
        sampler.close()
        agent.close()
        buffer_method(step_buffer, "close")
        buffer_method(step_buffer, "destroy")
        cage.close()


if __name__ == "__main__":

    log_dir = Path("log_data/cartpole-multiagent-independentppo/2022-08-15_15-39")

    eval_config = dict(
        max_steps = None,
        min_trajectories = 20,
    )

    config = load_config(log_dir)

    config |= eval_config

    model_checkpoint_path = log_dir / MODEL_FILENAME

    with build(config, model_checkpoint_path) as runner:
        runner.run()
