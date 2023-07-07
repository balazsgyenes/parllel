from contextlib import contextmanager
from os import PathLike
from pathlib import Path
from typing import Dict

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf, open_dict
import torch

from parllel.arrays import buffer_from_example, buffer_from_dict_example
from parllel.buffers import AgentSamples, EnvSamples, buffer_method, Samples
from parllel.cages import SerialCage, TrajInfo
from parllel.logger import JSONConfigSerializer
from parllel.runners import ShowPolicy
from parllel.samplers import EvalSampler
from parllel.torch.agents.categorical import CategoricalPgAgent
from parllel.torch.agents.ensemble import AgentProfile
from parllel.torch.agents.independent import IndependentPgAgents
from parllel.torch.distributions import Categorical
from parllel.torch.handler import TorchHandler

from hera_gym.builds.multi_agent_cartpole import build_multi_agent_cartpole
from models.atari_lstm_model import AtariLstmPgModel


@contextmanager
def build(config: Dict, model_checkpoint_path: PathLike) -> ShowPolicy:

    TrajInfo.set_discount(config["discount"])  # use the discount provided for evaluation

    cage_kwargs = dict(
        EnvClass=build_multi_agent_cartpole,
        env_kwargs=config["env"],
        TrajInfoClass=TrajInfo,
        reset_automatically=True,
    )

    cage = SerialCage(**cage_kwargs)

    # get example output from env
    cage.random_step_async()
    action, obs, reward, terminated, truncated, info = cage.await_step()

    obs_space, action_space = cage.spaces.observation, cage.spaces.action

    # allocate batch buffer based on examples
    step_observation = buffer_from_dict_example(obs, (1, 1), name="obs", padding=1)
    step_reward = buffer_from_dict_example(reward, (1, 1), name="reward")
    step_terminated = buffer_from_dict_example(terminated, (1, 1), name="done", padding=1)
    step_truncated = buffer_from_dict_example(truncated, (1, 1), name="done", padding=1)
    step_done = buffer_from_dict_example(terminated, (1, 1), name="done", padding=1)
    step_info = buffer_from_dict_example(info, (1, 1), name="envinfo")
    step_env = EnvSamples(step_observation, step_reward, step_done, step_terminated, step_truncated, step_info)

    step_action = buffer_from_dict_example(action, (1, 1), name="action")

    # write dict into namedarraytuple and read it back out. this ensures the
    # example is in a standard format (i.e. namedarraytuple).
    step_env.observation[0] = obs_space.sample()
    example_obs = step_env.observation[0]
    step_action[0] = action_space.sample()

    # instantiate model and agent
    device = config["device"] or ("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
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
        example_obs=example_obs,
        example_action=step_action.cart[0],
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
        example_obs=example_obs,
        example_action=step_action.camera[0],
        device=device,
        recurrent=True,
    )
    camera_profile = AgentProfile(instance=camera_agent, action_key="camera")

    agent = IndependentPgAgents(
        agent_profiles=[cart_profile, camera_profile],
        action_space=action_space,
    )
    agent = TorchHandler(agent=agent)

    agent.load_model(model_checkpoint_path, device)

    # get example output from agent
    _, agent_info = agent.step(example_obs)

    step_agent_info = buffer_from_example(agent_info, (1, 1))
    step_agent = AgentSamples(step_action, step_agent_info)

    step_buffer = Samples(step_agent, step_env)

    sampler = EvalSampler(
        max_traj_length=config["max_steps"],
        min_trajectories=config["min_trajectories"],
        envs=[cage],
        agent=agent,
        step_buffer=step_buffer,
        # TODO: add obs_transforms loaded from snapshot
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
        cage.close()


@hydra.main(version_base=None, config_path="conf", config_name="show_multiagent_ppo_recurrent")
def main(config: DictConfig) -> None:

    if config["max_steps"] is None:
        with open_dict(config):
            config["max_steps"] = np.iinfo(int).max

    training_config = JSONConfigSerializer().load(Path(config["log_dir"]) / "config.json")
    training_config = OmegaConf.create(training_config)

    config = OmegaConf.merge(training_config, config)

    model_checkpoint_path = Path(config["log_dir"]) / "model.pt"

    with build(config, model_checkpoint_path) as runner:
        runner.run()


if __name__ == "__main__":
    main()
