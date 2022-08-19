from contextlib import contextmanager
from datetime import datetime
import multiprocessing as mp
from pathlib import Path
from typing import Dict

import torch

from parllel.arrays import (Array, RotatingArray, SharedMemoryArray, 
    RotatingSharedMemoryArray, buffer_from_example)
from parllel.buffers import AgentSamples, buffer_method, Samples
from parllel.cages import TrajInfo
from parllel.configuration import add_default_config_fields, add_metadata
from parllel.logging import init_log_folder, log_config
from parllel.patterns import (add_advantage_estimation, add_bootstrap_value, add_reward_clipping,
    add_reward_normalization, add_valid, build_cages_and_env_buffers,
    add_initial_rnn_state)
from parllel.runners import OnPolicyRunner
from parllel.samplers import RecurrentSampler
from parllel.torch.agents.categorical import CategoricalPgAgent
from parllel.torch.agents.ensemble import AgentProfile
from parllel.torch.agents.independent import IndependentPgAgents
from parllel.torch.algos.ppo import PPO, add_default_ppo_config
from parllel.torch.distributions import Categorical
from parllel.torch.handler import TorchHandler
from parllel.transforms import Compose
from parllel.types import BatchSpec

from hera_gym.builds.multi_agent_cartpole import build_multi_agent_cartpole
from models.atari_lstm_model import AtariLstmPgModel


@contextmanager
def build(config: Dict) -> OnPolicyRunner:

    init_log_folder(config["log_dir"])
    log_config(config, config["log_dir"])

    parallel = config["parallel"]
    batch_spec = BatchSpec(
        config["batch_T"],
        config["batch_B"],
    )
    traj_info_kwargs = {
        "discount": config["discount"],
    }

    if parallel:
        ArrayCls = SharedMemoryArray
        RotatingArrayCls = RotatingSharedMemoryArray
    else:
        ArrayCls = Array
        RotatingArrayCls = RotatingArray

    cages, batch_action, batch_env = build_cages_and_env_buffers(
        EnvClass=build_multi_agent_cartpole,
        env_kwargs=config["env"],
        TrajInfoClass=TrajInfo,
        traj_info_kwargs=traj_info_kwargs,
        wait_before_reset=True,
        batch_spec=batch_spec,
        parallel=parallel,
    )

    obs_space, action_space = cages[0].spaces

    # instantiate model and agent
    device = torch.device(config["device"])
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

    batch_buffer, batch_transforms = add_advantage_estimation(
        batch_buffer,
        batch_transforms,
        discount=config["discount"],
        gae_lambda=config["gae_lambda"],
        multiagent=True,
        normalize=config["normalize_advantage"],
    )

    sampler = RecurrentSampler(
        batch_spec=batch_spec,
        envs=cages,
        agent=agent,
        sample_buffer=batch_buffer,
        max_steps_decorrelate=config["max_steps_decorrelate"],
        get_bootstrap_value=True,
        batch_transform=Compose(batch_transforms),
    )

    optimizer = torch.optim.Adam(
        agent.model.parameters(),
        lr=config["learning_rate"],
        **config["optimizer"],
    )
    
    # create algorithm
    algorithm = PPO(
        batch_spec=batch_spec,
        agent=agent,
        optimizer=optimizer,
        **config["algo"],
    )

    # create runner
    runner = OnPolicyRunner(
        sampler=sampler,
        agent=agent,
        algorithm=algorithm,
        batch_spec=batch_spec,
        log_dir=config["log_dir"],
        **config["runner"],
    )

    try:
        yield runner
    
    finally:
        sampler.close()
        agent.close()
        for cage in cages:
            cage.close()
        buffer_method(batch_buffer, "close")
        buffer_method(batch_buffer, "destroy")


if __name__ == "__main__":
    mp.set_start_method("fork")

    model_config = dict(
        channels = [32, 64, 64],
        kernel_sizes = [8, 4, 3],
        strides = [4, 2, 1],
        paddings = [0, 0, 0],
        use_maxpool = False,
        post_conv_hidden_sizes = 256,
        post_conv_output_size = None,
        post_conv_nonlinearity = torch.nn.ReLU,
        lstm_size = 256,
        post_lstm_hidden_sizes = None,
        post_lstm_nonlinearity = torch.nn.ReLU,
    )

    config = dict(
        log_dir = Path(f"log_data/cartpole-multiagent-independentppo/{datetime.now().strftime('%Y-%m-%d_%H-%M')}"),
        parallel = False,
        batch_T = 64,
        batch_B = 16,
        discount = 0.99,
        learning_rate = 3e-4,
        gae_lambda = 0.8,
        reward_clip_min = -5,
        reward_clip_max = 5,
        normalize_advantage = True,
        max_steps_decorrelate = 50,
        render_during_training = False,
        env = dict(
            max_episode_steps = 1000,
            reward_type = "sparse",
            headless = True,
        ),
        device = "cuda:0" if torch.cuda.is_available() else "cpu",
        cart_model = model_config.copy(),
        camera_model = model_config.copy(),
        runner = dict(
            n_steps = 2e6,
            log_interval_steps = 1e4,
        ),
        meta = dict(
            name = "[Example 1] Multiagent Visual CartPole with Independent PPO",
        ),
    )

    if config.get("render_during_training", False):
        config["env"]["headless"] = False
        config["env"]["subprocess"] = config["parallel"]

    config = add_default_ppo_config(config)
    config = add_metadata(config, build)
    config = add_default_config_fields(config)

    with build(config) as runner:
        runner.run()
