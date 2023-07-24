from contextlib import contextmanager
from datetime import datetime
import multiprocessing as mp
from pathlib import Path
from typing import Dict

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import torch
import wandb

from parllel.arrays import buffer_from_example
from parllel.buffers import AgentSamples, buffer_asarray, buffer_method, Samples
from parllel.cages import TrajInfo
import parllel.logger as logger
from parllel.logger import Verbosity
from parllel.patterns import (add_advantage_estimation, add_bootstrap_value, add_reward_clipping,
    add_reward_normalization, add_valid, build_cages_and_sample_tree,
    add_initial_rnn_state)
from parllel.replays import BatchedDataLoader
from parllel.runners import OnPolicyRunner
from parllel.samplers import RecurrentSampler
from parllel.torch.agents.categorical import CategoricalPgAgent
from parllel.torch.agents.ensemble import AgentProfile
from parllel.torch.agents.independent import IndependentPgAgents
from parllel.torch.algos.ppo import PPO, build_loss_sample_tree
from parllel.torch.distributions import Categorical
from parllel.torch.handler import TorchHandler
from parllel.torch.utils import torchify_buffer, buffer_to_device
from parllel.transforms import Compose
from parllel.types import BatchSpec

from hera_gym.builds.multi_agent_cartpole import build_multi_agent_cartpole
from models.atari_lstm_model import AtariLstmPgModel


@contextmanager
def build(config: Dict) -> OnPolicyRunner:

    parallel = config["parallel"]
    batch_spec = BatchSpec(
        config["batch_T"],
        config["batch_B"],
    )
    TrajInfo.set_discount(config["algo"]["discount"])

    cages, batch_action, batch_env = build_cages_and_sample_tree(
        EnvClass=build_multi_agent_cartpole,
        env_kwargs=config["env"],
        TrajInfoClass=TrajInfo,
        reset_automatically=False,
        batch_spec=batch_spec,
        parallel=parallel,
    )

    spaces = cages[0].spaces
    obs_space, action_space = spaces.observation, spaces.action

    # write dict into namedarraytuple and read it back out. this ensures the
    # example is in a standard format (i.e. namedarraytuple).
    batch_env.observation[0] = obs_space.sample()
    example_obs = batch_env.observation[0]
    batch_action[0] = action_space.sample()

    # instantiate model and agent
    device = config["device"] or ("cuda:0" if torch.cuda.is_available() else "cpu")
    wandb.config.update({"device": device}, allow_val_change=True)
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
        example_action=batch_action.cart[0],
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
        example_action=batch_action.camera[0],
        device=device,
        recurrent=True,
    )
    camera_profile = AgentProfile(instance=camera_agent, action_key="camera")

    agent = IndependentPgAgents(
        agent_profiles=[cart_profile, camera_profile],
        action_space=action_space,
    )
    agent = TorchHandler(agent=agent)

    # get example output from agent
    _, agent_info = agent.step(example_obs)

    # allocate batch buffer based on examples
    batch_agent_info = buffer_from_example(agent_info, (batch_spec.T,))
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
        discount=config["algo"]["discount"],
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
        discount=config["algo"]["discount"],
        gae_lambda=config["algo"]["gae_lambda"],
        multiagent=True,
        normalize=config["algo"]["normalize_advantage"],
    )

    sampler = RecurrentSampler(
        batch_spec=batch_spec,
        envs=cages,
        agent=agent,
        sample_tree=batch_buffer,
        max_steps_decorrelate=config["max_steps_decorrelate"],
        get_bootstrap_value=True,
        batch_transform=Compose(batch_transforms),
    )

    dataloader_buffer = build_loss_sample_tree(batch_buffer, recurrent=True)
    dataloader_buffer = buffer_asarray(dataloader_buffer)
    dataloader_buffer = torchify_buffer(dataloader_buffer)

    dataloader = BatchedDataLoader(
        buffer=dataloader_buffer,
        sampler_batch_spec=batch_spec,
        n_batches=config["algo"]["minibatches"],
        batch_only_fields=["initial_rnn_state"],
        recurrent=True,
        pre_batches_transform=lambda x: buffer_to_device(x, device=device),
    )

    optimizer = torch.optim.Adam(
        agent.model.parameters(),
        lr=config["algo"]["learning_rate"],
        **config.get("optimizer", {}),
    )
    
    # create algorithm
    algorithm = PPO(
        agent=agent,
        dataloader=dataloader,
        optimizer=optimizer,
        **config["algo"],
    )

    # create runner
    runner = OnPolicyRunner(
        sampler=sampler,
        agent=agent,
        algorithm=algorithm,
        batch_spec=batch_spec,
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


@hydra.main(version_base=None, config_path="conf", config_name="train_multiagent_ppo_recurrent")
def main(config: DictConfig) -> None:

    mp.set_start_method("fork")

    if config.get("render_during_training", False):
        with open_dict(config):
            config["env"]["headless"] = False
            config["env"]["subprocess"] = config["parallel"]

    run = wandb.init(
        anonymous="must", # for this example, send to wandb dummy account
        project="CartPole",
        tags=["discrete", "image-based", "ppo", "recurrent", "multiagent"],
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
        sync_tensorboard=True,  # auto-upload any values logged to tensorboard
        monitor_gym=True,  # save videos to wandb
        save_code=True,  # save script used to start training, git commit, and patch
    )

    logger.init(
        wandb_run=run,
        # this log_dir is used if wandb is disabled (using `wandb disabled`)
        log_dir=Path(f"log_data/cartpole-multiagent-independentppo/{datetime.now().strftime('%Y-%m-%d_%H-%M')}"),
        tensorboard=True,
        output_files={
            "txt": "log.txt",
            # "csv": "progress.csv",
        },
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
        model_save_path="model.pt",
        # verbosity=Verbosity.DEBUG,
    )

    with build(config) as runner:
        runner.run()

    run.finish()


if __name__ == "__main__":
    main()
