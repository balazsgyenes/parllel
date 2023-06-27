import multiprocessing as mp
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict

import hydra
import torch
from envs.cartpole import build_cartpole
# from hera_gym.wrappers import add_human_render_wrapper, add_subprocess_wrapper
from models.lstm_model import CartPoleLstmPgModel
from omegaconf import DictConfig, OmegaConf

import parllel.logger as logger
import wandb
from parllel.arrays import buffer_from_example
from parllel.buffers import AgentSamples, Samples, buffer_method
from parllel.cages import TrajInfo
from parllel.logger import Verbosity
from parllel.patterns import (add_advantage_estimation, add_bootstrap_value,
                              add_initial_rnn_state, add_obs_normalization,
                              add_reward_clipping, add_reward_normalization,
                              add_valid, build_cages_and_env_buffers)
from parllel.replays import BatchedDataLoader
from parllel.runners.onpolicy import OnPolicyRunner
from parllel.samplers import RecurrentSampler
from parllel.torch.agents.categorical import CategoricalPgAgent
from parllel.torch.algos.ppo import PPO, build_dataloader_buffer
from parllel.torch.distributions import Categorical
from parllel.torch.handler import TorchHandler
from parllel.transforms import Compose
from parllel.types import BatchSpec


@contextmanager
def build(config: Dict) -> OnPolicyRunner:
    parallel = config["parallel"]
    batch_spec = BatchSpec(
        config["batch_T"],
        config["batch_B"],
    )
    TrajInfo.set_discount(config["algo"]["discount"])

    EnvClass = build_cartpole
    # if config["render_during_training"]:
    #     if parallel:
    #         EnvClass = add_subprocess_wrapper(EnvClass)
    #     EnvClass = add_human_render_wrapper(EnvClass)

    cages, batch_action, batch_env = build_cages_and_env_buffers(
        EnvClass=EnvClass,
        env_kwargs=config["env"],
        TrajInfoClass=TrajInfo,
        reset_automatically=False,
        batch_spec=batch_spec,
        parallel=parallel,
    )

    spaces = cages[0].spaces
    obs_space, action_space = spaces.observation, spaces.action

    # instantiate model and agent
    model = CartPoleLstmPgModel(
        obs_space=obs_space,
        action_space=action_space,
        **config["model"],
    )
    distribution = Categorical(dim=action_space.n)
    device = config["device"] or ("cuda:0" if torch.cuda.is_available() else "cpu")
    wandb.config.update({"device": device}, allow_val_change=True)
    device = torch.device(device)

    # instantiate model and agent
    agent = CategoricalPgAgent(
        model=model,
        distribution=distribution,
        observation_space=obs_space,
        action_space=action_space,
        n_states=batch_spec.B,
        device=device,
        recurrent=True,
    )
    agent = TorchHandler(agent=agent)

    # write dict into namedarraytuple and read it back out. this ensures the
    # example is in a standard format (i.e. namedarraytuple).
    batch_env.observation[0] = obs_space.sample()
    example_obs = batch_env.observation[0]

    # get example output from agent
    _, agent_info = agent.step(example_obs)

    # allocate batch buffer based on examples
    storage = "shared" if parallel else "local"
    batch_agent_info = buffer_from_example(agent_info, (batch_spec.T,), storage=storage)
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

    batch_buffer, step_transforms = add_obs_normalization(
        batch_buffer,
        step_transforms,
        initial_count=config["obs_norm_initial_count"],
    )

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

    # add advantage normalization, required for PPO
    batch_buffer, batch_transforms = add_advantage_estimation(
        batch_buffer,
        batch_transforms,
        discount=config["algo"]["discount"],
        gae_lambda=config["algo"]["gae_lambda"],
        normalize=config["algo"]["normalize_advantage"],
    )

    sampler = RecurrentSampler(
        batch_spec=batch_spec,
        envs=cages,
        agent=agent,
        sample_buffer=batch_buffer,
        max_steps_decorrelate=config["max_steps_decorrelate"],
        get_bootstrap_value=True,
        obs_transform=Compose(step_transforms),
        batch_transform=Compose(batch_transforms),
    )

    dataloader_buffer = build_dataloader_buffer(batch_buffer, recurrent=True)

    dataloader = BatchedDataLoader(
        buffer=dataloader_buffer,
        sampler_batch_spec=batch_spec,
        n_batches=config["algo"]["minibatches"],
        batch_only_fields=["init_rnn_state"],
        recurrent=True,
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
        buffer_method(batch_buffer, "destroy")


@hydra.main(version_base=None, config_path="conf", config_name="train_ppo_recurrent")
def main(config: DictConfig) -> None:
    mp.set_start_method("fork")

    run = wandb.init(
        anonymous="must",  # for this example, send to wandb dummy account
        project="CartPole",
        tags=["discrete", "state-based", "ppo", "recurrent"],
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
        sync_tensorboard=True,  # auto-upload any values logged to tensorboard
        save_code=True,  # save script used to start training, git commit, and patch
    )

    logger.init(
        wandb_run=run,
        # this log_dir is used if wandb is disabled (using `wandb disabled`)
        log_dir=Path(
            f"log_data/cartpole-ppo-recurrent/{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
        ),
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
