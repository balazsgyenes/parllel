import multiprocessing as mp
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import hydra
import torch
from envs.continuous_cartpole import build_cartpole
from gymnasium import spaces
from models.pg_model import GaussianCartPoleFfPgModel
from omegaconf import DictConfig, OmegaConf

import parllel.logger as logger
import wandb
from parllel.cages import TrajInfo
from parllel.logger import Verbosity
from parllel.patterns import (add_advantage_estimation, add_agent_info,
                              add_bootstrap_value, add_obs_normalization,
                              add_reward_clipping, add_reward_normalization,
                              build_cages_and_sample_tree, build_eval_sampler)
from parllel.replays import BatchedDataLoader
from parllel.runners import OnPolicyRunner
from parllel.samplers import BasicSampler
from parllel.torch.agents.gaussian import GaussianPgAgent
from parllel.torch.algos.ppo import PPO, build_loss_sample_tree
from parllel.torch.distributions import Gaussian
from parllel.transforms import Compose
from parllel.types import BatchSpec


@contextmanager
def build(config: DictConfig) -> OnPolicyRunner:
    parallel = config["parallel"]
    batch_spec = BatchSpec(
        config["batch_T"],
        config["batch_B"],
    )
    TrajInfo.set_discount(config["algo"]["discount"])

    cages, sample_tree, metadata = build_cages_and_sample_tree(
        EnvClass=build_cartpole,
        env_kwargs=config["env"],
        TrajInfoClass=TrajInfo,
        reset_automatically=True,
        batch_spec=batch_spec,
        parallel=parallel,
    )
    obs_space, action_space = metadata.obs_space, metadata.action_space
    assert isinstance(obs_space, spaces.Box)
    assert isinstance(action_space, spaces.Box)

    # instantiate model
    model = GaussianCartPoleFfPgModel(
        obs_space=obs_space,
        action_space=action_space,
        **config["model"],
    )
    distribution = Gaussian(
        dim=action_space.shape[0],
        deterministic_eval_mode=config["distribution"]["deterministic_eval_mode"],
    )
    device = config["device"] or ("cuda:0" if torch.cuda.is_available() else "cpu")
    wandb.config.update({"device": device}, allow_val_change=True)
    device = torch.device(device)

    # instantiate agent
    agent = GaussianPgAgent(
        model=model,
        distribution=distribution,
        example_obs=metadata.example_obs_batch,
        device=device,
    )

    # add agent info, which stores value predictions
    sample_tree = add_agent_info(sample_tree, agent, metadata.example_obs_batch)

    # for advantage estimation, we need to estimate the value of the last
    # state in the batch
    sample_tree = add_bootstrap_value(sample_tree)

    # add several helpful transforms
    batch_transforms, step_transforms = [], []

    sample_tree, step_transforms = add_obs_normalization(
        sample_tree,
        step_transforms,
        initial_count=config["obs_norm_initial_count"],
    )

    sample_tree, batch_transforms = add_reward_normalization(
        sample_tree,
        batch_transforms,
        discount=config["algo"]["discount"],
    )

    sample_tree, batch_transforms = add_reward_clipping(
        sample_tree,
        batch_transforms,
        reward_clip_min=config["reward_clip_min"],
        reward_clip_max=config["reward_clip_max"],
    )

    # add advantage normalization, required for PPO
    sample_tree, batch_transforms = add_advantage_estimation(
        sample_tree,
        batch_transforms,
        discount=config["algo"]["discount"],
        gae_lambda=config["algo"]["gae_lambda"],
        normalize=config["algo"]["normalize_advantage"],
    )

    sampler = BasicSampler(
        batch_spec=batch_spec,
        envs=cages,
        agent=agent,
        sample_tree=sample_tree,
        max_steps_decorrelate=config["max_steps_decorrelate"],
        get_bootstrap_value=True,
        obs_transform=Compose(step_transforms),
        batch_transform=Compose(batch_transforms),
    )

    loss_sample_tree = build_loss_sample_tree(sample_tree)
    loss_sample_tree = loss_sample_tree.to_ndarray()
    loss_sample_tree = loss_sample_tree.apply(torch.from_numpy)

    dataloader = BatchedDataLoader(
        tree=loss_sample_tree,
        sampler_batch_spec=batch_spec,
        n_batches=config["algo"]["minibatches"],
        pre_batches_transform=lambda tree: tree.to(device=device),
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

    eval_sampler, eval_sample_tree = build_eval_sampler(
        sample_tree=sample_tree,
        agent=agent,
        CageCls=type(cages[0]),
        EnvClass=build_cartpole,
        env_kwargs=config["env"],
        TrajInfoClass=TrajInfo,
        **config["eval_sampler"],
    )

    # create runner
    runner = OnPolicyRunner(
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
        sampler.close()
        agent.close()
        for cage in cages:
            cage.close()
        sample_tree.close()


@hydra.main(version_base=None, config_path="conf", config_name="train_ppo")
def main(config: DictConfig) -> None:
    mp.set_start_method("fork")

    run = wandb.init(
        anonymous="must",  # for this example, send to wandb dummy account
        project="Continuous CartPole",
        tags=["continuous", "state-based", "ppo", "feedforward"],
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
        sync_tensorboard=True,  # auto-upload any values logged to tensorboard
        save_code=True,  # save script used to start training, git commit, and patch
    )

    logger.init(
        wandb_run=run,
        # this log_dir is used if wandb is disabled (using `wandb disabled`)
        log_dir=Path(
            f"log_data/continuous-cartpole-ppo/{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
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
