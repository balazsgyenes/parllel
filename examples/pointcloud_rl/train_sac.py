# fmt: off
import itertools
import multiprocessing as mp
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

# isort: off
import hydra
import torch
import wandb
from gymnasium import spaces
from omegaconf import DictConfig, OmegaConf

# isort: on
import parllel.logger as logger
from parllel import Array, ArrayDict, dict_map
from parllel.cages import TrajInfo
from parllel.logger import Verbosity
from parllel.patterns import build_cages_and_sample_tree, build_eval_sampler
from parllel.replays.replay import ReplayBuffer
from parllel.runners import RLRunner
from parllel.samplers import BasicSampler
from parllel.torch.agents.sac_agent import SacAgent
from parllel.torch.algos.sac import SAC, build_replay_buffer_tree
from parllel.torch.distributions.squashed_gaussian import SquashedGaussian
from parllel.types import BatchSpec

# isort: split
from envs.dummy import DummyEnv
from models.modules import PointNetEncoder
from models.pointnet_q_and_pi import PointNetPiModel, PointNetQModel
from pointcloud import PointCloudSpace


# fmt: on
@contextmanager
def build(config: DictConfig) -> RLRunner:
    parallel = config["parallel"]
    batch_spec = BatchSpec(
        config["batch_T"],
        config["batch_B"],
    )
    TrajInfo.set_discount(config["algo"]["discount"])

    env_kwargs = {"actions": "continuous"} | {**config["env"]}
    cages, sample_tree, metadata = build_cages_and_sample_tree(
        EnvClass=DummyEnv,
        env_kwargs=env_kwargs,
        TrajInfoClass=TrajInfo,
        reset_automatically=True,
        batch_spec=batch_spec,
        parallel=parallel,
        full_size=config["algo"]["replay_length"],
        keys_to_skip="observation",  # we will allocate this ourselves
    )
    obs_space, action_space = metadata.obs_space, metadata.action_space
    assert isinstance(obs_space, PointCloudSpace)
    assert isinstance(action_space, spaces.Box)

    sample_tree["observation"] = dict_map(
        Array.from_numpy,
        metadata.example_obs,
        batch_shape=tuple(batch_spec),
        max_mean_num_elem=obs_space.max_num_points,
        feature_shape=obs_space.shape,
        kind="jagged",
        storage="shared" if parallel else "local",
        padding=1,
        full_size=config["algo"]["replay_length"],
    )

    # instantiate models
    pi_model = PointNetPiModel(
        encoding_size=config["encoder"]["encoding_size"],
        action_space=action_space,
        **config["pi_model"],
    )
    q1_model = PointNetQModel(
        encoding_size=config["encoder"]["encoding_size"],
        action_space=action_space,
        **config["q_model"],
    )
    q2_model = PointNetQModel(
        encoding_size=config["encoder"]["encoding_size"],
        action_space=action_space,
        **config["q_model"],
    )
    encoder = PointNetEncoder(
        obs_space=obs_space,
        **config["encoder"],
    )
    model = torch.nn.ModuleDict(
        {
            "pi": pi_model,
            "q1": q1_model,
            "q2": q2_model,
            "encoder": encoder,
        }
    )
    distribution = SquashedGaussian(
        dim=action_space.shape[0],
        scale=action_space.high[0],
    )
    device = config["device"] or ("cuda:0" if torch.cuda.is_available() else "cpu")
    wandb.config.update({"device": device}, allow_val_change=True)
    device = torch.device(device)

    # instantiate agent
    agent = SacAgent(
        model=model,
        distribution=distribution,
        device=device,
        learning_starts=config["algo"]["learning_starts"],
    )

    sampler = BasicSampler(
        batch_spec=batch_spec,
        envs=cages,
        agent=agent,
        sample_tree=sample_tree,
        max_steps_decorrelate=config["max_steps_decorrelate"],
        get_bootstrap_value=False,
    )

    replay_buffer_tree = build_replay_buffer_tree(sample_tree)

    def batch_transform(tree: ArrayDict[Array]) -> ArrayDict[torch.Tensor]:
        tree = tree.to_ndarray()
        tree = tree.apply(torch.from_numpy)
        return tree.to(device=device)

    replay_buffer = ReplayBuffer(
        tree=replay_buffer_tree,
        sampler_batch_spec=batch_spec,
        size_T=config["algo"]["replay_length"],
        replay_batch_size=config["algo"]["batch_size"],
        newest_n_samples_invalid=0,
        oldest_n_samples_invalid=1,  # TODO: temporary fix to prevent sampling from accessing overwritten point clouds
        batch_transform=batch_transform,
    )

    optimizers = {
        "pi": torch.optim.Adam(
            agent.model["pi"].parameters(),
            lr=config["algo"]["learning_rate"],
            **config.get("optimizer", {}),
        ),
        "q": torch.optim.Adam(
            itertools.chain(
                agent.model["q1"].parameters(),
                agent.model["q2"].parameters(),
            ),
            lr=config["algo"]["learning_rate"],
            **config.get("optimizer", {}),
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

    eval_sampler, eval_sample_tree = build_eval_sampler(
        sample_tree=sample_tree,
        agent=agent,
        CageCls=type(cages[0]),
        EnvClass=DummyEnv,
        env_kwargs=env_kwargs,
        TrajInfoClass=TrajInfo,
        **config["eval_sampler"],
    )

    # create runner
    runner = RLRunner(
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
        eval_sample_tree.close()

        sampler.close()
        agent.close()
        for cage in cages:
            cage.close()
        sample_tree.close()


@hydra.main(version_base=None, config_path="conf", config_name="train_sac")
def main(config: DictConfig) -> None:
    mp.set_start_method("fork")

    run = wandb.init(
        anonymous="must",  # for this example, send to wandb dummy account
        project="PointCloudRL",
        tags=["continuous", "state-based", "sac", "feedforward"],
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
        sync_tensorboard=True,  # auto-upload any values logged to tensorboard
        save_code=True,  # save script used to start training, git commit, and patch
        mode="disabled",
    )

    logger.init(
        wandb_run=run,
        # this log_dir is used if wandb is disabled (using `wandb disabled`)
        log_dir=Path(
            f"log_data/pointcloud-sac/{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
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
