# fmt: off
import multiprocessing as mp

# isort: off
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

# isort: on
import parllel.logger as logger

# isort: split
from build_parllel_sac import build


# fmt: on
@hydra.main(version_base=None, config_path="conf", config_name="sac_cheetah_parllel")
def main(config: DictConfig) -> None:
    run = wandb.init(
        project="parllel",
        tags=["sac", "halfcheetah"],
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
        sync_tensorboard=True,  # auto-upload any values logged to tensorboard
        save_code=True,  # save script used to start training, git commit, and patch
    )

    logger.init(
        wandb_run=run,
        tensorboard=True,
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
    )

    with build(config) as runner:
        runner.run()

    run.finish()


if __name__ == "__main__":
    mp.set_start_method("forkserver")
    main()
