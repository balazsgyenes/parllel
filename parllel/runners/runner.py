from abc import ABC
from pathlib import Path
from typing import List, Union

import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
    has_summary_writer = True
except ImportError:
    has_summary_writer = False

from parllel.cages.traj_info import TrajInfo
from parllel.logging import MODEL_FILENAME


class Runner(ABC):
    def __init__(self,
        log_dir: Union[Path, str, None] = None,
    ) -> None:
        
        if log_dir is not None:
            print(f"Saving model checkpoints to {log_dir}.")
            self.log_dir = Path(log_dir)
            if has_summary_writer:
                print("Saving learning statistics to tensorboard.")
                self.logger = SummaryWriter(log_dir=str(log_dir))
            else:
                print("WARNING: Tensorboard not installed, so no tensorboard "
                    "records will be created")
                self.logger = None
        else:
            print("WARNING: No log_dir was specified, so nothing from this "
                "run will be saved.")
            self.log_dir = None
            self.logger = None

    def log_progress(self, elapsed_steps: int, trajectories: List[TrajInfo]) -> None:
        traj_disc_returns = [traj.DiscountedReturn for traj in trajectories]
        traj_disc_returns = np.array(traj_disc_returns)
        mean_disc_return = traj_disc_returns.mean()

        if self.logger is not None:
            self.logger.add_scalar("DiscountedReturn", mean_disc_return, elapsed_steps)

        if self.log_dir is not None:
            # TODO: technically, the agent should be added to this base class
            self.agent.save_model(path=self.log_dir / MODEL_FILENAME)

        print(f"Average discounted return: {mean_disc_return:.3f}")
