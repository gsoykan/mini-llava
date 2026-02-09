from lightning import Callback
from lightning.pytorch.loggers import WandbLogger
from lightning_utilities.core.rank_zero import rank_zero_only


def get_wandb_logger(trainer) -> WandbLogger:
    """Safely get Weights&Biases logger from Trainer."""
    if trainer.fast_dev_run:
        raise Exception(
            "Cannot use wandb callbacks since pytorch lightning disables loggers in `fast_dev_run=true` mode."
        )

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    if isinstance(trainer.logger, list):
        for logger in trainer.logger:
            if isinstance(logger, WandbLogger):
                return logger

    raise Exception(
        "You are using wandb related callback, but WandbLogger was not found for some reason..."
    )


class WatchModel(Callback):
    """Make wandb watch model at the beginning of the run.

    This callback enables WandB's model watching feature which:
    - Logs gradients and parameter distributions
    - Visualizes the computational graph
    - Tracks model architecture in WandB UI

    Note: This can add overhead to training, use primarily for debugging.
    """

    def __init__(self, log_value: str = "gradients", log_freq: int = 100):
        """Initialize WatchModel callback.

        :param log_value: What to log - "gradients", "parameters", or "all"
        :param log_freq: Frequency (in batches) to log histograms
        """
        self.log_value = log_value
        self.log_freq = log_freq

    @rank_zero_only
    def on_train_start(self, trainer, pl_module) -> None:
        """Hook called at the start of training."""
        logger = get_wandb_logger(trainer=trainer)
        logger.watch(
            model=trainer.model,
            log=self.log_value,
            log_freq=self.log_freq,
            log_graph=True,
        )
