import wandb
from pytorch_lightning.callbacks import Callback


class DefineMetricCallback_cls(Callback):
    def setup(self, trainer, pl_module, stage):
        # Check if this is the main process
        if trainer.is_global_zero:
            wandb.define_metric("val_accuracy", summary="max")
            wandb.define_metric("val_loss", summary="min")

class DefineMetricCallback_reg(Callback):
    def setup(self, trainer, pl_module, stage):
        # Check if this is the main process
        if trainer.is_global_zero:
            wandb.define_metric("val_L1loss", summary="min")
            wandb.define_metric("val_loss", summary="min")
