import os

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from model import WakeWordDetector
from dataset.data_module import WakeWordDataModule
from torch.optim.lr_scheduler import _LRScheduler
from math import cos, pi

class CosineAnnealingWarmup(_LRScheduler):
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            min_lr: float = 0.00001,
            warmup_steps: int = 100,
            decay_steps: int = 500,
            last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.min_lr = min_lr
        self.last_epoch = last_epoch
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            mult = self.last_epoch / self.warmup_steps
        else:
            mult = 0.5 * (
                1 + cos(pi * (self.last_epoch - self.warmup_steps) / self.decay_steps)
            )
            return [
                self.min_lr + (base_lr - self.min_lr) * mult for base_lr in self.base_lrs
            ]
        
def main():
    model_name = "wakeword_detector_v13(+background)"
    # Prepare train dataset 
    data_module = WakeWordDataModule(16)

    wakeword_model = WakeWordDetector()
    
    output_folder = f"output/checkpoints/{model_name}"
    if not os.path.exists(output_folder + "/"):
        os.makedirs(output_folder + "/")
    
    callbacks = [
        ModelCheckpoint(
            dirpath=output_folder + "/",
            filename="{step}-{val_loss:.3f}",
            monitor="val_loss",
            mode="min",
            every_n_epochs=1,
            save_top_k=3,
        ),
    ]

    trainer = Trainer(
        accelerator="gpu",
        max_steps=20_000,
        precision=16,
        benchmark=True,
        callbacks=callbacks,
    )
    trainer.fit(
        wakeword_model,
        data_module,
    )


if __name__ == "__main__":
    main()