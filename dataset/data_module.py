import pytorch_lightning as pl
from dataset.collate_batch import make_dataloader
from dataset.dataset import WakeWordDataset
from torchaudio.transforms import MelSpectrogram


class WakeWordDataModule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage):
        if stage == "fit":
            transform = MelSpectrogram()
            transform = MelSpectrogram()
            self.wake_word_data_train = WakeWordDataset(positive_dir="train/positive_samples", 
                                                        negative_dir="train/negative_samples",
                                                        transform=transform,
                                                        augment=None,
                                                        )
            self.wake_word_data_val = WakeWordDataset(positive_dir="test/positive_samples", 
                                                      negative_dir="test/negative_samples",
                                                      transform=transform,
                                                      augment=None,
                                                      )
    
    def train_dataloader(self):
        return make_dataloader(self.wake_word_data_train, "train", batch_size=self.batch_size, num_workers=8)
    
    def val_dataloader(self):
        return make_dataloader(self.wake_word_data_val, "val", batch_size=self.batch_size, num_workers=8)