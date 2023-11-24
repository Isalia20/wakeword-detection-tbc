import os
import torchaudio
from torch.utils.data import Dataset
from torch.nn import functional as F
import torch


class WakeWordDataset(Dataset):
    def __init__(self, positive_dir, negative_dir, transform=None, augment=None):
        self.transform = transform
        self.augment = augment

        # Collect all audio file paths
        self.positive_file_paths = [os.path.join(positive_dir, f) for f in os.listdir(positive_dir) if f.endswith('.wav')]
        self.negative_file_paths = [os.path.join(negative_dir, f) for f in os.listdir(negative_dir) if f.endswith('.wav')]

        # Concatenate the file paths, and create an array of labels (1 for positive, 0 for negative)
        self.file_paths = self.positive_file_paths + self.negative_file_paths
        self.labels = [1] * len(self.positive_file_paths) + [0] * len(self.negative_file_paths)
        self.max_len = self.find_max_waveform()

    def find_max_waveform(self):
        # Find max wave form from positive file paths
        max_len = 0
        for idx, _ in enumerate(self.positive_file_paths):
            waveform, _ = torchaudio.load(self.file_paths[idx])
            spectrogram = self.transform(waveform) if self.transform else waveform
            if spectrogram.shape[-1] > max_len:
                max_len = spectrogram.shape[-1]
        return max_len

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load the audio file at the given index
        waveform, _ = torchaudio.load(self.file_paths[idx])
        # Get the label for this sample
        label = self.labels[idx]

        # Apply augmentations if any exist
        if self.augment:
            waveform = self.augment(samples=waveform.squeeze().numpy(), sample_rate=16000) # Assuming a sample rate of 16KHz
            waveform = torch.from_numpy(waveform).unsqueeze(0)

        # Apply transformations if any exist
        if self.transform:
            spectrogram = self.transform(waveform)

        # Zero-pad the waveform to a fixed size
        padding = self.max_len - spectrogram.shape[2]
        if padding >= 0:
            spectrogram = F.pad(input=spectrogram, pad=(0, padding, 0, 0))  # Pad the time dimension
        else:
            spectrogram = spectrogram[:, :, :self.max_len]
        
        if spectrogram.shape[0] == 2:
            return None
        
        return spectrogram.unsqueeze(0), torch.tensor(label)


class WakeWordDatasetValidation(WakeWordDataset):
    def __init__(self, positive_dir, negative_dir, transform, augment):
        super().__init__(positive_dir, negative_dir, transform, augment)
