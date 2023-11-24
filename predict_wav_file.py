from model import WakeWordDetector
import torch
import torchaudio
from torch.nn import functional as F
from dataset.dataset import WakeWordDataset

wakeword_model = WakeWordDetector()
weights = torch.load("output/checkpoints/wakeword_detector_v12(+background)/step=2622-val_loss=0.091.ckpt")["state_dict"]
wakeword_model.load_state_dict(weights)

# speech, a = torchaudio.load("output.wav")
speech, a = torchaudio.load("train/positive_samples/speech1.wav")


all_preds = []
transform = torchaudio.transforms.MelSpectrogram()
dataset = WakeWordDataset(positive_dir="wav_files/positive_samples_raw", 
                          negative_dir="wav_files/negative_samples_raw", 
                          transform=transform, 
                          augment=None
                          )
speech_spectrogram = transform(speech)

padding = dataset.max_len - speech_spectrogram.shape[-1]
if padding >= 0:
    speech_spectrogram = F.pad(input=speech_spectrogram, pad=(0, padding, 0, 0))
else:
    speech_spectrogram = speech_spectrogram[:, :, :dataset.max_len]

print(wakeword_model(speech_spectrogram.unsqueeze(0)))