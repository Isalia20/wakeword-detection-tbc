from model import WakeWordDetector
import torch
import torchaudio
from torch.nn import functional as F
import sounddevice as sd
import numpy as np
from dataset.dataset import WakeWordDataset

def initialize_model():
    wakeword_model = WakeWordDetector()
    weights = torch.load("output/checkpoints/wakeword_detector_v12(+background)/step=2622-val_loss=0.091.ckpt")["state_dict"]
    wakeword_model.load_state_dict(weights)
    wakeword_model = wakeword_model.eval()
    return wakeword_model

# Define sample rate and duration
sample_rate = 16000  # Modify if using a different sample rate
duration = 1.5  # Duration of recording in seconds
recording_length = int(sample_rate * duration)  # Length of recording in samples

# Buffer to store audio
buffer = np.zeros((recording_length,))
transform = torchaudio.transforms.MelSpectrogram()
model = initialize_model()
dataset = WakeWordDataset(positive_dir="wav_files/positive_samples_raw", 
                          negative_dir="wav_files/negative_samples_raw", 
                          transform=transform, 
                          augment=None
                          )

def audio_callback(indata, frames, time, status):
    global buffer, model, transform

    # Shift the buffer to add new data
    buffer[:-frames] = buffer[frames:]
    buffer[-frames:] = indata[:, 0]

    # Convert buffer to PyTorch tensor
    waveform = torch.from_numpy(buffer).float()
    spectrogram = transform(waveform)
    padding = dataset.max_len - spectrogram.shape[-1]
    spectrogram = F.pad(input=spectrogram, pad=(0, padding, 0, 0))

    # Predict with your model
    with torch.no_grad():
        prediction = model(spectrogram.unsqueeze(0).unsqueeze(0))

    # print("Prediction:", prediction)
    if prediction.item() > 0.5:
        print(1)


with sd.InputStream(callback=audio_callback, channels=1, samplerate=sample_rate, blocksize=int(sample_rate * 0.1)):
    while True:
        sd.sleep(100)
