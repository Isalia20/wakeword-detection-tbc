from archive.utils import graph_spectrogram, match_target_amplitude
import matplotlib.pyplot as plt
from pydub import AudioSegment
import numpy as np
import torch
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_shape):
        """
        Module for creating the model's graph.

        Argument:
        input_shape -- shape of the model's input data (using PyTorch conventions)

        """
        super(Model, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=input_shape[1], out_channels=196, kernel_size=15, stride=4).float()
        self.bn1 = nn.BatchNorm1d(1375) #TODO stop hardcoding this
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)

        self.gru1 = nn.GRU(input_size=196, hidden_size=128, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.bn2 = nn.BatchNorm1d(1375)

        self.gru2 = nn.GRU(input_size=128, hidden_size=128, batch_first=True)
        self.dropout3 = nn.Dropout(0.2)
        self.bn3 = nn.BatchNorm1d(1375)
        self.dropout4 = nn.Dropout(0.2)

        self.dense = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = x.swapdims(1, 2)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x.swapdims(1, 2))
        x = self.dropout1(x)

        x, _ = self.gru1(x)
        x = self.dropout2(x)
        x = self.bn2(x)

        x, _ = self.gru2(x)
        x = self.dropout3(x)
        x = self.bn3(x)
        x = self.dropout4(x)

        x = self.dense(x)
        return x

Tx = 5511
n_freq = 101

model = Model((Tx, n_freq))
model.load_state_dict(torch.load("trained_model.pth"))

def detect_triggerword(filename):
    plt.subplot(2, 1, 1)
    
    # Correct the amplitude of the input file before prediction 
    audio_clip = AudioSegment.from_wav(filename)
    audio_clip = match_target_amplitude(audio_clip, -20.0)
    filename = "tmp.wav"

    x = graph_spectrogram(filename)
    # the spectrogram outputs (freqs, Tx) and we want (Tx, freqs) to input into the model
    x  = x.swapaxes(0,1)
    x = np.expand_dims(x, axis=0)
    x = torch.tensor(x).float()
    predictions = model(x)
    predictions = torch.sigmoid(predictions)
    
    plt.subplot(2, 1, 2)
    plt.plot(predictions[0,:,0])
    plt.ylabel('probability')
    plt.show()
    return predictions

with torch.inference_mode():
    predictions = detect_triggerword("raw_data/dev/1.wav")
