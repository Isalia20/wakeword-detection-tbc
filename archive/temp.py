from archive.utils import graph_spectrogram, load_raw_audio, match_target_amplitude
import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np

x = graph_spectrogram("audio_examples/example_train.wav")
plt.show()


_, data = wavfile.read("audio_examples/example_train.wav")
print("Time steps in audio recording before spectrogram", data[:,0].shape)
print("Time steps in input after spectrogram", x.shape)

Tx = 5511 # The number of time steps input to the model from the spectrogram
n_freq = 101 # Number of frequencies input to the model at each time step of the spectrogram
Ty = 1375 # The number of time steps in the output of our model

# Load audio segments using pydub 
activates, negatives, backgrounds = load_raw_audio('./raw_data/')

print("background len should be 10,000, since it is a 10 sec clip\n" + str(len(backgrounds[0])),"\n")
print("activate[0] len may be around 1000, since an `activate` audio clip is usually around 1 second (but varies a lot) \n" + str(len(activates[0])),"\n")
print("activate[1] len: different `activate` clips can have different lengths\n" + str(len(activates[1])),"\n")

def get_random_time_segment(segment_ms):
    """
    Gets a random time segment of duration segment_ms in a 10,000 ms audio clip.
    
    Arguments:
    segment_ms -- the duration of the audio clip in ms ("ms" stands for "milliseconds")
    
    Returns:
    segment_time -- a tuple of (segment_start, segment_end) in ms
    """
    
    segment_start = np.random.randint(low=0, high=10000-segment_ms)   # Make sure segment doesn't run past the 10sec background 
    segment_end = segment_start + segment_ms - 1
    
    return (segment_start, segment_end)


def is_overlapping(segment_time, previous_segments):
    """
    Checks if the time of a segment overlaps with the times of existing segments.
    
    Arguments:
    segment_time -- a tuple of (segment_start, segment_end) for the new segment
    previous_segments -- a list of tuples of (segment_start, segment_end) for the existing segments
    
    Returns:
    True if the time segment overlaps with any of the existing segments, False otherwise
    """
    
    segment_start, segment_end = segment_time
    overlap = False
    
    for previous_start, previous_end in previous_segments:
        if segment_start <= previous_end and segment_end >= previous_start:
            overlap = True
            break

    return overlap


def insert_audio_clip(background, audio_clip, previous_segments):
    """
    Insert a new audio segment over the background noise at a random time step, ensuring that the 
    audio segment does not overlap with existing segments.
    
    Arguments:
    background -- a 10 second background audio recording.  
    audio_clip -- the audio clip to be inserted/overlaid. 
    previous_segments -- times where audio segments have already been placed
    
    Returns:
    new_background -- the updated background audio
    """
    
    # Get the duration of the audio clip in ms
    segment_ms = len(audio_clip)
    
    # Get a random time segment where we will insert the new audio clip
    segment_time = get_random_time_segment(segment_ms)
    
    # Check if this inserted segment overlaps with one of the segments. If so we randomly 
    # pick another time segment until they don't overlap. 
    retry = 5
    while is_overlapping(segment_time, previous_segments) and retry >= 0:
        segment_time = get_random_time_segment(segment_ms)
        retry = retry - 1

    # if last try is not overlaping, insert it to the background
    if not is_overlapping(segment_time, previous_segments):
        # Append the new segment_time to the list of previous_segments
        previous_segments.append(segment_time)
        # Superpose audio segment and background
        new_background = background.overlay(audio_clip, position = segment_time[0])
        print(segment_time)
    else:
        print("Timeouted")
        new_background = background
        segment_time = (10000, 10000)
    
    return new_background, segment_time

np.random.seed(5)
audio_clip, segment_time = insert_audio_clip(backgrounds[0], activates[0], [(3790, 4400)])
audio_clip.export("insert_test.wav", format="wav")
print("Segment Time: ", segment_time)


def insert_ones(y, segment_end_ms):
    """
    Update the label vector y. The labels of the 50 output steps strictly after the end of the segment 
    should be set to 1. By strictly we mean that the label of segment_end_y should be 0 while, the
    50 following labels should be ones.
    
    
    Arguments:
    y -- numpy array of shape (1, Ty), the labels of the training example
    segment_end_ms -- the end time of the segment in ms
    
    Returns:
    y -- updated labels
    """
    _, Ty = y.shape
    
    # duration of the background (in terms of spectrogram time-steps)
    segment_end_y = int(segment_end_ms * Ty / 10000.0)
    
    if segment_end_y < Ty:
        # Add 1 to the correct index in the background label (y)
        for i in range(segment_end_y+1, segment_end_y+51):
            if i < Ty:
                y[0, i] = 1
    return y

arr1 = insert_ones(np.zeros((1, Ty)), 9700)
plt.plot(insert_ones(arr1, 4251)[0,:])
plt.show()
print("sanity checks:", arr1[0][1333], arr1[0][634], arr1[0][635])


def create_training_example(background, activates, negatives, Ty):
    """
    Creates a training example with a given background, activates, and negatives.
    
    Arguments:
    background -- a 10 second background audio recording
    activates -- a list of audio segments of the word "activate"
    negatives -- a list of audio segments of random words that are not "activate"
    Ty -- The number of time steps in the output

    Returns:
    x -- the spectrogram of the training example
    y -- the label at each time step of the spectrogram
    """
    
    # Make background quieter
    background = background - 20

    # Initialize y (label vector) of zeros (≈ 1 line)
    y = np.zeros((1, Ty))

    # Initialize segment times as empty list (≈ 1 line)
    previous_segments = []
    
    # Select 0-4 random "activate" audio clips from the entire list of "activates" recordings
    number_of_activates = np.random.randint(0, 5)
    random_indices = np.random.randint(len(activates), size=number_of_activates)
    random_activates = [activates[i] for i in random_indices]
    
    # Loop over randomly selected "activate" clips and insert in background
    for random_activate in random_activates: # @KEEP
        # Insert the audio clip on the background
        background, segment_time = insert_audio_clip(background, random_activate, previous_segments)
        # Retrieve segment_start and segment_end from segment_time
        segment_start, segment_end = segment_time
        # Insert labels in "y" at segment_end
        y = insert_ones(y, segment_end)

    # Select 0-2 random negatives audio recordings from the entire list of "negatives" recordings
    number_of_negatives = np.random.randint(0, 3)
    random_indices = np.random.randint(len(negatives), size=number_of_negatives)
    random_negatives = [negatives[i] for i in random_indices]

    # Loop over randomly selected negative clips and insert in background
    for random_negative in random_negatives: # @KEEP
        # Insert the audio clip on the background 
        background, _ = insert_audio_clip(background, random_negative, previous_segments)
    
    # Standardize the volume of the audio clip 
    background = match_target_amplitude(background, -20.0)

    # Export new training example 
    # file_handle = background.export("train" + ".wav", format="wav")
    
    # Get and plot spectrogram of the new recording (background with superposition of positive and negatives)
    x = graph_spectrogram("train.wav")
    
    return x, y

# Set the random seed
np.random.seed(18)
x, y = create_training_example(backgrounds[0], activates, negatives, Ty)

np.random.seed(4543)
nsamples = 400
X = []
Y = []
for i in range(0, nsamples):
    if i%10 == 0:
        print(i)
    x, y = create_training_example(backgrounds[i % 2], activates, negatives, Ty)
    X.append(x.swapaxes(0,1))
    Y.append(y.swapaxes(0,1))
X = np.array(X)
Y = np.array(Y)

#------------------------------------------------------------------------------------------------------------------------------------------------
# MODEL DEFINITION 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


class Model(nn.Module):
    def __init__(self, input_shape):
        """
        PyTorch Module for creating the model's graph.

        Argument:
        input_shape -- shape of the model's input data (using PyTorch conventions)

        """
        super(Model, self).__init__()

        # Step 1: CONV layer
        self.conv1 = nn.Conv1d(in_channels=input_shape[1], out_channels=196, kernel_size=15, stride=4).float()
        self.bn1 = nn.BatchNorm1d(1375) #TODO stop hardcoding this
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)

        # Step 2: First GRU Layer
        self.gru1 = nn.GRU(input_size=196, hidden_size=128, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.bn2 = nn.BatchNorm1d(1375)

        # Step 3: Second GRU Layer
        self.gru2 = nn.GRU(input_size=128, hidden_size=128, batch_first=True)
        self.dropout3 = nn.Dropout(0.2)
        self.bn3 = nn.BatchNorm1d(1375)
        self.dropout4 = nn.Dropout(0.2)

        # Step 4: Time-distributed dense layer
        self.dense = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

        ### END CODE HERE ###

    def forward(self, x):
        # Step 1
        x = x.swapdims(1, 2)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x.swapdims(1, 2))
        x = self.dropout1(x)

        # Step 2
        x, _ = self.gru1(x)
        x = self.dropout2(x)
        x = self.bn2(x)

        # Step 3
        x, _ = self.gru2(x)
        x = self.dropout3(x)
        x = self.bn3(x)
        x = self.dropout4(x)

        # Step 4
        x = self.dense(x)
        return x
    
Tx = 5511
n_freq = 101

model = Model((Tx, n_freq))
X_torch = torch.tensor(X).float()

import torch.optim as optim
import torch.nn as nn

# Assuming your model is called "model"
optimizer = optim.Adam(model.parameters(), lr=1e-6, betas=(0.9, 0.999))
criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy


from torch.utils.data import TensorDataset, DataLoader

# Assuming X and Y are your data and labels and they are numpy arrays
# Convert them to PyTorch tensors and create a DataLoader
tensor_x = torch.Tensor(X)
tensor_y = torch.Tensor(Y)

# Create the dataset and dataloader
dataset = TensorDataset(tensor_x, tensor_y)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Define the training function
def train(model, dataloader, optimizer, criterion, epochs):
    model.train()
    
    for epoch in range(epochs):
        for inputs, labels in dataloader:
            # Send data to the same device as the model
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Reset the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, labels)
            print(loss)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
        
# Assuming 'device' is the device you're using (either a CPU or a GPU)
device = "cpu"
model = model.to(device)

# Now run the training
train(model, dataloader, optimizer, criterion, epochs=20)

torch.save(model.cpu().state_dict(), "trained_model.pth")


from pydub import AudioSegment

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


# Preprocess the audio to the correct format
def preprocess_audio(filename):
    # Trim or pad audio segment to 10000ms
    padding = AudioSegment.silent(duration=10000)
    segment = AudioSegment.from_wav(filename)[:10000]
    segment = padding.overlay(segment)
    # Set frame rate to 44100
    segment = segment.set_frame_rate(44100)
    # Export as wav
    segment.export(filename, format='wav')


with torch.inference_mode():
    prediction = detect_triggerword("raw_data/dev/1.wav")


print("X SHAPE IS ", X.shape)
print("Y SHAPE IS ", Y.shape)

