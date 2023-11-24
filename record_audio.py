# import sounddevice as sd
# from scipy.io.wavfile import write

# # Set the parameters
# fs = 16000  # Sample rate 
# seconds = 5  # Duration of recording

# # Record audio
# myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
# print("SAY SOMETHING")
# sd.wait()  # Wait until recording is finished

# # Save as WAV file 
# write('output.wav', fs, myrecording)

# Record audio for 1 hour

import sounddevice as sd
from scipy.io.wavfile import write
import os

# Set the parameters
fs = 16000  # Sample rate 
chunk_length_in_seconds = 1  # Length of chunks
total_duration_in_seconds = 3600  # Total duration (1 hour)

# Create directory to store the chunks if it doesn't exist
os.makedirs('background_sounds', exist_ok=True)

# Calculate number of chunks
num_chunks = total_duration_in_seconds // chunk_length_in_seconds

for i in range(881, num_chunks):
    # Record audio
    print(f"Recording chunk {i}...")
    myrecording = sd.rec(int(chunk_length_in_seconds * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished

    # Save as WAV file 
    filename = f'background_sounds/background_{i}.wav'
    write(filename, fs, myrecording)

print("Recording finished")