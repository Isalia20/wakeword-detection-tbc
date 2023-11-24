# import os
# import shutil
# import random

# # Define main directory and the ratio of train/test split
# main_dir = ""
# train_ratio = 0.8  # 80% of data will be used for training

# # Define subfolders
# folders = ["negative_samples_raw", "positive_samples_raw"]

# # Loop over both folders
# for folder in folders:
#     # Get the list of wav files
#     file_list = os.listdir(os.path.join(main_dir, "wav_files", folder))

#     # Shuffle the file list
#     random.shuffle(file_list)
    
#     # Split into train and test
#     train_files = file_list[:int(len(file_list) * train_ratio)]
#     test_files = file_list[int(len(file_list) * train_ratio):]

#     # Copy the files to the corresponding train/test folder
#     for file in train_files:
#         dst_folder = os.path.join(main_dir, "train", folder.replace("_raw", ""))
#         os.makedirs(dst_folder, exist_ok=True)  # Create directory if it doesn't exist
#         shutil.copy(
#             os.path.join(main_dir, "wav_files", folder, file), 
#             os.path.join(dst_folder, file)
#         )
    
#     for file in test_files:
#         dst_folder = os.path.join(main_dir, "test", folder.replace("_raw", ""))
#         os.makedirs(dst_folder, exist_ok=True)  # Create directory if it doesn't exist
#         shutil.copy(
#             os.path.join(main_dir, "wav_files", folder, file), 
#             os.path.join(dst_folder, file)
#         )
        

# background_voice = "train/negative_samples/background_1.wav"
# import torchaudio

# torchaudio.load(background_voice)[0].shape


# import librosa

# def convert_to_mono(filename):
#     # Load the audio file; this automatically converts stereo to mono
#     y, sr = librosa.load(filename, mono=True)
    
#     return y, sr


# import soundfile as sf

# background_voice = "train/negative_samples/background_2.wav"
# y, sr = convert_to_mono(background_voice)

# sf.write('background_2.wav', y, sr)



# import os
# import shutil
# import numpy as np

# # Define source and destination directories
# source_dir = 'background_sounds'
# train_dir = 'train/negative_samples'
# test_dir = 'test/negative_samples'

# # Make sure the destination directories exist
# os.makedirs(train_dir, exist_ok=True)
# os.makedirs(test_dir, exist_ok=True)

# # Get all file names in the source directory
# all_files = [f for f in os.listdir(source_dir) if f.endswith('.wav')]
# np.random.shuffle(all_files)

# # Calculate the number of files to put in the train set (80% of all files)
# train_count = int(0.8 * len(all_files))

# # Split files
# train_files = all_files[:train_count]
# test_files = all_files[train_count:]

# # Move files to respective directories
# for file in train_files:
#     shutil.move(os.path.join(source_dir, file), os.path.join(train_dir, file))
    
# for file in test_files:
#     shutil.move(os.path.join(source_dir, file), os.path.join(test_dir, file))


# y now contains the mono audio data
# sr is the sample rate of the audio file



import os
import shutil

# Define the source directories and target directory
source_dirs = ['train/negative_samples', 'test/negative_samples']
target_dir = 'archive_tts'

# First, ensure that the target directory exists
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# Iterate over all source directories
for source_dir in source_dirs:
    # Iterate over all files in the source directory
    for filename in os.listdir(source_dir):
        if 'background' not in filename:
            # Construct full file paths
            source = os.path.join(source_dir, filename)
            target = os.path.join(target_dir, filename)
            # Move the file
            shutil.move(source, target)
            