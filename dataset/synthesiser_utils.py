from typing import List
import torch
from transformers import pipeline
from datasets import load_dataset
import soundfile as sf
import random
from multiprocessing import Pool

def save_wav_file(speech, id, sample_type: str = "positive"):
    if sample_type == "positive":
        sf.write("wav_files/positive_samples_raw/" + str(id) + ".wav", speech["audio"], samplerate=speech["sampling_rate"])
    elif sample_type == "negative":
        sf.write("wav_files/negative_samples_raw/" + str(id) + ".wav", speech["audio"], samplerate=speech["sampling_rate"])
    else:
        raise ValueError(f"Sample type expected to be either 'positive' or 'negative' but got {sample_type}")


class TextToSpeech:
    def __init__(self):
        self.synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")
        self.embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")

    def generate_speech(self, text_to_generate: str, speaker_id: int):
        speaker_id = speaker_id % self.embeddings_dataset.num_rows
        speaker_embedding = torch.tensor(self.embeddings_dataset[speaker_id]["xvector"]).unsqueeze(0)
        speech = self.synthesiser(text_to_generate, forward_params={"speaker_embeddings": speaker_embedding})
        return speech

    def generate_speeches(self, text_to_generate: str, n_files_to_generate: int):
        with Pool(processes=4) as pool:  # Adjust this number to match your CPU cores
            pool.starmap(self.generate_speech_and_save, [(text_to_generate, i) for i in range(n_files_to_generate)])

    def generate_speech_and_save(self, text_to_generate: str, i: int):
        speech = self.generate_speech(text_to_generate, i)
        save_wav_file(speech, i)

    def generate_negative_speeches(self, text_corpus: List[str], n_files_to_generate_per_voice):
        with Pool(processes=4) as pool:  # Adjust this number to match your CPU cores
            pool.starmap(self.generate_negative_speech_and_save, [(text_corpus, i, j) for i in range(n_files_to_generate_per_voice) for j in range(self.embeddings_dataset.num_rows)])

    def generate_negative_speech_and_save(self, text_corpus: List[str], i: int, j: int):
        print(f"running generate negative speech and save for {i} and {j}")
        text_to_generate = text_corpus[random.randint(0, len(text_corpus))]
        speech = self.generate_speech(text_to_generate, i)
        save_wav_file(speech, str(j) + "_" + str(i), "negative")
