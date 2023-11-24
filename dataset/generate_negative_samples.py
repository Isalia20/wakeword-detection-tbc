
from dataset.synthesiser_utils import TextToSpeech
from datasets import load_dataset

def download_random_text_corpus():
    dataset = load_dataset("glue", "mnli_mismatched")
    val_dataset = dataset["validation"]
    test_dataset = dataset["test"]
    all_text_val = [i["premise"] for i in val_dataset if  "wake up" not in i["premise"].lower()]
    all_text_test = [i["premise"] for i in test_dataset if "wake up" not in i["premise"].lower()]
    all_text_val.extend(all_text_test)
    return all_text_val

def main():
    tts_generator = TextToSpeech()
    all_text = download_random_text_corpus()
    tts_generator.generate_negative_speeches(all_text, 100) # Generate 4 negative samples per voice

if __name__ == "__main__":
    main()
