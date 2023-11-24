
from dataset.synthesiser_utils import TextToSpeech


def main():
    tts_generator = TextToSpeech()
    tts_generator.generate_speeches("Wake up", 1000)

if __name__ == "__main__":
    main()
