import os

import torch

import sys
sys.path.append("../deps/toucan")
from toucanTTSInferface import ToucanTTSInterface

def read_texts(model_id, sentence, filename, device="cpu", language="en", speaker_reference=None, faster_vocoder=False):
    tts = ToucanTTSInterface(device=device, tts_model_path=model_id, faster_vocoder=faster_vocoder)
    tts.set_language(language)
    if speaker_reference is not None:
        tts.set_utterance_embedding(speaker_reference)
    if type(sentence) == str:
        sentence = [sentence]
    tts.read_to_file(text_list=sentence, file_location=filename)
    del tts


def the_raven(version, model_id="Meta", exec_device="cpu", speed_over_quality=True):
    os.makedirs("audios", exist_ok=True)

    read_texts(model_id=model_id,
               sentence="Welcome to Willow. Willow is a high-performance and flexible engine for voice interactivity with various integrations, including home assistant and devices.",
               filename=f"audios/output.wav",
               device=exec_device,
               language="en",
               speaker_reference=None,
               faster_vocoder=speed_over_quality)


if __name__ == '__main__':
    exec_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"running on {exec_device}")

    the_raven(version="MetaBaseline", model_id="Meta", exec_device=exec_device, speed_over_quality=exec_device != "cuda")
