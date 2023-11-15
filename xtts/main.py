import base64
import io
import os
import tempfile
from typing import List, Literal
import wave

import numpy as np
import torch
from fastapi import (
    FastAPI,
    UploadFile,
    Body,
)
from pydantic import BaseModel
from fastapi.responses import StreamingResponse

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.generic_utils import get_user_data_dir
from TTS.utils.manage import ModelManager

torch.set_num_threads(int(os.environ.get("NUM_THREADS", "2")))
device = torch.device("cuda")

model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
print("Downloading XTTS Model:",model_name,flush=True)
ModelManager().download_model(model_name)
model_path = os.path.join(get_user_data_dir("tts"), model_name.replace("/", "--"))
print("XTTS Model downloaded",flush=True)

print("Loading XTTS",flush=True)
config = XttsConfig()
config.load_json(os.path.join(model_path, "config.json"))
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir=model_path, eval=True, use_deepspeed=True)
model.to(device)
print("XTTS Loaded.",flush=True)

print("Running XTTS Server ...",flush=True)

##### Run fastapi #####
app = FastAPI(
    title="XTTS Streaming server",
    description="""XTTS Streaming server""",
    version="0.0.1",
    docs_url="/",
)


@app.post("/clone_speaker")
def predict_speaker(wav_file: UploadFile):
    """Compute conditioning inputs from reference audio file."""
    temp_audio_name = next(tempfile._get_candidate_names())
    with open(temp_audio_name, "wb") as temp, torch.inference_mode():
        temp.write(io.BytesIO(wav_file.file.read()).getbuffer())
        gpt_cond_latent, _, speaker_embedding = model.get_conditioning_latents(
            temp_audio_name
        )
    return {
        "gpt_cond_latent": gpt_cond_latent.cpu().squeeze().half().tolist(),
        "speaker_embedding": speaker_embedding.cpu().squeeze().half().tolist(),
    }


def postprocess(wav):
    """Post process the output waveform"""
    if isinstance(wav, list):
        wav = torch.cat(wav, dim=0)
    wav = wav.clone().detach().cpu().numpy()
    wav = wav[None, : int(wav.shape[0])]
    wav = np.clip(wav, -1, 1)
    wav = (wav * 32767).astype(np.int16)
    return wav


def encode_audio_common(
    frame_input, encode_base64=True, sample_rate=24000, sample_width=2, channels=1
):
    """Return base64 encoded audio"""
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)

    wav_buf.seek(0)
    if encode_base64:
        b64_encoded = base64.b64encode(wav_buf.getbuffer()).decode("utf-8")
        return b64_encoded
    else:
        return wav_buf.read()


class StreamingInputs(BaseModel):
    speaker_embedding: List[float]
    gpt_cond_latent: List[List[float]]
    text: str
    language: Literal[
        "en",
        "de",
        "fr",
        "es",
        "it",
        "pl",
        "pt",
        "tr",
        "ru",
        "nl",
        "cs",
        "ar",
        "zh-cn",
        "ja",
    ]
    add_wav_header: bool = True
    stream_chunk_size: str = "20"
    decoder: str = "ne_hifigan"


def predict_streaming_generator(parsed_input: dict = Body(...)):
    speaker_embedding = (
        torch.tensor(parsed_input.speaker_embedding).unsqueeze(0).unsqueeze(-1)
    )
    gpt_cond_latent = (
        torch.tensor(parsed_input.gpt_cond_latent).reshape((-1, 1024)).unsqueeze(0)
    )
    text = parsed_input.text
    language = parsed_input.language
    decoder = parsed_input.decoder

    if decoder not in ["ne_hifigan","hifigan"]:
        decoder = "ne_hifigan"

    stream_chunk_size = int(parsed_input.stream_chunk_size)
    add_wav_header = parsed_input.add_wav_header


    chunks = model.inference_stream(text, language, gpt_cond_latent, speaker_embedding, decoder=decoder,stream_chunk_size=stream_chunk_size)
    for i, chunk in enumerate(chunks):
        chunk = postprocess(chunk)
        if i == 0 and add_wav_header:
            yield encode_audio_common(b"", encode_base64=False)
            yield chunk.tobytes()
        else:
            yield chunk.tobytes()


@app.post("/tts_stream")
def predict_streaming_endpoint(parsed_input: StreamingInputs):
    return StreamingResponse(
        predict_streaming_generator(parsed_input),
        media_type="audio/wav",
    )

### Begin Willow
# Based on 999456989833c80be735a8252b440977589615d5
from typing import Optional
import json
import re

supported_languages = config.languages

def load_speaker(speaker):
    with open(f"/xtts/{speaker}.json") as f:
        default_speaker = json.load(f)
    gpt_cond_latent = default_speaker.get("gpt_cond_latent")
    default_speaker = default_speaker.get("speaker_embedding")

    speaker_embedding = (
        torch.tensor(default_speaker).unsqueeze(0).unsqueeze(-1)
    )
    gpt_cond_latent = (
        torch.tensor(gpt_cond_latent).reshape((-1, 1024)).unsqueeze(0)
    )

    return speaker_embedding, gpt_cond_latent

# Defaults to load on startup
default_speaker_embedding, default_gpt_cond_latent = load_speaker("default")

def predict_streaming_generator_get(text, language, decoder, stream_chunk_size, add_wav_header, speaker, temperature, repetition_penalty):
    speaker_embedding, gpt_cond_latent = default_speaker_embedding, default_gpt_cond_latent
    if speaker != "default":
        try:
            speaker_embedding, gpt_cond_latent = load_speaker(speaker)
        except:
            print(f"Could not load requested speaker '{speaker}' - using default")

    if language not in supported_languages:
        print(f"XTTS does not support requested language '{language}' - setting en")
        language = "en"

    if decoder not in ["ne_hifigan","hifigan"]:
        decoder = "ne_hifigan"

    chunks = model.inference_stream(text, language, gpt_cond_latent, speaker_embedding, decoder=decoder, stream_chunk_size=stream_chunk_size, temperature=temperature, repetition_penalty=repetition_penalty)
    for i, chunk in enumerate(chunks):
        chunk = postprocess(chunk)
        if i == 0 and add_wav_header:
            yield encode_audio_common(b"", encode_base64=False)
            yield chunk.tobytes()
        else:
            yield chunk.tobytes()

print('Warming TTS...')
predict_streaming_generator_get("Warming model", "en", "ne_hifigan", 20, True, "default", 0.85, 7.0)

@app.get("/api/tts")
def predict_streaming_endpoint_get(text, language: Optional[str] = "en", decoder: Optional[str] = "ne_hifigan", stream_chunk_size: Optional[int] = 20, add_wav_header: Optional[bool] = True, speaker: Optional[str] = "default", temperature: Optional[float] = 0.85, repetition_penalty: Optional[float] = 7.0):
    print(f"Coqui XTTS request with {text} {language} {decoder} {stream_chunk_size} {speaker} {temperature} {repetition_penalty}")
    # From their HF Space
    text = re.sub("([^\x00-\x7F]|\w)(\.|\。|\?)",r"\1 \2\2", text)

    return StreamingResponse(
        predict_streaming_generator_get(text, language, decoder, stream_chunk_size, add_wav_header, speaker, temperature, repetition_penalty),
        media_type="audio/wav",
    )