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

if torch.cuda.is_available():
    threads = int(os.environ.get("XTTS_NUM_THREADS", "2"))
    use_deepspeed = True
    device = "cuda"
else:
    threads = int(os.environ.get("XTTS_NUM_THREADS", os.cpu_count()))
    use_deepspeed = False
    device = "cpu"

print(f"Using {threads} threads for device {device}")
torch.set_num_threads(threads)

model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
print("Downloading XTTS Model:", model_name, flush=True)
ModelManager().download_model(model_name)
model_path = os.path.join(get_user_data_dir("tts"), model_name.replace("/", "--"))
print("XTTS Model downloaded", flush=True)

print("Loading XTTS", flush=True)
config = XttsConfig()
config.load_json(os.path.join(model_path, "config.json"))
model = Xtts.init_from_config(config)
model.load_checkpoint(
    config, checkpoint_dir=model_path, eval=True, use_deepspeed=use_deepspeed
)
model.to(device)
print("XTTS Loaded.", flush=True)

print("Running XTTS Server ...", flush=True)

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

    if decoder not in ["ne_hifigan", "hifigan"]:
        decoder = "ne_hifigan"

    stream_chunk_size = int(parsed_input.stream_chunk_size)
    add_wav_header = parsed_input.add_wav_header

    chunks = model.inference_stream(
        text,
        language,
        gpt_cond_latent,
        speaker_embedding,
        decoder=decoder,
        stream_chunk_size=stream_chunk_size,
    )
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
# Based on 23fd10a
from fastapi import Depends
from fastapi.responses import JSONResponse
from pydantic import Field
import json

# import re
import copy


def load_speaker(speaker):
    with open(f"/xtts/{speaker}.json") as f:
        default_speaker = json.load(f)
    gpt_cond_latent = default_speaker.get("gpt_cond_latent")
    default_speaker = default_speaker.get("speaker_embedding")

    speaker_embedding = torch.tensor(default_speaker).unsqueeze(0).unsqueeze(-1)
    gpt_cond_latent = torch.tensor(gpt_cond_latent).reshape((-1, 1024)).unsqueeze(0)

    return speaker_embedding, gpt_cond_latent


# Defaults to load on startup
default_speaker_embedding, default_gpt_cond_latent = load_speaker("default")


def predict_streaming_generator_get(**kwargs):

    chunks = model.inference_stream(**kwargs)
    for i, chunk in enumerate(chunks):
        chunk = postprocess(chunk)
        # We always add WAV header
        if i == 0:
            yield encode_audio_common(b"", encode_base64=False)
            yield chunk.tobytes()
        else:
            yield chunk.tobytes()


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


# To-do: inherit StreamingInputs
# temperature and repetition_penalty come from their HF Space
# Expose all params from https://raw.githubusercontent.com/coqui-ai/TTS/dev/TTS/tts/models/xtts.py
# Can be passed as URL args to GET /api/tts
class WillowStreamingInputs(BaseModel):
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
        "zh",
        "ja",
        "hu",
        "ko",
    ] = "en"
    stream_chunk_size: int = 20
    overlap_wav_len: int = 1024
    temperature: float = 0.1
    length_penalty: float = 1.0
    repetition_penalty: float = 7.0
    top_k: int = 50
    top_p: float = Field(0.8, ge=0.0, le=1.0)
    do_sample: bool = True
    speed: float = 1.0
    enable_text_splitting: bool = True
    # Ours
    decoder: Literal["ne_hifigan", "hifigan"] = "ne_hifigan"
    speaker: str = "default"


@app.get("/api/tts")
def predict_streaming_endpoint_get(stream: WillowStreamingInputs = Depends()):
    # From their HF Space - no longer needed as of 23fd10a
    # text = re.sub("([^\x00-\x7F]|\w)(\.|\。|\?)",r"\1 \2\2", stream.text)
    text = stream.text

    # We load speaker from existing json on disk
    speaker_embedding, gpt_cond_latent = (
        default_speaker_embedding,
        default_gpt_cond_latent,
    )
    if stream.speaker != "default":
        try:
            speaker_embedding, gpt_cond_latent = load_speaker(stream.speaker)
        except:
            print(
                f"Could not load requested speaker '{stream.speaker}' - using default"
            )

    generator_args = {
        "text": text,
        "language": stream.language,
        "gpt_cond_latent": gpt_cond_latent,
        "speaker_embedding": speaker_embedding,
        "stream_chunk_size": stream.stream_chunk_size,
        "overlap_wav_len": stream.overlap_wav_len,
        "temperature": stream.temperature,
        "length_penalty": stream.length_penalty,
        "repetition_penalty": stream.repetition_penalty,
        "top_k": stream.top_k,
        "top_p": stream.top_p,
        "do_sample": stream.do_sample,
        "speed": stream.speed,
        "enable_text_splitting": stream.enable_text_splitting,
        "decoder": stream.decoder,
        "speaker": stream.speaker,
    }

    # Log request details
    generator_args_str = copy.deepcopy(generator_args)
    del generator_args_str["speaker_embedding"]
    del generator_args_str["gpt_cond_latent"]
    print("Coqui XTTS request with args: " + str(generator_args_str))

    return StreamingResponse(
        predict_streaming_generator_get(**generator_args),
        media_type="audio/wav",
    )


# Speaker clone for Willow
@app.post("/api/tts")
def predict_speaker_post(audio_file: UploadFile, speaker):
    """Compute conditioning inputs from reference audio file."""
    temp_audio_name = next(tempfile._get_candidate_names())
    with open(temp_audio_name, "wb") as temp, torch.inference_mode():
        temp.write(io.BytesIO(audio_file.file.read()).getbuffer())
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
            temp_audio_name
        )

        # TODO: Clean this up
        gpt_cond_latent = gpt_cond_latent.cpu().squeeze()
        speaker_embedding = speaker_embedding.cpu().squeeze()
        gpt_cond_latent = gpt_cond_latent.squeeze().half().tolist()
        speaker_embedding = speaker_embedding.squeeze().half().tolist()
        os.remove(temp_audio_name)

    print(f"Got speaker name {speaker}")
    speaker_json = {
        "gpt_cond_latent": gpt_cond_latent,
        "speaker_embedding": speaker_embedding,
    }
    speaker_json = json.dumps(speaker_json, indent=2)
    with open(f"/xtts/{speaker}.json", "w") as f:
        f.write(str(speaker_json))

    return JSONResponse(content={"status": f"Added speaker '{speaker}'"})
