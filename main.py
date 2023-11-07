# Profiling
# import cProfile as profile
# import pstats
# Logging
import os
import logging
# FastAPI preprocessor
from fastapi import FastAPI, UploadFile, Request, Response, status
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import PlainTextResponse
from typing import Optional, Tuple
from pydantic import BaseModel
import types
import random
import base64
import binascii
import json
import datetime
import numpy as np
import warnings
import io
import re
import math
import functools
from typing import NamedTuple

# WebRTC
import asyncio

from aiortc import RTCPeerConnection, RTCSessionDescription, RTCRtpReceiver
from aiortc.rtp import RtcpByePacket
from wis.media import MediaRecorderLite


# Whisper
import ctranslate2
import librosa
import transformers
import wis.languages

import torch
import torchaudio

# SV
from torchaudio.sox_effects import apply_effects_tensor
import operator

# Import audio stuff adapted from ref Whisper implementation
from wis.audio import log_mel_spectrogram, pad_or_trim, chunk_iter, find_longest_common_sequence

# Willow
import wave
import av

logger = logging.getLogger("infer")
gunicorn_logger = logging.getLogger('gunicorn.error')
logger.handlers = gunicorn_logger.handlers
logger.setLevel(gunicorn_logger.level)

try:
    from custom_settings import get_api_settings
    settings = get_api_settings()
    logger.info(f"{settings.name} is starting with custom settings... Please wait.")
except Exception:
    from settings import get_api_settings
    settings = get_api_settings()
    logger.info(f"{settings.name} is starting... Please wait.")

warnings.simplefilter(action='ignore')

pcs = set()

# Whisper supported languages
whisper_languages = wis.languages.LANGUAGES

# Use soundfile so we can support WAV, FLAC, etc
torchaudio.set_audio_backend('soundfile')


# you can chunkit
def chunkit(lst, num):
    """Yield successive num-sized chunks from list lst."""
    for i in range(0, len(lst), num):
        yield lst[i:i + num]


# Function to create a wav file from stream data
def write_stream_wav(data, rate, bits, ch):
    file = io.BytesIO()
    wavfile = wave.open(file, 'wb')
    wavfile.setparams((ch, int(bits/8), rate, 0, 'NONE', 'NONE'))
    wavfile.writeframesraw(bytearray(data))
    wavfile.close()
    file.seek(0)
    return file


def audio_to_wav(file, rate: 16000):
    """Arbitrary media files to wav"""
    wav = io.BytesIO()
    with av.open(file) as in_container:
        in_stream = in_container.streams.audio[0]
        with av.open(wav, 'w', 'wav') as out_container:
            out_stream = out_container.add_stream(
                'pcm_s16le',
                rate=rate,
                layout='mono'
            )
            for frame in in_container.decode(in_stream):
                for packet in out_stream.encode(frame):
                    out_container.mux(packet)

    wav.seek(0)
    return wav


# Monkey patch aiortc
# sender.replaceTrack(null) sends a RtcpByePacket which we want to ignore
# in this case and keep connection open. XXX: Are there other cases we want to close?
old_handle_rtcp_packet = RTCRtpReceiver._handle_rtcp_packet


async def new_handle_rtcp_packet(self, packet):
    if isinstance(packet, RtcpByePacket):
        return
    return old_handle_rtcp_packet(self, packet)
RTCRtpReceiver._handle_rtcp_packet = new_handle_rtcp_packet

if settings.aiortc_debug:
    logger.debug('AIORTC: Debugging active')
    logging.basicConfig(level=logging.DEBUG)  # very useful debugging aiortc issues

local_ports = list(range(10000, 10000+50))  # Allowed ephemeral port range


def patch_loop_datagram():
    loop = asyncio.get_event_loop()
    if getattr(loop, '_patch_done', False):
        return

    # Monkey patch aiortc to control ephemeral ports
    old_create_datagram_endpoint = loop.create_datagram_endpoint

    async def create_datagram_endpoint(self, protocol_factory, local_addr: Tuple[str, int] = None, **kwargs):
        # if port is specified just use it
        if local_addr and local_addr[1]:
            return await old_create_datagram_endpoint(protocol_factory, local_addr=local_addr, **kwargs)
        if local_addr is None:
            return await old_create_datagram_endpoint(protocol_factory, local_addr=None, **kwargs)
        # if port is not specified make it use our range
        ports = list(local_ports)
        random.shuffle(ports)
        for port in ports:
            try:
                ret = await old_create_datagram_endpoint(
                    protocol_factory, local_addr=(local_addr[0], port), **kwargs
                )
                logger.debug(f'create_datagram_endpoint chose port {port}')
                return ret
            except OSError as exc:
                if port == ports[-1]:
                    # this was the last port, give up
                    raise exc
        raise ValueError("local_ports must not be empty")
    loop.create_datagram_endpoint = types.MethodType(create_datagram_endpoint, loop)
    loop._patch_done = True


patch_loop_datagram()  # not really needed here...

# XXX: rm these globals and use settings directly

# default beam_size - 5 is lib default, 1 for greedy
beam_size = settings.beam_size

# default beam size for longer transcriptions
long_beam_size = settings.long_beam_size
# Audio duration in ms to activate "long" mode
long_beam_size_threshold = settings.long_beam_size_threshold

# models to preload
preload_all_models = settings.preload_all_models
preload_whisper_model_tiny = settings.preload_whisper_model_tiny
preload_whisper_model_base = settings.preload_whisper_model_base
preload_whisper_model_small = settings.preload_whisper_model_small
preload_whisper_model_medium = settings.preload_whisper_model_medium
preload_whisper_model_large = settings.preload_whisper_model_large
preload_whisper_model_large_v3 = settings.preload_whisper_model_large_v3

# bit hacky but if we need to preload all models, we need to preload each model independently too
if preload_all_models:
    preload_whisper_model_tiny = True
    preload_whisper_model_base = True
    preload_whisper_model_small = True
    preload_whisper_model_medium = True
    preload_whisper_model_large = True
    preload_whisper_model_large_v3 = True

# model threads
model_threads = settings.model_threads

# Support for chunking
support_chunking = settings.support_chunking

# Support for SV
support_sv = settings.support_sv

# Default SV threshold
sv_threshold = settings.sv_threshold

whisper_model_default = settings.whisper_model_default

# Default language
language = settings.language

# Default detect language?
detect_language = settings.detect_language

concurrent_gpu_chunks = settings.concurrent_gpu_chunks

# Try CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    cuda_num_devices = torch.cuda.device_count()
    logger.info(f'CUDA: Detected {cuda_num_devices} device(s)')

    # Assume Volta or higher
    compute_type = "int8_float16"

    for cuda_dev_num in range(0, cuda_num_devices, 1):
        # Print CUDA device name
        cuda_device_name = torch.cuda.get_device_name(cuda_dev_num)
        logger.info(f'CUDA: Device {cuda_dev_num} name: {cuda_device_name}')

        # Get CUDA device capability
        cuda_device_capability = torch.cuda.get_device_capability(cuda_dev_num)
        cuda_device_capability = functools.reduce(lambda sub, ele: sub * 10 + ele, cuda_device_capability)
        logger.info(f'CUDA: Device {cuda_dev_num} capability: {cuda_device_capability}')

        # Get CUDA memory - returns in bytes
        cuda_total_memory = torch.cuda.mem_get_info(cuda_dev_num)[1]
        cuda_free_memory = torch.cuda.mem_get_info(cuda_dev_num)[0]
        logger.info(f'CUDA: Device {cuda_dev_num} total memory: {cuda_total_memory} bytes')
        logger.info(f'CUDA: Device {cuda_dev_num} free memory: {cuda_free_memory} bytes')

        # Disable chunking if card has too little VRAM
        # This can still encounter out of memory errors depending on audio length
        # XXX: we don't really need this anymore, but leaving it in to not affect short audio on small cards
        if cuda_free_memory <= settings.chunking_memory_threshold:
            logger.warning(f'CUDA: Device {cuda_dev_num} has low memory, disabling chunking support')
            support_chunking = False

        if cuda_free_memory <= settings.sv_memory_threshold:
            logger.warning(f'CUDA: Device {cuda_dev_num} has low memory, disabling SV support')
            support_sv = False

        # Override compute_type if at least one non-Volta card and override+warn on pre-pascal devices
        if cuda_device_capability in range(1, 59):
            logger.warning(f'CUDA: Device {cuda_dev_num} is pre-Pascal, forcing float32')
            logger.warning('CUDA: SUPPORT FOR PRE-PASCAL DEVICES IS UNSUPPORTED AND WILL BE REMOVED IN THE FUTURE')
            compute_type = "float32"
        elif cuda_device_capability < 70:
            logger.warning(f'CUDA: Device {cuda_dev_num} is pre-Volta, forcing int8')
            compute_type = "int8"

    # Set ctranslate device index based on number of supported devices
    device_index = [*range(0, cuda_num_devices, 1)]
else:
    num_cpu_cores = os.cpu_count()
    compute_type = "int8"
    # Tested to generally perform best on CPU
    intra_threads = num_cpu_cores // 2
    model_threads = num_cpu_cores // 2
    logger.info(f'CUDA: Not found - using CPU with {num_cpu_cores} cores')

# These models refuse to cooperate otherwise
# XXX move these to LazyModels too?
if support_sv:
    logger.info("Loading SV models...")
    sv_feature_extractor = transformers.Wav2Vec2FeatureExtractor.from_pretrained(
            "./models/microsoft-wavlm-base-plus-sv")
    sv_model = transformers.WavLMForXVector.from_pretrained("./models/microsoft-wavlm-base-plus-sv").to(device=device)
else:
    sv_feature_extractor = None
    sv_model = None


class LazyModels:
    def __init__(self):
        self._whisper_processor = None
        self._whisper_processor_v3 = None
        self._whisper_model_tiny = None
        self._whisper_model_base = None
        self._whisper_model_small = None
        self._whisper_model_medium = None
        self._whisper_model_large = None
        self._whisper_model_large_v3 = None

    @property
    def whisper_processor(self):
        if self._whisper_processor is None:
            self._whisper_processor = transformers.WhisperProcessor.from_pretrained("./models/tovera-wis-whisper-base")
        return self._whisper_processor

    @property
    def whisper_processor_v3(self):
        if self._whisper_processor_v3 is None:
            self._whisper_processor_v3 = transformers.WhisperProcessor.from_pretrained("./models/tovera-wis-whisper-large-v3")
        return self._whisper_processor_v3

    @property
    def whisper_model_tiny(self):
        if self._whisper_model_tiny is None:
            logger.info("Loading whisper model: tiny")
            if device == "cuda":
                self._whisper_model_tiny = ctranslate2.models.Whisper(
                    'models/tovera-wis-whisper-tiny',
                    device=device,
                    compute_type=compute_type,
                    inter_threads=model_threads,
                    device_index=device_index,
                )
            else:
                self._whisper_model_tiny = ctranslate2.models.Whisper(
                    'models/tovera-wis-whisper-tiny',
                    device=device,
                    compute_type=compute_type,
                    inter_threads=model_threads,
                    intra_threads=intra_threads,
                )
        return self._whisper_model_tiny

    @property
    def whisper_model_base(self):
        if self._whisper_model_base is None:
            logger.info("Loading whisper model: base")
            if device == "cuda":
                self._whisper_model_base = ctranslate2.models.Whisper(
                    'models/tovera-wis-whisper-base',
                    device=device,
                    compute_type=compute_type,
                    inter_threads=model_threads,
                    device_index=device_index,
                )
            else:
                self._whisper_model_base = ctranslate2.models.Whisper(
                    'models/tovera-wis-whisper-base',
                    device=device,
                    compute_type=compute_type,
                    inter_threads=model_threads,
                    intra_threads=intra_threads,
                )
        return self._whisper_model_base

    @property
    def whisper_model_small(self):
        if self._whisper_model_small is None:
            logger.info("Loading whisper model: small")
            if device == "cuda":
                self._whisper_model_small = ctranslate2.models.Whisper(
                    'models/tovera-wis-whisper-small',
                    device=device,
                    compute_type=compute_type,
                    inter_threads=model_threads,
                    device_index=device_index,
                )
            else:
                self._whisper_model_small = ctranslate2.models.Whisper(
                    'models/tovera-wis-whisper-small',
                    device=device,
                    compute_type=compute_type,
                    inter_threads=model_threads,
                    intra_threads=intra_threads,
                )
        return self._whisper_model_small

    @property
    def whisper_model_medium(self):
        if self._whisper_model_medium is None:
            logger.info("Loading whisper model: medium")
            if device == "cuda":
                self._whisper_model_medium = ctranslate2.models.Whisper(
                    'models/tovera-wis-whisper-medium',
                    device=device,
                    compute_type=compute_type,
                    inter_threads=model_threads,
                    device_index=device_index,
                )
            else:
                self._whisper_model_medium = ctranslate2.models.Whisper(
                    'models/tovera-wis-whisper-medium',
                    device=device,
                    compute_type=compute_type,
                    inter_threads=model_threads,
                    intra_threads=intra_threads,
                )
        return self._whisper_model_medium

    @property
    def whisper_model_large(self):
        if self._whisper_model_large is None:
            logger.info("Loading whisper model: large")
            if device == "cuda":
                self._whisper_model_large = ctranslate2.models.Whisper(
                    'models/tovera-wis-whisper-large-v2',
                    device=device,
                    compute_type=compute_type,
                    inter_threads=model_threads,
                    device_index=device_index,
                )
            else:
                self._whisper_model_large = ctranslate2.models.Whisper(
                    'models/tovera-wis-whisper-large-v2',
                    device=device,
                    compute_type=compute_type,
                    inter_threads=model_threads,
                    intra_threads=intra_threads,
                )
        return self._whisper_model_large

    @property
    def whisper_model_large_v3(self):
        if self._whisper_model_large_v3 is None:
            logger.info("Loading whisper model: large-v3")
            if device == "cuda":
                self._whisper_model_large_v3 = ctranslate2.models.Whisper(
                    'models/tovera-wis-whisper-large-v3',
                    device=device,
                    compute_type=compute_type,
                    inter_threads=model_threads,
                    device_index=device_index,
                )
            else:
                self._whisper_model_large_v3 = ctranslate2.models.Whisper(
                    'models/tovera-wis-whisper-large-v3',
                    device=device,
                    compute_type=compute_type,
                    inter_threads=model_threads,
                    intra_threads=intra_threads,
                )
        return self._whisper_model_large_v3

# this is a singleton instance that is never re-assigned
models: LazyModels = LazyModels()


def load_models() -> LazyModels:
    # Turn up log_level for ctranslate2
    # ctranslate2.set_log_level(logger.DEBUG)
    supported_compute_types = str(ctranslate2.get_supported_compute_types(device))
    logger.info(f'CTRANSLATE: Supported compute types for device {device} are {supported_compute_types}'
                f'- using configured {compute_type}')

    # preload models if requested
    if preload_whisper_model_tiny:
        models.whisper_processor
        models.whisper_model_tiny
    if preload_whisper_model_base:
        models.whisper_processor
        models.whisper_model_base
    if preload_whisper_model_small:
        models.whisper_processor
        models.whisper_model_small
    if preload_whisper_model_medium:
        models.whisper_processor
        models.whisper_model_medium
    if preload_whisper_model_large:
        models.whisper_processor
        models.whisper_model_large
    if preload_whisper_model_large_v3:
        models.whisper_processor_v3
        models.whisper_model_large_v3

    return models


def warm_models():
    if device == "cuda":
        logger.info("Warming models...")
        for x in range(3):
            if preload_whisper_model_tiny and models.whisper_model_tiny is not None:
                do_whisper("client/3sec.flac", "tiny", beam_size, "transcribe", False, "en")
            if preload_whisper_model_base and models.whisper_model_base is not None:
                do_whisper("client/3sec.flac", "base", beam_size, "transcribe", False, "en")
            if preload_whisper_model_small and models.whisper_model_small is not None:
                do_whisper("client/3sec.flac", "small", beam_size, "transcribe", False, "en")
            if preload_whisper_model_medium and models.whisper_model_medium is not None:
                do_whisper("client/3sec.flac", "medium", beam_size, "transcribe", False, "en")
            if preload_whisper_model_large and models.whisper_model_large is not None:
                do_whisper("client/3sec.flac", "large", beam_size, "transcribe", False, "en")
            if preload_whisper_model_large_v3 and models.whisper_model_large_v3 is not None:
                do_whisper("client/3sec.flac", "large-v3", beam_size, "transcribe", False, "en")
            if sv_model is not None:
                # XXX sv_model is loaded at top level instead of in LazyModels
                # XXX so it is always preloaded & warmed if supported
                do_sv("client/3sec.flac")

    else:
        logger.info("Skipping warm_models for CPU")
        return


def do_translate(whisper_model, whisper_processor, features, total_chunk_count, language, beam_size):
    # Set task in token format for processor
    task = 'translate'
    logger.debug(f'WHISPER: Doing translation with {language}, beam size {beam_size}, chunk count {total_chunk_count}')
    processor_task = f'<|{task}|>'

    # Describe the task in the prompt.
    # See the prompt format in https://github.com/openai/whisper.
    prompt = whisper_processor.tokenizer.convert_tokens_to_ids(
        [
            "<|startoftranscript|>",
            language,
            processor_task,
            "<|notimestamps|>",  # Remove this token to generate timestamps.
        ]
    )

    # Run generation for the 30-second window.
    time_start = datetime.datetime.now()
    results = whisper_model.generate(features, [prompt]*total_chunk_count, beam_size=beam_size)
    time_end = datetime.datetime.now()
    infer_time = time_end - time_start
    infer_time_milliseconds = infer_time.total_seconds() * 1000
    logger.debug('WHISPER: Translate inference took ' + str(infer_time_milliseconds) + ' ms')
    results = whisper_processor.decode(results[0].sequences_ids[0])
    logger.debug(results)

    return results


def check_language(language):
    return language in whisper_languages


def do_whisper(audio_file, model: str, beam_size: int = beam_size, task: str = "transcribe",
               detect_language: bool = False, force_language: str = None, translate: bool = False):
    # Set defaults
    n_mels = 80
    whisper_processor = models.whisper_processor
    # Point to model object depending on passed model string
    if model == "large":
        whisper_model = models.whisper_model_large
    elif model == "large-v3":
        n_mels = 128
        whisper_model = models.whisper_model_large_v3
        whisper_processor = models.whisper_processor_v3
    elif model == "medium":
        whisper_model = models.whisper_model_medium
    elif model == "small":
        whisper_model = models.whisper_model_small
    elif model == "base":
        whisper_model = models.whisper_model_base
    elif model == "tiny":
        whisper_model = models.whisper_model_tiny

    processor_task = f'<|{task}|>'
    first_time_start = datetime.datetime.now()

    # Whisper STEP 1 - load audio and extract features
    audio, audio_sr = librosa.load(audio_file, sr=16000, mono=True)
    audio_duration = librosa.get_duration(y=audio, sr=audio_sr) * 1000
    audio_duration = int(audio_duration)
    if audio_duration >= long_beam_size_threshold:
        logger.debug(f'WHISPER: Audio duration is {audio_duration} ms - activating long mode')
        beam_size = long_beam_size
    use_chunking = False
    if audio_duration > 30*1000:
        if support_chunking:
            logger.debug('WHISPER: Audio duration is > 30s - activating chunking')
            use_chunking = True
        else:
            logger.warning('WHISPER: Audio duration is > 30s but chunking is not available. Will truncate!')

    time_end = datetime.datetime.now()
    infer_time = time_end - first_time_start
    infer_time_milliseconds = infer_time.total_seconds() * 1000
    logger.debug('WHISPER: Loading audio took ' + str(infer_time_milliseconds) + ' ms')

    time_start = datetime.datetime.now()
    if use_chunking:
        chunks = []
        strides = []
        for chunk, stride in chunk_iter(audio):
            chunk = pad_or_trim(chunk)
            chunks.append(log_mel_spectrogram(chunk, n_mels=n_mels).numpy())
            strides.append(stride)
        mel_features = np.stack(chunks)
        total_chunk_count = len(chunks)
    else:
        mel_audio = pad_or_trim(audio)
        mel_features = log_mel_spectrogram(mel_audio, n_mels=n_mels).numpy()
        # Ref Whisper returns shape (80|128, 3000) but model expects (1, 80|128, 3000)
        mel_features = np.expand_dims(mel_features, axis=0)
        total_chunk_count = 1

    time_end = datetime.datetime.now()
    infer_time = time_end - time_start
    infer_time_milliseconds = infer_time.total_seconds() * 1000
    logger.debug('WHISPER: Feature extraction took ' + str(infer_time_milliseconds) + ' ms')

    # Whisper STEP 2 - optionally actually detect the language or default to configuration
    time_start = datetime.datetime.now()

    # System default language by default
    language = settings.language
    processor_language = f'<|{language}|>'

    if detect_language and not force_language:
        # load the first mel_features batch into the GPU
        # just for language detection
        # important - this is named gpu_features so it will be unloaded during our batch processing later
        first_mel_features = mel_features[0:1, :, :]
        gpu_features = ctranslate2.StorageView.from_array(first_mel_features)
        results = whisper_model.detect_language(gpu_features)
        language, probability = results[0][0]
        processor_language = language
        logger.debug(f"WHISPER: Detected language {language} with probability {probability}")

    else:
        if force_language:
            language = force_language
            logger.debug(f'WHISPER: Forcing language {language}')
            processor_language = f'<|{language}|>'
        else:
            logger.debug(f'WHISPER: Using system default language {language}')

    # Describe the task in the prompt.
    # See the prompt format in https://github.com/openai/whisper.
    prompt = whisper_processor.tokenizer.convert_tokens_to_ids(
        [
            "<|startoftranscript|>",
            processor_language,
            processor_task,
            "<|notimestamps|>",  # Remove this token to generate timestamps.
        ]
    )
    time_end = datetime.datetime.now()
    infer_time = time_end - time_start
    infer_time_milliseconds = infer_time.total_seconds() * 1000
    if detect_language:
        logger.debug('WHISPER: Language detection took ' + str(infer_time_milliseconds) + ' ms')

    # Whisper STEP 3 - run model
    time_start = datetime.datetime.now()
    logger.debug(f'WHISPER: Using model {model} with beam size {beam_size}')

    results = []
    for i, mel_features_batch in enumerate(
        chunkit(mel_features, concurrent_gpu_chunks)
    ):
        logger.debug("WHISPER: Processing batch %s of expected %s", i+1, len(mel_features) // concurrent_gpu_chunks + 1)
        gpu_features = ctranslate2.StorageView.from_array(mel_features_batch)
        results.extend(whisper_model.generate(
            gpu_features,
            [prompt]*len(mel_features_batch),
            beam_size=beam_size,
            return_scores=False,
        ))
    assert len(results) == total_chunk_count, "Result length doesn't match expected total_chunk_count"

    time_end = datetime.datetime.now()
    infer_time = time_end - time_start
    infer_time_milliseconds = infer_time.total_seconds() * 1000
    logger.debug('WHISPER: Model took ' + str(infer_time_milliseconds) + ' ms')

    time_start = datetime.datetime.now()
    if use_chunking:
        assert strides, 'strides needed to compute final tokens when chunking'
        tokens = [(results[i].sequences_ids[0], strides[i]) for i in range(total_chunk_count)]
        tokens = find_longest_common_sequence(tokens, whisper_processor.tokenizer)
    else:
        tokens = results[0].sequences_ids[0]
    results = whisper_processor.decode(tokens)
    time_end = datetime.datetime.now()
    infer_time = time_end - time_start
    infer_time_milliseconds = infer_time.total_seconds() * 1000
    logger.debug('WHISPER: Decode took ' + str(infer_time_milliseconds) + ' ms')
    logger.debug('WHISPER: ASR transcript: ' + results)

    # Strip out token stuff
    pattern = re.compile("[A-Za-z0-9]+", )
    language = pattern.findall(language)[0]

    # the gpu_features were loaded above when we ran the initial whisper model
    # so we don't need to reload them to the GPU here
    if translate and len(total_chunk_count) > concurrent_gpu_chunks:
        logger.warning("Cannot translate because too much audio for the GPU memory")
        translation = None
    elif translate:
        logger.debug(f'WHISPER: Detected non-preferred language {language}, translating')
        translation = do_translate(whisper_model, whisper_processor, gpu_features, total_chunk_count, language, beam_size=beam_size)
        # Strip tokens from translation output - brittle but works right now
        translation = translation.split('>')[2]
        translation = translation.strip()
        logger.debug(f'WHISPER: ASR translation: {translation}')
    else:
        translation = None

    # Strip tokens from results output - brittle but works right now
    # if detect_language:
    #     results = results.split('>')[2]
    # Remove trailing and leading spaces
    results = results.strip()

    time_end = datetime.datetime.now()
    infer_time = time_end - first_time_start
    infer_time_milliseconds = infer_time.total_seconds() * 1000
    logger.debug('WHISPER: Inference took ' + str(infer_time_milliseconds) + ' ms')
    infer_speedup = math.floor(audio_duration / infer_time_milliseconds)
    logger.debug('WHISPER: Inference speedup: ' + str(infer_speedup) + 'x')

    return language, results, infer_time_milliseconds, translation, infer_speedup, audio_duration


# Handy function for converting numbers to the individual word
def num_to_word(text):
    dct = {'0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
           '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'}
    newstr = ''
    for ch in text:
        if ch.isdigit() is True:
            dw = dct[ch]
            newstr = newstr+dw
        else:
            newstr = newstr+ch
    return newstr


def do_sv(audio_file, threshold=sv_threshold):
    logger.debug(f'SV: Got request with threshold {threshold}')

    if sv_model is None:
        logger.warn('SV: Speaker verification support disabled')
        return

    time_initial_start = datetime.datetime.now()
    # Effects for processing of incoming audio
    sv_effects = [
        ["norm", "8"],
        ["trim", "0", "10"],
    ]
    # ["silence", "1", "0.1", "0.1%", "-1", "0.1", "0.1%"],
    # Use cosine simularity for comparison
    cosine_sim = torch.nn.CosineSimilarity(dim=-1)

    # Load audio from request
    audio, audio_sr = librosa.load(audio_file, sr=16000, mono=True)

    # Process incoming audio
    audio_wav, _ = apply_effects_tensor(
        torch.tensor(audio).unsqueeze(0), audio_sr, sv_effects)

    audio_input = sv_feature_extractor(audio_wav.squeeze(0), return_tensors="pt", sampling_rate=audio_sr
                                       ).input_values.to(device)

    with torch.inference_mode():
        audio_emb = sv_model(audio_input).embeddings
    audio_emb = torch.nn.functional.normalize(audio_emb, dim=-1).cpu()

    # Get all defined speakers
    emb_names = []
    emb_res = []
    examples = []

    emb_names.clear
    emb_res.clear
    examples.clear

    dir = "speakers/voice_auth"
    for f in os.listdir(dir):
        if (f.endswith(".npy")):
            name = re.sub(r'(.npy)$', '', f)
            examples.append(dir+f)
            file_path = f"{dir}/{name}.npy"
            if os.path.isfile(file_path):
                speaker_numpy = file_path
            emb = np.load(speaker_numpy)
            emb = torch.tensor(emb).unsqueeze(0).cpu()
            emb_res.append(emb)
            emb_names.append(name)
            logger.debug(f'SV: Loaded speaker {name}')

    result = {}
    for i in range(0, len(emb_names)):
        emb1 = emb_res[i]
        similarity = cosine_sim(emb1, audio_emb).numpy()[0]
        # result[emb_names[i]] = "{:.3f}".format(similarity)
        if similarity >= sv_threshold:
            result[emb_names[i]] = "{:.3f}".format(similarity)

    # Sort result from highest probability
    result = sorted(result.items(), key=operator.itemgetter(1), reverse=True)

    # Return dict
    result = dict(result)

    if result:
        result_string = str(result)
        logger.debug(f'SV: Got known speaker(s) {result_string}')
    else:
        logger.debug('SV: Unknown speaker')

    time_end = datetime.datetime.now()
    infer_time = time_end - time_initial_start
    infer_time_milliseconds = infer_time.total_seconds() * 1000
    logger.debug('SV: Total time took ' + str(infer_time_milliseconds) + ' ms')

    # Return result
    return result


class DataChannelMessage(NamedTuple):
    type: str
    message: Optional[str] = None
    obj: Optional[any] = None


def send_dc_response(channel, *args, **kargs):
    response = DataChannelMessage(*args, **kargs)
    channel.send(json.dumps(response._asdict()))


# Function for WebRTC handling
async def rtc_offer(request, model, beam_size, task, detect_language):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    recorder = None
    top_track = None

    logger.debug(f'RTC: Created for {request.client.host}')

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            logger.debug("RTC DC: message: " + message)
            if not isinstance(message, str):
                logger.warn('RTC DC:  got non string message', message)
                return
            try:
                message = json.loads(message)
                message = DataChannelMessage(**message)
            except Exception:
                logger.exception("could not parse datachannel message", message)
                send_dc_response(channel, "error", "could not parse message")
                return
            if message.type == "ping":
                send_dc_response(channel, "pong", message.message)
            elif message.type == "start":
                # XXX what if top_track is still None i.e. we got start before we got track?
                logger.debug(f'RTC DC: Recording started with track {top_track}')
                nonlocal recorder
                recorder = MediaRecorderLite()
                recorder.addTrack(top_track)
                recorder.start()
                send_dc_response(channel, "log", "ASR Recording - start talking and press stop when done")
            elif message.type == "stop":
                logger.debug('RTC DC: Recording stopped')
                if not recorder:
                    send_dc_response(channel, "error", "Recording not yet started")
                    return
                obj = message.obj or {}
                model = obj.get('model') or settings.whisper_model_default
                beam_size = obj.get('beam_size') or settings.beam_size
                detect_language = obj.get('detect_language') or settings.detect_language
                logger.debug(f'RTC DC: Debug Stop Vars model {model} beam size {beam_size} detect language {detect_language}')
                logger.debug("RTC DC: Recording stopped")
                time_start_base = datetime.datetime.now()
                time_end = datetime.datetime.now()
                infer_time = time_end - time_start_base
                infer_time_milliseconds = infer_time.total_seconds() * 1000
                recorder.stop()
                logger.debug('RTC DC: Recorder stop took ' + str(infer_time_milliseconds) + ' ms')
                # Tell client what we are doing
                send_dc_response(channel, "log", f'Doing ASR with model {model} beam size {beam_size} detect language {detect_language} - please wait')
                # Finally call Whisper
                recorder.file.seek(0)
                language, results, infer_time, translation, infer_speedup, audio_duration = do_whisper(recorder.file,
                                                                                                       model, beam_size,
                                                                                                       task,
                                                                                                       detect_language)
                logger.debug("RTC DC: " + results)
                send_dc_response(channel, "infer", obj=dict(text=results))
                if translation:
                    send_dc_response(channel, "log", f'ASR Translation from {language}:  {translation}')
                infer_time = str(infer_time)
                audio_duration = str(audio_duration)
                infer_speedup = str(infer_speedup)
                send_dc_response(channel, "log", f'ASR Infer time: {infer_time} ms')
                send_dc_response(channel, "log", f'ASR Audio Duration: {audio_duration} ms')
                send_dc_response(channel, "log", f'ASR Speedup: {infer_speedup}x faster than realtime')
                recorder = None
            else:
                send_dc_response(channel, "error", f'unknown message type "{message.type}"')

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.debug(f'RTC: Connection state is {pc.connectionState}')
        if pc.connectionState == "failed" or pc.connectionState == "closed":
            try:
                await recorder.stop()
            except Exception:
                pass
            else:
                logger.debug("RTC: Recording stopped")
            await pc.close()
            pcs.discard(pc)
            # XXX: close recorders?
            logger.debug("RTC: Connection ended")

    @pc.on("track")
    def on_track(track):
        logger.debug(f'RTC: Track received {track.kind}')
        if track.kind == "audio":
            logger.debug("RTC: Setting top track")
            nonlocal top_track
            top_track = track

        @track.on("ended")
        async def on_ended():
            logger.debug(f'RTC: Track ended {track.kind}')

    # handle offer
    await pc.setRemoteDescription(offer)

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

app = FastAPI(title=settings.name,
              description=settings.description,
              version=settings.version,
              openapi_url="/api/openapi.json",
              docs_url="/api/docs",
              redoc_url="/api/redoc")

if settings.cors_allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

basic_auth_401_response = PlainTextResponse('Invalid credentials', status_code=401,
                                            headers={"WWW-Authenticate": "Basic"})


class HttpBasicAuth(BaseHTTPMiddleware):
    def __init__(self, app, username: str = None, password: str = None):
        super().__init__(app)
        self.username = username
        self.password = password

    async def dispatch(self, request: Request, call_next):
        if not request.headers.get('Authorization'):
            return basic_auth_401_response
        auth = request.headers["Authorization"]
        try:
            scheme, credentials = auth.split()
            if scheme.lower() != 'basic':
                return basic_auth_401_response
            decoded = base64.b64decode(credentials).decode("ascii")
        except (ValueError, UnicodeDecodeError, binascii.Error):
            return basic_auth_401_response
        username, _, password = decoded.partition(":")
        if self.username and username != self.username or self.password and password != self.password:
            return basic_auth_401_response
        return await call_next(request)


if settings.basic_auth_pass and settings.basic_auth_user:
    logger.info(f"{settings.name} is configured for HTTP Basic Authentication")
    app.add_middleware(HttpBasicAuth, username=settings.basic_auth_user, password=settings.basic_auth_pass)


@app.on_event("startup")
def startup_event():
    load_models()
    warm_models()
    logger.info(f"{settings.name} is ready for requests!")


@app.on_event("shutdown")
def shutdown_event():
    logger.info(f"{settings.name} is stopping (this can take a while)...")


# Mount static dir to serve files for WebRTC client
app.mount("/rtc", StaticFiles(directory="nginx/static/rtc", html=True), name="rtc_files")

# Mount static dir to serve files for dictation client
app.mount("/dict", StaticFiles(directory="nginx/static/dict", html=True), name="dict_files")

# Expose audio mount in the event willow is configured to save
app.mount("/audio", StaticFiles(directory="nginx/static/audio", html=True), name="audio_files")


class Ping(BaseModel):
    message: str


@app.get("/api/ping", response_model=Ping, summary="Ping for connectivity check", response_description="pong")
async def ping():
    response = {"message": "pong"}
    return JSONResponse(content=response)


@app.post("/api/rtc/asr", summary="Return SDP for WebRTC clients", response_description="SDP for WebRTC clients")
async def rtc_asr(request: Request, response: Response, model: Optional[str] = whisper_model_default,
                  task: Optional[str] = "transcribe", detect_language: Optional[bool] = detect_language,
                  beam_size: Optional[int] = beam_size):
    patch_loop_datagram()
    response = await rtc_offer(request, model, beam_size, task, detect_language)
    return JSONResponse(content=response)


class ASR(BaseModel):
    language: str
    results: str
    infer_time: float
    translation: Optional[str]
    infer_speedup: int
    audio_duration: int
    text: str


@app.post("/api/asr", response_model=ASR, summary="Submit audio file for ASR", response_description="ASR engine output")
async def asr(request: Request, audio_file: UploadFile, response: Response,
              model: Optional[str] = whisper_model_default, detect_language: Optional[bool] = detect_language,
              beam_size: Optional[int] = beam_size, force_language: Optional[str] = None,
              translate: Optional[bool] = False):
    # prof = profile.Profile()
    # prof.enable()
    logger.debug(f'FASTAPI: Got ASR request for model {model} beam size {beam_size} '
                 f'language detection {detect_language}')
    task = "transcribe"

    if force_language:
        if not check_language(force_language):
            logger.debug("FASTAPI: Invalid force_language in request - returning HTTP 400")
            response = {"error": "Invalid force_language"}
            return JSONResponse(content=response, status_code=status.HTTP_400_BAD_REQUEST)

    # Setup access to file
    audio_file = io.BytesIO(await audio_file.read())
    # Do Whisper
    language, results, infer_time, translation, infer_speedup, audio_duration = do_whisper(audio_file, model,
                                                                                           beam_size, task,
                                                                                           detect_language,
                                                                                           force_language, translate)

    # Create final response
    final_response = {"infer_time": infer_time, "infer_speedup": infer_speedup, "audio_duration": audio_duration,
                      "language": language, "text": results}

    # Handle translation in one response
    if translation:
        final_response['translation'] = translation

    # prof.disable()
    # print profiling output
    # stats = pstats.Stats(prof).strip_dirs().sort_stats("cumtime")
    # stats.print_stats(10) # top 10 rows
    return JSONResponse(content=final_response)


@app.post("/api/willow", response_model=ASR, summary="Stream Willow audio for ASR",
          response_description="ASR engine output")
async def willow(request: Request, response: Response, model: Optional[str] = whisper_model_default,
                 detect_language: Optional[bool] = detect_language, beam_size: Optional[int] = beam_size,
                 force_language: Optional[str] = None, translate: Optional[bool] = False,
                 save_audio: Optional[bool] = False, stats: Optional[bool] = False,
                 voice_auth: Optional[bool] = False):
    logger.debug(f'FASTAPI: Got WILLOW request for model {model} beam size {beam_size} '
                 f'language detection {detect_language}')
    task = "transcribe"

    if force_language:
        if not check_language(force_language):
            logger.debug("WILLOW: Invalid force_language in request - returning HTTP 400")
            response = {"error": "Invalid force_language"}
            return JSONResponse(content=response, status_code=status.HTTP_400_BAD_REQUEST)

    # Set defaults - use strings because we parse HTTP headers and convert to int later anyway
    sample_rate = "16000"
    bits = "16"
    channel = "1"
    codec = "pcm"

    sample_rate = request.headers.get('x-audio-sample-rate', '').lower()
    bits = request.headers.get('x-audio-bits', '').lower()
    channel = request.headers.get('x-audio-channel', '').lower()
    codec = request.headers.get('x-audio-codec', '').lower()
    willow_id = request.headers.get('x-willow-id', '').lower()

    logger.debug(f'WILLOW: Audio information: sample rate: {sample_rate}, bits: {bits}, channel(s): {channel}, '
                 f'codec: {codec}')

    if willow_id:
        logger.debug(f"WILLOW: Got Willow ID {willow_id}")

    body = []
    async for chunk in request.stream():
        body.append(chunk)
    body = b''.join(body)

    try:
        if codec == "pcm":
            logger.debug("WILLOW: Source audio is raw PCM, creating WAV container")
            audio_file = write_stream_wav(body, int(sample_rate), int(bits), int(channel))
        elif codec == "wav":
            logger.debug("WILLOW: Source audio is wav")
            audio_file = io.BytesIO(body)
            audio_file.seek(0)
        else:
            logger.debug(f"WILLOW: Converting {codec} to wav")
            file = io.BytesIO(body)
            file.seek(0)
            audio_file = audio_to_wav(file, int(sample_rate))
    except Exception:
        logger.debug("WILLOW: Invalid audio in request - returning HTTP 400")
        response = {"error": "Invalid audio"}
        return JSONResponse(content=response, status_code=status.HTTP_400_BAD_REQUEST)

    # Save received audio if requested - defaults to false
    if save_audio:
        save_filename = 'nginx/static/audio/willow.wav'
        logger.debug(f"WILLOW: Saving audio to {save_filename}")
        with open(save_filename, 'wb') as f:
            f.write(audio_file.getbuffer())

    # Optionally authenticate voice
    if voice_auth:
        stats = True
        sv_results = do_sv(audio_file)
        if sv_results:
            logger.debug("WILLOW: Authenticated voice")
            # Seek back to 0 for Whisper later
            audio_file.seek(0)
            speaker = list(sv_results.keys())[0]
            speaker_status = (f"I heard {speaker} say:")
        else:
            logger.debug("WILLOW: Unknown or unauthorized voice - returning HTTP 406")
            return PlainTextResponse('Unauthorized voice', status_code=406)

    # Do Whisper
    language, results, infer_time, translation, infer_speedup, audio_duration = do_whisper(audio_file, model,
                                                                                           beam_size, task,
                                                                                           detect_language,
                                                                                           force_language, translate)

    # Create final response
    if stats:
        if voice_auth:
            final_response = {"infer_time": infer_time, "infer_speedup": infer_speedup,
                              "audio_duration": audio_duration, "language": language, "text": results,
                              "voice_auth": sv_results, "speaker_status": speaker_status}
        else:
            final_response = {"infer_time": infer_time, "infer_speedup": infer_speedup,
                              "audio_duration": audio_duration, "language": language, "text": results}
    else:
        final_response = {"language": language, "text": results}

    # Handle translation in one response
    if translation:
        final_response['translation'] = translation

    return JSONResponse(content=final_response)
