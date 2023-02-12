# Profiling
import cProfile as profile
import pstats
# FastAPI preprocessor
from fastapi import FastAPI, File, Form, UploadFile, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Optional
import datetime
import numpy as np
from PIL import Image
import warnings
warnings.simplefilter(action='ignore')
from transformers import AutoImageProcessor
import tritonclient.grpc as grpcclient
import json
import io
import os
import re

# WebRTC
import asyncio
import uuid
import resample
import resampy

from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay

# Only need this because of MediaRecorder...
ROOT = os.path.dirname(__file__)

pcs = set()
relay = MediaRelay()

# Whisper
import ctranslate2
import librosa
import transformers
import datetime
import logging
import torch
import torchaudio

# Configs

# default return language
return_language = "en"

# default beam_size - 5 is lib default, 1 for greedy
beam_size = 5

# model threads
model_threads = 4
# CUDA params
device = "cuda"
device_index = [0]
compute_type = "float16"

## Testing CPU
#device = "cpu"
#device_index = [0]
#compute_type = "auto"

# Turn up log_level for ctranslate2
ctranslate2.set_log_level(logging.INFO)

# Load processor from transformers
processor = transformers.WhisperProcessor.from_pretrained("openai/whisper-large-v2")

# Show supported compute types
compute_types = str(ctranslate2.get_supported_compute_types("cuda"))
print("Supported compute types are: " + compute_types)

# Load all models - thanks for quantization ctranslate2
whisper_model_base = ctranslate2.models.Whisper('models/openai-whisper-base', device=device, device_index=device_index, compute_type=compute_type, inter_threads=model_threads)
whisper_model_medium = ctranslate2.models.Whisper('models/openai-whisper-medium', device=device, device_index=device_index, compute_type=compute_type, inter_threads=model_threads)
whisper_model_large = ctranslate2.models.Whisper('models/openai-whisper-large-v2', device=device, device_index=device_index, compute_type=compute_type, inter_threads=model_threads)

# Go big or go home by default
whisper_model_default = 'large'

# Default detect language?
detect_language = False

# Triton
triton_url = os.environ.get('triton_url', 'hw0-mke.tovera.com:18001')
triton_model = os.environ.get('triton_model', 'medvit-trt-fp32')

def warm_models():
    print("Warming models...")
    for x in range(5):
        do_whisper("3sec.flac", "base", 5, "transcribe", True, "en")
        do_whisper("3sec.flac", "medium", 5, "transcribe", True, "en")
        do_whisper("3sec.flac", "large", 5, "transcribe", True, "en")

def do_translate(features, language):
    # Set task in token format for processor
    task = 'translate'
    processor_task = f'<|{task}|>'

    # Describe the task in the prompt.
    # See the prompt format in https://github.com/openai/whisper.
    prompt = processor.tokenizer.convert_tokens_to_ids(
        [
            "<|startoftranscript|>",
            language,
            processor_task,
            "<|notimestamps|>",  # Remove this token to generate timestamps.
        ]
    )

    # Run generation for the 30-second window.
    time_start = datetime.datetime.now()
    results = whisper_model.generate(features, [prompt], beam_size=beam_size)
    time_end = datetime.datetime.now()
    infer_time = time_end - time_start
    infer_time_milliseconds = infer_time.total_seconds() * 1000
    print('Translate inference took ' + str(infer_time_milliseconds) + ' ms')
    results = processor.decode(results[0].sequences_ids[0])
    print(results)

    return results

def do_whisper(audio_file, model, beam_size, task, detect_language, return_language):
    # Point to model object depending on passed model string
    if model == "large":
        whisper_model = whisper_model_large
    elif model == "medium":
        whisper_model = whisper_model_medium
    elif model == "base":
        whisper_model = whisper_model_base

    processor_task = f'<|{task}|>'
    first_time_start = datetime.datetime.now()

    # Whisper STEP 1 - load audio and extract features
    audio, _ = librosa.load(audio_file, sr=16000, mono=True)
    time_end = datetime.datetime.now()
    infer_time = time_end - first_time_start
    infer_time_milliseconds = infer_time.total_seconds() * 1000
    print('Loading audio took ' + str(infer_time_milliseconds) + ' ms')

    time_start = datetime.datetime.now()
    inputs = processor(audio, return_tensors="np", sampling_rate=16000)
    features = ctranslate2.StorageView.from_array(inputs.input_features)
    time_end = datetime.datetime.now()
    infer_time = time_end - time_start
    infer_time_milliseconds = infer_time.total_seconds() * 1000
    print('Feature extraction took ' + str(infer_time_milliseconds) + ' ms')

    # Whisper STEP 2 - optionally actually detect the language or default to en
    time_start = datetime.datetime.now()
    if detect_language:
        results = whisper_model.detect_language(features)
        language, probability = results[0][0]
        print("Detected language %s with probability %f" % (language, probability))

    else:
        print('Hardcoding language to en')
        # Hardcode language
        language = '<|en|>'

    # Describe the task in the prompt.
    # See the prompt format in https://github.com/openai/whisper.
    prompt = processor.tokenizer.convert_tokens_to_ids(
        [
            "<|startoftranscript|>",
            language,
            processor_task,
            "<|notimestamps|>",  # Remove this token to generate timestamps.
        ]
    )
    time_end = datetime.datetime.now()
    infer_time = time_end - time_start
    infer_time_milliseconds = infer_time.total_seconds() * 1000
    print('Language detection took ' + str(infer_time_milliseconds) + ' ms')

    # Whisper STEP 3 - run model
    time_start = datetime.datetime.now()
    print(f'Using model {model} with beam size {beam_size}')
    results = whisper_model.generate(features, [prompt], beam_size=beam_size)
    time_end = datetime.datetime.now()
    infer_time = time_end - time_start
    infer_time_milliseconds = infer_time.total_seconds() * 1000
    print('Model took ' + str(infer_time_milliseconds) + ' ms')
    
    time_start = datetime.datetime.now()
    results = processor.decode(results[0].sequences_ids[0])
    time_end = datetime.datetime.now()
    infer_time = time_end - time_start
    infer_time_milliseconds = infer_time.total_seconds() * 1000
    print('Decode took ' + str(infer_time_milliseconds) + ' ms')
    print(results)

    # Strip out token stuff
    pattern = re.compile("[A-Za-z0-9]+", )
    language = pattern.findall(language)[0]

    if not language == return_language:
        print(f'Detected non-preferred language {language}, translating to {return_language}')
        translation = do_translate(features, language)
        translation = translation.strip()
    else:
        translation = None

    # Remove trailing and leading spaces
    results = results.strip()

    used_macros = None
    try:
        results.find('period')
        macro_results = results.replace("period", "" )
        macro_results = results.replace("PERIOD", "" )
        #used_macros = 'format_period'
    except:
        pass

    time_end = datetime.datetime.now()
    infer_time = time_end - first_time_start
    infer_time_milliseconds = infer_time.total_seconds() * 1000
    print('Inference took ' + str(infer_time_milliseconds) + ' ms')

    return language, results, infer_time_milliseconds, translation, used_macros

# transformers
def get_transform(img):
    processor = AutoImageProcessor.from_pretrained('preprocessor_config.json')
    image = Image.open(img)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    processed_image = processor(image, return_tensors='np').pixel_values
    return processed_image

def do_infer(img, triton_model):
    # Set up Triton GRPC inference client
    client = grpcclient.InferenceServerClient(url=triton_url, verbose=False, ssl=False)

    # Use transformers
    input = get_transform(img)

    # Hack shape for now - hopefully we can get it dynamically from transformers or something
    shape=(1, 3, 224, 224)

    inputs = grpcclient.InferInput('input__0', shape, datatype='FP32')
    inputs.set_data_from_numpy(input)
    outputs = grpcclient.InferRequestedOutput('output__0', class_count=10)

    # Send for inference
    print('Doing triton inference with model ' + triton_model + " and url " + triton_url)
    time_start = datetime.datetime.now()
    results = client.infer(model_name=triton_model, inputs=[inputs], outputs=[outputs])

    time_end = datetime.datetime.now()
    infer_time = time_end - time_start
    infer_time_milliseconds = infer_time.total_seconds() * 1000
    print('Inference took ' + str(infer_time_milliseconds) + ' ms')

    inference_output = results.as_numpy('output__0')

    # Display and return full results
    print(str(inference_output))
    inference_output_dict = dict(enumerate(inference_output.flatten(), 1))
    print(inference_output_dict)
    return inference_output_dict, infer_time_milliseconds

# Function for WebRTC handling
async def rtc_offer(request, model, beam_size, task, detect_language, return_language):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    # TODO: This seems broken but still works
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    print("RTC: Created for", request.client.host)

    # prepare local media
    #recorder_file = os.path.join(ROOT, "recorder.wav")
    #recorder_file = io.BytesIO()
    recorder_file = "/tmp/recorder.wav"
    recorder = MediaRecorder(recorder_file)

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            print("RTC DC message: " + message)
            if isinstance(message, str) and message.startswith("ping"):
                channel.send("pong" + message[4:])
            if isinstance(message, str) and message.startswith("stop"):
                print("RTC: Recording stopped")
                time_start_base = datetime.datetime.now()
                time_end = datetime.datetime.now()
                infer_time = time_end - time_start_base
                infer_time_milliseconds = infer_time.total_seconds() * 1000
                recorder.stop()
                recorder_file = "/tmp/recorder.wav"
                print('Recorder stop took ' + str(infer_time_milliseconds) + ' ms')
                print("RTC: Got buffer")
                # Tell client what we are doing
                channel.send(f'Doing ASR with model {model} beam size {beam_size} detect language {detect_language}')
                # Compat with standard whisper function all
                audio_file = recorder_file
                # Finally call Whisper
                language, results, infer_time, translation, used_macros = do_whisper(audio_file, model, beam_size, task, detect_language, return_language)
                print("RTC: " + results)
                channel.send('ASR Transcript: ' + results)
                if translation:
                    channel.send(f'ASR Translation from {language}:  {translation}')
                infer_time = str(infer_time)
                channel.send(f'ASR Infer time: {infer_time} ms')

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("RTC: Connection state is", pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        print("RTC: Track received", track.kind)

        if track.kind == "audio":
            recorder.addTrack(track)

        @track.on("ended")
        async def on_ended():
            print("RTC: Track ended", track.kind)

    # handle offer
    await pc.setRemoteDescription(offer)
    await recorder.start()

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

# Warm models
warm_models()

app = FastAPI()
# Mount static dir to serve files for aiortc client
app.mount("/rtc", StaticFiles(directory="rtc"), name="rtc_files")

@app.get("/ping")
async def ping():
    return {"message": "Pong"}

@app.post("/api/infer")
async def infer(request: Request, file: UploadFile, response: Response, model: Optional[str] = triton_model):
    # Setup access to file
    audio_file = io.BytesIO(await file.read())
    response, infer_time = do_infer(img, model)
    final_response = {"infer_time": infer_time, "results": [response]}
    json_compatible_item_data = jsonable_encoder(final_response)
    return JSONResponse(content=json_compatible_item_data)

@app.post("/api/rtc/asr")
async def rtc_asr(request: Request, response: Response, model: Optional[str] = whisper_model_default, task: Optional[str] = "transcribe", detect_language: Optional[bool] = detect_language, return_language: Optional[str] = return_language, beam_size: Optional[int] = beam_size):
    response = await rtc_offer(request, model, beam_size, task, detect_language, return_language)
    return JSONResponse(content=response)

@app.post("/api/asr")
async def asr(request: Request, audio_file: UploadFile, response: Response, model: Optional[str] = whisper_model_default, task: Optional[str] = "transcribe", detect_language: Optional[bool] = detect_language, return_language: Optional[str] = return_language, beam_size: Optional[int] = beam_size):
    prof = profile.Profile()
    prof.enable()

    print(f"Got ASR request for model {model} beam size {beam_size} language detection {detect_language}")
    # Setup access to file
    audio_file = io.BytesIO(await audio_file.read())
    # Do Whisper
    language, results, infer_time, translation, used_macros = do_whisper(audio_file, model, beam_size, task, detect_language, return_language)

    # Create final response
    final_response = {"infer_time": infer_time, "language": language, "text": results}

    # Handle translation in one response
    if translation:
        final_response['translation']=translation

    # If we detected a custom macro, tell them what it was
    if used_macros:
        final_response['used_macros']=used_macros

    json_compatible_item_data = jsonable_encoder(final_response)
    prof.disable()
    # print profiling output
    stats = pstats.Stats(prof).strip_dirs().sort_stats("cumtime")
    stats.print_stats(10) # top 10 rows
    return JSONResponse(content=json_compatible_item_data)