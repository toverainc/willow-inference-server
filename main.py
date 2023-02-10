# FastAPI preprocessor
from fastapi import FastAPI, File, Form, UploadFile, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
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

# Whisper
import ctranslate2
import librosa
import transformers
import datetime
import logging

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

# Load the model on CUDA
asr_model = "openai-whisper-large-v2"
#asr_model = "openai-whisper-medium"

whisper_model_path = "models" + "/"+ asr_model

whisper_model = ctranslate2.models.Whisper(whisper_model_path, device=device, device_index=device_index, compute_type=compute_type, inter_threads=model_threads)

# Triton
triton_url = os.environ.get('triton_url', 'hw0-mke.tovera.com:18001')
triton_model = os.environ.get('triton_model', 'medvit-trt-fp32')

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

def do_whisper(audio_file, model, task, return_language):
    time_start = datetime.datetime.now()

    # Load audio
    audio, _ = librosa.load(audio_file, sr=16000, mono=True)

    inputs = processor(audio, return_tensors="np", sampling_rate=16000)
    features = ctranslate2.StorageView.from_array(inputs.input_features)

    # Detect the language.
    results = whisper_model.detect_language(features)
    language, probability = results[0][0]

    print("Detected language %s with probability %f" % (language, probability))

    # Set task in token format for processor
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
    results = processor.decode(results[0].sequences_ids[0])
    print(results)

    # Strip out token stuff
    pattern = re.compile("[A-Za-z0-9]+", )
    language = pattern.findall(language)[0]

    if not language == return_language:
        print(f'Detected non-preferred language {language}, translating to {return_language}')
        translation = do_translate(features, language)
    else:
        translation = None

    time_end = datetime.datetime.now()
    infer_time = time_end - time_start
    infer_time_milliseconds = infer_time.total_seconds() * 1000
    print('Inference took ' + str(infer_time_milliseconds) + ' ms')

    # Remove trailing and leading spaces
    results = results.strip()
    translation = translation.strip()

    return language, results, infer_time_milliseconds, translation

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

app = FastAPI()

@app.get("/ping")
async def root():
    return {"message": "Pong"}

@app.post("/api/infer")
async def infer(request: Request, file: UploadFile, response: Response, model: Optional[str] = triton_model):
    # Setup access to file
    audio_file = io.BytesIO(await file.read())
    response, infer_time = do_infer(img, model)
    final_response = {"infer_time": infer_time, "results": [response]}
    json_compatible_item_data = jsonable_encoder(final_response)
    return JSONResponse(content=json_compatible_item_data)

@app.post("/api/asr")
async def infer(request: Request, audio_file: UploadFile, response: Response, model: Optional[str] = asr_model, task: Optional[str] = "transcribe", return_language: Optional[str] = return_language):
    # Setup access to file
    print("Got ASR request for model: " + model)
    audio_file = io.BytesIO(await audio_file.read())
    language, results, infer_time, translation = do_whisper(audio_file, model, task, return_language)

    final_response = {"infer_time": infer_time, "language": language, "text": results}

    # Handle translation in one response
    if translation:
        final_response['translation']=translation

    json_compatible_item_data = jsonable_encoder(final_response)
    return JSONResponse(content=json_compatible_item_data)