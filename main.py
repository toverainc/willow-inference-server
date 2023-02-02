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

triton_url = os.environ.get('triton_url', 'hw0-mke.tovera.com:18001')
triton_model = os.environ.get('triton_model', 'medvit')

# transformers
def get_transform(img):
    processor = AutoImageProcessor.from_pretrained('preprocessor_config.json')
    image = Image.open(img)
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

    # Display results - hacky
    inference_output = inference_output[0][0]
    result = inference_output.decode('utf-8')

    return result, infer_time_milliseconds

app = FastAPI()

@app.get("/ping")
async def root():
    return {"message": "Pong"}

@app.post("/api/infer")
async def infer(request: Request, file: UploadFile, response: Response, model: Optional[str] = triton_model):
    try:
        # Setup access to file
        img = io.BytesIO(await file.read())
        triton_model = model
        response, infer_time = do_infer(img, model)
        print(response)
        # Build JSON - NASTY
        response_split = response.split(':', -1)
        prob = response_split[0]
        index = response_split[1]
        label = response_split[2]
        finalResponse = {'model': model, 'prob': prob, 'index': index, 'label': label, 'infer_time': infer_time}
        json_compatible_item_data = jsonable_encoder(finalResponse)
        return JSONResponse(content=json_compatible_item_data)
    except:
        return {'status': 'inference failed'}