# FastAPI preprocessor
from fastapi import FastAPI, File, Form, UploadFile, Request, Response
import datetime
import numpy as np
from PIL import Image
import warnings
warnings.simplefilter(action='ignore')
from transformers import AutoImageProcessor
import tritonclient.grpc as grpcclient
import json
import io

triton_url = "hw0-mke.tovera.com:18001"
triton_model = "medvit"

# transformers
def get_transform(img):
    processor = AutoImageProcessor.from_pretrained('preprocessor_config.json')
    image = Image.open(img)
    processed_image = processor(image, return_tensors='np').pixel_values
    return processed_image

def do_vit(img):
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

    return result

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/api/infer")
async def do_infer(request: Request, file: UploadFile, response: Response):
#    # Setup content-type
#    content_type = file.content_type

    # Setup access to file
    img = io.BytesIO(await file.read())
    response = do_vit(img)
    return response