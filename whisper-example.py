import ctranslate2
import librosa
import transformers
import datetime
import logging

ctranslate2.set_log_level(logging.INFO)

# Load and resample the audio file.
audio, _ = librosa.load("client/asr.flac", sr=16000, mono=True)

# Compute the features of the first 30 seconds of audio.
processor = transformers.WhisperProcessor.from_pretrained("openai/whisper-large-v2")
inputs = processor(audio, return_tensors="np", sampling_rate=16000)
features = ctranslate2.StorageView.from_array(inputs.input_features)

# Show supported compute types
compute_types = str(ctranslate2.get_supported_compute_types("cuda"))
print("Supported compute types are: " + compute_types)

# Load the model on CUDA
model = ctranslate2.models.Whisper("models/openai-whisper-large-v2", device="cuda")

time_start = datetime.datetime.now()

# Detect the language.
results = model.detect_language(features)
language, probability = results[0][0]
print("Detected language %s with probability %f" % (language, probability))

# Describe the task in the prompt.
# See the prompt format in https://github.com/openai/whisper.
prompt = processor.tokenizer.convert_tokens_to_ids(
    [
        "<|startoftranscript|>",
        language,
        "<|transcribe|>",
        "<|notimestamps|>",  # Remove this token to generate timestamps.
    ]
)

# Run generation for the 30-second window.
time_start = datetime.datetime.now()
results = model.generate(features, [prompt])
time_end = datetime.datetime.now()
infer_time = time_end - time_start
infer_time_milliseconds = infer_time.total_seconds() * 1000
print('Inference took ' + str(infer_time_milliseconds) + ' ms')
transcription = processor.decode(results[0].sequences_ids[0])
print(transcription)
