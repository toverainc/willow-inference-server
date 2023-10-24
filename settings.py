from functools import lru_cache
from pydantic import BaseSettings
from typing import List


class APISettings(BaseSettings):
    # Project metadata
    name: str = "Willow Inference Server"
    description: str = "High Performance Language Inference API"
    version: str = "1.0"

    # default beam_size - 5 is lib default, 1 for greedy
    beam_size: int = 1
    # default beam size for longer transcriptions
    long_beam_size: int = 3
    # Audio duration in ms to activate "long" mode
    long_beam_size_threshold: int = 12000
    model_threads: int = 10

    # Default language
    language: str = "en"

    # Default detect language?
    detect_language: bool = False

    # TTS CUDA memory threshold - equivalent of 4GB GPUs
    tts_memory_threshold: int = 3798205849

    # SV CUDA memory threshold - equivalent of 6GB GPUs
    sv_memory_threshold: int = 5798205849

    # Enable chunking support
    support_chunking: bool = True

    # There is really no reason to disable chunking anymore
    # But if you still want to, you can set this threshold higher
    # current value is equivalent of 4GB GPUs
    chunking_memory_threshold: int = 3798205849

    # Maximum number of chunks that are loaded into the GPU at once
    # This will need to be tweaked based on GPU ram
    # 8GB GPUs should support at least 2 chunks so starting with that
    concurrent_gpu_chunks: int = 2

    # Enable TTS
    support_tts: bool = True

    # Enable SV
    support_sv: bool = False

    # SV threshold
    sv_threshold: float = 0.75

    # The default whisper model to use
    whisper_model_default: str = 'medium'

    # Default TTS format to use
    tts_default_format: str = "FLAC"

    # Default TTS speaker to use. CLB is US female
    tts_default_speaker: str = "CLB"

    # List of allowed origins for WebRTC. See https://fastapi.tiangolo.com/tutorial/cors/#use-corsmiddleware
    cors_allowed_origins: List[str] = []

    # If basic_auth_pass or basic_auth_user are set all endpoints are guarded by basic auth
    # If basic_auth_user is falsy it will not be checked. If basic_auth_pass is falsy it will not be checked.
    basic_auth_user: str = None
    basic_auth_pass: str = None

    # Support chatbot
    support_chatbot: bool = False

    # Path to chatbot model - download from HuggingFace at runtime by default (gets cached)
    chatbot_model_path: str = 'TheBloke/vicuna-13b-v1.3-GPTQ'

    # Chatbot model basename
    chatbot_model_basename: str = 'vicuna-13b-v1.3-GPTQ-4bit-128g.no-act.order'

    # Chatbot device
    chatbot_device: str = 'cuda:0'

    # Chatbot pipeline default temperature
    chatbot_temperature: float = 0.7

    # Chatbot pipeline default top_p
    chatbot_top_p: float = 0.95

    # Chatbot pipeline default repetition penalty
    chatbot_repetition_penalty: float = 1.15

    # Chatbot pipeline default max new tokens
    chatbot_max_new_tokens: int = 512

    # airotc debug for connectivity and other WebRTC debugging
    aiortc_debug: bool = False

    class Config:
        env_prefix = ""
        case_sensitive = False


@lru_cache()
def get_api_settings() -> APISettings:
    return APISettings()  # reads variables from environment
