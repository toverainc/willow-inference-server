from functools import lru_cache
from pydantic import BaseSettings
from typing import List

class APISettings(BaseSettings):
    # Project metadata
    name = "Willow Inference Server"
    description = "High Performance Language Inference API"
    version = "1.0"

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

    # Enable TTS
    support_tts: bool = True

    # Enable SV
    support_sv: bool = False

    # SV threshold
    sv_threshold: float = 0.75

    # The default whisper model to use
    whisper_model_default: str = 'medium'

    # Default TTS speaker to use. CLB is US female
    tts_default_speaker: str = "CLB"

    # List of allowed origins for WebRTC. See https://fastapi.tiangolo.com/tutorial/cors/#use-corsmiddleware
    cors_allowed_origins: List[str] = []

    # If basic_auth_pass or basic_auth_user are set all endpoints are guarded by basic auth
    # If basic_auth_user is falsy it will not be checked. If basic_auth_pass is falsy it will not be checked.
    basic_auth_user: str = None
    basic_auth_pass: str = None

    # Path to chatbot model
    chatbot_model_path: str = 'models/vicuna'

    # Chatbot model max length
    chatbot_max_length: int = 5000

    # airotc debug for connectivity and other WebRTC debugging
    aiortc_debug: bool = False

    class Config:
        env_prefix = ""
        case_sensitive = False

@lru_cache()
def get_api_settings() -> APISettings:
    return APISettings()  # reads variables from environment