from functools import lru_cache
from pydantic import BaseSettings

class APISettings(BaseSettings):
    # default return language
    return_language: str = "en"
    # default beam_size - 5 is lib default, 1 for greedy
    beam_size: int = 2
    # default beam size for longer transcriptions
    long_beam_size: int = 5
    # Audio duration in ms to activate "long" mode
    long_beam_size_threshold: int = 12000
    model_threads: int = 10
    # Default detect language?
    detect_language: bool = False

    class Config:
        env_prefix = ""
        case_sensitive = False

@lru_cache()
def get_api_settings() -> APISettings:
    return APISettings()  # reads variables from environment