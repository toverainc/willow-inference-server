#!/usr/bin/env bash
set -e
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"
set -x

python -c 'import transformers; processor=transformers.SpeechT5Processor.from_pretrained("microsoft/speecht5_tts"); processor.save_pretrained("./models/microsoft-speecht5_tts")'
python -c 'import transformers; model=transformers.SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts"); model.save_pretrained("./models/microsoft-speecht5_tts")'
python -c 'import transformers; vocoder=transformers.SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan"); vocoder.save_pretrained("./models/microsoft-speecht5_hifigan")'