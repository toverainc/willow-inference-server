#!/bin/bash

export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

if [ -r venv/bin/activate ]; then
    source venv/bin/activate
else
    echo "No venv found - exiting. You need to create a venv with a python3 local installation."
    exit 1
fi

case $1 in

models)
    for i in tiny base small medium large-v2;do
        MODEL="tovera/wis-whisper-$i"
        echo "Setting up WIS model $MODEL..."
        MODEL_OUT=`echo $MODEL | sed -e 's,/,-,g'`
        git clone https://huggingface.co/"$MODEL" models/"$MODEL_OUT"
        rm -rf "$MODEL_OUT"/.git
    done

    python -c 'import transformers; processor=transformers.SpeechT5Processor.from_pretrained("microsoft/speecht5_tts"); processor.save_pretrained("./models/microsoft-speecht5_tts")'
    python -c 'import transformers; model=transformers.SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts"); model.save_pretrained("./models/microsoft-speecht5_tts")'
    python -c 'import transformers; vocoder=transformers.SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan"); vocoder.save_pretrained("./models/microsoft-speecht5_hifigan")'

    python -c 'import transformers; feature_extractor=transformers.AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv"); feature_extractor.save_pretrained("./models/microsoft-wavlm-base-plus-sv")'
    python -c 'import transformers; model=transformers.AutoModelForAudioXVector.from_pretrained("microsoft/wavlm-base-plus-sv"); model.save_pretrained("./models/microsoft-wavlm-base-plus-sv")'
;;

install)
    pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
    pip install -r requirements-mac.txt
;;

run)
    ./entrypoint.sh
;;

esac