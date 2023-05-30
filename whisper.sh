#!/usr/bin/env bash
set -e
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"
set -x

if [ $1 ]; then
    MODEL="$1"
else
    MODEL="openai/whisper-tiny"
fi

MODEL_OUT=`echo $MODEL | sed -e 's,/,-,g'`

# Verbose
export CT2_VERBOSE=1
export QUANT="float16"

ct2-transformers-converter --force --model "$MODEL" --quantization "$QUANT" --output_dir models/"$MODEL_OUT"
python -c 'import transformers; processor=transformers.WhisperProcessor.from_pretrained("'$MODEL'"); processor.save_pretrained("./models/'$MODEL_OUT'")'