#!/bin/bash

if [ $1 ]; then
    MODEL="$1"
else
    MODEL="openai/whisper-tiny"
fi

MODEL_OUT=`echo $MODEL | sed -e 's,/,-,g'`

# Verbose
export CT2_VERBOSE=1
export QUANT="float16"

ct2-transformers-converter --model "$MODEL" --quantization "$QUANT" --output_dir models/"$MODEL_OUT"