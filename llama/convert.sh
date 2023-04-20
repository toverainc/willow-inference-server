#!/bin/bash
set -e
set -x

# Deps workaround
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

VICUNA_VER="v1.1"

if [ "$1" ]; then
    SIZE="$1"
else
    SIZE="13B"
fi

echo "Using size $SIZE"
SIZE_LOWER=$(echo $SIZE | tr '[:upper:]' '[:lower:]')

if [ ! -d "transformers" ]; then
    echo "Fetching transformers source for conversion"
    git clone https://github.com/huggingface/transformers.git
fi

echo "Converting base LLaMA to HF..."
python transformers/src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /data/ml/llama-meta/LLaMA --model_size "$SIZE" --output_dir /data/ml/llama-"$SIZE"-hf

echo "Appying Vicuna delta..."
python -m fastchat.model.apply_delta \
    --base /data/ml/llama-"$SIZE"-hf \
    --target /data/ml/vicuna-"$SIZE" \
    --delta lmsys/vicuna-"$SIZE_LOWER"-delta-"$VICUNA_VER"

echo "Success!"
ls -lh /data/ml/vicuna-"$SIZE"