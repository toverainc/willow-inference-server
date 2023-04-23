#!/bin/bash
set -e
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"

VICUNA_VER="v1.1"

if [ "$2" ]; then
    SIZE="$1"
else
    SIZE="13B"
fi

check_llama() {
    if [ ! -d "llama/$SIZE" ]; then
        echo "You need to get the original LLaMA weights - exiting"
        exit 1
    fi
}

mkdir -p ../models

convert_llama() {
    echo "Using size $SIZE"
    SIZE_LOWER=$(echo $SIZE | tr '[:upper:]' '[:lower:]')

    echo "Converting base LLaMA to HF..."
    python convert_llama_weights_to_hf.py \
        --input_dir llama --model_size "$SIZE" --output_dir llama-"$SIZE"-hf

    echo "Appying Vicuna delta..."
    python -m fastchat.model.apply_delta \
        --base llama-"$SIZE"-hf \
        --target vicuna-"$SIZE"-hf \
        --delta lmsys/vicuna-"$SIZE_LOWER"-delta-"$VICUNA_VER"
}

quant_vicuna() {
    echo "Quantizing Vicuna model - this will take a while..."
    mkdir -p vicuna
    python quant-vicuna.py --src vicuna-"$SIZE"-hf --dest vicuna
}

copy_tokenizer() {
    for i in generation_config.json special_tokens_map.json tokenizer.model tokenizer_config.json; do
        cp vicuna-"$SIZE"-hf/"$i" vicuna/
    done
}

case $1 in

clean)
    rm -rf llama-* vicuna vicuna-hf
;;

dist)
    tar -C ../models -cvf - vicuna | zstd -T0 > vicuna.tar.zstd
;;

install-dist)
    zstdcat vicuna.tar.zstd | tar -xvf -
    mv vicuna ../models/
;;

install)
    check_llama
    convert_llama
    quant_vicuna
    copy_tokenizer
    mv vicuna ../models/
    echo "Vicuna installed to models path"
;;

esac