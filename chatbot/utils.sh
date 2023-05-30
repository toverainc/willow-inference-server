#!/usr/bin/env bash
set -e
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"

VICUNA_VER="v1.1"

if [ "$2" ]; then
    SIZE="$2"
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

convert_llama_hf() {
    echo "Using size $SIZE"
    SIZE_LOWER=$(echo $SIZE | tr '[:upper:]' '[:lower:]')

    echo "Converting base LLaMA to HF..."
    python convert_llama_weights_to_hf.py \
        --input_dir llama --model_size "$SIZE" --output_dir llama-"$SIZE"-hf
}

apply_vicuna() {
    echo "Appying Vicuna delta..."
    python -m fastchat.model.apply_delta \
        --base llama-"$SIZE"-hf \
        --target vicuna-"$SIZE"-hf \
        --delta lmsys/vicuna-"$SIZE_LOWER"-delta-"$VICUNA_VER"
}

quant_vicuna() {
    echo "Quantizing Vicuna model - this will take a while..."
    mkdir -p vicuna
    python quant-vicuna.py -s vicuna-"$SIZE"-hf -d vicuna
}

install_dist() {
    zstdcat -T0 vicuna.tar.zstd | tar -xvf -
}

clean() {
    rm -rf llama-* vicuna-*
}

dist() {
    tar -cvf - vicuna | zstd -T0 > vicuna.tar.zstd
}

case $1 in

clean)
    clean
;;

dist)
    dist
;;

install)
    if [ -r "vicuna.tar.zstd" ]; then
        echo "Found dist tarball - extracting..."
        install_dist
    else
        check_llama
        convert_llama_hf
        apply_vicuna
        quant_vicuna
        clean
        dist
    fi
    mv vicuna ../models/
    echo "Vicuna installed to models path"
;;

esac