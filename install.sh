#!/bin/bash

IFS='.' read -ra cuda_version <<< $(nvidia-smi | grep -io 'cuda.*' | tr -cd '[:digit:].')
if [ ${#cuda_version[@]} -ge 2 ] && ([ ${cuda_version[0]} -gt 12 ] || [ ${cuda_version[0]} -eq 12 ] && [ ${cuda_version[1]} -ge 1 ]); then
    processor="cuda12.1"
else
    processor="cpu"
fi

INSTALL_DIR=~/Fltr
mkdir -p "$INSTALL_DIR"
curl -sSL https://github.com/moritztng/fltr/releases/download/v0.1.2/fltr-0.1.2-$(uname -m)-${processor}-$(uname | tr '[:upper:]' '[:lower:]').gz | gunzip > "$INSTALL_DIR/fltr"

MODEL_URL=https://huggingface.co/moritztng/fltr/resolve/main
curl -L "$MODEL_URL/tokenizer.json" -o "$INSTALL_DIR/tokenizer.json"
if [[ ",$1," == *",small,"* ]]; then
    curl -L "$MODEL_URL/mistral-7b-instruct-v0.2.bin" -o "$INSTALL_DIR/small.bin"
fi
if [[ ",$1," == *",large,"* ]]; then
    curl -L "$MODEL_URL/mixtral-8x7b-instruct-v0.1.bin" -o "$INSTALL_DIR/large.bin"
fi
chmod +x "$INSTALL_DIR/fltr"

if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
    echo "export PATH=\"\$PATH:$INSTALL_DIR\"" >> ~/.$(basename $SHELL)rc
fi
