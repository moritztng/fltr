#!/bin/bash

IFS='.' read -ra cuda_version <<< $(nvidia-smi | grep -io 'cuda.*' | tr -cd '[:digit:].')
if [ ${#cuda_version[@]} -ge 2 ] && ([ ${cuda_version[0]} -gt 12 ] || [ ${cuda_version[0]} -eq 12 ] && [ ${cuda_version[1]} -ge 1 ]); then
    processor="cuda12.1"
else
    processor="cpu"
fi

curl -sSL https://github.com/moritztng/fltr/releases/download/v0.1-alpha/fltr-0.1-x86_64-${processor}.gz | gunzip > /usr/local/bin/fltr  \
&& curl -L https://huggingface.co/moritztng/Mixtral-8x7B-Instruct-v0.1/resolve/main/{weights.bin,tokenizer.json} --create-dirs -o /usr/share/fltr/weights.bin -o /usr/share/fltr/tokenizer.json \
&& chmod +x /usr/local/bin/fltr /usr/share/fltr
