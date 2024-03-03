#!/bin/bash

sudo curl -L https://github.com/moritztng/fltr/releases/download/v0.1-alpha/fltr-0.1-x86_64-cuda -o /usr/local/fltr
sudo chmod +x /usr/local/bin/fltr
sudo curl -L https://huggingface.co/moritztng/Mixtral-8x7B-Instruct-v0.1/resolve/main/{weights.bin,tokenizer.json} --create-dirs -o /usr/share/fltr/weights.bin -o /usr/share/fltr/tokenizer.json
