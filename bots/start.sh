#!/bin/bash

target/release/mixtral server &
source env/bin/activate
while true; do python3 bots/x_bot.py; sleep 60; done &
python3 bots/fetch_arxiv.py
sleep 100
sudo shudown -h now
