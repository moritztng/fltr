#!/bin/bash

target/release/mixtral server &
while true; do env/bin/python3 bots/x_bot.py; sleep 60; done &
env/bin/python3 bots/fetch_arxiv.py
sleep 100
sudo shudown -h now
