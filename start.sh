#!/bin/bash

target/release/llamars server &
env/bin/python3 fetch_arxiv.py
shudown -h now
