Supports SIMD and multiple cores. I got 8 tokens/second on CPU. It requires ~48GB of memory for fast inference - otherwise it's slower.

# Installation
```bash
git clone https://github.com/moritztng/mixtral.git
cd mixtral
# Download 8 bit quantized weights
curl --create-dirs -o weights/weights.bin -o weights/tokenizer.json -L https://huggingface.co/moritztng/Mixtral-8x7B-Instruct-v0.1/resolve/main/{weights.bin,tokenizer.json}
```

# Quickstart
## Command Line Interface
```bash
cargo run --release generate --weights weights --prompt "Who is Satoshi Nakamoto?" --length 256 --autostop
```
## Server
Adjust the parameters in `config.toml` to your needs
```bash
cargo run --release server
```
