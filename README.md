Supports SIMD and multiple cores. I got 5.7 tokens/second on CPU. It requires ~48GB of memory for fast inference - otherwise it's slower.

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

# Paul Papers
I'm a dog and read all llm papers on arxiv. When a paper is about running llms more efficiently, I'll let you know on [x.com](https://x.com/paulpapers). This repo is my brain.

![_c6a2fbd1-996d-4618-ac7c-67341940022e](https://github.com/moritztng/mixtral/assets/19519902/49966e48-ab3d-4e30-bfd5-2cf80ab596a5)
