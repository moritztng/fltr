Like grep but for natural language questions. Based on Mistral 7B or 8x7B. ~40 tokens/s or ~23 tokens/s on Nvidia RTX 3070 with 8GB memory.

# Installation
## Linux & macOS
If nvidia driver that supports cuda 12.1 exists, it installs cuda version, else cpu version. Replace `small` with `large` to install Mixtral 8x7B. It's ~7GB or ~48GB.
```bash
curl https://raw.githubusercontent.com/moritztng/fltr/main/install.sh -o install.sh && bash install.sh small && export PATH=$PATH:~/Fltr
```

# Quickstart
Add `--large` for Mixtral 8x7B.
```bash
fltr --file emails.txt --prompt "Is the following email spam? Email:" --batch-size 32
```
It will output all lines in the file where the answer is yes.
