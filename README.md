Like grep but for natural language questions. Based on Mixtral 8x7B. ~15 tokens/s on Nvidia RTX 3070 with 8GB memory.

# Installation
## Linux x86_64
If nvidia driver that supports cuda 12.1 exists, it installs cuda version, else cpu version. It's ~48GB. 
```bash
curl -sSL https://raw.githubusercontent.com/moritztng/fltr/main/install.sh | sudo bash
```

# Quickstart
```bash
fltr --file emails.txt --prompt "Is the following email spam? Email:" --batch-size 32
```
It will output all lines in the file where the answer is yes.
