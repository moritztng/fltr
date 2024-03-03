Like grep but for natural language questions. Based on Mixtral 8x7B.

# Installation
```bash
curl -sSL https://raw.githubusercontent.com/moritztng/fltr/main/install.sh | bash
```

# Quickstart
```bash
fltr --file emails.txt --prompt "Is the following email spam? Email:" --batch-size 32
```
