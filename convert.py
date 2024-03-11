# adapted from karpathy's llama2.c

import torch, struct
import numpy as np
from argparse import ArgumentParser


def serialize_fp32(file, tensor):
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f"{len(d)}f", *d)
    file.write(b)


def serialize_int8(file, tensor):
    d = tensor.detach().cpu().view(-1).numpy().astype(np.int8)
    b = struct.pack(f"{len(d)}b", *d)
    file.write(b)


def quantize_serialize(f, w, group_size):
    assert w.numel() % group_size == 0
    w = w.float()
    w = w.reshape(-1, group_size)
    wmax = torch.abs(w).max(dim=1).values
    scale = wmax / 127.0
    quant = w / scale[:, None]
    int8val = torch.round(quant).to(torch.int8)
    fp32val = (int8val.float() * scale[:, None]).view(-1)
    fp32valr = fp32val.reshape(-1, group_size)
    err = torch.abs(fp32valr - w).max(dim=1).values
    maxerr = err.max().item()
    serialize_int8(f, int8val)
    serialize_fp32(f, scale)
    return maxerr


parser = ArgumentParser(prog="mistral converter")
parser.add_argument("output_path", type=str)
parser.add_argument("--checkpoint", type=str)
parser.add_argument("--group-size", default=64, type=int)
parser.add_argument("--moe", action="store_true")
parser.add_argument("--cuda", action="store_true")
args = parser.parse_args()

state_dict = torch.load(
    args.checkpoint, map_location="cuda" if args.cuda else "cpu", mmap=True
)
with open(args.output_path, "wb") as f:
    serialize_fp32(f, state_dict["norm.weight"])
    print("norm.weight")
    err = quantize_serialize(f, state_dict["tok_embeddings.weight"], args.group_size)
    print(f"tok_embeddings.weight, error: {err}")
    err = quantize_serialize(f, state_dict["output.weight"], args.group_size)
    print(f"output.weight, error: {err}")
    for i in range(32):
        layer_prefix = f"layers.{i}."
        print(layer_prefix)
        for name in ["attention_norm.weight", "ffn_norm.weight"]:
            serialize_fp32(f, state_dict[layer_prefix + name])
            print(name)
        for name in [
            "attention.wq.weight",
            "attention.wk.weight",
            "attention.wv.weight",
            "attention.wo.weight",
        ] + (
            ["feed_forward.gate.weight"]
            if args.moe
            else [
                "feed_forward.w1.weight",
                "feed_forward.w2.weight",
                "feed_forward.w3.weight",
            ]
        ):
            err = quantize_serialize(
                f, state_dict[layer_prefix + name], args.group_size
            )
            print(f"{name}, error: {err}")
        if args.moe:
            for e in range(8):
                expert_prefix = layer_prefix + f"feed_forward.experts.{e}."
                print(expert_prefix)
                for name in ["w1.weight", "w2.weight", "w3.weight"]:
                    err = quantize_serialize(
                        f, state_dict[expert_prefix + name], args.group_size
                    )
                    print(f"{name}, error: {err}")
