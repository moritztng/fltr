import torch, struct
import numpy as np
from argparse import ArgumentParser

def serialize_fp32(file, tensor):
    """ writes one fp32 tensor to file that is open in wb mode """
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f'{len(d)}f', *d)
    file.write(b)

def serialize_int8(file, tensor):
    """ writes one int8 tensor to file that is open in wb mode """
    d = tensor.detach().cpu().view(-1).numpy().astype(np.int8)
    b = struct.pack(f'{len(d)}b', *d)
    file.write(b)

def quantize_serialize(f, w, group_size):
    """
    takes a tensor and returns the Q8_0 quantized version
    i.e. symmetric quantization into int8, range [-127,127]
    """
    assert w.numel() % group_size == 0
    ori_shape = w.shape
    w = w.float() # convert to float32
    w = w.reshape(-1, group_size)
    # find the max in each group
    wmax = torch.abs(w).max(dim=1).values
    # calculate the scaling factor such that float = quant * scale
    scale = wmax / 127.0
    # scale into range [-127, 127]
    quant = w / scale[:,None]
    # round to nearest integer
    int8val = torch.round(quant).to(torch.int8)
    # dequantize by rescaling
    fp32val = (int8val.float() * scale[:,None]).view(-1)
    fp32valr = fp32val.reshape(-1, group_size)
    # calculate the max error in each group
    err = torch.abs(fp32valr - w).max(dim=1).values
    # find the max error across all groups
    maxerr = err.max().item()
    
    serialize_int8(f, int8val)
    serialize_fp32(f, scale)
    return maxerr

parser = ArgumentParser(prog="llama 2 converter")
parser.add_argument("output_path", type=str)
parser.add_argument("--checkpoint", type=str)
parser.add_argument("--group-size", default=64, type=int)
args = parser.parse_args()

state_dict = torch.load(args.checkpoint, map_location="cpu", mmap=True)
with open(args.output_path, "wb") as f:
    serialize_fp32(f, state_dict['norm.weight'])
    print("norm.weight")
    err = quantize_serialize(f, state_dict['tok_embeddings.weight'], args.group_size)
    print(f"tok_embeddings.weight, error: {err}")
    err = quantize_serialize(f, state_dict['output.weight'], args.group_size)
    print(f"output.weight, error: {err}")
    for i in range(32):
        layer_prefix = f'layers.{i}.'
        print(layer_prefix)
        for name in ['attention_norm.weight', 'ffn_norm.weight']:
            serialize_fp32(f, state_dict[layer_prefix + name])
            print(name)
        for name in ['attention.wq.weight', 'attention.wk.weight', 'attention.wv.weight', 'attention.wo.weight', 'feed_forward.gate.weight']:
            err = quantize_serialize(f, state_dict[layer_prefix + name], args.group_size)
            print(f"{name}, error: {err}")
        for e in range(8):
            expert_prefix = layer_prefix + f"feed_forward.experts.{e}."
            print(expert_prefix)
            for name in ['w1.weight', 'w2.weight', 'w3.weight']:
                err = quantize_serialize(f, state_dict[expert_prefix + name], args.group_size)
                print(f"{name}, error: {err}")
