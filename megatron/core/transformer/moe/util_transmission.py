import torch
import struct

# Metadata结构: 32 bytes
def pack_metadata(md):
    shape = md['shape']
    shape_pad = list(shape) + [0] * (3 - len(shape))  # pad to 3
    return struct.pack(
        'BB3iIq',                  # dtype_id, ndim, shape0~2, device, offset
        dtype_to_code(md['dtype']),
        len(shape),
        *shape_pad,
        md['device'],
        md['offset_bytes']
    )

def unpack_metadata(data):
    dtype_id, ndim, s0, s1, s2, device, offset = struct.unpack('BB3iIq', data)
    return {
        'dtype': code_to_dtype(dtype_id),
        'shape': tuple([s0, s1, s2][:ndim]),
        'device': device,
        'offset_bytes': offset
    }


def dtype_to_code(dtype):
    mapping = {
        torch.float32: 0,
        torch.float16: 1,
        torch.int64: 2,
        torch.bfloat16: 3,
        torch.int32: 4,
        torch.uint8: 5
    }
    return mapping[dtype]


def code_to_dtype(dtype):
    mapping = {
        0: torch.float32,
        1: torch.float16,
        2: torch.int64,
        3: torch.bfloat16,
        4: torch.int32,
        5: torch.uint8
    }
    return mapping[dtype]