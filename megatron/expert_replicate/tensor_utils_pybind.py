# vllm/ipc_handler_demo/tensor_utils_pybind.py
import torch
import sys
import os

# Add the path to the built module
cuda_tools_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                             'ipc_handler_demo/cuda_tools')
sys.path.append(cuda_tools_path)

try:
    import ipc_tensor_pybind
except ImportError as e:
    print(f"Error importing ipc_tensor_pybind: {e}")
    print(f"Make sure you've built the module by running:")
    print(f"cd {cuda_tools_path} && pip install -e .")
    raise ImportError("Failed to import ipc_tensor_pybind. Make sure it's built and in the Python path.")

# Map string dtype names to torch.dtype objects
DTYPE_MAP = {
    'torch.float32': torch.float32,
    'torch.int32': torch.int32,
    'torch.int64': torch.int64,
    'torch.float64': torch.float64,
    'torch.uint8': torch.uint8,
    'torch.float16': torch.float16,
    'torch.bfloat16': torch.bfloat16,
}

class IPCHandleManager:
    def __init__(self, handle_bytes, device):
        self.dev_ptr = ipc_tensor_pybind.open_ipc_handle(handle_bytes, device)
        self.device = device
        
    def create_tensor(self, meta, make_contiguous=False):
        shape = meta['shape']
        dtype_str = meta['dtype']
        offset_bytes = meta['offset_bytes']
        
        dtype = DTYPE_MAP.get(dtype_str)
        if dtype is None:
            raise ValueError(f"Unsupported dtype: {dtype_str}")
        
        return ipc_tensor_pybind.create_tensor_from_ptr(
            self.dev_ptr,
            offset_bytes,
            list(shape),
            dtype,
            self.device,
            make_contiguous
        )
        
    def close_ipc_handle(self):
        if hasattr(self, 'dev_ptr'):
            ipc_tensor_pybind.close_ipc_handle(self.dev_ptr)
        
    def __del__(self):
        self.close_ipc_handle()

def tensor_restore_from_handler_pybind(handle_manager, meta, make_contiguous=False):
    """
    Restore a tensor from an IPC handle using PyBind11.
    
    Args:
        handle_manager: IPCHandleManager instance
        meta: Dictionary containing tensor metadata (shape, dtype, device)
        make_contiguous: Whether to make the tensor contiguous (creates a copy)
    """
    return handle_manager.create_tensor(meta, make_contiguous)

def get_ipc_handle_pybind(tensor: torch.Tensor, ensure_contiguous=False):
    """
    Get an IPC handle from a tensor using PyBind11.
    
    Args:
        tensor: The tensor to export
        ensure_contiguous: Whether to ensure the tensor is contiguous before exporting
    
    Returns:
        tuple: (ipc_handle_bytes, meta) where meta contains tensor metadata
    """
    if not tensor.is_cuda:
        raise ValueError("Tensor must be on a CUDA device")
    
    # Synchronize to ensure all operations on the tensor are complete
    torch.cuda.synchronize(tensor.device)

    ipc_handle_bytes = ipc_tensor_pybind.export_ipc_handle(tensor, ensure_contiguous)

    meta = {
        "shape": tensor.shape,
        "dtype": str(tensor.dtype),
        "device": tensor.device.index,
        "numel": tensor.numel(),
        "strides": tensor.stride(),
        "is_contiguous": tensor.is_contiguous(),
    }
    
    # Get IPC handle using PyBind11 module
    
    return ipc_handle_bytes, meta

def merge_tensors_and_export_ipc_handle(tensors, index):
    return ipc_tensor_pybind.merge_tensors_and_export_ipc_handle(tensors,index)