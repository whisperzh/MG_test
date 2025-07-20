#!/bin/bash
set -e

echo "Building PyBind11 module for CUDA IPC tensor handling..."

# Check if PyTorch is installed
# if ! python -c "import torch" &> /dev/null; then
#     echo "PyTorch is not installed. Please install it first."
#     exit 1
# fi

# Check if CUDA is available
# if ! python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
#     echo "CUDA is not available. Please check your CUDA installation."
#     exit 1
# fi

# Build the module
pip install --no-build-isolation -e .

echo "Build completed successfully!"
echo "You can now import ipc_tensor_pybind in your Python code." 