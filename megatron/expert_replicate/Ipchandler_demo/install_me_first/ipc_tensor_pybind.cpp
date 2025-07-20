    // vllm/ipc_handler_demo/cuda_tools/ipc_tensor_pybind.cpp
    #include <pybind11/pybind11.h>
    #include <pybind11/numpy.h>
    #include <pybind11/stl.h>
    #include <cuda_runtime.h>
    #include <torch/extension.h>
    #include <string>
    #include <vector>

    namespace py = pybind11;

    void* open_ipc_handle_pybind(py::bytes handle_bytes, int device) {
        // Get the size of the bytes object using PyBytes_Size
        Py_ssize_t size = PyBytes_Size(handle_bytes.ptr());
        if (size != sizeof(cudaIpcMemHandle_t)) {
            throw std::runtime_error("Invalid IPC handle size");
        }

        // Set CUDA device
        cudaError_t err = cudaSetDevice(device);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaSetDevice failed: ") + cudaGetErrorString(err));
        }

        // Copy handle from bytes
        cudaIpcMemHandle_t handle;
        std::memcpy(&handle, PyBytes_AsString(handle_bytes.ptr()), sizeof(cudaIpcMemHandle_t));

        void* dev_ptr = nullptr;
        cudaError_t err1 = cudaDeviceSynchronize();
        if (err1 != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaIpcOpenMemHandle failed: ") + cudaGetErrorString(err1));
        }


        err = cudaIpcOpenMemHandle(&dev_ptr, handle, cudaIpcMemLazyEnablePeerAccess);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaIpcOpenMemHandle failed: ") + cudaGetErrorString(err));
        }

        return dev_ptr;
    }

    // 新增：从已打开的handle创建tensor的函数
    torch::Tensor create_tensor_from_ptr(void* dev_ptr,
                                    size_t offset_bytes,
                                    std::vector<int64_t> shape,
                                    torch::Dtype dtype,
                                    int device,
                                    bool make_contiguous) {
        void* offset_ptr = static_cast<void*>(static_cast<char*>(dev_ptr) + offset_bytes);
        
        std::cout << "offset_ptr: " << offset_ptr << std::endl;
        // Synchronize CUDA device to ensure memory is available
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaDeviceSynchronize failed: ") + cudaGetErrorString(err));
        }

        // Create PyTorch tensor from device pointer
        // auto options = torch::TensorOptions()
        //     .dtype(dtype)
        //     .device(torch::kCUDA, device);
        
        // // Create tensor without copying data
        // auto tensor = torch::from_blob(offset_ptr, shape, 
        //     // Empty deleter since we'll handle IPC handle closing separately
        //     [](void*) {}, 
        //     options);

        auto options = torch::TensorOptions().dtype(dtype).
        device(torch::kCUDA, device);

        // Create an empty tensor with the specified shape and dtype
        auto tensor = torch::empty(shape, options);

        // Copy the data from the device pointer to the newly allocated tensor
        cudaMemcpy(tensor.data_ptr(), offset_ptr, tensor.nbytes(), 
        cudaMemcpyDeviceToDevice);


        std::cout << "tensor.data_ptr: " << tensor.data_ptr() << std::endl;

        if (make_contiguous) {
            return tensor.clone();
        } else {
            return tensor;
        }

    }

    // Function to export an IPC handle from a PyTorch tensor
    py::bytes export_ipc_handle_pybind(torch::Tensor tensor, bool ensure_contiguous) {
        if (!tensor.is_cuda()) {
            throw std::runtime_error("Tensor must be on a CUDA device");
        }
        
        // Make tensor contiguous if requested
        if (ensure_contiguous && !tensor.is_contiguous()) {
            tensor = tensor.contiguous();
        }
        
        // Synchronize device to ensure memory is ready
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaDeviceSynchronize failed: ") + cudaGetErrorString(err));
        }

        void* dev_ptr = tensor.data_ptr();
        std::cout << "dev_ptr: " << dev_ptr << std::endl;
        cudaIpcMemHandle_t handle;
        
        err = cudaIpcGetMemHandle(&handle, dev_ptr);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaIpcGetMemHandle failed: ") + cudaGetErrorString(err));
        }

        // Convert handle to bytes
        return py::bytes(reinterpret_cast<const char*>(&handle), sizeof(cudaIpcMemHandle_t));
    }

    py::bytes merge_tensors_and_export_ipc_handle(const std::vector<torch::Tensor>& tensors, int device) {
        // Step 1: Calculate the total size needed for all tensors
        size_t total_size = 0;
        size_t max_dtype_size = 0;

        for (const auto& tensor : tensors) {
            size_t dtype_size = tensor.element_size();
            total_size += tensor.numel() * dtype_size;
            max_dtype_size = std::max(max_dtype_size, dtype_size);
        }

        // Step 2: Manually allocate memory using CUDA
        void* dev_ptr = nullptr;
        cudaError_t err = cudaMalloc(&dev_ptr, total_size);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA memory allocation failed: " + std::string(cudaGetErrorString(err)));
        }

        // Step 3: Copy data from each tensor to the allocated memory
        size_t current_offset = 0;
        
        for (const auto& tensor : tensors) {
            auto flattened_tensor = tensor.flatten();
            size_t tensor_size = flattened_tensor.numel() * flattened_tensor.element_size();
            
            err = cudaMemcpy(static_cast<char*>(dev_ptr) + current_offset, 
                           flattened_tensor.data_ptr(),
                           tensor_size, 
                           cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess) {
                cudaFree(dev_ptr);
                throw std::runtime_error("CUDA memcpy failed: " + std::string(cudaGetErrorString(err)));
            }

            current_offset += tensor_size;
        }

        // Step 4: Export the IPC handle
        cudaIpcMemHandle_t handle;
        err = cudaIpcGetMemHandle(&handle, dev_ptr);
        if (err != cudaSuccess) {
            cudaFree(dev_ptr);
            throw std::runtime_error("cudaIpcGetMemHandle failed: " + std::string(cudaGetErrorString(err)));
        }

        // Convert the handle to bytes and return
        return py::bytes(reinterpret_cast<const char*>(&handle), sizeof(cudaIpcMemHandle_t));
    }

    void close_ipc_handle_pybind(void* dev_ptr) {
        if (dev_ptr != nullptr) {
            cudaError_t err = cudaIpcCloseMemHandle(dev_ptr);
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    std::string("cudaIpcCloseMemHandle failed: ") + cudaGetErrorString(err));
            }
    }
}

    PYBIND11_MODULE(ipc_tensor_pybind, m) {
        m.doc() = "PyBind11 module for CUDA IPC tensor handling";
        
        m.def("open_ipc_handle", &open_ipc_handle_pybind, 
            "Open an IPC handle and return a device pointer",
            py::arg("handle_bytes"), py::arg("device"));
        
        m.def("create_tensor_from_ptr", &create_tensor_from_ptr,
            "Create a tensor from a device pointer",
            py::arg("dev_ptr"), py::arg("offset_bytes"),
            py::arg("shape"), py::arg("dtype"),
            py::arg("device"), py::arg("make_contiguous") = false);
        
        m.def("merge_tensors_and_export_ipc_handle", &merge_tensors_and_export_ipc_handle,
             "Merge multiple tensors and export the IPC handle",
            py::arg("tensors"), py::arg("device"));

        m.def("close_ipc_handle", &close_ipc_handle_pybind,
            "Close an IPC handle",
            py::arg("dev_ptr"));

        m.def("export_ipc_handle", &export_ipc_handle_pybind,
            "Export an IPC handle from a PyTorch tensor",
            py::arg("tensor"), py::arg("ensure_contiguous") = true);
    }