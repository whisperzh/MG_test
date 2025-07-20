from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
    name='ipc_tensor_pybind',
    ext_modules=[
        cpp_extension.CUDAExtension(
            name='ipc_tensor_pybind',
            sources=['ipc_tensor_pybind.cpp'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3']
            }
        )
    ],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension
    }
)