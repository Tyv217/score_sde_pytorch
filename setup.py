from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='upfirdn2d',
    ext_modules=[
        CUDAExtension('upfirdn2d', [
            os.path.join(module_path, "upfirdn2d.cpp"),
            os.path.join(module_path, "upfirdn2d_kernel.cu"),
        ],
        extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']}),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)