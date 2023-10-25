from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
  name='sub1_cuda',
  ext_modules=[cpp_extension.CUDAExtension(
    'sub1_cuda', ['sub1_cuda.cpp', 'sub1_cuda_kernel.cu']
  )],
  cmdclass={'build_ext': cpp_extension.BuildExtension}
)
