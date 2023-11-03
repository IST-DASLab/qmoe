# QMoE

This repository contains the full code of the paper [QMoE: Practical Sub-1-Bit Compression of Trillion-Parameter Models](https://arxiv.org/abs/2310.16795).

It is organized as follows:

* `datautils.py`: utilities for dataset loading
* `gptq.py`: robust batch-implementation of GPTQ
* `quant.py`: quantization utilities
* `sub1.py`: efficient inference of compressed models 
* `sub1_cuda_kernel.cu`: CUDA kernels
* `switch.py`: the efficient QMoE compression framework
* `test.py`: per-layer benchmarks and ideal compression rates 

## Dependencies

The project was developed with:

* `torch==2.0.0+cu117`
* `transformers==4.28.0`
* `datasets==2.10.1` 
* CUDA 11.4 GPU drivers

CUDA kernels for compressed storage and inference can be installed via:

```
python setup_cuda.py install
```

## Usage

Now follows a list of sample commands for running different experiments.

```
# BF16 baseline eval on C4 
CUDA_VISIBLE_DEVICES=0 python switch.py google/switch-base-128 
# BF16 baseline eval on additional datasets 
CUDA_VISIBLE_DEVICES=0 python switch.py google/switch-base-128 --detaileval
# ternary round to nearest baseline 
CUDA_VISIBLE_DEVICES=0 python switch.py google/switch-base-128 --wbits 1.5 --nearest 

# ternary compression with QMoE, saving the compressed model for later inference
CUDA_VISIBLE_DEVICES=0 python switch.py google/switch-base-128 --wbits 1.5 --trainsamples 10000 --save PATH_TO_COMP_MODEL
# 2-bit compression with QMoE
CUDA_VISIBLE_DEVICES=0 python switch.py google/switch-base-128 --wbits 2 --trainsamples 10000

# test kernels and compute ideal compression rates 
CUDA_VISIBLE_DEVICES=0 python test.py
# run per-layer benchmarks
CUDA_VISIBLE_DEVICES=0 python test.py --benchmark

# run eval of stored compressed model
CUDA_VISIBLE_DEVICES=0 python sub1.py PATH_TO_COMP_MODEL --valsamples 128 
# run end-to-end benchmark
CUDA_VISIBLE_DEVICES=0 python sub1.py PATH_TO_COMP_MODEL --gentokens 128
# run simulated end-to-end benchmark for BF16
CUDA_VISIBLE_DEVICES=0 python sub1.py PATH_TO_COMP_MODEL --gentokens 128 --simul
```

In general, you can pass `google/switch-large-128` and `google/switch-c-2048` to run on large-128 and c-2048, respectively. We note that other SwitchTransformer models than those 3 may not work out-of-the-box due to Hugging Face bugs.

Always specify `CUDA_VISIBLE_DEVICES` since some commands, like `sub1.py`, will otherwise attempt to use all available GPUs.

## Compressed Models

Our models in compressed custom QMoE format are available on Hugging Face: [base-128](https://huggingface.co/ISTA-DASLab/switch-base-128_qmoe), [large-128](https://huggingface.co/ISTA-DASLab/switch-large-128_qmoe) and [c-2048](https://huggingface.co/ISTA-DASLab/switch-c-2048_qmoe). To use them, clone the repository and then simply pass their path to `sub1.py`.

## Cite

If you found this work useful, please consider citing: 

```
@article{frantar-qmoe,
  title={{QMoE}: Practical Sub-1-Bit Compression of Trillion-Parameter Models}
  author={Elias Frantar and Dan Alistarh},
  year={2023},
  journal={arXiv preprint, arxiv:2310.16795}
}
```
