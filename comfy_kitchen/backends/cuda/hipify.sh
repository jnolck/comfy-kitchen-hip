#!/bin/bash
CUDA_PATH="/usr/local/cuda-12.8"
#TORCH_PREFIX=$(python -c "import torch; print(torch.__path__[0] + '/include')")
#TORCH_CPP=$(python3 -c "import torch.utils.cpp_extension; print(torch.utils.cpp_extension.include_paths()[1])")
PYTHON_INCLUDE=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
#VENV_ROOT="/home/jnolck/Downloads/software/python-venv/comfy/ComfyUI-0.10.0/.venv"
NB_INC="/home/jnolck/Documents/src/ai/.venv/lib/python3.12/site-packages/nanobind/include/"
CLANG_RESOURCE_DIR=$(clang -print-resource-dir)

hipify-clang -v ops/int8_linear.cu \
        --cuda-gpu-arch=sm_80 \
        --cuda-path="$CUDA_PATH" \
        --clang-resource-directory="$CLANG_RESOURCE_DIR" \
        -- \
        -x cuda \
        --cuda-host-only \
        -std=c++17 \
        -U__CUDA_ARCH__ \
        -I"$PYTHON_INCLUDE" \
        -I"$NB_INC" \
        -I"$CUDA_PATH/include" \
        -I.
