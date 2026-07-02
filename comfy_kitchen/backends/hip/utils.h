/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef COMFY_UTILS_CUH_
#define COMFY_UTILS_CUH_

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
#if CUDA_VERSION >= 12080
#include <hip/hip_fp4.h>
#endif

#include <mutex>
#include <stdexcept>
#include <type_traits>

namespace comfy {

////////////////////////////////////////////////////////////////////////////////

constexpr int kThreadsPerWarp = 32;

////////////////////////////////////////////////////////////////////////////////
// NOTE: This file previously contained ATen-dependent type traits and macros.
// Those have been removed to eliminate all PyTorch C++ dependencies.
// The kernels that used these utilities are not compiled in pure DLPack mode.
////////////////////////////////////////////////////////////////////////////////

/* Use CUDA const memory to store scalar 1 and 0 for hipblas usage
 */
__device__ __constant__ float one_device;
__device__ __constant__ float zero_device;

// Helper macro for CUDA error checking (replaces C10_CUDA_CHECK)
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    hipError_t err = call;                                                     \
    if (err != hipSuccess) {                                                   \
      throw std::runtime_error(std::string("CUDA error: ") +                   \
                               hipGetErrorString(err));                        \
    }                                                                          \
  } while (0)

inline float *GetScalarOne() {
  static std::once_flag init_flag;
  std::call_once(init_flag, []() {
    float one = 1.0f;
    CUDA_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(one_device), &one, sizeof(float)));
  });
  // return address by cudaGetSymbolAddress
  float *dev_ptr;
  CUDA_CHECK(hipGetSymbolAddress((void **)&dev_ptr, HIP_SYMBOL(one_device)));
  return dev_ptr;
}

inline float *GetScalarZero() {
  static std::once_flag init_flag;
  std::call_once(init_flag, []() {
    float zero = 0.0f;
    CUDA_CHECK(
        hipMemcpyToSymbol(HIP_SYMBOL(zero_device), &zero, sizeof(float)));
  });
  // return address by cudaGetSymbolAddress
  float *dev_ptr;
  CUDA_CHECK(hipGetSymbolAddress((void **)&dev_ptr, HIP_SYMBOL(zero_device)));
  return dev_ptr;
}

} // namespace comfy

#endif // COMFY_UTILS_CUH_
