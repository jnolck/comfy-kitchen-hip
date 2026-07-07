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

#include "dtype_dispatch.h"
#include "float_utils.h"
#include "utils.h"
#include <hip/hip_runtime.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace comfy {

constexpr int kQMaxKernelThreads = 128;
constexpr int kE4M3Alignment = 16;

namespace {

__forceinline__ __device__ float clamp(float val, float min, float max) {
  return fminf(max, fmaxf(min, val));
}

//
__forceinline__ __device__ uint8_t encode_std_e4m3(float val) {
  constexpr float kMax = 448.0f;
  val = fminf(fmaxf(val, -kMax), kMax);

  uint8_t sign = (val < 0.0f) ? 0x80 : 0x00;
  float abs_val = fabsf(val);

  if (abs_val == 0.0f)
    return 0x00;

  int exp;
  float sig = frexpf(abs_val, &exp);
  int exp_bits =
      exp +
      6; // bias=7: frexp gives sig*2^exp, we need 2^(exp_bits-7)*(1+mant/8)

  if (exp_bits <= 0) {
    // Subnormal: 2^-6 * mant/8 = abs_val
    int mant = (int)roundf(abs_val * 64.0f * 8.0f); // / (2^-6/8) = * 512
    if (mant > 7)
      mant = 7;
    if (mant == 0)
      return 0x00;
    return sign | mant;
  }

  if (exp_bits >= 15) {
    return sign | (15 << 3) | 6; // max: 0x7E or 0xFE
  }

  // Normal: abs_val = (1 + mant/8) * 2^(exp_bits - 7)
  // sig * 2^exp = (1 + mant/8) * 2^(exp_bits - 7)
  // sig * 2 = 1 + mant/8  →  mant = (sig * 2 - 1) * 8
  int mant = (int)roundf((sig * 2.0f - 1.0f) * 8.0f + 1e-7f);
  if (mant == 8) {
    mant = 0;
    exp_bits++;
    if (exp_bits >= 15)
      return sign | (15 << 3) | 6;
  }

  return sign | (exp_bits << 3) | mant;
}

template <typename InputType, typename OutputType>
__global__ void
quantize_fp8_tensor_kernel(const InputType *src, OutputType *dst,
                           const float *scale_f, const uint32_t size) {

  constexpr float kFP8Max = FP8LimitsTrait<OutputType>::max;
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  idx *= kE4M3Alignment;
  if (idx >= size) {
    return;
  }
  constexpr int n_src_load = sizeof(float4) / sizeof(InputType); // 8
  constexpr int n_loads = kE4M3Alignment / n_src_load;           // 2

  union {
    float4 f4;
    OutputType f8[kE4M3Alignment];
  } f4_e4m3;

#pragma unroll
  for (int i = 0; i < n_loads; i++) {
    float4 _src_f4 =
        *reinterpret_cast<const float4 *>(src + idx + i * n_src_load);
    InputType *_src_ptr = reinterpret_cast<InputType *>(&_src_f4);
#pragma unroll
    for (int j = 0; j < n_src_load; j++) {
      float scaled_val = static_cast<float>(_src_ptr[j]) / *scale_f;
      scaled_val = clamp(scaled_val, -kFP8Max, kFP8Max);
      // f4_e4m3.f8[i * n_src_load + j] = static_cast<OutputType>(scaled_val);
      uint8_t fp8_byte = encode_std_e4m3(scaled_val);
      reinterpret_cast<uint8_t *>(&f4_e4m3.f8)[i * n_src_load + j] = fp8_byte;
    }
  }
  *reinterpret_cast<float4 *>(dst + idx) = f4_e4m3.f4;
}

template <typename InputType, typename OutputType>
__global__ void
dequantize_fp8_tensor_kernel(const InputType *src, OutputType *dst,
                             const float *scale_f, const uint32_t size) {

  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr uint32_t n_src_load = sizeof(float2) / sizeof(InputType); // 8

  idx *= n_src_load;
  if (idx >= size) {
    return;
  }

  const uint8_t *bytes = reinterpret_cast<const uint8_t *>(src + idx);
  float vals[8];

  // Decode standard E4M3 (bias=7): 1 sign, 4 exp, 3 mant
#pragma unroll
  for (int i = 0; i < n_src_load; i++) {
    uint8_t b = bytes[i];
    int sign = (b >> 7) & 1;
    int exp = (b >> 3) & 0x0F;
    int mant = b & 0x07;

    float val;
    if (exp == 0) {
      val = (mant / 8.0f) * 0.015625f; // 2^-6
    } else {
      val = (1.0f + mant / 8.0f) * exp2f((float)(exp - 7)); // bias=7
    }
    vals[i] = sign ? -val : val;
  }
  // For 16-bit output types (FP16/BF16), use vectorized float4 store (8 values
  // = 16 bytes) For 32-bit output types (FP32), use two float4 stores (8 values
  // = 32 bytes)
  if constexpr (sizeof(OutputType) == 2) {
    union {
      float4 f4;
      OutputType f16[n_src_load];
    } f4_f16;

#pragma unroll
    for (int i = 0; i < n_src_load; i++) {
      f4_f16.f16[i] = static_cast<OutputType>((vals[i]) * *scale_f);
    }

    *reinterpret_cast<float4 *>(dst + idx) = f4_f16.f4;
  } else {
    // FP32 output: store 8 float values as 2 float4
    union {
      float4 f4[2];
      float f32[n_src_load];
    } f4_f32;

#pragma unroll
    for (int i = 0; i < n_src_load; i++) {
      f4_f32.f32[i] = static_cast<float>(vals[i]) * *scale_f;
    }

    reinterpret_cast<float4 *>(dst + idx)[0] = f4_f32.f4[0];
    reinterpret_cast<float4 *>(dst + idx)[1] = f4_f32.f4[1];
  }
}

template <typename InputType, typename OutputType>
__global__ void stochastic_round_fp8_kernel(uint8_t *rng_and_dst,
                                            const InputType *src,
                                            const uint32_t size) {

  constexpr float kFP8Max = FP8LimitsTrait<OutputType>::max;
  constexpr int kExponentBits =
      std::is_same_v<OutputType, __hip_fp8_e4m3_fnuz> ? 4 : 5;
  constexpr int kMantissaBits =
      std::is_same_v<OutputType, __hip_fp8_e4m3_fnuz> ? 3 : 2;
  constexpr int kExponentBias =
      std::is_same_v<OutputType, __hip_fp8_e4m3_fnuz> ? 7 : 15;
  constexpr float kMantissaLevels = static_cast<float>(1 << kMantissaBits);
  const float subnormal_mantissa_scale =
      exp2f(-kExponentBias + 1 - kMantissaBits);
  const float subnormal_value_scale = exp2f(-kExponentBias + 1);

  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }

  const float value = __half2float(__float2half(static_cast<float>(src[idx])));
  const float abs_value = fabsf(value);
  const float sign = value < 0.0f ? -1.0f : (value > 0.0f ? 1.0f : 0.0f);

  float exponent = floorf(log2f(abs_value)) + kExponentBias;
  exponent =
      clamp(exponent, 0.0f, static_cast<float>((1 << kExponentBits) - 1));
  const bool normal = exponent != 0.0f;
  const float exponent_scale = exp2f(exponent - kExponentBias);

  const float normal_mantissa =
      (abs_value / exponent_scale - 1.0f) * kMantissaLevels;
  const float subnormal_mantissa = abs_value / subnormal_mantissa_scale;
  const float random = static_cast<float>(rng_and_dst[idx]) * (1.0f / 256.0f);
  const float mantissa_float =
      floorf((normal ? normal_mantissa : subnormal_mantissa) + random);
  const int mantissa_int = (int)mantissa_float;
  const float mantissa = mantissa_float / kMantissaLevels;
  const float rounded = sign * (normal ? exponent_scale * (1.0f + mantissa)
                                       : subnormal_value_scale * mantissa);
  // union {
  //   OutputType fp8;
  //   uint8_t u8;
  // } out;
  // out.fp8 = static_cast<OutputType>(clamp(rounded, -kFP8Max, kFP8Max));
  // rng_and_dst[idx] = out.u8;
  float clamped = clamp(rounded, -kFP8Max, kFP8Max);
  uint8_t sign_bit = (clamped < 0.0f) ? 0x80 : 0x00;

  if (fabsf(clamped) == 0.0f) {
    rng_and_dst[idx] = 0x00;
  } else {
    int exp_bits = (int)exponent;
    int mant_bits = mantissa_int;

    // Handle mantissa carry (when random pushes mantissa to kMantissaLevels)
    if (mant_bits >= (int)kMantissaLevels) {
      mant_bits = 0;
      exp_bits++;
    }

    if (exp_bits > 15)
      exp_bits = 15;
    rng_and_dst[idx] = sign_bit | ((exp_bits & 0x0F) << 3) | (mant_bits & 0x07);
  }
}

} // anonymous namespace

} // namespace comfy

// C interface for DLPack bindings
extern "C" {

void launch_quantize_fp8_kernel(const void *input, void *output,
                                const void *scale, int64_t numel,
                                int input_dtype_code, int output_dtype_code,
                                hipStream_t stream) {

  if (numel == 0) {
    return;
  }

  const float *scale_f = static_cast<const float *>(scale);

  constexpr int vals_per_thread = comfy::kE4M3Alignment;
  constexpr int vals_per_block = vals_per_thread * comfy::kQMaxKernelThreads;
  const int blocks =
      static_cast<int>((numel + vals_per_block - 1) / vals_per_block);

  // Dispatch based on input and output dtypes
  // Input dtype codes: 0=float32, 1=float16, 2=bfloat16
  // Output dtype codes: 5=float8_e4m3fn, 6=float8_e5m2

  DISPATCH_INPUT_FP8_OUTPUT_DTYPES(
      input_dtype_code, output_dtype_code, InputType, OutputType, [&] {
        comfy::quantize_fp8_tensor_kernel<InputType, OutputType>
            <<<blocks, comfy::kQMaxKernelThreads, 0, stream>>>(
                static_cast<const InputType *>(input),
                static_cast<OutputType *>(output), scale_f, numel);
      });

  // Check for kernel launch errors
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    throw std::runtime_error(std::string("CUDA kernel launch failed: ") +
                             hipGetErrorString(err));
  }
}

void launch_dequantize_fp8_kernel(const void *input, void *output,
                                  const void *scale, int64_t numel,
                                  int input_dtype_code, int output_dtype_code,
                                  hipStream_t stream) {

  if (numel == 0) {
    return;
  }

  const float *scale_f = static_cast<const float *>(scale);

  constexpr int vals_per_thread = 8;
  constexpr int vals_per_block = vals_per_thread * comfy::kQMaxKernelThreads;
  const int blocks =
      static_cast<int>((numel + vals_per_block - 1) / vals_per_block);

  // Dispatch based on input and output dtypes
  // Input dtype codes: 5=float8_e4m3fn, 6=float8_e5m2
  // Output dtype codes: 0=float32, 1=float16, 2=bfloat16
  DISPATCH_FP8_INPUT_FP_OUTPUT_DTYPES(
      input_dtype_code, output_dtype_code, InputType, OutputType, [&] {
        comfy::dequantize_fp8_tensor_kernel<InputType, OutputType>
            <<<blocks, comfy::kQMaxKernelThreads, 0, stream>>>(
                static_cast<const InputType *>(input),
                static_cast<OutputType *>(output), scale_f, numel);
      });

  // Check for kernel launch errors
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    throw std::runtime_error(std::string("CUDA kernel launch failed: ") +
                             hipGetErrorString(err));
  }
}

void launch_stochastic_round_fp8_kernel(void *rng_and_output, const void *input,
                                        int64_t numel, int rng_dtype_code,
                                        int input_dtype_code,
                                        int output_dtype_code,
                                        hipStream_t stream) {

  if (numel == 0) {
    return;
  }

  if (rng_dtype_code != 3) {
    throw std::runtime_error("stochastic_round_fp8 requires uint8 RNG storage");
  }

  constexpr int vals_per_thread = 1;
  constexpr int vals_per_block = vals_per_thread * comfy::kQMaxKernelThreads;
  const int blocks =
      static_cast<int>((numel + vals_per_block - 1) / vals_per_block);

  DISPATCH_INPUT_FP8_OUTPUT_DTYPES(
      input_dtype_code, output_dtype_code, InputType, OutputType, [&] {
        comfy::stochastic_round_fp8_kernel<InputType, OutputType>
            <<<blocks, comfy::kQMaxKernelThreads, 0, stream>>>(
                static_cast<uint8_t *>(rng_and_output),
                static_cast<const InputType *>(input), numel);
      });

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    throw std::runtime_error(std::string("CUDA kernel launch failed: ") +
                             hipGetErrorString(err));
  }
}

} // extern "C"
