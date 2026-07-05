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
#pragma once
#include <hip/hip_runtime.h>

#include <hipblaslt.h>
#include <mutex>
#include <string>

// #ifdef _WIN32
// #include <windows.h>
// #else
#include <dlfcn.h>
// #endif

namespace comfy {
class HipblasLtRuntime {
public:
  // Function pointer types for cuBLASLt functions we need
  using hipblasLtCreate_t = hipblasStatus_t (*)(hipblasLtHandle_t *);
  using hipblasLtDestroy_t = hipblasStatus_t (*)(hipblasLtHandle_t);
  using hipblasLtMatmul_t = hipblasStatus_t (*)(
      hipblasLtHandle_t, hipblasLtMatmulDesc_t, const void *, const void *,
      hipblasLtMatrixLayout_t, const void *, hipblasLtMatrixLayout_t,
      const void *, const void *, hipblasLtMatrixLayout_t, void *,
      hipblasLtMatrixLayout_t, const hipblasLtMatmulAlgo_t *, void *, size_t,
      hipStream_t);
  using hipblasLtMatmulDescCreate_t = hipblasStatus_t (*)(
      hipblasLtMatmulDesc_t *, hipblasComputeType_t, hipDataType);
  using hipblasLtMatmulDescDestroy_t =
      hipblasStatus_t (*)(hipblasLtMatmulDesc_t);
  using hipblasLtMatmulDescSetAttribute_t = hipblasStatus_t (*)(
      hipblasLtMatmulDesc_t, hipblasLtMatmulDescAttributes_t, const void *,
      size_t);
  using hipblasLtMatrixLayoutCreate_t = hipblasStatus_t (*)(
      hipblasLtMatrixLayout_t *, hipDataType, uint64_t, uint64_t, int64_t);
  using hipblasLtMatrixLayoutDestroy_t =
      hipblasStatus_t (*)(hipblasLtMatrixLayout_t);
  using hipblasLtMatmulPreferenceCreate_t =
      hipblasStatus_t (*)(hipblasLtMatmulPreference_t *);
  using hipblasLtMatmulPreferenceDestroy_t =
      hipblasStatus_t (*)(hipblasLtMatmulPreference_t);
  using hipblasLtMatmulPreferenceSetAttribute_t = hipblasStatus_t (*)(
      hipblasLtMatmulPreference_t, hipblasLtMatmulPreferenceAttributes_t,
      const void *, size_t);
  using hipblasLtMatmulAlgoGetHeuristic_t = hipblasStatus_t (*)(
      hipblasLtHandle_t, hipblasLtMatmulDesc_t, hipblasLtMatrixLayout_t,
      hipblasLtMatrixLayout_t, hipblasLtMatrixLayout_t, hipblasLtMatrixLayout_t,
      hipblasLtMatmulPreference_t, int, hipblasLtMatmulHeuristicResult_t *,
      int *);

  static HipblasLtRuntime &instance() {
    static HipblasLtRuntime runtime;
    return runtime;
  }

  bool is_available() const { return available_; }
  const std::string &error_message() const { return error_message_; }

  // Function pointers - only valid if is_available() returns true
  hipblasLtCreate_t hipblasLtCreate = nullptr;
  hipblasLtDestroy_t hipblasLtDestroy = nullptr;
  hipblasLtMatmul_t hipblasLtMatmul = nullptr;
  hipblasLtMatmulDescCreate_t hipblasLtMatmulDescCreate = nullptr;
  hipblasLtMatmulDescDestroy_t hipblasLtMatmulDescDestroy = nullptr;
  hipblasLtMatmulDescSetAttribute_t hipblasLtMatmulDescSetAttribute = nullptr;
  hipblasLtMatrixLayoutCreate_t hipblasLtMatrixLayoutCreate = nullptr;
  hipblasLtMatrixLayoutDestroy_t hipblasLtMatrixLayoutDestroy = nullptr;
  hipblasLtMatmulPreferenceCreate_t hipblasLtMatmulPreferenceCreate = nullptr;
  hipblasLtMatmulPreferenceDestroy_t hipblasLtMatmulPreferenceDestroy = nullptr;
  hipblasLtMatmulPreferenceSetAttribute_t
      hipblasLtMatmulPreferenceSetAttribute = nullptr;
  hipblasLtMatmulAlgoGetHeuristic_t hipblasLtMatmulAlgoGetHeuristic = nullptr;

private:
  HipblasLtRuntime() { load(); }

  ~HipblasLtRuntime() { unload(); }

  // Delete copy/move
  HipblasLtRuntime(const HipblasLtRuntime &) = delete;
  HipblasLtRuntime &operator=(const HipblasLtRuntime &) = delete;

  void load() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (load_attempted_)
      return;
    load_attempted_ = true;

    // #ifdef _WIN32
    //     // Windows: cuBLAS 13.x (CUDA 13+)
    //     const char *lib_names[] = {"hipblasLt64_13.dll",
    //                                "hipblasLt64.dll", // Fallback (may
    //                                be 13.x) nullptr};
    //
    //     for (const char **name = lib_names; *name != nullptr; ++name) {
    //       handle_ = LoadLibraryA(*name);
    //       if (handle_)
    //         break;
    //     }
    //
    //     if (!handle_) {
    //       error_message_ = "cuBLASLt 13.x library not found (requires CUDA
    //       13+)"; return;
    //     }
    // #else
    // Linux: cuBLAS 13.x (CUDA 13+)
    const char *lib_names[] = {"libhipblaslt.so.1.2", "libhipblaslt.so.1",
                               "libhipblaslt.so", // Fallback (may be 13.x)
                               nullptr};

    for (const char **name = lib_names; *name != nullptr; ++name) {
      handle_ = dlopen(*name, RTLD_NOW | RTLD_GLOBAL);
      if (handle_)
        break;
    }

    if (!handle_) {
      error_message_ = std::string("HipBLASLt library not found: ") + dlerror();
      return;
    }
    // #endif

    // Load all required function pointers
    if (!load_symbols()) {
      unload();
      return;
    }

    available_ = true;
  }

  bool load_symbols() {
// #ifdef _WIN32
// #define LOAD_SYMBOL(name) \
//   name = reinterpret_cast<name##_t>( \
//       GetProcAddress(static_cast<HMODULE>(handle_), #name)); \
//   if (!name) { \
//     error_message_ = "Failed to load symbol: " #name; \
//     return false; \
//   }
// #else
#define LOAD_SYMBOL(name)                                                      \
  name = reinterpret_cast<name##_t>(dlsym(handle_, #name));                    \
  if (!name) {                                                                 \
    error_message_ =                                                           \
        std::string("Failed to load symbol: " #name ": ") + dlerror();         \
    return false;                                                              \
  }
    // #endif

    LOAD_SYMBOL(hipblasLtCreate);
    LOAD_SYMBOL(hipblasLtDestroy);
    LOAD_SYMBOL(hipblasLtMatmul);
    LOAD_SYMBOL(hipblasLtMatmulDescCreate);
    LOAD_SYMBOL(hipblasLtMatmulDescDestroy);
    LOAD_SYMBOL(hipblasLtMatmulDescSetAttribute);
    LOAD_SYMBOL(hipblasLtMatrixLayoutCreate);
    LOAD_SYMBOL(hipblasLtMatrixLayoutDestroy);
    LOAD_SYMBOL(hipblasLtMatmulPreferenceCreate);
    LOAD_SYMBOL(hipblasLtMatmulPreferenceDestroy);
    LOAD_SYMBOL(hipblasLtMatmulPreferenceSetAttribute);
    LOAD_SYMBOL(hipblasLtMatmulAlgoGetHeuristic);

#undef LOAD_SYMBOL
    return true;
  }

  void unload() {
    if (handle_) {
      // #ifdef _WIN32
      //       FreeLibrary(static_cast<HMODULE>(handle_));
      // #else
      dlclose(handle_);
      // #endif
      handle_ = nullptr;
    }
    available_ = false;

    // Clear function pointers
    hipblasLtCreate = nullptr;
    hipblasLtDestroy = nullptr;
    hipblasLtMatmul = nullptr;
    hipblasLtMatmulDescCreate = nullptr;
    hipblasLtMatmulDescDestroy = nullptr;
    hipblasLtMatmulDescSetAttribute = nullptr;
    hipblasLtMatrixLayoutCreate = nullptr;
    hipblasLtMatrixLayoutDestroy = nullptr;
    hipblasLtMatmulPreferenceCreate = nullptr;
    hipblasLtMatmulPreferenceDestroy = nullptr;
    hipblasLtMatmulPreferenceSetAttribute = nullptr;
    hipblasLtMatmulAlgoGetHeuristic = nullptr;
  }

  void *handle_ = nullptr;
  bool available_ = false;
  bool load_attempted_ = false;
  std::string error_message_;
  std::mutex mutex_;
};

} // namespace comfy
