/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <assert.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <stdexcept>

class CudaException : public std::runtime_error {
public:
    CudaException(const char* file, int line, cudaError_t error)
        : std::runtime_error(std::string("CUDA error (") + file + ":" + 
                            std::to_string(line) + "): " + 
                            cudaGetErrorString(error)) {}
};

#define CHECK_CUDA_THROW(call)                                    \
    do {                                                          \
        cudaError_t status_ = call;                              \
        if (status_ != cudaSuccess) {                            \
            throw CudaException(__FILE__, __LINE__, status_);    \
        }                                                         \
    } while(0)

#define CHECK_CUDA_KERNEL_LAUNCH_THROW() CHECK_CUDA_THROW(cudaGetLastError())


// #define CHECK_CUDA(call)                        \
//     do {                                                                                                  \
//         cudaError_t status_ = call;                                                                       \
//         if (status_ != cudaSuccess) {                                                                     \
//             fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(status_)); \
//             exit(1);                                                                                      \
//         }                                                                                                 \
//     } while(0)

// #define CHECK_CUDA_KERNEL_LAUNCH() CHECK_CUDA(cudaGetLastError())
