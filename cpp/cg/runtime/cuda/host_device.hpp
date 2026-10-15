#pragma once

#if defined(__CUDACC__) || defined(__CUDA__)
#define CG_CUDA_HOST_DEVICE __host__ __device__
#else
#define CG_CUDA_HOST_DEVICE
#endif
