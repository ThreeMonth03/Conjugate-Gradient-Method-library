# Shared by native presets and isolated wheel builds. Explicit -D overrides
# take precedence over these project-local defaults.
set(CUDAToolkit_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.deps/cuda-12.8"
    CACHE PATH "CUDA Toolkit location")
set(CMAKE_CUDA_COMPILER "${CUDAToolkit_ROOT}/bin/nvcc"
    CACHE FILEPATH "CUDA compiler")
set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++-13"
    CACHE FILEPATH "CUDA host compiler")
set(CMAKE_CXX_COMPILER "${CMAKE_CUDA_HOST_COMPILER}"
    CACHE FILEPATH "C++ compiler for the CUDA build")
set(CMAKE_CUDA_ARCHITECTURES "75" CACHE STRING "Explicit CUDA architectures")
if(IS_ABSOLUTE "${CMAKE_CUDA_COMPILER}")
    file(REAL_PATH "${CUDAToolkit_ROOT}/bin/nvcc" _cg_expected_nvcc)
    file(REAL_PATH "${CMAKE_CUDA_COMPILER}" _cg_selected_nvcc)
    if(NOT _cg_selected_nvcc STREQUAL _cg_expected_nvcc)
        message(FATAL_ERROR
            "CUDAToolkit_ROOT does not match the selected nvcc. Use a fresh build directory or explicitly select its compiler.")
    endif()
endif()
list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES
    CUDAToolkit_ROOT CMAKE_CUDA_COMPILER CMAKE_CUDA_HOST_COMPILER CMAKE_CXX_COMPILER)
