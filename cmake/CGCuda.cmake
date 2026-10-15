if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "CG_ENABLE_CUDA currently supports native Linux builds")
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "75" CACHE STRING "Explicit CUDA target architectures")
endif()
if(NOT CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES MATCHES "native|all")
    message(FATAL_ERROR "Set an explicit numeric CMAKE_CUDA_ARCHITECTURES list (default: 75)")
endif()
if(NOT CMAKE_CUDA_COMPILER AND CUDAToolkit_ROOT)
    set(CMAKE_CUDA_COMPILER "${CUDAToolkit_ROOT}/bin/nvcc" CACHE FILEPATH "CUDA compiler")
endif()
enable_language(CUDA)
find_package(CUDAToolkit 12.8 REQUIRED)
if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.0")
    message(FATAL_ERROR "CG_ENABLE_CUDA requires CUDA Toolkit >=12.8 and <13 to match cu12 runtimes")
endif()
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "12.8" OR CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "13.0")
    message(FATAL_ERROR "The selected nvcc compiler must also be CUDA >=12.8 and <13")
endif()
add_library(cg_cuda STATIC
    cpp/cg/storage/cuda/vector.cu
    cpp/cg/storage/cuda/allocation.cu
    cpp/cg/storage/cuda/detail/validation.cu
    cpp/cg/matrix/cuda/dense_matrix.cu
    cpp/cg/matrix/cuda/csr_matrix.cu
    cpp/cg/matrix/cuda/detail/validation.cu
    cpp/cg/solver/cuda/workspace.cu
    cpp/cg/solver/cuda/conjugate_gradient.cu
    cpp/cg/solver/cuda/device_execution.cu
    cpp/cg/solver/cuda/prepared_solver.cu
    cpp/cg/preconditioner/cuda/jacobi.cu
    cpp/cg/preconditioner/cuda/kernels.cu
    cpp/cg/backend/cuda/backend.cu
    cpp/cg/backend/cuda/execution_resources.cu
    cpp/cg/backend/cuda/vector.cu
    cpp/cg/backend/cuda/reduction.cu
    cpp/cg/backend/cuda/dense.cu
    cpp/cg/backend/cuda/csr.cu
    cpp/cg/runtime/cuda/context.cu
)
add_library(cg::cuda ALIAS cg_cuda)
target_compile_features(cg_cuda PUBLIC cxx_std_20)
set_target_properties(cg_cuda PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_RUNTIME_LIBRARY Shared
    CXX_VISIBILITY_PRESET hidden
    CUDA_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)
target_link_libraries(cg_cuda PUBLIC cg_core CUDA::cudart CUDA::cublas)
set(CG_CUDA_PUBLIC_HEADERS
    cuda.hpp
    runtime/cuda/context.hpp
    runtime/cuda/host_device.hpp
    backend/cuda/dense_kernel.hpp
    storage/cuda/vector.hpp
    storage/cuda/device_memory.hpp
    storage/cuda/vector_view.hpp
    matrix/cuda/csr_matrix.hpp
    matrix/cuda/csr_matrix_view.hpp
    matrix/cuda/dense_matrix.hpp
    matrix/cuda/dense_matrix_view.hpp
    solver/cuda/workspace.hpp
    solver/cuda/conjugate_gradient.hpp
    solver/cuda/prepared_solver.hpp
    preconditioner/cuda/jacobi.hpp
)
install(TARGETS cg_cuda ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT development)
foreach(header IN LISTS CG_CUDA_PUBLIC_HEADERS)
    get_filename_component(directory "${header}" DIRECTORY)
    install(FILES "cpp/cg/${header}"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/cg/${directory}" COMPONENT development)
endforeach()
