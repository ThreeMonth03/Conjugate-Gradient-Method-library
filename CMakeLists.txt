cmake_minimum_required(VERSION 3.24)
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml" CG_VERSION_LINE REGEX "^version = ")
string(REGEX REPLACE "^version = \"([^\"]+)\"$" "\\1" CG_VERSION "${CG_VERSION_LINE}")
project(cgpy VERSION "${CG_VERSION}" LANGUAGES CXX)

include(CTest)
include(GNUInstallDirs)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(CG_BUILD_PYTHON "Build the Python extension modules" ON)
option(CG_ENABLE_OPENMP "Enable OpenMP CPU operations" OFF)
option(CG_ENABLE_BLAS "Build the private sequential OpenBLAS dense backend" OFF)
option(CG_ENABLE_CUDA "Build the optional CUDA backend and Python extension" OFF)
option(CG_ENABLE_SANITIZERS "Enable address and undefined-behavior sanitizers" OFF)
option(CG_ENABLE_CLANG_TIDY "Run clang-tidy while compiling project sources" OFF)

find_package(Threads REQUIRED)
add_library(cg_core STATIC
    cpp/cg/storage/vector.cpp
    cpp/cg/storage/validation.cpp
    cpp/cg/matrix/dense_matrix.cpp
    cpp/cg/matrix/csr_matrix.cpp
    cpp/cg/solver/workspace.cpp
    cpp/cg/solver/conjugate_gradient.cpp
    cpp/cg/preconditioner/jacobi.cpp
    cpp/cg/backend/cpu/backend.cpp
    cpp/cg/backend/cpu/csr.cpp
    cpp/cg/backend/cpu/blas.cpp
)
add_library(cg::core ALIAS cg_core)
target_compile_features(cg_core PUBLIC cxx_std_17)
set_target_properties(cg_core PROPERTIES POSITION_INDEPENDENT_CODE ON CXX_EXTENSIONS OFF)
target_include_directories(cg_core PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpp>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)
target_link_libraries(cg_core PUBLIC Threads::Threads)
set(CG_PUBLIC_HEADERS
    cg.hpp
    storage/vector.hpp
    storage/vector_view.hpp
    matrix/dense_matrix.hpp
    matrix/dense_matrix_view.hpp
    matrix/csr_matrix.hpp
    solver/workspace.hpp
    solver/options.hpp
    solver/conjugate_gradient.hpp
    preconditioner/jacobi.hpp
    backend/cpu/backend.hpp
)
install(TARGETS cg_core ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT development)
foreach(header IN LISTS CG_PUBLIC_HEADERS)
    get_filename_component(directory "${header}" DIRECTORY)
    install(FILES "cpp/cg/${header}"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/cg/${directory}" COMPONENT development)
endforeach()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cg_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(CG_ENABLE_BLAS)
    include(cmake/CGOpenBLAS.cmake)
    target_link_libraries(cg_core PRIVATE cg_private_blas)
    target_compile_definitions(cg_core PUBLIC CG_ENABLE_BLAS=1)
    target_compile_definitions(cg_core PRIVATE
        cblas_dgemv=cgpy_cblas_dgemv
        CG_BLAS_PROVIDER_INFO="${CG_BLAS_PROVIDER_INFO}"
    )
endif()
if(CG_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
    target_link_libraries(cg_core PUBLIC OpenMP::OpenMP_CXX)
endif()
if(CG_ENABLE_SANITIZERS)
    if(CG_ENABLE_CUDA OR CG_BUILD_PYTHON)
        message(FATAL_ERROR "Sanitizers require CG_ENABLE_CUDA=OFF and CG_BUILD_PYTHON=OFF")
    endif()
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        message(FATAL_ERROR "CG_ENABLE_SANITIZERS requires GCC or Clang")
    endif()
    target_compile_options(cg_core PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(cg_core PUBLIC -fsanitize=address,undefined)
endif()
if(CG_ENABLE_CLANG_TIDY)
    find_program(CG_CLANG_TIDY NAMES clang-tidy REQUIRED)
    set_target_properties(cg_core PROPERTIES CXX_CLANG_TIDY "${CG_CLANG_TIDY}")
endif()

if(CG_ENABLE_CUDA)
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
        cpp/cg/storage/cuda/validation.cu
        cpp/cg/matrix/cuda/dense_matrix.cu
        cpp/cg/matrix/cuda/csr_matrix.cu
        cpp/cg/solver/cuda/workspace.cu
        cpp/cg/solver/cuda/conjugate_gradient.cu
        cpp/cg/preconditioner/cuda/jacobi.cu
        cpp/cg/backend/cuda/backend.cu
        cpp/cg/backend/cuda/context.cu
    )
    add_library(cg::cuda ALIAS cg_cuda)
    target_compile_features(cg_cuda PUBLIC cxx_std_17)
    set_target_properties(cg_cuda PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_RUNTIME_LIBRARY Shared
        CXX_VISIBILITY_PRESET hidden
        CUDA_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
    )
    target_link_libraries(cg_cuda PUBLIC cg_core CUDA::cudart CUDA::cublas)
    set(CG_CUDA_PUBLIC_HEADERS
        cuda.hpp
        backend/cuda/context.hpp
        backend/cuda/backend.hpp
        storage/cuda/vector.hpp
        storage/cuda/vector_view.hpp
        matrix/cuda/csr_matrix.hpp
        matrix/cuda/csr_matrix_view.hpp
        matrix/cuda/dense_matrix.hpp
        matrix/cuda/dense_matrix_view.hpp
        solver/cuda/workspace.hpp
        solver/cuda/conjugate_gradient.hpp
        preconditioner/cuda/jacobi.hpp
    )
    install(TARGETS cg_cuda ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT development)
    foreach(header IN LISTS CG_CUDA_PUBLIC_HEADERS)
        get_filename_component(directory "${header}" DIRECTORY)
        install(FILES "cpp/cg/${header}"
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/cg/${directory}" COMPONENT development)
    endforeach()
endif()

if(CG_BUILD_PYTHON)
    set(PYBIND11_FINDPYTHON ON)
    set(Python_FIND_IMPLEMENTATIONS CPython)
    find_package(Python 3.12...<3.13 REQUIRED COMPONENTS Interpreter Development.Module)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE CG_PYBIND11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY
    )
    find_package(pybind11 3.0.1 CONFIG REQUIRED PATHS "${CG_PYBIND11_DIR}" NO_DEFAULT_PATH)
    # Stage the package beside its private extensions for in-tree execution.
    file(GLOB_RECURSE CG_PYTHON_SOURCES CONFIGURE_DEPENDS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/python"
        "${CMAKE_CURRENT_SOURCE_DIR}/python/cgpy/*.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/python/cgbench/*.py"
    )
    file(GLOB_RECURSE CG_STAGED_PYTHON RELATIVE "${CMAKE_CURRENT_BINARY_DIR}/python"
        "${CMAKE_CURRENT_BINARY_DIR}/python/*.py")
    foreach(source IN LISTS CG_STAGED_PYTHON)
        if(NOT source IN_LIST CG_PYTHON_SOURCES)
            file(REMOVE "${CMAKE_CURRENT_BINARY_DIR}/python/${source}")
        endif()
    endforeach()
    foreach(source IN LISTS CG_PYTHON_SOURCES)
        configure_file("python/${source}" "python/${source}" COPYONLY)
    endforeach()
    pybind11_add_module(_core MODULE
        cpp/cg/python/module.cpp
        cpp/cg/python/runtime.cpp
        cpp/cg/python/array_conversion.cpp
        cpp/cg/matrix/pymod/module.cpp
        cpp/cg/solver/pymod/module.cpp
    )
    target_link_libraries(_core PRIVATE cg_core)
    string(TOUPPER "${CMAKE_BUILD_TYPE}" CG_BUILD_CONFIG)
    target_compile_definitions(_core PRIVATE
        CG_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
        CG_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
        CG_BUILD_FLAGS="${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${CG_BUILD_CONFIG}}"
    )
    set_target_properties(_core PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/python/cgpy"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/python/cgpy"
    )
    install(TARGETS _core
        LIBRARY DESTINATION cgpy COMPONENT python
        RUNTIME DESTINATION cgpy COMPONENT python)
    if(CG_ENABLE_CUDA)
        pybind11_add_module(_cuda MODULE
            cpp/cg/python/cuda_module.cpp
            cpp/cg/python/cuda_runtime.cpp
            cpp/cg/python/array_conversion.cpp
            cpp/cg/python/cuda_array_conversion.cpp
            cpp/cg/storage/pymod/cuda.cpp
            cpp/cg/matrix/pymod/cuda.cpp
            cpp/cg/solver/pymod/cuda.cpp
        )
        target_link_libraries(_cuda PRIVATE cg_cuda)
        string(REPLACE ";" "," CG_CUDA_ARCHITECTURES_INFO "${CMAKE_CUDA_ARCHITECTURES}")
        string(TOUPPER "${CMAKE_BUILD_TYPE}" CG_CUDA_BUILD_CONFIG)
        set_target_properties(_cuda PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/python/cgpy"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/python/cgpy"
            INSTALL_RPATH ""
            INSTALL_RPATH_USE_LINK_PATH OFF
        )
        target_compile_definitions(_cuda PRIVATE
            CG_CUDA_TOOLKIT_VERSION="${CUDAToolkit_VERSION}"
            CG_CUDA_ARCHITECTURES="${CG_CUDA_ARCHITECTURES_INFO}"
            CG_CUDA_COMPILER="${CMAKE_CUDA_COMPILER_ID} ${CMAKE_CUDA_COMPILER_VERSION}"
            CG_CUDA_HOST_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
            CG_CUDA_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
            CG_CUDA_BUILD_FLAGS="${CMAKE_CUDA_FLAGS} ${CMAKE_CUDA_FLAGS_${CG_CUDA_BUILD_CONFIG}}"
        )
        install(TARGETS _cuda
            LIBRARY DESTINATION cgpy COMPONENT python
            RUNTIME DESTINATION cgpy COMPONENT python)
    endif()
endif()

if(BUILD_TESTING)
    include(FetchContent)
    set(FETCHCONTENT_BASE_DIR "${CMAKE_CURRENT_BINARY_DIR}/_deps" CACHE PATH "Local dependency directory")
    FetchContent_Declare(googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
        URL_HASH SHA256=65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    add_executable(cg_gtests
        tests/cpp/solver/test_core.cpp
        tests/cpp/matrix/test_csr.cpp
        tests/cpp/solver/test_pcg.cpp
        tests/cpp/backend/test_blas.cpp
    )
    target_link_libraries(cg_gtests PRIVATE cg_core GTest::gtest_main)
    include(GoogleTest)
    gtest_discover_tests(cg_gtests PROPERTIES LABELS cpp)
    add_custom_target(cg_native_tests DEPENDS cg_gtests)
    if(CG_ENABLE_CUDA)
        add_executable(cg_cuda_gtests tests/cpp/cuda/test_cuda.cpp)
        target_link_libraries(cg_cuda_gtests PRIVATE cg_cuda GTest::gtest_main)
        gtest_discover_tests(cg_cuda_gtests PROPERTIES LABELS "cpp;cuda")
        add_dependencies(cg_native_tests cg_cuda_gtests)
    endif()
    if(CG_BUILD_PYTHON)
        add_test(NAME pytest
            COMMAND "${Python_EXECUTABLE}" -m pytest "${CMAKE_CURRENT_SOURCE_DIR}/tests/python" -q
        )
        set_tests_properties(pytest PROPERTIES
            LABELS python
            ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python"
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        )
    endif()
endif()
