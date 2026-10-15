cmake_minimum_required(VERSION 3.24)
project(cgpy LANGUAGES CXX)

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
    cpp/cg/storage/detail/validation.cpp
    cpp/cg/matrix/dense_matrix.cpp
    cpp/cg/matrix/csr_matrix.cpp
    cpp/cg/solver/workspace.cpp
    cpp/cg/solver/conjugate_gradient.cpp
    cpp/cg/solver/prepared_solver.cpp
    cpp/cg/solver/solve_info.cpp
    cpp/cg/preconditioner/jacobi.cpp
    cpp/cg/backend/cpu/backend.cpp
    cpp/cg/backend/cpu/vector.cpp
    cpp/cg/backend/cpu/reduction.cpp
    cpp/cg/backend/cpu/dense.cpp
    cpp/cg/backend/cpu/csr.cpp
    cpp/cg/backend/cpu/blas.cpp
)
add_library(cg::core ALIAS cg_core)
target_compile_features(cg_core PUBLIC cxx_std_20)
set_target_properties(cg_core PROPERTIES POSITION_INDEPENDENT_CODE ON CXX_EXTENSIONS OFF)
target_include_directories(cg_core PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpp>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)
target_link_libraries(cg_core PUBLIC Threads::Threads)
set(CG_PUBLIC_HEADERS
    cg.hpp
    storage/vector_view.hpp
    matrix/dense_matrix.hpp
    matrix/dense_matrix_view.hpp
    matrix/csr_matrix.hpp
    solver/workspace.hpp
    solver/options.hpp
    solver/conjugate_gradient.hpp
    solver/prepared_solver.hpp
    solver/solve_info.hpp
    preconditioner/jacobi.hpp
    preconditioner/kind.hpp
    backend/residual_sign.hpp
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
    include(cmake/CGCuda.cmake)
endif()

if(CG_BUILD_PYTHON)
    include(cmake/CGPython.cmake)
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
        tests/cpp/storage/test_cuda_views.cpp
        tests/cpp/storage/test_sizes.cpp
        tests/cpp/solver/test_core.cpp
        tests/cpp/matrix/test_csr.cpp
        tests/cpp/solver/test_pcg.cpp
        tests/cpp/solver/test_prepared.cpp
        tests/cpp/backend/test_blas.cpp
    )
    target_link_libraries(cg_gtests PRIVATE cg_core GTest::gtest_main)
    include(GoogleTest)
    gtest_discover_tests(cg_gtests PROPERTIES LABELS cpp)
    add_executable(cg_allocation_gtests
        tests/cpp/matrix/test_copy_assignment.cpp
        tests/cpp/solver/test_workspace_allocation.cpp
        tests/cpp/support/allocation_failure.cpp
    )
    target_include_directories(cg_allocation_gtests PRIVATE tests/cpp)
    target_link_libraries(cg_allocation_gtests PRIVATE cg_core GTest::gtest_main)
    gtest_discover_tests(cg_allocation_gtests PROPERTIES LABELS cpp)
    add_custom_target(cg_native_tests DEPENDS cg_gtests cg_allocation_gtests)
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
