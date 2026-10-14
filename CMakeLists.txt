cmake_minimum_required(VERSION 3.24)
project(cgpy VERSION 0.2.0 LANGUAGES CXX)

include(CTest)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(CG_BUILD_PYTHON "Build the Python extension modules" ON)
option(CG_ENABLE_OPENMP "Enable OpenMP CPU operations" OFF)
option(CG_ENABLE_SANITIZERS "Enable address and undefined-behavior sanitizers" OFF)
option(CG_ENABLE_CLANG_TIDY "Run clang-tidy while compiling project sources" OFF)

find_package(Threads REQUIRED)
add_library(cg_legacy STATIC
    cpp/matrix/_matrix.cpp
    cpp/cg_method/_cg_method.cpp
)
target_compile_features(cg_legacy PUBLIC cxx_std_17)
set_target_properties(cg_legacy PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(cg_legacy PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpp/matrix>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpp/cg_method>"
)
target_link_libraries(cg_legacy PUBLIC Threads::Threads)

add_library(cg_core STATIC src/storage.cpp src/backend.cpp src/solver.cpp src/csr.cpp src/preconditioner.cpp)
target_compile_features(cg_core PUBLIC cxx_std_17)
set_target_properties(cg_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(cg_core PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")
target_link_libraries(cg_legacy PUBLIC cg_core)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cg_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cg_legacy PRIVATE -Wall -Wextra -Wpedantic)
    if(NOT CG_ENABLE_OPENMP)
        target_compile_options(cg_legacy PRIVATE -Wno-unknown-pragmas)
    endif()
endif()

if(CG_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
    target_link_libraries(cg_legacy PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(cg_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(cg_legacy PUBLIC CG_ENABLE_OPENMP=1)
endif()

if(CG_ENABLE_SANITIZERS)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        message(FATAL_ERROR "CG_ENABLE_SANITIZERS requires GCC or Clang")
    endif()
    if(CG_BUILD_PYTHON)
        message(FATAL_ERROR "Use CG_BUILD_PYTHON=OFF for the native sanitizer test suite")
    endif()
    target_compile_options(cg_legacy PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(cg_legacy PUBLIC -fsanitize=address,undefined)
    target_compile_options(cg_core PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(cg_core PUBLIC -fsanitize=address,undefined)
endif()

if(CG_ENABLE_CLANG_TIDY)
    find_program(CG_CLANG_TIDY NAMES clang-tidy REQUIRED)
    set_target_properties(cg_legacy PROPERTIES CXX_CLANG_TIDY "${CG_CLANG_TIDY}")
    set_target_properties(cg_core PROPERTIES CXX_CLANG_TIDY "${CG_CLANG_TIDY}")
endif()

if(CG_BUILD_PYTHON)
    set(PYBIND11_FINDPYTHON ON)
    find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE CG_PYBIND11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY
    )
    find_package(pybind11 3.0.1 CONFIG REQUIRED PATHS "${CG_PYBIND11_DIR}" NO_DEFAULT_PATH)
    pybind11_add_module(_cgpy MODULE cpp/pybind/_pybind.cpp)
    target_link_libraries(_cgpy PRIVATE cg_legacy)
    set_target_properties(_cgpy PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/python"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/python"
    )
    install(TARGETS _cgpy LIBRARY DESTINATION . RUNTIME DESTINATION .)
    pybind11_add_module(_cgcore MODULE bindings/cpu.cpp)
    target_link_libraries(_cgcore PRIVATE cg_core)
    string(TOUPPER "${CMAKE_BUILD_TYPE}" CG_BUILD_CONFIG)
    target_compile_definitions(_cgcore PRIVATE
        CG_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
        CG_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
        CG_BUILD_FLAGS="${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${CG_BUILD_CONFIG}}"
    )
    set_target_properties(_cgcore PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/python")
    install(TARGETS _cgcore LIBRARY DESTINATION . RUNTIME DESTINATION .)
    install(FILES python/utils.py DESTINATION .)
endif()

if(BUILD_TESTING)
    include(FetchContent)
    # Every download and extracted dependency stays in this build tree.
    set(FETCHCONTENT_BASE_DIR "${CMAKE_CURRENT_BINARY_DIR}/_deps" CACHE PATH "Local dependency directory")
    FetchContent_Declare(googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
        URL_HASH SHA256=65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    add_executable(cg_gtests tests/cpp/test_legacy.cpp tests/cpp/test_core.cpp tests/cpp/test_csr.cpp tests/cpp/test_pcg.cpp)
    target_link_libraries(cg_gtests PRIVATE cg_legacy cg_core GTest::gtest_main)
    include(GoogleTest)
    gtest_discover_tests(cg_gtests PROPERTIES LABELS cpp)

    if(CG_BUILD_PYTHON)
        add_test(NAME pytest
            COMMAND "${Python_EXECUTABLE}" -m pytest "${CMAKE_CURRENT_SOURCE_DIR}/tests" -q
        )
        if(WIN32)
            set(CG_TEST_PYTHONPATH "${CMAKE_CURRENT_BINARY_DIR}/python\\;${CMAKE_CURRENT_SOURCE_DIR}/python")
        else()
            set(CG_TEST_PYTHONPATH "${CMAKE_CURRENT_BINARY_DIR}/python:${CMAKE_CURRENT_SOURCE_DIR}/python")
        endif()
        set_tests_properties(pytest PROPERTIES
            LABELS python
            ENVIRONMENT "PYTHONPATH=${CG_TEST_PYTHONPATH}"
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        )
    endif()
endif()
