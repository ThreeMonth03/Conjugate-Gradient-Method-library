cmake_minimum_required(VERSION 3.24)
project(cg_consumer LANGUAGES CXX)

option(CG_CONSUMER_CUDA "Build the CUDA consumer as well" OFF)
option(CG_CONSUMER_REQUIRE_CUDA "Require a CUDA device when running tests" OFF)

if(CG_CONSUMER_REQUIRE_CUDA AND NOT CG_CONSUMER_CUDA)
    message(FATAL_ERROR "CG_CONSUMER_REQUIRE_CUDA requires CG_CONSUMER_CUDA")
endif()

if(CG_CONSUMER_CUDA)
    find_package(cgpy CONFIG REQUIRED COMPONENTS CUDA)
else()
    find_package(cgpy CONFIG REQUIRED)
endif()

add_executable(cg_consumer main.cpp)
target_link_libraries(cg_consumer PRIVATE cg::core)

include(CTest)
if(BUILD_TESTING)
    add_test(NAME cg_consumer COMMAND cg_consumer)
endif()

if(CG_CONSUMER_CUDA)
    add_executable(cg_cuda_consumer cuda.cpp)
    target_link_libraries(cg_cuda_consumer PRIVATE cg::cuda)
    if(BUILD_TESTING)
        add_test(NAME cg_cuda_consumer COMMAND cg_cuda_consumer)
        if(NOT CG_CONSUMER_REQUIRE_CUDA)
            set_tests_properties(cg_cuda_consumer PROPERTIES SKIP_RETURN_CODE 77)
        endif()
    endif()
endif()
