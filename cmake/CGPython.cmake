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
