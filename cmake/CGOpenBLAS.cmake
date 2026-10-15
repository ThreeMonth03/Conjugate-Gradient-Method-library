# A private, pinned provider. Never discover a system or Python-wheel BLAS.
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "CG_ENABLE_BLAS currently supports Linux with GNU/LLVM nm and objcopy")
endif()
if(CMAKE_CROSSCOMPILING)
    message(FATAL_ERROR "The private OpenBLAS provider currently requires a native build")
endif()

enable_language(C)
include(ExternalProject)
find_program(CG_BLAS_NM NAMES llvm-nm nm REQUIRED)
find_program(CG_BLAS_OBJCOPY NAMES llvm-objcopy objcopy REQUIRED)
set(CG_BLAS_BUILD_JOBS 2 CACHE STRING "Maximum jobs in the private OpenBLAS build")
if(NOT CG_BLAS_BUILD_JOBS MATCHES "^[1-9][0-9]*$")
    message(FATAL_ERROR "CG_BLAS_BUILD_JOBS must be a positive integer")
endif()

set(CG_BLAS_ROOT "${CMAKE_CURRENT_BINARY_DIR}/_deps/cg-openblas")
set(CG_BLAS_INCLUDE "${CG_BLAS_ROOT}/private/include")
set(CG_BLAS_ARCHIVE "${CG_BLAS_ROOT}/private/lib/libcgpy_openblas.a")
file(MAKE_DIRECTORY "${CG_BLAS_INCLUDE}")

# CORE2 kernels require SSE3/SSSE3; Haswell kernels are selected only at runtime
# on supporting CPUs. Other Linux architectures use
# OpenBLAS's portable native target detection.
set(CG_BLAS_ARCH_ARGS)
set(CG_BLAS_BASELINE "native target detection")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
    list(APPEND CG_BLAS_ARCH_ARGS
        -DTARGET=CORE2 -DDYNAMIC_ARCH=ON -DDYNAMIC_LIST=CORE2,HASWELL)
    set(CG_BLAS_BASELINE "CORE2 SSE3/SSSE3 baseline, runtime CORE2/HASWELL")
endif()
set(CG_BLAS_PROVIDER_INFO "OpenBLAS 0.3.30, sequential, USE_LOCKING=1, LP64, BLAS-only, prefix=cgpy_, ${CG_BLAS_BASELINE}")

ExternalProject_Add(cg_openblas_build
    PREFIX "${CG_BLAS_ROOT}/ep"
    URL https://github.com/OpenMathLib/OpenBLAS/archive/refs/tags/v0.3.30.tar.gz
    URL_HASH SHA256=27342cff518646afb4c2b976d809102e368957974c250a25ccc965e53063c95d
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SOURCE_DIR "${CG_BLAS_ROOT}/source"
    BINARY_DIR "${CG_BLAS_ROOT}/build"
    UPDATE_COMMAND ""
    LIST_SEPARATOR ,
    CMAKE_ARGS
        -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
        -DCMAKE_MAKE_PROGRAM:FILEPATH=${CMAKE_MAKE_PROGRAM}
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DCMAKE_C_VISIBILITY_PRESET=hidden
        -DBUILD_STATIC_LIBS=ON -DBUILD_SHARED_LIBS=OFF -DFIXED_LIBNAME=ON
        -DBUILD_WITHOUT_LAPACK=ON -DBUILD_WITHOUT_LAPACKE=ON
        -DNOFORTRAN=ON -DBUILD_TESTING=OFF -DBUILD_BENCHMARKS=OFF
        -DUSE_THREAD=OFF -DUSE_OPENMP=OFF -DUSE_LOCKING=ON -DNUM_THREADS=1
        -DINTERFACE64=OFF -DNO_AFFINITY=ON
        -DBUILD_DOUBLE=ON -DBUILD_SINGLE=OFF
        -DBUILD_COMPLEX=OFF -DBUILD_COMPLEX16=OFF
        ${CG_BLAS_ARCH_ARGS}
    # Upstream adds some CBLAS tests even with BUILD_TESTING=OFF; build only
    # the archive, avoiding tests for intentionally disabled precisions.
    BUILD_COMMAND "${CMAKE_COMMAND}" --build <BINARY_DIR> --config Release --target openblas_static --parallel ${CG_BLAS_BUILD_JOBS}
    INSTALL_COMMAND "${CMAKE_COMMAND}"
        -DCG_BLAS_BINARY=<BINARY_DIR>
        -DCG_BLAS_SOURCE=<SOURCE_DIR>
        -DCG_BLAS_PRIVATE=${CG_BLAS_ROOT}/private
        -DCG_BLAS_NM=${CG_BLAS_NM}
        -DCG_BLAS_OBJCOPY=${CG_BLAS_OBJCOPY}
        -P "${CMAKE_CURRENT_LIST_DIR}/NamespaceOpenBLAS.cmake"
    BUILD_BYPRODUCTS "${CG_BLAS_ARCHIVE}"
    LOG_DOWNLOAD ON LOG_CONFIGURE ON LOG_BUILD ON LOG_INSTALL ON
    LOG_OUTPUT_ON_FAILURE ON
)

add_library(cg_private_blas STATIC IMPORTED GLOBAL)
set_target_properties(cg_private_blas PROPERTIES
    IMPORTED_LOCATION "${CG_BLAS_ARCHIVE}"
    INTERFACE_INCLUDE_DIRECTORIES "${CG_BLAS_INCLUDE}"
    INTERFACE_LINK_LIBRARIES "Threads::Threads;m;${CMAKE_DL_LIBS}"
    INTERFACE_LINK_OPTIONS "LINKER:--exclude-libs,libcgpy_openblas.a"
)
add_dependencies(cg_private_blas cg_openblas_build)
# Redistribution license travels with either Python extension wheel.
install(FILES "${CG_BLAS_ROOT}/source/LICENSE"
    DESTINATION cgpy/_licenses RENAME OpenBLAS.txt COMPONENT python)
install(FILES "${CG_BLAS_ARCHIVE}"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT development)
install(FILES "${CG_BLAS_ROOT}/source/LICENSE"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/cgpy"
    RENAME OpenBLAS.txt COMPONENT development)
